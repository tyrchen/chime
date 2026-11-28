//! Optional Confidence-Gated Two-Tier Local MLX Reranker (`Qwen3-1.7B/4B-Base` 4-bit MLX).
//!
//! When Tier-1 (`BI-18L`, 190 MB) is uncertain (`score[0] - score[1] < margin_threshold`),
//! [`Tier2Reranker`] scores the top-$K$ already-aligned candidates in a single batched Apple
//! Silicon Metal forward pass over a shared left-context [`MlxPrefixCache`], interpolating
//! `(1 - alpha) * lm_tier1 + alpha * lm_tier2 + chan_tier1 + extra` while protecting Tier-1's
//! grammatical `{的, 地, 得}` calibration.

use std::{collections::HashMap, fs, path::Path};

use chime_phonetics::ShuangpinScheme;
use serde::{Deserialize, Serialize};

use crate::{
    MlxPrefixCache, MlxTransformer, ModelConfig, NeuralBeamCandidate, NeuralError, ScoringMode,
};

/// Configuration for confidence-gated Two-Tier local reranking.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tier2Config {
    /// Escalate to Tier-2 only when `cands[0].score - cands[1].score < margin_threshold`.
    pub margin_threshold: f32,
    /// Interpolation weight for Tier-2 LM log-probability (`0.65` => `0.35 * T1 + 0.65 * T2`).
    pub alpha: f32,
    /// Maximum number of top Tier-1 candidates to rescore in one batched GPU forward pass.
    pub top_k: usize,
    /// Right-boundary utterance-closure weight in `ClosedSentence` mode.
    pub closure_weight: f32,
    /// Preserve Tier-1's grammatical `{的, 地, 得}` decision when candidates differ only in those
    /// three structural particles.
    pub guard_de_di_de: bool,
}

impl Default for Tier2Config {
    fn default() -> Self {
        Self {
            margin_threshold: 3.0,
            alpha: 0.65,
            top_k: 5,
            closure_weight: 0.25,
            guard_de_di_de: true,
        }
    }
}

/// Pure-Rust `HuggingFace` GPT-2 / Qwen3 byte-level BPE tokenizer (`vocab.json` + `merges.txt`)
/// with zero external regex/tokenizer crate dependencies.
#[derive(Debug, Clone)]
pub struct HfBpeTokenizer {
    vocab: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    bos_token_id: u32,
}

impl HfBpeTokenizer {
    /// Loads `vocab.json` and `merges.txt` from a standard `HuggingFace` / `mlx-lm` model
    /// directory.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError`] if `vocab.json` or `merges.txt` cannot be read or parsed.
    pub fn load_dir(model_dir: &Path) -> Result<Self, NeuralError> {
        let vocab_bytes = fs::read(model_dir.join("vocab.json"))?;
        let vocab: HashMap<String, u32> = serde_json::from_slice(&vocab_bytes)?;

        let merges_text = fs::read_to_string(model_dir.join("merges.txt"))?;
        let mut merges = HashMap::with_capacity(merges_text.lines().count());
        let mut rank = 0usize;
        for line in merges_text.lines() {
            let trimmed = line.trim_end_matches('\r');
            if trimmed.is_empty() || trimmed.starts_with("#version:") {
                continue;
            }
            if let Some((left, right)) = trimmed.split_once(' ') {
                merges.insert((left.to_string(), right.to_string()), rank);
                rank += 1;
            }
        }

        let bos_token_id = vocab
            .get("<|endoftext|>")
            .or_else(|| vocab.get("<|im_start|>"))
            .copied()
            .unwrap_or(151_643);

        Ok(Self {
            vocab,
            merges,
            byte_encoder: build_gpt2_byte_encoder(),
            bos_token_id,
        })
    }

    /// Encodes `text` using Qwen3 byte-level BPE pre-tokenization and merge ranks.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 2 + 4);
        for chunk in pretokenize_qwen3(text) {
            self.encode_chunk(chunk, &mut out);
        }
        out
    }

    /// Looks up a single exact token string (UTF-8 converted via GPT-2 byte mapping).
    #[must_use]
    pub fn encode_single_piece(&self, piece: &str) -> Option<u32> {
        let encoded: String = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_encoder[usize::from(b)])
            .collect();
        self.vocab.get(&encoded).copied()
    }

    /// Returns the fallback BOS/EOS token ID.
    #[must_use]
    pub const fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    fn encode_chunk(&self, chunk: &str, out: &mut Vec<u32>) {
        if chunk.is_empty() {
            return;
        }
        let mut word: Vec<String> = chunk
            .as_bytes()
            .iter()
            .map(|&b| self.byte_encoder[usize::from(b)].to_string())
            .collect();

        while word.len() >= 2 {
            let mut best_rank = usize::MAX;
            let mut best_pair_idx = None;
            for i in 0..(word.len() - 1) {
                // Borrow lookup without reallocating every probe when possible.
                if let Some(&r) = self.merges.get(&(word[i].clone(), word[i + 1].clone()))
                    && r < best_rank
                {
                    best_rank = r;
                    best_pair_idx = Some(i);
                }
            }
            let Some(idx) = best_pair_idx else {
                break;
            };
            let first = word[idx].clone();
            let second = word[idx + 1].clone();
            let merged = format!("{first}{second}");

            let mut next = Vec::with_capacity(word.len() - 1);
            let mut i = 0usize;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    next.push(merged.clone());
                    i += 2;
                } else {
                    next.push(word[i].clone());
                    i += 1;
                }
            }
            word = next;
        }

        for piece in word {
            if let Some(&id) = self.vocab.get(&piece) {
                out.push(id);
            }
        }
    }
}

fn build_gpt2_byte_encoder() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0u16..=255 {
        let byte = u8::try_from(b).unwrap_or(0);
        let is_direct = matches!(byte, 33..=126 | 161..=172 | 174..=255);
        if is_direct {
            table[usize::from(byte)] = char::from(byte);
        } else {
            table[usize::from(byte)] = char::from_u32(256 + extra).unwrap_or('?');
            extra += 1;
        }
    }
    table
}

#[inline]
fn is_qwen_letter(c: char) -> bool {
    c.is_alphabetic()
}

#[inline]
fn is_qwen_number(c: char) -> bool {
    c.is_numeric()
}

#[inline]
fn is_qwen_punct_or_sym(c: char) -> bool {
    !c.is_whitespace() && !is_qwen_letter(c) && !is_qwen_number(c)
}

fn match_ascii_contraction(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first().copied() != Some(b'\'') || bytes.len() < 2 {
        return None;
    }
    if bytes.len() >= 3 {
        let b1 = bytes[1].to_ascii_lowercase();
        let b2 = bytes[2].to_ascii_lowercase();
        if matches!((b1, b2), (b'r' | b'v', b'e') | (b'l', b'l')) {
            return Some(3);
        }
    }
    let b1 = bytes[1].to_ascii_lowercase();
    if matches!(b1, b's' | b't' | b'm' | b'd') {
        return Some(2);
    }
    None
}

/// Qwen2 / Qwen3 byte-level BPE pre-tokenizer matching the official cl100k-derived regex:
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
/// ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
#[allow(clippy::too_many_lines)]
fn pretokenize_qwen3(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut i = 0usize;
    let n = text.len();

    while i < n {
        let rest = &text[i..];
        // 1. English contraction
        if let Some(len) = match_ascii_contraction(rest) {
            chunks.push(&text[i..i + len]);
            i += len;
            continue;
        }

        let mut it = rest.char_indices();
        let Some((_, c0)) = it.next() else {
            break;
        };
        let c0_len = c0.len_utf8();
        let c1_opt = it.next().map(|(_, c)| c);

        // 2. `[^\r\n\p{L}\p{N}]?\p{L}+`
        if is_qwen_letter(c0) {
            let mut end = i + c0_len;
            for (off, ch) in rest[c0_len..].char_indices() {
                if is_qwen_letter(ch) {
                    end = i + c0_len + off + ch.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..end]);
            i = end;
            continue;
        }
        if c0 != '\r' && c0 != '\n' && !is_qwen_number(c0) && c1_opt.is_some_and(is_qwen_letter) {
            let mut end = i + c0_len;
            for (off, ch) in rest[c0_len..].char_indices() {
                if is_qwen_letter(ch) {
                    end = i + c0_len + off + ch.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..end]);
            i = end;
            continue;
        }

        // 3. Single digit `\p{N}`
        if is_qwen_number(c0) {
            chunks.push(&text[i..i + c0_len]);
            i += c0_len;
            continue;
        }

        // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        let starts_punct =
            is_qwen_punct_or_sym(c0) || (c0 == ' ' && c1_opt.is_some_and(is_qwen_punct_or_sym));
        if starts_punct {
            let mut pos = if c0 == ' ' { i + 1 } else { i };
            while pos < n {
                if let Some(ch) = text[pos..].chars().next()
                    && is_qwen_punct_or_sym(ch)
                {
                    pos += ch.len_utf8();
                } else {
                    break;
                }
            }
            while pos < n {
                if let Some(ch) = text[pos..].chars().next()
                    && (ch == '\r' || ch == '\n')
                {
                    pos += ch.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..pos]);
            i = pos;
            continue;
        }

        // 5 & 6. Whitespace: `\s*[\r\n]+` or `\s+(?!\S)` or `\s+`
        if c0.is_whitespace() {
            let mut ws_end = i;
            let mut last_nl_end = None;
            for (off, ch) in rest.char_indices() {
                if ch.is_whitespace() {
                    let e = i + off + ch.len_utf8();
                    ws_end = e;
                    if ch == '\r' || ch == '\n' {
                        last_nl_end = Some(e);
                    }
                } else {
                    break;
                }
            }
            if let Some(nl_end) = last_nl_end {
                chunks.push(&text[i..nl_end]);
                i = nl_end;
            } else if ws_end < n {
                // `\s+(?!\S)`: leave the last whitespace char for the following token if >= 2 chars
                let ws_slice = &text[i..ws_end];
                if let Some((last_off, _)) = ws_slice.char_indices().last()
                    && last_off > 0
                {
                    chunks.push(&text[i..i + last_off]);
                    i += last_off;
                } else {
                    chunks.push(&text[i..ws_end]);
                    i = ws_end;
                }
            } else {
                chunks.push(&text[i..ws_end]);
                i = ws_end;
            }
            continue;
        }

        chunks.push(&text[i..i + c0_len]);
        i += c0_len;
    }

    chunks
}

/// Returns `true` when `a` and `b` have identical character length, differ in at least one
/// position, and EVERY differing character pair belongs exclusively to `{的, 地, 得}`.
#[must_use]
pub fn is_pure_de_di_de_pair(a: &str, b: &str) -> bool {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    if ac.len() != bc.len() {
        return false;
    }
    let mut diffs = 0usize;
    for (&ca, &cb) in ac.iter().zip(bc.iter()) {
        if ca != cb {
            if matches!(ca, '的' | '地' | '得') && matches!(cb, '的' | '地' | '得') {
                diffs += 1;
            } else {
                return false;
            }
        }
    }
    diffs > 0
}

/// Confidence-gated Tier-2 local MLX reranker owning a larger quantized Qwen3-Base model (`1.7B`
/// or `4B`) with persistent shared left-context [`MlxPrefixCache`].
#[derive(Debug)]
pub struct Tier2Reranker {
    mlx_model: MlxTransformer,
    tokenizer: HfBpeTokenizer,
    kv_cache: MlxPrefixCache,
    closure_token_ids: Vec<u32>,
    config: Tier2Config,
    escalations: u64,
    total_queries: u64,
}

impl Tier2Reranker {
    /// Loads an MLX 4-bit Qwen3-Base model directory (`config.json`, `vocab.json`, `merges.txt`,
    /// `model.safetensors` or sharded index) onto Apple Silicon Metal unified memory.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError`] if loading config, tokenizer, or safetensors weights fails.
    pub fn load_dir(model_dir: &Path, config: Tier2Config) -> Result<Self, NeuralError> {
        let cfg_bytes = fs::read(model_dir.join("config.json"))?;
        let model_cfg: ModelConfig = serde_json::from_slice(&cfg_bytes)?;
        let tokenizer = HfBpeTokenizer::load_dir(model_dir)?;
        let mlx_model = MlxTransformer::load_safetensors_dir(&model_cfg, model_dir)?;

        let closure_token_ids: Vec<u32> = ["。", "，", "！", "？", "\n"]
            .into_iter()
            .filter_map(|p| tokenizer.encode_single_piece(p))
            .collect();

        let mut kv_cache = MlxPrefixCache::new();
        // Warm up prefix prefill and batched `[K, L]` Metal shaders during load.
        let warm_prefix = tokenizer.encode("。，");
        if !warm_prefix.is_empty() {
            let _ = mlx_model.ensure_prefix_cached_with_scales(&warm_prefix, &[], &mut kv_cache);
            let warm_cands = vec![
                tokenizer.encode("今天天气不错"),
                tokenizer.encode("今天天气不差"),
            ];
            let _ = mlx_model.score_candidates_with_closure(
                &kv_cache,
                &warm_cands,
                &[],
                &closure_token_ids,
                config.closure_weight,
            );
            kv_cache = MlxPrefixCache::new();
        }

        Ok(Self {
            mlx_model,
            tokenizer,
            kv_cache,
            closure_token_ids,
            config,
            escalations: 0,
            total_queries: 0,
        })
    }

    /// Returns the active [`Tier2Config`].
    #[must_use]
    pub const fn config(&self) -> Tier2Config {
        self.config
    }

    /// Updates [`Tier2Config`] in place without reloading model weights.
    pub const fn set_config(&mut self, config: Tier2Config) {
        self.config = config;
    }

    /// Returns `(escalations, total_queries)` observed by this instance.
    #[must_use]
    pub const fn stats(&self) -> (u64, u64) {
        (self.escalations, self.total_queries)
    }

    /// Returns `true` if Tier-2 should escalate for `cands` based on the configured confidence
    /// margin and `{的, 地, 得}` grammatical guard (microsecond CPU check, zero GPU work).
    #[must_use]
    pub fn should_escalate(&self, cands: &[NeuralBeamCandidate]) -> bool {
        let k = cands.len().min(self.config.top_k);
        if k < 2 {
            return false;
        }
        let margin = cands[0].score - cands[1].score;
        if !margin.is_finite() || margin >= self.config.margin_threshold {
            return false;
        }
        if self.config.guard_de_di_de
            && is_pure_de_di_de_pair(&cands[0].text, &cands[1].text)
            && (k < 3 || cands[0].score - cands[2].score >= self.config.margin_threshold)
        {
            return false;
        }
        true
    }

    /// Conditionally reranks `cands` in place when Tier-1's Top-1/Top-2 margin falls below
    /// `config.margin_threshold`. Returns `true` if Tier-2 ran on this query.
    #[allow(clippy::too_many_lines)]
    pub fn rerank_candidates(
        &mut self,
        left_context: &str,
        raw_input: &str,
        scheme: ShuangpinScheme,
        mode: ScoringMode,
        cands: &mut [NeuralBeamCandidate],
    ) -> bool {
        self.total_queries += 1;
        let k = cands.len().min(self.config.top_k);
        if !self.should_escalate(cands) {
            return false;
        }

        // Keep the most recent 48 chars (~2-3 clauses) of left context so cold-prefix prefill on
        // 36-layer 4B stays fast without losing local discourse cues.
        let trimmed_ctx = left_context.trim();
        let prefix_str = if trimmed_ctx.is_empty() {
            "。，".to_string()
        } else {
            let chars: Vec<char> = trimmed_ctx.chars().collect();
            let start = chars.len().saturating_sub(48);
            let slice: String = chars[start..].iter().collect();
            if slice.starts_with('。') {
                slice
            } else {
                format!("。{slice}")
            }
        };

        let mut prefix_ids = self.tokenizer.encode(&prefix_str);
        if prefix_ids.is_empty() {
            prefix_ids.push(self.tokenizer.bos_token_id());
        }
        if self
            .mlx_model
            .ensure_prefix_cached_with_scales(&prefix_ids, &[], &mut self.kv_cache)
            .is_err()
        {
            return false;
        }

        let is_open = mode != ScoringMode::ClosedSentence
            || (!trimmed_ctx.is_empty()
                && (raw_input.len() <= 12
                    || (scheme == ShuangpinScheme::Full
                        && !raw_input.contains('\'')
                        && chime_phonetics::is_valid_syllable(&raw_input.to_ascii_lowercase()))));
        let eff_closure_weight = if is_open {
            0.0
        } else {
            self.config.closure_weight
        };

        // Fast-path candidate tokenization: when `prefix_str` ends with sentence punctuation or
        // whitespace, Qwen3's pre-tokenizer never merges across the boundary, avoiding 5x full
        // re-tokenization of `prefix_str`.
        let boundary_isolated = prefix_str.chars().last().is_some_and(|ch| {
            matches!(
                ch,
                '。' | '，' | '！' | '？' | '；' | '：' | '、' | '”' | '）' | '】' | '》'
            ) || ch.is_whitespace()
        });
        let mut cand_seqs: Vec<Vec<u32>> = Vec::with_capacity(k);
        for cand in &cands[..k] {
            let seq = if boundary_isolated {
                self.tokenizer.encode(&cand.text)
            } else {
                let combined = format!("{prefix_str}{}", cand.text);
                let full_ids = self.tokenizer.encode(&combined);
                if full_ids.len() > prefix_ids.len()
                    && full_ids[..prefix_ids.len()] == prefix_ids[..]
                {
                    full_ids[prefix_ids.len()..].to_vec()
                } else {
                    self.tokenizer.encode(&cand.text)
                }
            };
            cand_seqs.push(seq);
        }

        let Ok(t2_lps) = self.mlx_model.score_candidates_with_closure(
            &self.kv_cache,
            &cand_seqs,
            &[],
            &self.closure_token_ids,
            eff_closure_weight,
        ) else {
            return false;
        };

        self.escalations += 1;
        let orig_top1_text = cands[0].text.clone();
        let alpha = self.config.alpha.clamp(0.0, 1.0);

        let mut rescored: Vec<(usize, f32, f32)> = Vec::with_capacity(k);
        for (idx, (cand, &lp2)) in cands[..k].iter().zip(t2_lps.iter()).enumerate() {
            if !lp2.is_finite() {
                rescored.push((idx, cand.score, cand.lm_log_prob));
                continue;
            }
            let extra = cand.score - (cand.lm_log_prob + cand.channel_cost);
            let new_lm = (1.0 - alpha) * cand.lm_log_prob + alpha * lp2;
            let new_score = new_lm + cand.channel_cost + extra;
            rescored.push((idx, new_score, new_lm));
        }

        // Guard against raw pre-trained Tier-2 flipping Tier-1's calibrated `{的, 地, 得}` choice.
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for &(idx, sc, _) in &rescored {
            if sc > best_score {
                best_score = sc;
                best_idx = idx;
            }
        }
        if self.config.guard_de_di_de
            && best_idx != 0
            && is_pure_de_di_de_pair(&orig_top1_text, &cands[best_idx].text)
        {
            return true;
        }

        for (idx, new_score, new_lm) in rescored {
            cands[idx].score = new_score;
            cands[idx].lm_log_prob = new_lm;
        }
        cands[..k].sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.text.cmp(&b.text))
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_de_di_de_pair_detects_structural_particles_only() {
        assert!(is_pure_de_di_de_pair("开心地笑了起来", "开心的笑了起来"));
        assert!(is_pure_de_di_de_pair(
            "他说得非常有道理",
            "他说的非常有道理"
        ));
        assert!(!is_pure_de_di_de_pair("开心地笑了起来", "开心地笑了起来"));
        assert!(!is_pure_de_di_de_pair("我的眼镜", "我的眼睛"));
        assert!(!is_pure_de_di_de_pair("急得团团转", "记得团团转"));
    }

    #[test]
    fn pretokenizer_matches_qwen3_boundaries() {
        assert_eq!(pretokenize_qwen3("。，"), vec!["。，"]);
        assert_eq!(pretokenize_qwen3("。今天天气不错"), vec!["。今天天气不错"]);
        assert_eq!(
            pretokenize_qwen3("。，在 VSCode 里调试代码"),
            vec!["。，", "在", " VSCode", " 里调试代码"]
        );
        assert_eq!(pretokenize_qwen3("123元"), vec!["1", "2", "3", "元"]);
        assert_eq!(
            pretokenize_qwen3("it's Rust/Swift!"),
            vec!["it", "'s", " Rust", "/Swift", "!"]
        );
    }

    #[test]
    fn gpt2_byte_encoder_maps_space_and_newline() {
        let enc = build_gpt2_byte_encoder();
        assert_eq!(enc[usize::from(b' ')], 'Ġ');
        assert_eq!(enc[usize::from(b'\n')], 'Ċ');
        assert_eq!(enc[usize::from(b'A')], 'A');
    }
}
