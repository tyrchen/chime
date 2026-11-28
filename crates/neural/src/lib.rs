//! On-device Causal Small Language Model (SLM) reranker powered by Apple Silicon Metal (`mlx-rs`)
//! with persistent `MlxPrefixCache` and actor-isolated background worker for sub-5ms rescoring.

#![forbid(unsafe_code)]

pub mod beam_decoder;
pub mod mlx_backend;
pub mod pinyin_trie;
pub mod transformer;
pub mod worker;

use std::{collections::HashMap, path::Path};

use chime_phonetics::{FuzzyConfig, ShuangpinScheme};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::{
    beam_decoder::{
        BeamSearchConfig, NeuralBeamCandidate, RouteABeamCache, decode_constrained_beam,
    },
    mlx_backend::{
        LoraMountSpec, LoraRouting, MlxBeamKv, MlxLoraAdapter, MlxPrefixCache, MlxTransformer,
    },
    pinyin_trie::{ChannelConfig, TokenEdge, TokenPinyinIndex},
    transformer::{
        ModelConfig, PrefixKvCache, QuantizationConfig, TinyModelBuilder, TransformerWeights,
    },
    worker::{AsyncRerankHandle, RerankJob, RerankResult, RouteADecodeRequest, RouteAResult},
};

/// Scoring mode controlling right-boundary punctuation closure and logit calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoringMode {
    /// Complete candidate phrase or sentence: applies right-boundary punctuation closure.
    #[default]
    ClosedSentence,
    /// Mid-sentence single-syllable or OOV stepwise continuation: disables right-boundary
    /// punctuation closure (`closure_weight = 0.0`) while retaining full contextual scale (`1.18`)
    /// so verbs, copulas (`是`, `有`, `在`), and cross-boundary characters (`玉` + `桂`) are never
    /// penalized for not ending a clause.
    OpenContextual,
    /// Lightweight mid-beam prefix guidance (`closure_weight = 0.0`, `scale = 0.42`).
    BeamPrefixGuidance,
}

/// Errors raised when loading or running a `.safetensors` neural reranker bundle.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NeuralError {
    /// Failed to read model files from disk.
    #[error("neural model I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Invalid `config.json` or `vocab.json`.
    #[error("invalid neural config/vocab JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Invalid or missing tensor in `model.safetensors` or MLX runtime failure.
    #[error("safetensors / MLX error: {0}")]
    Safetensors(String),
}

/// Compact vocabulary mapping UTF-8 word/character/subword pieces to token indices
/// (`0..vocab_size`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactVocab {
    pieces: HashMap<String, u32>,
    unk_token: u32,
    bos_token: u32,
}

impl CompactVocab {
    /// Constructs a vocabulary from an ordered slice of token strings (`[BOS, UNK, ...]`).
    #[must_use]
    pub fn from_tokens(tokens: &[String]) -> Self {
        let mut pieces = HashMap::with_capacity(tokens.len());
        for (idx, tok) in tokens.iter().enumerate() {
            if let Ok(id) = u32::try_from(idx) {
                pieces.insert(tok.clone(), id);
            }
        }
        Self {
            pieces,
            bos_token: 0,
            unk_token: 1.min(u32::try_from(tokens.len().saturating_sub(1)).unwrap_or(0)),
        }
    }

    /// Greedy longest-match subword/character tokenizer over `text` (up to 6-char pieces).
    #[must_use]
    pub fn encode(&self, text: &str, prepend_bos: bool) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.chars().count() + usize::from(prepend_bos));
        if prepend_bos {
            ids.push(self.bos_token);
        }
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            if chars[i].is_whitespace() {
                i += 1;
                continue;
            }
            let max_span = (chars.len() - i).min(6);
            let mut matched = None;
            for span in (1..=max_span).rev() {
                let piece: String = chars[i..i + span].iter().collect();
                if let Some(&id) = self.pieces.get(&piece) {
                    matched = Some((id, span));
                    break;
                }
            }
            if let Some((id, span)) = matched {
                ids.push(id);
                i += span;
            } else {
                ids.push(self.unk_token);
                i += 1;
            }
        }
        ids
    }

    /// Looks up the exact token ID for a single piece (e.g. clause-boundary punctuation).
    #[must_use]
    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.pieces.get(piece).copied()
    }
}

/// Thread-confined Apple Silicon Metal (`mlx-rs`) Qwen2.5 reranker owned exclusively by the worker
/// actor.
#[derive(Debug)]
pub struct SafetensorsReranker {
    mlx_model: MlxTransformer,
    vocab: CompactVocab,
    kv_cache: MlxPrefixCache,
    beam_cache: RouteABeamCache,
    pinyin_index: Option<TokenPinyinIndex>,
    closure_token_ids: Vec<u32>,
    closure_weight: f32,
    weight_scale: f32,
}

impl SafetensorsReranker {
    /// Loads a `Chime` neural reranker directory (`config.json`, `vocab.json`, `model.safetensors`)
    /// directly onto Apple Silicon Metal unified memory!
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError`] if any required file or tensor is missing or malformed.
    pub fn load_dir(model_dir: &Path) -> Result<Self, NeuralError> {
        Self::load_dir_with_loras(model_dir, &[])
    }

    /// Loads the frozen base neural reranker directory and mounts zero or more `Multi-LoRA` sidecar
    /// adapters (`lora.safetensors`) onto Apple Silicon Metal unified memory!
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError`] if loading the base model fails. Corrupt optional `LoRAs` are
    /// skipped.
    pub fn load_dir_with_loras(
        model_dir: &Path,
        loras: &[LoraMountSpec],
    ) -> Result<Self, NeuralError> {
        for candidate in [
            Some(model_dir.join("mlx.metallib")),
            model_dir.parent().map(|p| p.join("mlx.metallib")),
        ]
        .into_iter()
        .flatten()
        {
            if candidate.is_file()
                && let Some(path_str) = candidate.to_str()
            {
                let _ = mlx_rs::metal::set_metallib_path(path_str);
                break;
            }
        }

        let cfg_bytes = std::fs::read(model_dir.join("config.json"))?;
        let config: ModelConfig = serde_json::from_slice(&cfg_bytes)?;

        let vocab_bytes = std::fs::read(model_dir.join("vocab.json"))?;
        let tokens: Vec<String> = serde_json::from_slice(&vocab_bytes)?;
        let vocab = CompactVocab::from_tokens(&tokens);
        let pinyin_index = if model_dir.join("token_pinyin.json").is_file() {
            TokenPinyinIndex::load_from_dir(model_dir, tokens).ok()
        } else {
            None
        };
        let closure_token_ids: Vec<u32> = ["。", "，", "？", "！"]
            .into_iter()
            .filter_map(|p| vocab.token_id(p))
            .collect();
        let closure_weight = 1.05_f32;

        let mut mlx_model =
            MlxTransformer::load_safetensors_file(&config, &model_dir.join("model.safetensors"))?;
        for spec in loras {
            if spec.path.exists() {
                let _ = mlx_model.mount_lora_safetensors(spec);
            }
        }
        let mut kv_cache = MlxPrefixCache::new();
        // Warm up all Metal GPU shader shape buckets (`past_kv=None` causal, `past_kv=Some`
        // single-step, and `past_kv=Some` multi-step causal + closure gather across typical batch
        // sizes) during background initialization so the first keystroke never hits Metal JIT!
        if config.vocab_size >= 2 {
            let t1 = 1.min(u32::try_from(config.vocab_size - 1).unwrap_or(0));
            let default_scales: Vec<f32> = loras.iter().map(|s| s.scale).collect();
            let _ =
                mlx_model.ensure_prefix_cached_with_scales(&[0], &default_scales, &mut kv_cache);
            let _ = mlx_model.ensure_prefix_cached_with_scales(
                &[0, t1, t1],
                &default_scales,
                &mut kv_cache,
            );
            let _ = mlx_model.ensure_prefix_cached_with_scales(
                &[t1; 16],
                &default_scales,
                &mut kv_cache,
            );
            for b in [2usize, 6, 8] {
                let dummy_batch: Vec<Vec<u32>> = vec![vec![0, t1, t1, 0]; b];
                let _ = mlx_model.score_candidates_with_closure(
                    &kv_cache,
                    &dummy_batch,
                    &default_scales,
                    &closure_token_ids,
                    closure_weight,
                );
            }
            kv_cache = MlxPrefixCache::new();
        }

        Ok(Self {
            mlx_model,
            vocab,
            kv_cache,
            beam_cache: RouteABeamCache::default(),
            pinyin_index,
            closure_token_ids,
            closure_weight,
            weight_scale: 1.02,
        })
    }

    /// Returns true if `token_pinyin.json` was loaded and Route A constrained neural beam
    /// decoding is available.
    #[must_use]
    pub const fn has_constrained_decoder(&self) -> bool {
        self.pinyin_index.is_some()
    }

    /// Returns true if the loaded model bundle uses full-sentence Pinyin lookahead conditioning
    /// (`pinyin_char_v1`).
    #[must_use]
    pub fn is_pinyin_conditioned(&self) -> bool {
        self.mlx_model.config().is_pinyin_conditioned()
    }

    fn build_route_a_prefix_ids(
        &self,
        left_context: &str,
        raw_input: &str,
        scheme: ShuangpinScheme,
    ) -> Vec<u32> {
        let trimmed_ctx = left_context.trim();
        let unk = self.vocab.unk_token;
        if self.is_pinyin_conditioned() {
            // Exact `pinyin_char_v1` contract:
            // `[0] + ctx_ids + [51('|'), 52('~'), scheme_tid, 51('|')] + char_ids(X) + [52('~'),
            // 50('=')]`
            let mut prefix_ids = Vec::with_capacity(raw_input.len() + 48);
            prefix_ids.push(self.vocab.bos_token);
            if !trimmed_ctx.is_empty() {
                let chars: Vec<char> = trimmed_ctx.chars().collect();
                let start = chars.len().saturating_sub(88);
                let slice: String = chars[start..].iter().collect();
                let mut c_ids = self.vocab.encode(&slice, false);
                c_ids.retain(|&id| id != unk);
                if c_ids.len() > 96 {
                    let drop = c_ids.len() - 96;
                    c_ids.drain(..drop);
                }
                prefix_ids.extend_from_slice(&c_ids);
            }
            let bar = self.vocab.token_id("|").unwrap_or(51);
            let tilde = self.vocab.token_id("~").unwrap_or(52);
            let eq = self.vocab.token_id("=").unwrap_or(50);
            let scheme_str = match scheme {
                ShuangpinScheme::Full => "p",
                ShuangpinScheme::Xiaohe => "x",
                ShuangpinScheme::Ziranma => "z",
                ShuangpinScheme::Microsoft => "m",
            };
            let scheme_tid = self
                .vocab
                .token_id(scheme_str)
                .unwrap_or(self.vocab.bos_token);
            prefix_ids.extend_from_slice(&[bar, tilde, scheme_tid, bar]);
            let mut char_buf = [0u8; 4];
            for ch in raw_input.to_ascii_lowercase().chars() {
                let s = ch.encode_utf8(&mut char_buf);
                if let Some(tid) = self.vocab.token_id(s) {
                    prefix_ids.push(tid);
                } else if ch == ';'
                    && let Some(tid) = self.vocab.token_id(":")
                {
                    prefix_ids.push(tid);
                }
            }
            prefix_ids.extend_from_slice(&[tilde, eq]);
            return prefix_ids;
        }

        let has_left_ctx = !trimmed_ctx.is_empty();
        let bounded_ctx: String = if has_left_ctx {
            let chars: Vec<char> = trimmed_ctx.chars().collect();
            let start = chars.len().saturating_sub(96);
            let slice: String = chars[start..].iter().collect();
            if slice.starts_with('。') {
                slice
            } else {
                format!("。{slice}")
            }
        } else {
            "。，".to_string()
        };
        let mut prefix_ids = self.vocab.encode(&bounded_ctx, false);
        prefix_ids.retain(|&id| id != unk);
        if prefix_ids.is_empty() {
            prefix_ids.push(self.vocab.bos_token);
        }
        prefix_ids
    }

    /// Runs Route A (First-Principles Pinyin-Constrained Autoregressive Neural Beam Search)
    /// directly inside the on-device MLX Transformer + Multi-LoRA!
    #[allow(clippy::too_many_arguments)]
    pub fn decode_constrained(
        &mut self,
        left_context: &str,
        app_bundle_id: &str,
        raw_input: &str,
        scheme: ShuangpinScheme,
        fuzzy: FuzzyConfig,
        pangu_spacing: bool,
        mode: ScoringMode,
        beam_width: usize,
    ) -> Vec<NeuralBeamCandidate> {
        let Some(pinyin_index) = &self.pinyin_index else {
            return Vec::new();
        };
        let is_pc = self.is_pinyin_conditioned();
        let prefix_ids = self.build_route_a_prefix_ids(left_context, raw_input, scheme);
        let lora_scales = self
            .mlx_model
            .compute_lora_scales(app_bundle_id, left_context, &[]);

        let eos_slice = [self.vocab.bos_token];
        let (eff_closure_ids, eff_closure_w): (&[u32], f32) = if is_pc {
            (&eos_slice, 1.0)
        } else {
            (&self.closure_token_ids, self.closure_weight)
        };
        let cache_opt = if is_pc {
            None
        } else {
            Some(&mut self.beam_cache)
        };

        decode_constrained_beam(
            &self.mlx_model,
            &mut self.kv_cache,
            pinyin_index,
            left_context,
            app_bundle_id,
            &prefix_ids,
            &lora_scales,
            eff_closure_ids,
            eff_closure_w,
            raw_input,
            scheme,
            fuzzy,
            pangu_spacing,
            mode,
            beam_width,
            is_pc,
            cache_opt,
        )
    }

    /// Dual-Proposal Exact Bayesian Decoding: combines Route A autoregressive BPE neural beam
    /// search with complementary Stage-1 word-level Viterbi proposals (`Exact`, `Fuzzy`,
    /// `TypoCorrected`), scoring every novel proposal under the exact same Bayesian objective
    /// `log P_LM(Y | C) + closure + exact_channel_cost(X | Y)`.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn decode_constrained_with_proposals(
        &mut self,
        left_context: &str,
        app_bundle_id: &str,
        raw_input: &str,
        scheme: ShuangpinScheme,
        fuzzy: FuzzyConfig,
        pangu_spacing: bool,
        mode: ScoringMode,
        beam_width: usize,
        extra_stage1_candidates: &[(String, f32, bool)],
    ) -> Vec<NeuralBeamCandidate> {
        let mut ncands = self.decode_constrained(
            left_context,
            app_bundle_id,
            raw_input,
            scheme,
            fuzzy,
            pangu_spacing,
            mode,
            beam_width,
        );
        if extra_stage1_candidates.is_empty() {
            return ncands;
        }

        let mut to_score: Vec<(String, f32, bool)> = Vec::new();
        let ch_cfg = pinyin_trie::ChannelConfig::default();
        let mut primary_lattice: Option<Vec<Vec<pinyin_trie::TokenEdge>>> = None;
        let mut neighbor_lattice: Option<Vec<Vec<pinyin_trie::TokenEdge>>> = None;
        let has_vowel_pinyin = raw_input.bytes().any(|b| {
            matches!(
                b.to_ascii_lowercase(),
                b'a' | b'e' | b'i' | b'o' | b'u' | b'v'
            )
        });
        let unk = self.vocab.unk_token;
        for (txt, fallback_chan, fallback_exact) in extra_stage1_candidates {
            if ncands.iter().any(|c| c.text == *txt)
                || to_score.iter().any(|(existing, _, _)| existing == txt)
            {
                continue;
            }
            let aligned = self.pinyin_index.as_ref().and_then(|idx| {
                let prim = primary_lattice.get_or_insert_with(|| {
                    idx.build_token_lattice_with_config(raw_input, scheme, fuzzy, false, ch_cfg)
                });
                if let Some(res) = idx.exact_channel_cost_with_lattice(raw_input, txt, prim) {
                    return Some(res);
                }
                if scheme == ShuangpinScheme::Full {
                    let neigh = neighbor_lattice.get_or_insert_with(|| {
                        idx.build_token_lattice_with_config(
                            raw_input,
                            scheme,
                            fuzzy,
                            false,
                            pinyin_trie::ChannelConfig {
                                enable_canonical_neighbor_typos: true,
                                canonical_neighbor_window: None,
                                ..ch_cfg
                            },
                        )
                    });
                    return idx.exact_channel_cost_with_lattice(raw_input, txt, neigh);
                }
                None
            });
            let (chan_cost, is_exact) = if let Some(res) = aligned {
                res
            } else {
                // Skip unaligned non-exact Stage-1 proposals (`Fuzzy`/`TypoCorrected` like
                // `nanning -> 南京`) that neither `primary_lattice` nor `neighbor_lattice`
                // accepted, as well as standalone single-char archaic/bound ghost readings from
                // `core.imedict` (`kuai -> 会`, `ji -> 给`) that `token_pinyin.json` excluded,
                // while keeping multi-char Stage-1 exact compounds (`会计`, `便宜`, `给予`) intact!
                let is_single_cjk = txt.chars().count() == 1
                    && txt.chars().all(|c| ('\u{4E00}'..='\u{9FA5}').contains(&c));
                let all_in_vocab = !self.vocab.encode(txt, false).contains(&unk);
                if !ncands.is_empty()
                    && all_in_vocab
                    && (!*fallback_exact || (has_vowel_pinyin && is_single_cjk))
                {
                    continue;
                }
                (*fallback_chan, *fallback_exact)
            };
            to_score.push((txt.clone(), chan_cost, is_exact));
        }

        if !to_score.is_empty() {
            let bcfg = beam_decoder::BeamSearchConfig::default();
            let lower_raw = raw_input.to_ascii_lowercase();
            let is_single_syl = scheme == ShuangpinScheme::Full
                && !raw_input.contains('\'')
                && chime_phonetics::is_valid_syllable(&lower_raw);
            let eff_mode = if is_single_syl
                || (!left_context.trim().is_empty()
                    && raw_input.len() <= bcfg.contextual_short_phrase_max_bytes)
            {
                ScoringMode::OpenContextual
            } else {
                mode
            };
            let texts: Vec<String> = to_score.iter().map(|(t, _, _)| t.clone()).collect();
            let lps = self.score_candidates_exact_bayesian_with_pinyin(
                left_context,
                app_bundle_id,
                raw_input,
                scheme,
                &texts,
                eff_mode,
            );
            let mut extra_scored: Vec<NeuralBeamCandidate> = Vec::new();
            for ((txt, chan_cost, is_exact), lp) in to_score.into_iter().zip(lps) {
                if lp.is_finite() {
                    extra_scored.push(NeuralBeamCandidate {
                        text: txt,
                        score: lp + chan_cost,
                        lm_log_prob: lp,
                        channel_cost: chan_cost,
                        is_exact,
                        consumed_bytes: raw_input.len(),
                    });
                }
            }
            if !extra_scored.is_empty() {
                // Apply the same Cache-LM left-context repetition prior to Stage-1 proposals so
                // Route A and Stage-1 candidates compete on an identical Bayesian manifold!
                beam_decoder::apply_left_context_cache_bonus(left_context, &mut extra_scored);
                ncands.extend(extra_scored);
                ncands.sort_by(|a, b| b.score.total_cmp(&a.score));
                beam_decoder::apply_pronoun_symmetry(left_context, &mut ncands);
            }
        }
        ncands
    }

    /// Computes exact unnormalized Bayesian log-likelihoods `log P_LM(Y | C) (+ closure)` on the
    /// exact same `"。，"` / `left_context` manifold as [`Self::decode_constrained`], used to score
    /// exact-pinyin multi-character lexicon / `.imepack` / custom phrase hypotheses without any
    /// static unigram or bigram bias!
    pub fn score_candidates_exact_bayesian(
        &mut self,
        left_context: &str,
        app_bundle_id: &str,
        candidate_texts: &[String],
        mode: ScoringMode,
    ) -> Vec<f32> {
        self.score_candidates_exact_bayesian_with_pinyin(
            left_context,
            app_bundle_id,
            "",
            ShuangpinScheme::Full,
            candidate_texts,
            mode,
        )
    }

    /// Computes exact unnormalized Bayesian log-likelihoods `log P(Y | C, X) (+ closure)` matching
    /// both legacy text-LM and `pinyin_char_v1` Pinyin-conditioned bundles.
    pub fn score_candidates_exact_bayesian_with_pinyin(
        &mut self,
        left_context: &str,
        app_bundle_id: &str,
        raw_input: &str,
        scheme: ShuangpinScheme,
        candidate_texts: &[String],
        mode: ScoringMode,
    ) -> Vec<f32> {
        if candidate_texts.is_empty() {
            return Vec::new();
        }
        let is_pc = self.is_pinyin_conditioned();
        let unk = self.vocab.unk_token;
        let prefix_ids = self.build_route_a_prefix_ids(left_context, raw_input, scheme);

        let lora_scales =
            self.mlx_model
                .compute_lora_scales(app_bundle_id, left_context, candidate_texts);

        if self
            .mlx_model
            .ensure_prefix_cached_with_scales(&prefix_ids, &lora_scales, &mut self.kv_cache)
            .is_err()
        {
            return vec![f32::NEG_INFINITY; candidate_texts.len()];
        }

        let mut valid_mask = Vec::with_capacity(candidate_texts.len());
        let mut encoded = Vec::with_capacity(candidate_texts.len());
        for text in candidate_texts {
            let ids = self.vocab.encode(text, false);
            let valid = !ids.is_empty() && !ids.contains(&unk);
            valid_mask.push(valid);
            encoded.push(if valid { ids } else { Vec::new() });
        }

        let eos_slice = [self.vocab.bos_token];
        let (eff_closure_ids, eff_closure_weight): (&[u32], f32) = if is_pc {
            (&eos_slice, 1.0)
        } else {
            let w = match mode {
                ScoringMode::ClosedSentence => self.closure_weight,
                ScoringMode::OpenContextual | ScoringMode::BeamPrefixGuidance => 0.0,
            };
            (&self.closure_token_ids, w)
        };
        let Ok(joint_lps) = self.mlx_model.score_candidates_with_closure(
            &self.kv_cache,
            &encoded,
            &lora_scales,
            eff_closure_ids,
            eff_closure_weight,
        ) else {
            return vec![f32::NEG_INFINITY; candidate_texts.len()];
        };

        joint_lps
            .into_iter()
            .zip(valid_mask)
            .map(|(lp, ok)| if ok { lp } else { f32::NEG_INFINITY })
            .collect()
    }

    /// Number of mounted `Multi-LoRA` sidecar adapters.
    #[must_use]
    pub fn lora_count(&self) -> usize {
        self.mlx_model.lora_count()
    }

    /// Computes zero-meaned conditional log-likelihood bonuses `log P(cand | left_context)` on
    /// Metal GPU while mutating the owned `MlxPrefixCache` in place (zero locks!).
    pub fn score_candidates(&mut self, left_context: &str, candidate_texts: &[String]) -> Vec<f32> {
        self.score_candidates_with_app(left_context, "", candidate_texts)
    }

    /// Computes zero-meaned conditional log-likelihood bonuses with dynamic `Multi-LoRA` routing
    /// conditioned on `(app_bundle_id, left_context, candidate_texts)`.
    pub fn score_candidates_with_app(
        &mut self,
        left_context: &str,
        app_bundle_id: &str,
        candidate_texts: &[String],
    ) -> Vec<f32> {
        self.score_candidates_with_mode(left_context, app_bundle_id, candidate_texts, false)
    }

    /// Computes zero-meaned conditional log-likelihood bonuses, optionally disabling right-boundary
    /// punctuation closure when `open_prefix` is true (used for mid-sentence beam guidance).
    pub fn score_candidates_with_mode(
        &mut self,
        left_context: &str,
        app_bundle_id: &str,
        candidate_texts: &[String],
        open_prefix: bool,
    ) -> Vec<f32> {
        let mode = if open_prefix {
            ScoringMode::BeamPrefixGuidance
        } else {
            ScoringMode::ClosedSentence
        };
        self.score_candidates_with_scoring_mode(left_context, app_bundle_id, candidate_texts, mode)
    }

    /// Computes zero-meaned conditional log-likelihood bonuses under an explicit [`ScoringMode`].
    #[allow(clippy::too_many_lines)]
    pub fn score_candidates_with_scoring_mode(
        &mut self,
        left_context: &str,
        app_bundle_id: &str,
        candidate_texts: &[String],
        mode: ScoringMode,
    ) -> Vec<f32> {
        if candidate_texts.is_empty() {
            return Vec::new();
        }
        let trimmed_ctx = left_context.trim();
        let has_left_ctx = !trimmed_ctx.is_empty();
        let unk = self.vocab.unk_token;

        let mut valid_mask = Vec::with_capacity(candidate_texts.len());
        let mut char_lens = Vec::with_capacity(candidate_texts.len());
        let mut encoded = Vec::with_capacity(candidate_texts.len());
        let mut max_char_len = 0usize;
        for text in candidate_texts {
            let ids = self.vocab.encode(text, false);
            let c_len = text.chars().filter(|c| !c.is_whitespace()).count().max(1);
            let valid = !ids.is_empty() && !ids.contains(&unk);
            if valid && text.chars().any(|c| ('\u{4E00}'..='\u{9FA5}').contains(&c)) {
                max_char_len = max_char_len.max(c_len);
            }
            valid_mask.push(valid);
            char_lens.push(c_len);
            if valid {
                encoded.push(ids);
            } else {
                encoded.push(Vec::new());
            }
        }

        // For short 1-3 character standalone words with no left context (`zidao`, `gen`, `sihu`),
        // Stage-1 `core.imedict` already holds calibrated multi-corpus + 28-layer Qwen3 priors!
        let valid_count = valid_mask.iter().filter(|&&v| v).count();
        if valid_count < 2 || (!has_left_ctx && max_char_len <= 3) {
            return vec![0.0; candidate_texts.len()];
        }

        // Bound sliding context window to the trailing 96 characters and always anchor RoPE
        // position 0 with `"。"` so prefix-conditioned scoring shares the exact same positional
        // manifold as standalone scoring and `Qwen3-6L` distillation!
        let bounded_ctx: String = if has_left_ctx {
            let chars: Vec<char> = trimmed_ctx.chars().collect();
            let start = chars.len().saturating_sub(96);
            let slice: String = chars[start..].iter().collect();
            if slice.starts_with('。') {
                slice
            } else {
                format!("。{slice}")
            }
        } else {
            "。".to_string()
        };

        let mut prefix_ids = self.vocab.encode(&bounded_ctx, false);
        prefix_ids.retain(|&id| id != unk);
        if prefix_ids.is_empty() {
            prefix_ids.push(self.vocab.bos_token);
        }

        let lora_scales =
            self.mlx_model
                .compute_lora_scales(app_bundle_id, left_context, candidate_texts);

        // Candidate Longest Common Prefix (LCP) KV-Cache reuse: when all valid candidates share
        // the same character length and at least 1 common leading token (typical for long-sentence
        // homophone beams where only the last 2-3 chars diverge), prefill `left_context + LCP`
        // once at batch=1 and score only the diverging suffixes! Mathematically 100% exact.
        let (min_valid_clen, max_valid_clen, min_tok_len) = encoded
            .iter()
            .zip(char_lens.iter())
            .zip(valid_mask.iter())
            .filter(|&((_, _), &v)| v)
            .fold(
                (usize::MAX, 0usize, usize::MAX),
                |(min_c, max_c, min_t), ((ids, &clen), _)| {
                    (min_c.min(clen), max_c.max(clen), min_t.min(ids.len()))
                },
            );
        let lcp_len = if min_valid_clen == max_valid_clen && min_tok_len >= 2 {
            let max_lcp = min_tok_len - 1;
            let mut valid_iter = encoded
                .iter()
                .zip(valid_mask.iter())
                .filter_map(|(ids, &v)| v.then_some(ids.as_slice()));
            if let Some(first) = valid_iter.next() {
                let mut common = max_lcp;
                for other in valid_iter {
                    common = first[..common]
                        .iter()
                        .zip(other.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    if common == 0 {
                        break;
                    }
                }
                common
            } else {
                0
            }
        } else {
            0
        };

        let (effective_prefix, scoring_encoded): (Vec<u32>, Vec<Vec<u32>>) = if lcp_len > 0 {
            let first_valid = encoded
                .iter()
                .zip(valid_mask.iter())
                .find_map(|(ids, &v)| v.then_some(ids.as_slice()))
                .unwrap_or(&[]);
            let mut ext_prefix = Vec::with_capacity(prefix_ids.len() + lcp_len);
            ext_prefix.extend_from_slice(&prefix_ids);
            ext_prefix.extend_from_slice(&first_valid[..lcp_len]);
            let suffixes = encoded
                .iter()
                .zip(valid_mask.iter())
                .map(|(ids, &v)| {
                    if v {
                        ids[lcp_len..].to_vec()
                    } else {
                        Vec::new()
                    }
                })
                .collect();
            (ext_prefix, suffixes)
        } else {
            (prefix_ids, encoded.clone())
        };

        if self
            .mlx_model
            .ensure_prefix_cached_with_scales(&effective_prefix, &lora_scales, &mut self.kv_cache)
            .is_err()
        {
            return vec![0.0; candidate_texts.len()];
        }

        let eff_closure_weight = match mode {
            ScoringMode::ClosedSentence => self.closure_weight,
            ScoringMode::OpenContextual | ScoringMode::BeamPrefixGuidance => 0.0,
        };
        let Ok(joint_lps) = self.mlx_model.score_candidates_with_closure(
            &self.kv_cache,
            &scoring_encoded,
            &lora_scales,
            &self.closure_token_ids,
            eff_closure_weight,
        ) else {
            return vec![0.0; candidate_texts.len()];
        };

        // Lossless-CJK vocabularies (`>= 32K`, retaining 100% of all Qwen3 CJK BPE tokens) have
        // no pruned-word subword-split inflation, so `split_pen = 0.0`; compact pruned `< 32K`
        // vocabularies retain the `1.85` subword-compression compensation.
        let split_pen = if self.mlx_model.config().vocab_size >= 32_768 {
            0.0
        } else {
            1.85
        };
        let (norm_scores, mean_lp) = Self::normalize_with_bpe_compression_prior(
            joint_lps,
            &char_lens,
            &encoded,
            &valid_mask,
            valid_count,
            split_pen,
        );

        let eff_scale = match mode {
            ScoringMode::BeamPrefixGuidance => 0.42,
            ScoringMode::OpenContextual => 1.18,
            ScoringMode::ClosedSentence => {
                if has_left_ctx {
                    1.18
                } else {
                    self.weight_scale
                }
            }
        };

        norm_scores
            .into_iter()
            .zip(valid_mask)
            .map(|(lp, valid)| {
                if valid {
                    (lp - mean_lp) * eff_scale
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn normalize_with_bpe_compression_prior(
        joint_lps: Vec<f32>,
        char_lens: &[usize],
        encoded: &[Vec<u32>],
        valid_mask: &[bool],
        valid_count: usize,
        split_pen: f32,
    ) -> (Vec<f32>, f32) {
        #[allow(clippy::cast_precision_loss)]
        let mean_clen: f32 = char_lens
            .iter()
            .zip(valid_mask.iter())
            .filter_map(|(&c, &v)| v.then_some(c as f32))
            .sum::<f32>()
            / (valid_count as f32);
        #[allow(clippy::cast_precision_loss)]
        let min_tok_per_char: f32 = encoded
            .iter()
            .zip(char_lens.iter())
            .zip(valid_mask.iter())
            .filter_map(|((ids, &clen), &v)| {
                (v && clen >= 2).then_some((ids.len() as f32) / (clen as f32))
            })
            .fold(1.0_f32, f32::min);
        let norm_scores: Vec<f32> = joint_lps
            .into_iter()
            .zip(char_lens.iter())
            .zip(encoded.iter())
            .map(|((lp, &clen), ids)| {
                #[allow(clippy::cast_precision_loss)]
                let rel_denom = ((clen as f32) / mean_clen.max(1.0)).powf(0.42);
                #[allow(clippy::cast_precision_loss)]
                let extra_splits = if clen >= 2 {
                    ((ids.len() as f32) - min_tok_per_char * (clen as f32)).max(0.0)
                } else {
                    0.0
                };
                (lp - split_pen * extra_splits) / rel_denom
            })
            .collect();
        #[allow(clippy::cast_precision_loss)]
        let mean_lp: f32 = norm_scores
            .iter()
            .zip(valid_mask.iter())
            .filter_map(|(&s, &v)| v.then_some(s))
            .sum::<f32>()
            / (valid_count as f32);
        (norm_scores, mean_lp)
    }

    /// Returns the loaded model configuration.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        self.mlx_model.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_load_safetensors_on_mlx_metal_and_batch_score() {
        let dir = tempfile::tempdir().unwrap_or_else(|_| std::process::abort());
        TinyModelBuilder::write_demo_bundle(dir.path(), &[("端侧", "量化")])
            .unwrap_or_else(|_| std::process::abort());

        let mut reranker =
            SafetensorsReranker::load_dir(dir.path()).unwrap_or_else(|_| std::process::abort());

        let cands = vec!["量化".to_string(), "亮化".to_string()];
        let scores1 = reranker.score_candidates("端侧模型", &cands);
        let scores2 = reranker.score_candidates("端侧模型", &cands);
        assert_eq!(scores1.len(), 2);
        assert!((scores1[0] - scores2[0]).abs() < 1e-4);
        assert!(
            scores1[0] > scores1[1],
            "trained pair 量化 should outscore 亮化 on MLX Metal: {scores1:?}"
        );
    }

    #[test]
    fn test_should_mount_multi_lora_sidecar_and_shift_scores_with_routing() {
        let dir = tempfile::tempdir().unwrap_or_else(|_| std::process::abort());
        TinyModelBuilder::write_demo_bundle(dir.path(), &[("端侧", "量化")])
            .unwrap_or_else(|_| std::process::abort());

        let mut base_reranker =
            SafetensorsReranker::load_dir(dir.path()).unwrap_or_else(|_| std::process::abort());
        let d = i32::try_from(base_reranker.config().hidden_size).unwrap_or(16);
        let cands = vec!["量化".to_string(), "亮化".to_string()];
        let base_scores =
            base_reranker.score_candidates_with_app("端侧模型", "md.obsidian", &cands);

        // Synthesize a Rank-4 LoRA adapter across layer 0 `gate_proj`, `up_proj`, and `down_proj`!
        let inter = i32::try_from(base_reranker.config().intermediate_size).unwrap_or(32);
        let rank = 4_i32;
        let mlp_in_low: Vec<f32> = (0..usize::try_from(rank * d).unwrap_or(64))
            .map(|i| if i % 2 == 0 { 0.45 } else { -0.25 })
            .collect();
        let mlp_out_low: Vec<f32> = (0..usize::try_from(inter * rank).unwrap_or(128))
            .map(|i| if i % 3 == 0 { 0.55 } else { 0.15 })
            .collect();
        let down_in_low: Vec<f32> = (0..usize::try_from(rank * inter).unwrap_or(128))
            .map(|i| if i % 3 == 0 { 0.4 } else { -0.2 })
            .collect();
        let down_out_low: Vec<f32> = (0..usize::try_from(d * rank).unwrap_or(64))
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.35 })
            .collect();
        let lora_path = dir.path().join("lora.safetensors");
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.layers.0.mlp.gate_proj.lora_A".to_string(),
            mlx_rs::Array::from_slice(&mlp_in_low, &[rank, d]),
        );
        tensors.insert(
            "model.layers.0.mlp.gate_proj.lora_B".to_string(),
            mlx_rs::Array::from_slice(&mlp_out_low, &[inter, rank]),
        );
        tensors.insert(
            "model.layers.0.mlp.up_proj.lora_A".to_string(),
            mlx_rs::Array::from_slice(&mlp_in_low, &[rank, d]),
        );
        tensors.insert(
            "model.layers.0.mlp.up_proj.lora_B".to_string(),
            mlx_rs::Array::from_slice(&mlp_out_low, &[inter, rank]),
        );
        tensors.insert(
            "model.layers.0.mlp.down_proj.lora_A".to_string(),
            mlx_rs::Array::from_slice(&down_in_low, &[rank, inter]),
        );
        tensors.insert(
            "model.layers.0.mlp.down_proj.lora_B".to_string(),
            mlx_rs::Array::from_slice(&down_out_low, &[d, rank]),
        );
        mlx_rs::Array::save_safetensors(&tensors, None, &lora_path)
            .unwrap_or_else(|_| std::process::abort());

        let spec = LoraMountSpec {
            id: "dev-test".to_string(),
            path: lora_path,
            scale: 2.0,
            routing: LoraRouting::Developer,
        };
        let mut lora_reranker = SafetensorsReranker::load_dir_with_loras(dir.path(), &[spec])
            .unwrap_or_else(|_| std::process::abort());
        assert_eq!(lora_reranker.lora_count(), 1);

        let prose_scores =
            lora_reranker.score_candidates_with_app("端侧模型", "md.obsidian", &cands);
        let dev_scores =
            lora_reranker.score_candidates_with_app("端侧模型", "com.mitchellh.ghostty", &cands);
        assert!(
            (dev_scores[0] - base_scores[0]).abs() > 1e-4,
            "mounted LoRA should alter Metal log-probs: base={base_scores:?} dev={dev_scores:?}"
        );
        assert!(
            (dev_scores[0] - prose_scores[0]).abs() > 1e-5,
            "Developer routing should apply stronger weight in Ghostty than Obsidian"
        );
    }

    #[test]
    fn test_batched_candidate_scoring_matches_individual_and_poetic_cadence_routing() {
        let dir = tempfile::tempdir().unwrap_or_else(|_| std::process::abort());
        TinyModelBuilder::write_demo_bundle(dir.path(), &[("今天", "使用"), ("使用", "模型")])
            .unwrap_or_else(|_| std::process::abort());
        let reranker =
            SafetensorsReranker::load_dir(dir.path()).unwrap_or_else(|_| std::process::abort());
        let mut cache = MlxPrefixCache::new();
        reranker
            .mlx_model
            .ensure_prefix_cached(&[0, 2, 3], &mut cache)
            .unwrap_or_else(|_| std::process::abort());

        let batch = vec![vec![2_u32], vec![2, 3], vec![3, 2, 3, 2]];
        let batched_lps = reranker
            .mlx_model
            .score_candidates_joint_lps(&cache, &batch)
            .unwrap_or_else(|_| std::process::abort());
        for (idx, single) in batch.iter().enumerate() {
            let solo = reranker
                .mlx_model
                .score_candidates_joint_lps(&cache, std::slice::from_ref(single))
                .unwrap_or_else(|_| std::process::abort());
            assert!(
                (batched_lps[idx] - solo[0]).abs() < 5e-3,
                "batched score {} != solo {} at idx {idx}",
                batched_lps[idx],
                solo[0]
            );
        }

        // Verify 5/7-char poetic cadence routing vs colloquial modern Mandarin
        let poem_scale = LoraRouting::Classical.effective_scale(
            1.0,
            "md.obsidian",
            "空山新雨后，",
            &["天气晚来秋".to_string()],
        );
        let colloq_scale = LoraRouting::Classical.effective_scale(
            1.0,
            "md.obsidian",
            "今天吃什么，",
            &["火锅".to_string()],
        );
        assert!(
            poem_scale >= 0.70,
            "expected poetic cadence activation, got {poem_scale}"
        );
        assert!(
            colloq_scale.abs() < 1e-6,
            "expected 0.0 on colloquial prose, got {colloq_scale}"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_route_a_constrained_beam_and_incremental_kv_cache() {
        let dir = tempfile::tempdir().unwrap_or_else(|_| std::process::abort());
        TinyModelBuilder::write_demo_bundle(dir.path(), &[("今天", "使用"), ("使用", "模型")])
            .unwrap_or_else(|_| std::process::abort());

        let vocab_bytes =
            std::fs::read(dir.path().join("vocab.json")).unwrap_or_else(|_| std::process::abort());
        let vocab_tokens: Vec<String> =
            serde_json::from_slice(&vocab_bytes).unwrap_or_else(|_| std::process::abort());
        let tid_of = |s: &str| -> u32 {
            u32::try_from(vocab_tokens.iter().position(|t| t == s).unwrap_or(0)).unwrap_or(0)
        };
        let tp_json = serde_json::json!({
            "vocab_size": vocab_tokens.len(),
            "cjk": [
                [tid_of("今天"), [["jin", "tian"]]],
                [tid_of("使用"), [["shi", "yong"]]],
                [tid_of("模型"), [["mo", "xing"]]],
                [tid_of("端侧"), [["duan", "ce"]]],
                [tid_of("量化"), [["liang", "hua"]]],
                [tid_of("亮化"), [["liang", "hua"]]]
            ],
            "trad": [],
            "en": [],
            "en_multi": []
        });
        std::fs::write(
            dir.path().join("token_pinyin.json"),
            serde_json::to_string(&tp_json).unwrap_or_else(|_| std::process::abort()),
        )
        .unwrap_or_else(|_| std::process::abort());

        let mut slm_fresh =
            SafetensorsReranker::load_dir(dir.path()).unwrap_or_else(|_| std::process::abort());
        assert!(slm_fresh.has_constrained_decoder());

        let fresh = slm_fresh.decode_constrained(
            "",
            "md.obsidian",
            "jintianshiyongmoxing",
            ShuangpinScheme::Full,
            FuzzyConfig::default(),
            true,
            ScoringMode::ClosedSentence,
            6,
        );
        assert!(
            fresh.first().is_some_and(|c| c.text == "今天使用模型"),
            "Route A should decode trained sequence: {fresh:?}"
        );

        // Incremental keystroke path (`jintianshiyongmo` -> `jintianshiyongmoxing`) must match
        // fresh decode via `RouteABeamCache`!
        let mut slm_inc =
            SafetensorsReranker::load_dir(dir.path()).unwrap_or_else(|_| std::process::abort());
        let _prefix = slm_inc.decode_constrained(
            "",
            "md.obsidian",
            "jintianshiyongmo",
            ShuangpinScheme::Full,
            FuzzyConfig::default(),
            true,
            ScoringMode::ClosedSentence,
            6,
        );
        let inc = slm_inc.decode_constrained(
            "",
            "md.obsidian",
            "jintianshiyongmoxing",
            ShuangpinScheme::Full,
            FuzzyConfig::default(),
            true,
            ScoringMode::ClosedSentence,
            6,
        );
        assert_eq!(inc.first().map(|c| c.text.as_str()), Some("今天使用模型"));
        assert!(
            (inc[0].score - fresh[0].score).abs() < 5e-3,
            "incremental RouteABeamCache score {} != fresh {}",
            inc[0].score,
            fresh[0].score
        );

        // Divergent sentence sharing a >6-byte prefix (`jintianshi...`) must never be corrupted by
        // `RouteABeamCache` (regression guard for `feikuaidesudu...` -> `feikuaidipaoxiang...`).
        let diverged = slm_inc.decode_constrained(
            "",
            "md.obsidian",
            "jintianshiyongduance",
            ShuangpinScheme::Full,
            FuzzyConfig::default(),
            true,
            ScoringMode::ClosedSentence,
            6,
        );
        let diverged_fresh = slm_fresh.decode_constrained(
            "",
            "md.obsidian",
            "jintianshiyongduance",
            ShuangpinScheme::Full,
            FuzzyConfig::default(),
            true,
            ScoringMode::ClosedSentence,
            6,
        );
        assert_eq!(
            diverged.first().map(|c| c.text.as_str()),
            Some("今天使用端侧"),
            "divergent prefix decode should not lose cross-boundary prefix tokens"
        );
        assert_eq!(
            diverged.first().map(|c| c.text.as_str()),
            diverged_fresh.first().map(|c| c.text.as_str()),
            "divergent prefix decode diverged from fresh decode"
        );
        assert!(
            (diverged[0].score - diverged_fresh[0].score).abs() < 5e-3,
            "divergent prefix score {} != fresh {}",
            diverged[0].score,
            diverged_fresh[0].score
        );
    }
}
