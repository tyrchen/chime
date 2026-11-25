//! On-device Causal Small Language Model (SLM) reranker powered by Apple Silicon Metal (`mlx-rs`)
//! with persistent `MlxPrefixCache` and actor-isolated background worker for sub-5ms rescoring.

#![forbid(unsafe_code)]

pub mod mlx_backend;
pub mod transformer;
pub mod worker;

use std::{collections::HashMap, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::{
    mlx_backend::{MlxPrefixCache, MlxTransformer},
    transformer::{ModelConfig, PrefixKvCache, TinyModelBuilder, TransformerWeights},
    worker::{AsyncRerankHandle, RerankJob, RerankResult},
};

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
}

/// Thread-confined Apple Silicon Metal (`mlx-rs`) Qwen2.5 reranker owned exclusively by the worker
/// actor.
#[derive(Debug)]
pub struct SafetensorsReranker {
    mlx_model: MlxTransformer,
    vocab: CompactVocab,
    kv_cache: MlxPrefixCache,
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
        let cfg_bytes = std::fs::read(model_dir.join("config.json"))?;
        let config: ModelConfig = serde_json::from_slice(&cfg_bytes)?;

        let vocab_bytes = std::fs::read(model_dir.join("vocab.json"))?;
        let tokens: Vec<String> = serde_json::from_slice(&vocab_bytes)?;
        let vocab = CompactVocab::from_tokens(&tokens);

        let mlx_model =
            MlxTransformer::load_safetensors_file(&config, &model_dir.join("model.safetensors"))?;
        let mut kv_cache = MlxPrefixCache::new();
        // Warm up all Metal GPU shader shape buckets (`past_kv=None` causal, `past_kv=Some`
        // single-step, and `past_kv=Some` multi-step causal) during background initialization!
        if config.vocab_size >= 2 {
            let t1 = 1.min(u32::try_from(config.vocab_size - 1).unwrap_or(0));
            let _ = mlx_model.ensure_prefix_cached(&[0], &mut kv_cache);
            let _ = mlx_model.ensure_prefix_cached(&[0, t1, t1], &mut kv_cache);
            let _ = mlx_model.score_candidates_batch(
                &kv_cache,
                &[vec![0, t1], vec![0, t1, t1, 0]],
                true,
            );
            kv_cache = MlxPrefixCache::new();
        }

        Ok(Self {
            mlx_model,
            vocab,
            kv_cache,
            weight_scale: 0.42,
        })
    }

    /// Computes zero-meaned conditional log-likelihood bonuses `log P(cand | left_context)` on
    /// Metal GPU while mutating the owned `MlxPrefixCache` in place (zero locks!).
    pub fn score_candidates(&mut self, left_context: &str, candidate_texts: &[String]) -> Vec<f32> {
        if candidate_texts.is_empty() {
            return Vec::new();
        }
        let has_left_ctx = !left_context.trim().is_empty();
        let unk = self.vocab.unk_token;

        let mut valid_mask = Vec::with_capacity(candidate_texts.len());
        let mut encoded = Vec::with_capacity(candidate_texts.len());
        for text in candidate_texts {
            let ids = self.vocab.encode(text, false);
            let valid = !ids.is_empty() && !ids.contains(&unk) && (has_left_ctx || ids.len() >= 2);
            valid_mask.push(valid);
            if valid {
                encoded.push(ids);
            } else {
                encoded.push(Vec::new());
            }
        }

        // Fast path: when left_context is empty and all candidates are single lexicon words (or OOV
        // identifiers), Stage-1 corpus frequencies are already exact — skip GPU work in 0 µs!
        let valid_count = valid_mask.iter().filter(|&&v| v).count();
        if valid_count < 2 {
            return vec![0.0; candidate_texts.len()];
        }

        let prefix_ids = self.vocab.encode(left_context, true);
        if self
            .mlx_model
            .ensure_prefix_cached(&prefix_ids, &mut self.kv_cache)
            .is_err()
        {
            return vec![0.0; candidate_texts.len()];
        }

        let Ok(raw_scores) =
            self.mlx_model
                .score_candidates_batch(&self.kv_cache, &encoded, has_left_ctx)
        else {
            return vec![0.0; candidate_texts.len()];
        };

        // Zero-mean center across valid candidates so `<unk>` / skipped candidates stay neutral
        // (0.0) and length offsets never penalize valid words against skipped entries!
        #[allow(clippy::cast_precision_loss)]
        let mean_lp: f32 = raw_scores
            .iter()
            .zip(valid_mask.iter())
            .filter_map(|(&s, &v)| v.then_some(s))
            .sum::<f32>()
            / (valid_count as f32);

        let scale = if has_left_ctx {
            self.weight_scale
        } else {
            self.weight_scale * 0.55
        };

        raw_scores
            .into_iter()
            .zip(valid_mask)
            .map(|(lp, valid)| if valid { (lp - mean_lp) * scale } else { 0.0 })
            .collect()
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
}
