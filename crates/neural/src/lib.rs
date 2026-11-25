//! On-device Causal Small Language Model (SLM) reranker with persistent `PrefixKvCache`
//! and non-blocking background worker for sub-10ms candidate rescoring.

#![forbid(unsafe_code)]

pub mod transformer;
pub mod worker;

use std::{collections::HashMap, path::Path, sync::Mutex};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::{
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
    /// Invalid or missing tensor in `model.safetensors`.
    #[error("safetensors error: {0}")]
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

/// Thread-safe on-device Causal Transformer reranker maintaining a hot `PrefixKvCache`.
#[derive(Debug)]
pub struct SafetensorsReranker {
    weights: TransformerWeights,
    vocab: CompactVocab,
    kv_cache: Mutex<PrefixKvCache>,
    weight_scale: f32,
}

impl SafetensorsReranker {
    /// Loads a `Chime` neural reranker directory containing `config.json`, `vocab.json`, and
    /// `model.safetensors`.
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

        let st_bytes = std::fs::read(model_dir.join("model.safetensors"))?;
        let weights = TransformerWeights::from_safetensors_bytes(&config, &st_bytes)?;
        let kv_cache = Mutex::new(PrefixKvCache::new(&config));

        Ok(Self {
            weights,
            vocab,
            kv_cache,
            weight_scale: 0.35,
        })
    }

    /// Computes additive conditional log-likelihood bonuses `log P(cand | left_context)`
    /// for a batch of candidate strings while reusing the cached prefix KV states!
    #[must_use]
    pub fn score_candidates(&self, left_context: &str, candidate_texts: &[String]) -> Vec<f32> {
        if candidate_texts.is_empty() {
            return Vec::new();
        }
        let prefix_ids = self.vocab.encode(left_context, true);
        let Ok(mut cache) = self.kv_cache.lock() else {
            return vec![0.0; candidate_texts.len()];
        };

        let last_prefix_hidden = self.weights.prefill_with_cache(&prefix_ids, &mut cache);

        let mut scores = Vec::with_capacity(candidate_texts.len());
        for text in candidate_texts {
            let cand_ids = self.vocab.encode(text, false);
            if cand_ids.is_empty() {
                scores.push(0.0);
                continue;
            }
            let avg_lp = self
                .weights
                .score_continuation(&cache, &last_prefix_hidden, &cand_ids);
            scores.push(avg_lp * self.weight_scale);
        }
        scores
    }

    /// Returns the loaded model configuration.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        self.weights.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_load_safetensors_prefill_kv_cache_and_batch_score() {
        let dir = tempfile::tempdir().unwrap_or_else(|_| std::process::abort());
        TinyModelBuilder::write_demo_bundle(dir.path(), &[("端侧", "量化")])
            .unwrap_or_else(|_| std::process::abort());

        let reranker =
            SafetensorsReranker::load_dir(dir.path()).unwrap_or_else(|_| std::process::abort());

        let cands = vec!["量化".to_string(), "亮化".to_string()];
        let scores1 = reranker.score_candidates("端侧模型", &cands);
        let scores2 = reranker.score_candidates("端侧模型", &cands);
        assert_eq!(scores1.len(), 2);
        assert!((scores1[0] - scores2[0]).abs() < 1e-5);
        assert!(
            scores1[0] > scores1[1],
            "trained pair 量化 should outscore 亮化: {scores1:?}"
        );
    }
}
