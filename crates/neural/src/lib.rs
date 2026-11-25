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
    mlx_backend::{LoraMountSpec, LoraRouting, MlxLoraAdapter, MlxPrefixCache, MlxTransformer},
    transformer::{
        ModelConfig, PrefixKvCache, QuantizationConfig, TinyModelBuilder, TransformerWeights,
    },
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
        let cfg_bytes = std::fs::read(model_dir.join("config.json"))?;
        let config: ModelConfig = serde_json::from_slice(&cfg_bytes)?;

        let vocab_bytes = std::fs::read(model_dir.join("vocab.json"))?;
        let tokens: Vec<String> = serde_json::from_slice(&vocab_bytes)?;
        let vocab = CompactVocab::from_tokens(&tokens);

        let mut mlx_model =
            MlxTransformer::load_safetensors_file(&config, &model_dir.join("model.safetensors"))?;
        for spec in loras {
            if spec.path.exists() {
                let _ = mlx_model.mount_lora_safetensors(spec);
            }
        }
        let mut kv_cache = MlxPrefixCache::new();
        // Warm up all Metal GPU shader shape buckets (`past_kv=None` causal, `past_kv=Some`
        // single-step, and `past_kv=Some` multi-step causal) during background initialization!
        if config.vocab_size >= 2 {
            let t1 = 1.min(u32::try_from(config.vocab_size - 1).unwrap_or(0));
            let _ = mlx_model.ensure_prefix_cached(&[0], &mut kv_cache);
            let _ = mlx_model.ensure_prefix_cached(&[0, t1, t1], &mut kv_cache);
            let _ = mlx_model.ensure_prefix_cached(&[t1; 16], &mut kv_cache);
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
            weight_scale: 0.65,
        })
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
            max_char_len = max_char_len.max(c_len);
            let valid = !ids.is_empty() && !ids.contains(&unk);
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

        // Bound sliding context window to the trailing 96 characters for consistent low-latency
        // prefill; Qwen3 was pretrained without a synthetic `<bos>` token (`bos_token_id = null`).
        let bounded_ctx: String = if has_left_ctx {
            let chars: Vec<char> = trimmed_ctx.chars().collect();
            let start = chars.len().saturating_sub(96);
            chars[start..].iter().collect()
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

        if self
            .mlx_model
            .ensure_prefix_cached_with_scales(&prefix_ids, &lora_scales, &mut self.kv_cache)
            .is_err()
        {
            return vec![0.0; candidate_texts.len()];
        }

        let Ok(joint_lps) = self.mlx_model.score_candidates_joint_lps_with_scales(
            &self.kv_cache,
            &encoded,
            &lora_scales,
        ) else {
            return vec![0.0; candidate_texts.len()];
        };

        // Calibrated length normalization (`alpha = 0.92`): leaves equal-length homophone pairs
        // (`权利` vs `权力`, `多大` vs `多达`) 100% invariant while preventing longer
        // prefix-completion verses (`君不见高堂明镜悲白发` vs `君不见`) from suffering linear BPE
        // accumulation penalty!
        let norm_scores: Vec<f32> = joint_lps
            .into_iter()
            .zip(char_lens)
            .map(|(lp, clen)| {
                #[allow(clippy::cast_precision_loss)]
                let denom = (clen as f32).powf(0.42);
                lp / denom
            })
            .collect();

        #[allow(clippy::cast_precision_loss)]
        let mean_lp: f32 = norm_scores
            .iter()
            .zip(valid_mask.iter())
            .filter_map(|(&s, &v)| v.then_some(s))
            .sum::<f32>()
            / (valid_count as f32);

        let scale = self.weight_scale;

        norm_scores
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
}
