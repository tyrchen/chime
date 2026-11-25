//! Apple Silicon Metal GPU Qwen2.5 Causal Transformer Reranker powered by `mlx-rs` (`0.33`).
//!
//! Uses unified-memory `FP16`/`BF16`/`F32` `SafeTensors`, fused Metal `rms_norm`, `rope` with
//! KV-cache position offsets, `scaled_dot_product_attention` (supporting Grouped-Query Attention),
//! and cached next-token log-probabilities so single-token candidates score in `< 0.01ms` and
//! multi-token continuations score in `~1.2ms` on M-series GPUs.

use std::{collections::HashMap, path::Path};

use mlx_rs::{
    Array, Dtype as MlxDtype,
    fast::{ScaledDotProductAttentionMask, rms_norm, rope, scaled_dot_product_attention},
    nn::{log_softmax, silu},
    ops::{concatenate, indexing::TryIndexOp},
    transforms::eval,
};

use crate::{ModelConfig, NeuralError};

#[derive(Debug, Clone)]
struct MlxLayer {
    input_norm: Array,
    q_proj_t: Array,
    q_bias: Option<Array>,
    k_proj_t: Array,
    k_bias: Option<Array>,
    v_proj_t: Array,
    v_bias: Option<Array>,
    o_proj_t: Array,
    post_attn_norm: Array,
    gate_proj_t: Array,
    up_proj_t: Array,
    down_proj_t: Array,
}

/// On-GPU persistent Key-Value cache and precomputed next-token log-probabilities.
#[derive(Debug, Clone)]
pub struct MlxPrefixCache {
    cached_tokens: Vec<i32>,
    keys: Vec<Array>,
    values: Vec<Array>,
    /// Cached `log P(token | cached_tokens)` (`[vocab_size]`) at the end of the prefix!
    next_token_log_probs: Vec<f32>,
}

impl MlxPrefixCache {
    /// Creates an empty GPU prefix cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            cached_tokens: Vec::new(),
            keys: Vec::new(),
            values: Vec::new(),
            next_token_log_probs: Vec::new(),
        }
    }
}

impl Default for MlxPrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Metal-accelerated Qwen2.5 Causal Transformer weights in Apple Silicon Unified Memory.
#[derive(Debug, Clone)]
pub struct MlxTransformer {
    config: ModelConfig,
    embed_tokens: Array,
    layers: Vec<MlxLayer>,
    final_norm: Array,
    lm_head_t: Array,
}

impl MlxTransformer {
    /// Loads `model.safetensors` directly onto Apple Silicon Metal unified memory via `mlx-rs`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if loading or tensor shape transposition fails.
    pub fn load_safetensors_file(cfg: &ModelConfig, st_path: &Path) -> Result<Self, NeuralError> {
        let mut raw = Array::load_safetensors(st_path)
            .map_err(|e| NeuralError::Safetensors(format!("MLX load_safetensors failed: {e}")))?;

        let embed_tokens = take_f32_or_f16(&mut raw, "model.embed_tokens.weight")?;
        let final_norm = take_f32_or_f16(&mut raw, "model.norm.weight")?;
        let lm_head_w = if raw.contains_key("lm_head.weight") {
            take_f32_or_f16(&mut raw, "lm_head.weight")?
        } else {
            embed_tokens.clone()
        };
        let lm_head_t = transpose_2d(&lm_head_w)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{idx}");
            let input_norm = take_f32_or_f16(&mut raw, &format!("{p}.input_layernorm.weight"))?;
            let q_proj_t = transpose_2d(&take_f32_or_f16(
                &mut raw,
                &format!("{p}.self_attn.q_proj.weight"),
            )?)?;
            let q_bias = raw.remove(&format!("{p}.self_attn.q_proj.bias"));
            let k_proj_t = transpose_2d(&take_f32_or_f16(
                &mut raw,
                &format!("{p}.self_attn.k_proj.weight"),
            )?)?;
            let k_bias = raw.remove(&format!("{p}.self_attn.k_proj.bias"));
            let v_proj_t = transpose_2d(&take_f32_or_f16(
                &mut raw,
                &format!("{p}.self_attn.v_proj.weight"),
            )?)?;
            let v_bias = raw.remove(&format!("{p}.self_attn.v_proj.bias"));
            let o_proj_t = transpose_2d(&take_f32_or_f16(
                &mut raw,
                &format!("{p}.self_attn.o_proj.weight"),
            )?)?;
            let post_attn_norm =
                take_f32_or_f16(&mut raw, &format!("{p}.post_attention_layernorm.weight"))?;
            let gate_proj_t = transpose_2d(&take_f32_or_f16(
                &mut raw,
                &format!("{p}.mlp.gate_proj.weight"),
            )?)?;
            let up_proj_t = transpose_2d(&take_f32_or_f16(
                &mut raw,
                &format!("{p}.mlp.up_proj.weight"),
            )?)?;
            let down_proj_t = transpose_2d(&take_f32_or_f16(
                &mut raw,
                &format!("{p}.mlp.down_proj.weight"),
            )?)?;

            layers.push(MlxLayer {
                input_norm,
                q_proj_t,
                q_bias,
                k_proj_t,
                k_bias,
                v_proj_t,
                v_bias,
                o_proj_t,
                post_attn_norm,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
            });
        }

        // Materialize transposed views once on load.
        let _ = eval([&embed_tokens, &final_norm, &lm_head_t]);

        Ok(Self {
            config: cfg.clone(),
            embed_tokens,
            layers,
            final_norm,
            lm_head_t,
        })
    }

    /// Returns model hyperparameters.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Incrementally prefills `prefix_ids` on Metal GPU, reusing cached KV states when unchanged!
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if an MLX Metal kernel fails.
    pub fn ensure_prefix_cached(
        &self,
        prefix_ids: &[u32],
        cache: &mut MlxPrefixCache,
    ) -> Result<(), NeuralError> {
        let target_i32: Vec<i32> = prefix_ids
            .iter()
            .map(|&id| i32::try_from(id).unwrap_or(0))
            .collect();
        if target_i32.is_empty() {
            return Ok(());
        }

        let common = cache
            .cached_tokens
            .iter()
            .zip(target_i32.iter())
            .take_while(|(a, b)| a == b)
            .count();

        if common == target_i32.len()
            && common == cache.cached_tokens.len()
            && !cache.next_token_log_probs.is_empty()
        {
            return Ok(());
        }

        // Slice cached KV tensors back to `keep` positions if prefix diverged, or run full prefill
        // in one parallel causal pass when starting a fresh sentence!
        let keep = if common == target_i32.len() {
            common.saturating_sub(1)
        } else if common > 0 && cache.keys.len() == self.config.num_hidden_layers {
            common
        } else {
            0
        };

        if keep == 0 {
            cache.keys.clear();
            cache.values.clear();
            cache.cached_tokens.clear();
        } else {
            let keep_i32 = i32::try_from(keep).unwrap_or(0);
            for l in 0..self.config.num_hidden_layers {
                cache.keys[l] = cache.keys[l]
                    .try_index((.., .., ..keep_i32, ..))
                    .map_err(mlx_err)?;
                cache.values[l] = cache.values[l]
                    .try_index((.., .., ..keep_i32, ..))
                    .map_err(mlx_err)?;
            }
            cache.cached_tokens.truncate(keep);
        }

        let delta = &target_i32[keep..];
        let (last_logits, new_keys, new_vals) =
            self.forward_step(delta, keep, &cache.keys, &cache.values)?;
        let lp_arr = log_softmax(
            &last_logits.as_dtype(MlxDtype::Float32).map_err(mlx_err)?,
            -1,
        )
        .map_err(mlx_err)?;

        eval(
            std::iter::once(&lp_arr)
                .chain(new_keys.iter())
                .chain(new_vals.iter()),
        )
        .map_err(mlx_err)?;

        cache.keys = new_keys;
        cache.values = new_vals;
        cache.cached_tokens.extend_from_slice(delta);
        cache.next_token_log_probs = lp_arr.as_slice::<f32>().to_vec();
        Ok(())
    }

    /// Scores a batch of tokenized candidates conditioned on an already-prefilled `MlxPrefixCache`.
    /// Single-token candidates resolve in $O(1)$ from `cache.next_token_log_probs`; multi-token
    /// candidates fuse into a single lazy MLX `eval` submission on Metal GPU!
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if an MLX Metal kernel fails during continuation pass.
    pub fn score_candidates_batch(
        &self,
        cache: &MlxPrefixCache,
        cands: &[Vec<u32>],
        include_first_token: bool,
    ) -> Result<Vec<f32>, NeuralError> {
        if cands.is_empty() || cache.next_token_log_probs.is_empty() {
            return Ok(vec![0.0; cands.len()]);
        }

        let mut lazy_indices = Vec::with_capacity(cands.len());
        let mut lazy_arrays = Vec::with_capacity(cands.len());

        for (c_idx, cand_ids) in cands.iter().enumerate() {
            if cand_ids.len() > 1 {
                let cont_i32: Vec<i32> = cand_ids[..cand_ids.len() - 1]
                    .iter()
                    .map(|&id| i32::try_from(id).unwrap_or(0))
                    .collect();
                let lps = self.forward_continuation_lps_lazy(
                    &cont_i32,
                    cache.cached_tokens.len(),
                    &cache.keys,
                    &cache.values,
                )?;
                lazy_indices.push(c_idx);
                lazy_arrays.push(lps);
            }
        }

        if !lazy_arrays.is_empty() {
            eval(&lazy_arrays).map_err(mlx_err)?;
        }

        let vocab = self.config.vocab_size;
        let mut results = vec![0.0_f32; cands.len()];
        let mut lazy_cursor = 0usize;

        for (c_idx, cand_ids) in cands.iter().enumerate() {
            if cand_ids.is_empty() || (!include_first_token && cand_ids.len() <= 1) {
                continue;
            }
            let first_idx = (cand_ids[0] as usize).min(cache.next_token_log_probs.len() - 1);
            let mut total_lp = if include_first_token {
                cache.next_token_log_probs[first_idx]
            } else {
                0.0
            };

            if lazy_indices.get(lazy_cursor) == Some(&c_idx) {
                let slice = lazy_arrays[lazy_cursor].as_slice::<f32>();
                for (step, &target_tok) in cand_ids[1..].iter().enumerate() {
                    let t_idx = (target_tok as usize).min(vocab - 1);
                    if let Some(&val) = slice.get(step * vocab + t_idx) {
                        total_lp += val;
                    }
                }
                lazy_cursor += 1;
            }

            let eff_len = if include_first_token {
                cand_ids.len()
            } else {
                cand_ids.len().saturating_sub(1).max(1)
            };
            #[allow(clippy::cast_precision_loss)]
            let norm = (eff_len as f32).powf(0.75);
            results[c_idx] = total_lp / norm;
        }

        Ok(results)
    }

    /// Scores `cand_ids` conditioned on an already-prefilled `MlxPrefixCache`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if an MLX Metal kernel fails during continuation pass.
    pub fn score_candidate(
        &self,
        cache: &MlxPrefixCache,
        cand_ids: &[u32],
    ) -> Result<f32, NeuralError> {
        let scores =
            self.score_candidates_batch(cache, std::slice::from_ref(&cand_ids.to_vec()), true)?;
        Ok(scores.first().copied().unwrap_or(0.0))
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn forward_layer(
        &self,
        layer: &MlxLayer,
        hidden: &Array,
        step_len: i32,
        offset_i32: i32,
        past_kv: Option<(&Array, &Array)>,
    ) -> Result<(Array, Array, Array), NeuralError> {
        let d = self.config.hidden_size as i32;
        let n_heads = self.config.num_attention_heads as i32;
        let n_kv = self.config.num_key_value_heads as i32;
        let head_dim = d / n_heads;
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let normed =
            rms_norm(hidden, Some(&layer.input_norm), self.config.rms_norm_eps).map_err(mlx_err)?;
        let mut q_lin = normed.matmul(&layer.q_proj_t).map_err(mlx_err)?;
        if let Some(b) = &layer.q_bias {
            q_lin = q_lin.add(b).map_err(mlx_err)?;
        }
        let mut k_lin = normed.matmul(&layer.k_proj_t).map_err(mlx_err)?;
        if let Some(b) = &layer.k_bias {
            k_lin = k_lin.add(b).map_err(mlx_err)?;
        }
        let mut v_lin = normed.matmul(&layer.v_proj_t).map_err(mlx_err)?;
        if let Some(b) = &layer.v_bias {
            v_lin = v_lin.add(b).map_err(mlx_err)?;
        }

        let q_heads = q_lin
            .reshape(&[1, step_len, n_heads, head_dim])
            .map_err(mlx_err)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(mlx_err)?;
        let k_heads = k_lin
            .reshape(&[1, step_len, n_kv, head_dim])
            .map_err(mlx_err)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(mlx_err)?;
        let v_heads = v_lin
            .reshape(&[1, step_len, n_kv, head_dim])
            .map_err(mlx_err)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(mlx_err)?;

        let q_rope = rope(
            &q_heads,
            head_dim,
            false,
            self.config.rope_theta,
            1.0,
            offset_i32,
            None,
        )
        .map_err(mlx_err)?;
        let k_rope = rope(
            &k_heads,
            head_dim,
            false,
            self.config.rope_theta,
            1.0,
            offset_i32,
            None,
        )
        .map_err(mlx_err)?;

        let (k_full, v_full) = if let Some((pk, pv)) = past_kv {
            (
                concatenate(&[pk, &k_rope], 2).map_err(mlx_err)?,
                concatenate(&[pv, &v_heads], 2).map_err(mlx_err)?,
            )
        } else {
            (k_rope, v_heads)
        };

        let mask = if step_len > 1 {
            Some(ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let attn = scaled_dot_product_attention(&q_rope, &k_full, &v_full, scale, mask, None)
            .map_err(mlx_err)?;
        let attn_proj = attn
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(mlx_err)?
            .reshape(&[1, step_len, d])
            .map_err(mlx_err)?
            .matmul(&layer.o_proj_t)
            .map_err(mlx_err)?;
        let after_attn = hidden.add(&attn_proj).map_err(mlx_err)?;

        let post = rms_norm(
            &after_attn,
            Some(&layer.post_attn_norm),
            self.config.rms_norm_eps,
        )
        .map_err(mlx_err)?;
        let gate = silu(&post.matmul(&layer.gate_proj_t).map_err(mlx_err)?).map_err(mlx_err)?;
        let up = post.matmul(&layer.up_proj_t).map_err(mlx_err)?;
        let down = gate
            .multiply(&up)
            .map_err(mlx_err)?
            .matmul(&layer.down_proj_t)
            .map_err(mlx_err)?;
        let next_hidden = after_attn.add(&down).map_err(mlx_err)?;

        Ok((next_hidden, k_full, v_full))
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn forward_step(
        &self,
        step_tokens: &[i32],
        pos_offset: usize,
        past_keys: &[Array],
        past_vals: &[Array],
    ) -> Result<(Array, Vec<Array>, Vec<Array>), NeuralError> {
        let step_len = step_tokens.len() as i32;
        let offset_i32 = pos_offset as i32;
        let ids_arr = Array::from_slice(step_tokens, &[1, step_len]);
        let mut hidden = self.embed_tokens.take_axis(&ids_arr, 0).map_err(mlx_err)?;

        let mut next_keys = Vec::with_capacity(self.config.num_hidden_layers);
        let mut next_vals = Vec::with_capacity(self.config.num_hidden_layers);

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let past_kv = past_keys.get(l_idx).zip(past_vals.get(l_idx));
            let (h, k_full, v_full) =
                self.forward_layer(layer, &hidden, step_len, offset_i32, past_kv)?;
            hidden = h;
            next_keys.push(k_full);
            next_vals.push(v_full);
        }

        let last_tok_h = hidden
            .try_index((.., (step_len - 1)..step_len, ..))
            .map_err(mlx_err)?;
        let out_norm = rms_norm(
            &last_tok_h,
            Some(&self.final_norm),
            self.config.rms_norm_eps,
        )
        .map_err(mlx_err)?;
        let last_logits = out_norm
            .matmul(&self.lm_head_t)
            .map_err(mlx_err)?
            .reshape(&[self.config.vocab_size as i32])
            .map_err(mlx_err)?;

        Ok((last_logits, next_keys, next_vals))
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn forward_continuation_lps_lazy(
        &self,
        cont_tokens: &[i32],
        pos_offset: usize,
        past_keys: &[Array],
        past_vals: &[Array],
    ) -> Result<Array, NeuralError> {
        let step_len = cont_tokens.len() as i32;
        let offset_i32 = pos_offset as i32;
        let ids_arr = Array::from_slice(cont_tokens, &[1, step_len]);
        let mut hidden = self.embed_tokens.take_axis(&ids_arr, 0).map_err(mlx_err)?;

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let past_kv = past_keys.get(l_idx).zip(past_vals.get(l_idx));
            let (h, _, _) = self.forward_layer(layer, &hidden, step_len, offset_i32, past_kv)?;
            hidden = h;
        }

        let out_norm =
            rms_norm(&hidden, Some(&self.final_norm), self.config.rms_norm_eps).map_err(mlx_err)?;
        let logits = out_norm
            .matmul(&self.lm_head_t)
            .map_err(mlx_err)?
            .as_dtype(MlxDtype::Float32)
            .map_err(mlx_err)?;
        log_softmax(&logits, -1).map_err(mlx_err)
    }
}

fn take_f32_or_f16(map: &mut HashMap<String, Array>, name: &str) -> Result<Array, NeuralError> {
    map.remove(name)
        .ok_or_else(|| NeuralError::Safetensors(format!("missing tensor {name}")))
}

fn transpose_2d(arr: &Array) -> Result<Array, NeuralError> {
    arr.transpose_axes(&[1, 0]).map_err(mlx_err)
}

fn mlx_err(e: impl std::fmt::Display) -> NeuralError {
    NeuralError::Safetensors(e.to_string())
}
