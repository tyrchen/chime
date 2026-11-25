//! Apple Silicon Metal GPU Qwen3 / Qwen2.5 Causal Transformer Reranker powered by `mlx-rs`
//! (`0.33`), supporting both native **4-bit MLX affine quantized** (`quantized_matmul` +
//! `dequantize`, ~60 MB) and `FP16`/`F32` `SafeTensors` weights!

use std::{collections::HashMap, path::Path};

use mlx_rs::{
    Array, Dtype as MlxDtype,
    fast::{ScaledDotProductAttentionMask, rms_norm, rope, scaled_dot_product_attention},
    nn::{log_softmax, silu},
    ops::{concatenate, dequantize, indexing::TryIndexOp, quantized_matmul},
    transforms::eval,
};

use crate::{ModelConfig, NeuralError, QuantizationConfig};

#[derive(Debug, Clone)]
enum MlxLinear {
    Dense {
        weight_t: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    },
}

impl MlxLinear {
    fn load(
        raw: &mut HashMap<String, Array>,
        prefix: &str,
        qcfg: Option<QuantizationConfig>,
    ) -> Result<Self, NeuralError> {
        let w = take_tensor(raw, &format!("{prefix}.weight"))?;
        if let Some(qc) = qcfg
            && let Some(scales) = raw.remove(&format!("{prefix}.scales"))
            && let Some(biases) = raw.remove(&format!("{prefix}.biases"))
        {
            Ok(Self::Quantized {
                weight: w,
                scales,
                biases,
                group_size: qc.group_size,
                bits: qc.bits,
            })
        } else {
            Ok(Self::Dense {
                weight_t: transpose_2d(&w)?,
            })
        }
    }

    fn forward(&self, x: &Array) -> Result<Array, NeuralError> {
        match self {
            Self::Dense { weight_t } => x.matmul(weight_t).map_err(mlx_err),
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => quantized_matmul(x, weight, scales, Some(biases), true, *group_size, *bits)
                .map_err(mlx_err),
        }
    }
}

#[derive(Debug, Clone)]
enum MlxEmbedding {
    Dense(Array),
    Quantized {
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    },
}

impl MlxEmbedding {
    fn load(
        raw: &mut HashMap<String, Array>,
        prefix: &str,
        qcfg: Option<QuantizationConfig>,
    ) -> Result<Self, NeuralError> {
        let w = take_tensor(raw, &format!("{prefix}.weight"))?;
        if let Some(qc) = qcfg
            && let Some(scales) = raw.remove(&format!("{prefix}.scales"))
            && let Some(biases) = raw.remove(&format!("{prefix}.biases"))
        {
            Ok(Self::Quantized {
                weight: w,
                scales,
                biases,
                group_size: qc.group_size,
                bits: qc.bits,
            })
        } else {
            Ok(Self::Dense(w))
        }
    }

    fn lookup(&self, step_tokens: &[i32], step_len: i32, d: i32) -> Result<Array, NeuralError> {
        match self {
            Self::Dense(arr) => {
                let ids_arr = Array::from_slice(step_tokens, &[1, step_len]);
                arr.take_axis(&ids_arr, 0).map_err(mlx_err)
            }
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => {
                let ids_1d = Array::from_slice(step_tokens, &[step_len]);
                let w_rows = weight.take_axis(&ids_1d, 0).map_err(mlx_err)?;
                let s_rows = scales.take_axis(&ids_1d, 0).map_err(mlx_err)?;
                let b_rows = biases.take_axis(&ids_1d, 0).map_err(mlx_err)?;
                dequantize(&w_rows, &s_rows, Some(&b_rows), *group_size, *bits)
                    .map_err(mlx_err)?
                    .reshape(&[1, step_len, d])
                    .map_err(mlx_err)
            }
        }
    }

    fn to_tied_lm_head(&self) -> Result<MlxLinear, NeuralError> {
        match self {
            Self::Dense(arr) => Ok(MlxLinear::Dense {
                weight_t: transpose_2d(arr)?,
            }),
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => Ok(MlxLinear::Quantized {
                weight: weight.clone(),
                scales: scales.clone(),
                biases: biases.clone(),
                group_size: *group_size,
                bits: *bits,
            }),
        }
    }
}

#[derive(Debug, Clone)]
struct MlxLayer {
    input_norm: Array,
    q_proj: MlxLinear,
    q_bias: Option<Array>,
    q_norm: Option<Array>,
    k_proj: MlxLinear,
    k_bias: Option<Array>,
    k_norm: Option<Array>,
    v_proj: MlxLinear,
    v_bias: Option<Array>,
    o_proj: MlxLinear,
    post_attn_norm: Array,
    gate_proj: MlxLinear,
    up_proj: MlxLinear,
    down_proj: MlxLinear,
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

/// Metal-accelerated Qwen3 / Qwen2.5 Causal Transformer weights in Apple Silicon Unified Memory.
#[derive(Debug, Clone)]
pub struct MlxTransformer {
    config: ModelConfig,
    embed_tokens: MlxEmbedding,
    layers: Vec<MlxLayer>,
    final_norm: Array,
    lm_head: MlxLinear,
}

impl MlxTransformer {
    /// Loads `model.safetensors` (either native 4-bit MLX quantized or `FP16`/`F32`) directly onto
    /// Apple Silicon Metal unified memory via `mlx-rs`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if loading or tensor validation fails.
    pub fn load_safetensors_file(cfg: &ModelConfig, st_path: &Path) -> Result<Self, NeuralError> {
        let mut raw = Array::load_safetensors(st_path)
            .map_err(|e| NeuralError::Safetensors(format!("MLX load_safetensors failed: {e}")))?;

        let qcfg = cfg.quantization;
        let embed_tokens = MlxEmbedding::load(&mut raw, "model.embed_tokens", qcfg)?;
        let final_norm = take_tensor(&mut raw, "model.norm.weight")?;
        let lm_head = if raw.contains_key("lm_head.weight") {
            MlxLinear::load(&mut raw, "lm_head", qcfg)?
        } else {
            embed_tokens.to_tied_lm_head()?
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{idx}");
            let input_norm = take_tensor(&mut raw, &format!("{p}.input_layernorm.weight"))?;
            let q_proj = MlxLinear::load(&mut raw, &format!("{p}.self_attn.q_proj"), qcfg)?;
            let q_bias = raw.remove(&format!("{p}.self_attn.q_proj.bias"));
            let q_norm = raw.remove(&format!("{p}.self_attn.q_norm.weight"));
            let k_proj = MlxLinear::load(&mut raw, &format!("{p}.self_attn.k_proj"), qcfg)?;
            let k_bias = raw.remove(&format!("{p}.self_attn.k_proj.bias"));
            let k_norm = raw.remove(&format!("{p}.self_attn.k_norm.weight"));
            let v_proj = MlxLinear::load(&mut raw, &format!("{p}.self_attn.v_proj"), qcfg)?;
            let v_bias = raw.remove(&format!("{p}.self_attn.v_proj.bias"));
            let o_proj = MlxLinear::load(&mut raw, &format!("{p}.self_attn.o_proj"), qcfg)?;
            let post_attn_norm =
                take_tensor(&mut raw, &format!("{p}.post_attention_layernorm.weight"))?;
            let gate_proj = MlxLinear::load(&mut raw, &format!("{p}.mlp.gate_proj"), qcfg)?;
            let up_proj = MlxLinear::load(&mut raw, &format!("{p}.mlp.up_proj"), qcfg)?;
            let down_proj = MlxLinear::load(&mut raw, &format!("{p}.mlp.down_proj"), qcfg)?;

            layers.push(MlxLayer {
                input_norm,
                q_proj,
                q_bias,
                q_norm,
                k_proj,
                k_bias,
                k_norm,
                v_proj,
                v_bias,
                o_proj,
                post_attn_norm,
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        let _ = eval([&final_norm]);

        Ok(Self {
            config: cfg.clone(),
            embed_tokens,
            layers,
            final_norm,
            lm_head,
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
        let joint = self.score_candidates_joint_lps(cache, cands)?;
        Ok(joint
            .into_iter()
            .zip(cands.iter())
            .map(|(lp, ids)| {
                if ids.is_empty() || (!include_first_token && ids.len() <= 1) {
                    return 0.0;
                }
                let eff_len = if include_first_token {
                    ids.len()
                } else {
                    ids.len().saturating_sub(1).max(1)
                };
                #[allow(clippy::cast_precision_loss)]
                let norm = (eff_len as f32).powf(0.75);
                lp / norm
            })
            .collect())
    }

    /// Computes exact unnormalized joint log-likelihoods (`sum_t log P(tok_t | prefix, tok_<t)`)
    /// for each tokenized candidate so callers can normalize by character count instead of BPE
    /// piece count!
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if an MLX Metal kernel fails during continuation pass.
    pub fn score_candidates_joint_lps(
        &self,
        cache: &MlxPrefixCache,
        cands: &[Vec<u32>],
    ) -> Result<Vec<f32>, NeuralError> {
        if cands.is_empty() || cache.next_token_log_probs.is_empty() {
            return Ok(vec![0.0; cands.len()]);
        }

        let mut unique_conts: HashMap<Vec<i32>, usize> = HashMap::with_capacity(cands.len());
        let mut lazy_arrays: Vec<Array> = Vec::with_capacity(cands.len());
        let mut cand_cont_slot: Vec<Option<usize>> = vec![None; cands.len()];

        for (c_idx, cand_ids) in cands.iter().enumerate() {
            if cand_ids.len() > 1 {
                let cont_i32: Vec<i32> = cand_ids[..cand_ids.len() - 1]
                    .iter()
                    .map(|&id| i32::try_from(id).unwrap_or(0))
                    .collect();
                let slot = if let Some(&existing) = unique_conts.get(&cont_i32) {
                    existing
                } else {
                    let lps = self.forward_continuation_lps_lazy(
                        &cont_i32,
                        cache.cached_tokens.len(),
                        &cache.keys,
                        &cache.values,
                    )?;
                    let idx = lazy_arrays.len();
                    lazy_arrays.push(lps);
                    unique_conts.insert(cont_i32, idx);
                    idx
                };
                cand_cont_slot[c_idx] = Some(slot);
            }
        }

        if !lazy_arrays.is_empty() {
            eval(&lazy_arrays).map_err(mlx_err)?;
        }

        let vocab = self.config.vocab_size;
        let mut results = vec![0.0_f32; cands.len()];

        for (c_idx, cand_ids) in cands.iter().enumerate() {
            if cand_ids.is_empty() {
                continue;
            }
            let first_idx = (cand_ids[0] as usize).min(cache.next_token_log_probs.len() - 1);
            let mut total_lp = cache.next_token_log_probs[first_idx];

            if let Some(slot) = cand_cont_slot[c_idx] {
                let slice = lazy_arrays[slot].as_slice::<f32>();
                for (step, &target_tok) in cand_ids[1..].iter().enumerate() {
                    let t_idx = (target_tok as usize).min(vocab - 1);
                    if let Some(&val) = slice.get(step * vocab + t_idx) {
                        total_lp += val;
                    }
                }
            }

            results[c_idx] = total_lp;
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
        let n_heads = self.config.num_attention_heads as i32;
        let n_kv = self.config.num_key_value_heads as i32;
        let head_dim = self.config.effective_head_dim() as i32;
        let q_out_dim = n_heads * head_dim;
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let normed =
            rms_norm(hidden, Some(&layer.input_norm), self.config.rms_norm_eps).map_err(mlx_err)?;
        let mut q_lin = layer.q_proj.forward(&normed)?;
        if let Some(b) = &layer.q_bias {
            q_lin = q_lin.add(b).map_err(mlx_err)?;
        }
        let mut k_lin = layer.k_proj.forward(&normed)?;
        if let Some(b) = &layer.k_bias {
            k_lin = k_lin.add(b).map_err(mlx_err)?;
        }
        let mut v_lin = layer.v_proj.forward(&normed)?;
        if let Some(b) = &layer.v_bias {
            v_lin = v_lin.add(b).map_err(mlx_err)?;
        }

        let mut q_reshaped = q_lin
            .reshape(&[1, step_len, n_heads, head_dim])
            .map_err(mlx_err)?;
        if let Some(qn) = &layer.q_norm {
            q_reshaped =
                rms_norm(&q_reshaped, Some(qn), self.config.rms_norm_eps).map_err(mlx_err)?;
        }
        let q_heads = q_reshaped.transpose_axes(&[0, 2, 1, 3]).map_err(mlx_err)?;

        let mut k_reshaped = k_lin
            .reshape(&[1, step_len, n_kv, head_dim])
            .map_err(mlx_err)?;
        if let Some(kn) = &layer.k_norm {
            k_reshaped =
                rms_norm(&k_reshaped, Some(kn), self.config.rms_norm_eps).map_err(mlx_err)?;
        }
        let k_heads = k_reshaped.transpose_axes(&[0, 2, 1, 3]).map_err(mlx_err)?;

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
        let attn_flat = attn
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(mlx_err)?
            .reshape(&[1, step_len, q_out_dim])
            .map_err(mlx_err)?;
        let attn_proj = layer.o_proj.forward(&attn_flat)?;
        let after_attn = hidden.add(&attn_proj).map_err(mlx_err)?;

        let post = rms_norm(
            &after_attn,
            Some(&layer.post_attn_norm),
            self.config.rms_norm_eps,
        )
        .map_err(mlx_err)?;
        let gate = silu(&layer.gate_proj.forward(&post)?).map_err(mlx_err)?;
        let up = layer.up_proj.forward(&post)?;
        let gated = gate.multiply(&up).map_err(mlx_err)?;
        let down = layer.down_proj.forward(&gated)?;
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
        let d = self.config.hidden_size as i32;
        let mut hidden = self.embed_tokens.lookup(step_tokens, step_len, d)?;

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
        let last_logits = self
            .lm_head
            .forward(&out_norm)?
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
        let d = self.config.hidden_size as i32;
        let mut hidden = self.embed_tokens.lookup(cont_tokens, step_len, d)?;

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let past_kv = past_keys.get(l_idx).zip(past_vals.get(l_idx));
            let (h, _, _) = self.forward_layer(layer, &hidden, step_len, offset_i32, past_kv)?;
            hidden = h;
        }

        let out_norm =
            rms_norm(&hidden, Some(&self.final_norm), self.config.rms_norm_eps).map_err(mlx_err)?;
        let logits = self
            .lm_head
            .forward(&out_norm)?
            .as_dtype(MlxDtype::Float32)
            .map_err(mlx_err)?;
        log_softmax(&logits, -1).map_err(mlx_err)
    }
}

fn take_tensor(map: &mut HashMap<String, Array>, name: &str) -> Result<Array, NeuralError> {
    map.remove(name)
        .ok_or_else(|| NeuralError::Safetensors(format!("missing tensor {name}")))
}

fn transpose_2d(arr: &Array) -> Result<Array, NeuralError> {
    arr.transpose_axes(&[1, 0]).map_err(mlx_err)
}

fn mlx_err(e: impl std::fmt::Display) -> NeuralError {
    NeuralError::Safetensors(e.to_string())
}
