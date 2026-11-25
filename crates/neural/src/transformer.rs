//! Qwen2.5-compatible Causal Transformer (`RMSNorm` + `RoPE` + `GQA` + `SwiGLU`)
//! with persistent `PrefixKvCache` and targeted `lm_head` log-probability scoring.

use std::{collections::HashMap, path::Path};

use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};

use crate::NeuralError;

/// Optional MLX affine quantization metadata (`group_size = 64, bits = 4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Group size along input dimension (typically `64`).
    pub group_size: i32,
    /// Bits per quantized weight (`4` or `8`).
    pub bits: i32,
}

/// Hyperparameters for the compact Causal Transformer reranker (`config.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hidden embedding dimension (`d_model`, e.g. 1024 for `Qwen3-0.6B`).
    pub hidden_size: usize,
    /// `SwiGLU` intermediate dimension (`d_ff`, e.g. 3072 for `Qwen3-0.6B`).
    pub intermediate_size: usize,
    /// Number of transformer decoder blocks.
    pub num_hidden_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (Grouped-Query Attention).
    pub num_key_value_heads: usize,
    /// Optional explicit per-head dimension (`128` in `Qwen3-0.6B`, `None` defaults to
    /// `hidden_size / num_attention_heads`).
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Compact vocabulary size (e.g. 24,576).
    pub vocab_size: usize,
    /// `RMSNorm` epsilon.
    pub rms_norm_eps: f32,
    /// Rotary embedding base frequency (`theta`).
    pub rope_theta: f32,
    /// Optional native MLX 4-bit/8-bit quantization config.
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl ModelConfig {
    /// Returns the explicit `head_dim` if configured (`Qwen3`), or `hidden_size /
    /// num_attention_heads`.
    #[must_use]
    pub const fn effective_head_dim(&self) -> usize {
        match self.head_dim {
            Some(hd) if hd > 0 => hd,
            _ => self.hidden_size / self.num_attention_heads,
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    input_norm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    post_attn_norm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

/// Persistent Key-Value cache storing computed prefix states across consecutive keystrokes.
#[derive(Debug, Clone)]
pub struct PrefixKvCache {
    cached_tokens: Vec<u32>,
    last_hidden: Vec<f32>,
    /// Per-layer flattened `[seq_len * kv_dim]` key states.
    keys: Vec<Vec<f32>>,
    /// Per-layer flattened `[seq_len * kv_dim]` value states.
    values: Vec<Vec<f32>>,
}

impl PrefixKvCache {
    /// Allocates an empty KV cache sized for `config`.
    #[must_use]
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            cached_tokens: Vec::with_capacity(64),
            last_hidden: vec![0.0; config.hidden_size],
            keys: vec![Vec::new(); config.num_hidden_layers],
            values: vec![Vec::new(); config.num_hidden_layers],
        }
    }

    /// Returns the number of currently cached prefix tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cached_tokens.len()
    }

    /// Returns true if no prefix tokens are cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cached_tokens.is_empty()
    }

    fn truncate_to(&mut self, common_len: usize, kv_dim: usize) {
        self.cached_tokens.truncate(common_len);
        for l in 0..self.keys.len() {
            self.keys[l].truncate(common_len * kv_dim);
            self.values[l].truncate(common_len * kv_dim);
        }
    }
}

/// Loaded FP32/FP16-decoded weights for a Qwen2.5-style Causal Transformer.
#[derive(Debug, Clone)]
pub struct TransformerWeights {
    config: ModelConfig,
    embed_tokens: Vec<f32>,
    layers: Vec<LayerWeights>,
    final_norm: Vec<f32>,
    lm_head: Vec<f32>,
}

impl TransformerWeights {
    /// Parses a `.safetensors` byte buffer (supporting `F32` and `F16`/`BF16` weights).
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if required tensors are missing or have mismatched
    /// shapes.
    pub fn from_safetensors_bytes(cfg: &ModelConfig, bytes: &[u8]) -> Result<Self, NeuralError> {
        let st =
            SafeTensors::deserialize(bytes).map_err(|e| NeuralError::Safetensors(e.to_string()))?;

        let d = cfg.hidden_size;
        let head_dim = d / cfg.num_attention_heads.max(1);
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let ff = cfg.intermediate_size;

        let embed_tokens = read_tensor_f32(&st, "model.embed_tokens.weight", cfg.vocab_size * d)?;
        let final_norm = read_tensor_f32(&st, "model.norm.weight", d)?;
        let lm_head = if st.tensor("lm_head.weight").is_ok() {
            read_tensor_f32(&st, "lm_head.weight", cfg.vocab_size * d)?
        } else {
            // Support tied word embeddings (`tie_word_embeddings = true` in Qwen2.5-0.5B!).
            embed_tokens.clone()
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(LayerWeights {
                input_norm: read_tensor_f32(&st, &format!("{p}.input_layernorm.weight"), d)?,
                q_proj: read_tensor_f32(&st, &format!("{p}.self_attn.q_proj.weight"), d * d)?,
                k_proj: read_tensor_f32(&st, &format!("{p}.self_attn.k_proj.weight"), kv_dim * d)?,
                v_proj: read_tensor_f32(&st, &format!("{p}.self_attn.v_proj.weight"), kv_dim * d)?,
                o_proj: read_tensor_f32(&st, &format!("{p}.self_attn.o_proj.weight"), d * d)?,
                post_attn_norm: read_tensor_f32(
                    &st,
                    &format!("{p}.post_attention_layernorm.weight"),
                    d,
                )?,
                gate_proj: read_tensor_f32(&st, &format!("{p}.mlp.gate_proj.weight"), ff * d)?,
                up_proj: read_tensor_f32(&st, &format!("{p}.mlp.up_proj.weight"), ff * d)?,
                down_proj: read_tensor_f32(&st, &format!("{p}.mlp.down_proj.weight"), d * ff)?,
            });
        }

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

    /// Incrementally prefills `prefix_ids` into `cache`, reusing any matching prefix!
    /// Returns the final normalized hidden state after the last prefix token.
    pub fn prefill_with_cache(&self, prefix_ids: &[u32], cache: &mut PrefixKvCache) -> Vec<f32> {
        let d = self.config.hidden_size;
        let head_dim = d / self.config.num_attention_heads;
        let kv_dim = self.config.num_key_value_heads * head_dim;

        let common = cache
            .cached_tokens
            .iter()
            .zip(prefix_ids.iter())
            .take_while(|(a, b)| a == b)
            .count();

        // If the entire prefix matches our cache, zero forward passes are needed for the prefix!
        if common == prefix_ids.len() && common == cache.cached_tokens.len() && common > 0 {
            return cache.last_hidden.clone();
        }

        // Keep at least `common.saturating_sub(1)` if we need to recompute the final hidden state
        // when `prefix_ids` was truncated shorter than the old cache.
        let keep = if common == prefix_ids.len() {
            common.saturating_sub(1)
        } else {
            common
        };
        cache.truncate_to(keep, kv_dim);

        let mut last_normed = vec![0.0; d];
        for &tok in &prefix_ids[keep..] {
            let pos = cache.cached_tokens.len();
            cache.cached_tokens.push(tok);
            let h = self.step_single_token(tok, pos, &mut cache.keys, &mut cache.values);
            last_normed = rms_norm(&h, &self.final_norm, self.config.rms_norm_eps);
        }
        cache.last_hidden.clone_from(&last_normed);
        last_normed
    }

    /// Scores a candidate token sequence `cand_ids` conditioned on an immutable `prefix_cache`!
    #[must_use]
    pub fn score_continuation(
        &self,
        prefix_cache: &PrefixKvCache,
        first_normed_hidden: &[f32],
        cand_ids: &[u32],
    ) -> f32 {
        if cand_ids.is_empty() {
            return 0.0;
        }
        let mut total_log_prob = 0.0f32;
        total_log_prob += self.target_log_prob(first_normed_hidden, cand_ids[0]);

        if cand_ids.len() > 1 {
            // Clone lightweight scratch KV tails only when candidate has >= 2 subword/character
            // tokens.
            let mut tmp_keys = prefix_cache.keys.clone();
            let mut tmp_vals = prefix_cache.values.clone();
            let base_pos = prefix_cache.cached_tokens.len();

            for (step, &tok) in cand_ids[..cand_ids.len() - 1].iter().enumerate() {
                let h = self.step_single_token(tok, base_pos + step, &mut tmp_keys, &mut tmp_vals);
                let normed = rms_norm(&h, &self.final_norm, self.config.rms_norm_eps);
                total_log_prob += self.target_log_prob(&normed, cand_ids[step + 1]);
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let length_norm = (cand_ids.len() as f32).powf(0.75);
        total_log_prob / length_norm
    }

    fn step_single_token(
        &self,
        token_id: u32,
        pos: usize,
        k_caches: &mut [Vec<f32>],
        v_caches: &mut [Vec<f32>],
    ) -> Vec<f32> {
        let dim = self.config.hidden_size;
        let n_heads = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let head_dim = dim / n_heads;
        let kv_dim = n_kv * head_dim;
        let group_size = n_heads / n_kv;

        let tok_idx = (token_id as usize).min(self.config.vocab_size.saturating_sub(1));
        let mut hidden = self.embed_tokens[tok_idx * dim..(tok_idx + 1) * dim].to_vec();

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let x_norm = rms_norm(&hidden, &layer.input_norm, self.config.rms_norm_eps);
            let mut query_vec = matvec(&layer.q_proj, &x_norm, dim, dim);
            let mut key_vec = matvec(&layer.k_proj, &x_norm, kv_dim, dim);
            let val_vec = matvec(&layer.v_proj, &x_norm, kv_dim, dim);

            apply_rope(
                &mut query_vec,
                n_heads,
                head_dim,
                pos,
                self.config.rope_theta,
            );
            apply_rope(&mut key_vec, n_kv, head_dim, pos, self.config.rope_theta);

            k_caches[l_idx].extend_from_slice(&key_vec);
            v_caches[l_idx].extend_from_slice(&val_vec);

            let seq_len = k_caches[l_idx].len() / kv_dim;
            #[allow(clippy::cast_precision_loss)]
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn_out = vec![0.0f32; dim];

            for head_idx in 0..n_heads {
                let kv_h = head_idx / group_size;
                let q_slice = &query_vec[head_idx * head_dim..(head_idx + 1) * head_dim];
                let mut scores = Vec::with_capacity(seq_len);
                let mut max_s = f32::NEG_INFINITY;
                for step_t in 0..seq_len {
                    let k_off = step_t * kv_dim + kv_h * head_dim;
                    let k_slice = &k_caches[l_idx][k_off..k_off + head_dim];
                    let score = dot(q_slice, k_slice) * scale;
                    if score > max_s {
                        max_s = score;
                    }
                    scores.push(score);
                }

                let mut sum_exp = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max_s).exp();
                    sum_exp += *score;
                }
                let inv_sum = 1.0 / sum_exp.max(1e-12);
                let out_head = &mut attn_out[head_idx * head_dim..(head_idx + 1) * head_dim];
                for (step_t, &prob_unnorm) in scores.iter().enumerate() {
                    let prob = prob_unnorm * inv_sum;
                    let v_off = step_t * kv_dim + kv_h * head_dim;
                    let v_slice = &v_caches[l_idx][v_off..v_off + head_dim];
                    for (dst, &val) in out_head.iter_mut().zip(v_slice.iter()) {
                        *dst = prob.mul_add(val, *dst);
                    }
                }
            }

            let proj_attn = matvec(&layer.o_proj, &attn_out, dim, dim);
            for (xi, &ai) in hidden.iter_mut().zip(proj_attn.iter()) {
                *xi += ai;
            }

            let post_norm = rms_norm(&hidden, &layer.post_attn_norm, self.config.rms_norm_eps);
            let gate = matvec(
                &layer.gate_proj,
                &post_norm,
                self.config.intermediate_size,
                dim,
            );
            let up = matvec(
                &layer.up_proj,
                &post_norm,
                self.config.intermediate_size,
                dim,
            );
            let mut hidden_ff = Vec::with_capacity(self.config.intermediate_size);
            for (&gate_v, &up_v) in gate.iter().zip(up.iter()) {
                let silu = gate_v / (1.0 + (-gate_v).exp());
                hidden_ff.push(silu * up_v);
            }
            let down = matvec(
                &layer.down_proj,
                &hidden_ff,
                dim,
                self.config.intermediate_size,
            );
            for (xi, &di) in hidden.iter_mut().zip(down.iter()) {
                *xi += di;
            }
        }
        hidden
    }

    fn target_log_prob(&self, normed_hidden: &[f32], target_id: u32) -> f32 {
        let d = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let t_idx = (target_id as usize).min(vocab.saturating_sub(1));

        let mut max_logit = f32::NEG_INFINITY;
        let mut logits = Vec::with_capacity(vocab);
        for row in 0..vocab {
            let l = dot(&self.lm_head[row * d..(row + 1) * d], normed_hidden);
            if l > max_logit {
                max_logit = l;
            }
            logits.push(l);
        }
        let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        (logits[t_idx] - max_logit) - sum_exp.max(1e-12).ln()
    }
}

/// Helper to create a valid Qwen2.5-arch `.safetensors` reranker bundle for tests and local
/// bootstrapping.
#[derive(Debug)]
pub struct TinyModelBuilder;

impl TinyModelBuilder {
    /// Writes a self-contained tiny Qwen2.5-arch bundle (`config.json`, `vocab.json`,
    /// `model.safetensors`) encoding contextual associations `associations` (e.g. `[("端侧",
    /// "量化")]`).
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError`] if serialization or file writing fails.
    #[allow(clippy::too_many_lines)]
    pub fn write_demo_bundle(
        out_dir: &Path,
        associations: &[(&str, &str)],
    ) -> Result<(), NeuralError> {
        std::fs::create_dir_all(out_dir)?;
        let mut vocab_list: Vec<String> = vec![
            "<bos>".into(),
            "<unk>".into(),
            "端侧".into(),
            "模型".into(),
            "量化".into(),
            "亮化".into(),
            "帮我".into(),
            "review".into(),
            "这个".into(),
            "PR".into(),
            "个人".into(),
            "macOS".into(),
            "GitHub".into(),
        ];
        for &(ctx, tgt) in associations {
            if !vocab_list.iter().any(|s| s == ctx) {
                vocab_list.push(ctx.to_string());
            }
            if !vocab_list.iter().any(|s| s == tgt) {
                vocab_list.push(tgt.to_string());
            }
        }

        let d = 16usize;
        let ff = 32usize;
        let v = vocab_list.len();
        let cfg = ModelConfig {
            hidden_size: d,
            intermediate_size: ff,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: None,
            vocab_size: v,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            quantization: None,
        };

        // Construct orthogonal basis embeddings so context -> target associations score
        // deterministically higher!
        let mut embed = vec![0.0f32; v * d];
        for tok_idx in 0..v {
            embed[tok_idx * d + (tok_idx % d)] = 1.0;
        }
        let mut lm_head = embed.clone();
        for &(ctx_word, tgt_word) in associations {
            if let (Some(c_idx), Some(t_idx)) = (
                vocab_list.iter().position(|w| w == ctx_word),
                vocab_list.iter().position(|w| w == tgt_word),
            ) {
                lm_head[t_idx * d + (c_idx % d)] += 3.5;
            }
        }

        let ones_d = vec![1.0f32; d];
        let zeros_dd = vec![0.0f32; d * d];
        let zeros_ffd = vec![0.0f32; ff * d];

        let mut raw_tensors: HashMap<String, (Vec<usize>, Vec<u8>)> = HashMap::new();
        insert_f32_tensor(
            &mut raw_tensors,
            "model.embed_tokens.weight",
            vec![v, d],
            &embed,
        );
        insert_f32_tensor(&mut raw_tensors, "model.norm.weight", vec![d], &ones_d);
        insert_f32_tensor(&mut raw_tensors, "lm_head.weight", vec![v, d], &lm_head);
        insert_f32_tensor(
            &mut raw_tensors,
            "model.layers.0.input_layernorm.weight",
            vec![d],
            &ones_d,
        );
        insert_f32_tensor(
            &mut raw_tensors,
            "model.layers.0.self_attn.q_proj.weight",
            vec![d, d],
            &zeros_dd,
        );
        insert_f32_tensor(
            &mut raw_tensors,
            "model.layers.0.self_attn.k_proj.weight",
            vec![d, d],
            &zeros_dd,
        );
        // Set v_proj and o_proj slightly positive so attention averages context!
        let mut eye_dd = vec![0.0f32; d * d];
        for i in 0..d {
            eye_dd[i * d + i] = 0.6;
        }
        insert_f32_tensor(
            &mut raw_tensors,
            "model.layers.0.self_attn.v_proj.weight",
            vec![d, d],
            &eye_dd,
        );
        insert_f32_tensor(
            &mut raw_tensors,
            "model.layers.0.self_attn.o_proj.weight",
            vec![d, d],
            &eye_dd,
        );
        insert_f32_tensor(
            &mut raw_tensors,
            "model.layers.0.post_attention_layernorm.weight",
            vec![d],
            &ones_d,
        );
        insert_f32_tensor(
            &mut raw_tensors,
            "model.layers.0.mlp.gate_proj.weight",
            vec![ff, d],
            &zeros_ffd,
        );
        insert_f32_tensor(
            &mut raw_tensors,
            "model.layers.0.mlp.up_proj.weight",
            vec![ff, d],
            &zeros_ffd,
        );
        insert_f32_tensor(
            &mut raw_tensors,
            "model.layers.0.mlp.down_proj.weight",
            vec![d, ff],
            &zeros_ffd,
        );

        let views: Vec<(&str, TensorView<'_>)> = raw_tensors
            .iter()
            .filter_map(|(name, (shape, bytes))| {
                TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .ok()
                    .map(|tv| (name.as_str(), tv))
            })
            .collect();
        let serialized = safetensors::serialize(views, None)
            .map_err(|e| NeuralError::Safetensors(e.to_string()))?;

        std::fs::write(
            out_dir.join("config.json"),
            serde_json::to_vec_pretty(&cfg)?,
        )?;
        std::fs::write(
            out_dir.join("vocab.json"),
            serde_json::to_vec_pretty(&vocab_list)?,
        )?;
        std::fs::write(out_dir.join("model.safetensors"), serialized)?;
        Ok(())
    }
}

fn insert_f32_tensor(
    map: &mut HashMap<String, (Vec<usize>, Vec<u8>)>,
    name: &str,
    shape: Vec<usize>,
    values: &[f32],
) {
    let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
    map.insert(name.to_string(), (shape, bytes));
}

fn read_tensor_f32(
    st: &SafeTensors<'_>,
    name: &str,
    expected_elems: usize,
) -> Result<Vec<f32>, NeuralError> {
    let view = st
        .tensor(name)
        .map_err(|e| NeuralError::Safetensors(format!("missing tensor {name}: {e}")))?;
    let data = view.data();
    let out: Vec<f32> = match view.dtype() {
        Dtype::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect(),
        Dtype::BF16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| f32::from_bits(u32::from(u16::from_le_bytes(b)) << 16))
            .collect(),
        Dtype::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| decode_ieee_f16(u16::from_le_bytes(b)))
            .collect(),
        other => {
            return Err(NeuralError::Safetensors(format!(
                "unsupported dtype {other:?} for tensor {name}"
            )));
        }
    };
    if out.len() != expected_elems {
        return Err(NeuralError::Safetensors(format!(
            "tensor {name} has {} elements, expected {expected_elems}",
            out.len()
        )));
    }
    Ok(out)
}

fn decode_ieee_f16(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127u32 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let (chunks_a, rem_a) = a.as_chunks::<4>();
    let (chunks_b, rem_b) = b.as_chunks::<4>();
    for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
        sum0 = ca[0].mul_add(cb[0], sum0);
        sum1 = ca[1].mul_add(cb[1], sum1);
        sum2 = ca[2].mul_add(cb[2], sum2);
        sum3 = ca[3].mul_add(cb[3], sum3);
    }
    let mut total = (sum0 + sum1) + (sum2 + sum3);
    for (&xa, &xb) in rem_a.iter().zip(rem_b.iter()) {
        total = xa.mul_add(xb, total);
    }
    total
}

fn matvec(weights: &[f32], x: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(out_rows);
    for r in 0..out_rows {
        let row = &weights[r * in_cols..(r + 1) * in_cols];
        out.push(dot(row, x));
    }
    out
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    #[allow(clippy::cast_precision_loss)]
    let mean_sq = dot(x, x) / (x.len().max(1) as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}

fn apply_rope(vec: &mut [f32], num_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;
    #[allow(clippy::cast_precision_loss)]
    let pos_f = pos as f32;
    for h in 0..num_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            #[allow(clippy::cast_precision_loss)]
            let freq = 1.0 / theta.powf((2 * i) as f32 / (head_dim as f32));
            let angle = pos_f * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos_a - x1 * sin_a;
            head[i + half] = x1 * cos_a + x0 * sin_a;
        }
    }
}
