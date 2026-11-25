//! Apple Silicon Metal GPU Qwen3 / Qwen2.5 Causal Transformer Reranker powered by `mlx-rs`
//! (`0.34`), supporting native **4-bit MLX affine quantized** (`quantized_matmul` +
//! `dequantize`, ~64 MB), `FP16`/`F32` `SafeTensors` weights, and **Multi-LoRA Sidecar**
//! hot-mounted low-rank adapters (`y = Base(x) + sum_k s_k * (x @ A_k^T) @ B_k^T`).

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use mlx_rs::{
    Array, Dtype as MlxDtype,
    fast::{ScaledDotProductAttentionMask, rms_norm, rope, scaled_dot_product_attention},
    nn::{log_softmax, silu},
    ops::{
        DequantizeOptions, QuantizedMatmulOptions, broadcast_to, concatenate, dequantize,
        indexing::TryIndexOp, quantized_matmul,
    },
    transforms::eval,
};
use serde::{Deserialize, Serialize};

use crate::{ModelConfig, NeuralError, QuantizationConfig};

/// Dynamic context and host-application routing policy for a mounted `LoRA` adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LoraRouting {
    /// Always active at `base_scale` across every app and context (personal habit packs).
    #[default]
    Always,
    /// Full `base_scale` in terminals, IDEs, and bilingual code-switching; gentle `0.25x` in prose.
    Developer,
    /// Full `base_scale` when classical poetry/prose cues or multi-char verses appear; `0.30x`
    /// otherwise.
    Classical,
}

impl LoraRouting {
    /// Computes the raw pre-simplex `LoRA` multiplier given the active host app, left context, and
    /// candidate pool.
    #[must_use]
    pub fn effective_scale(
        self,
        base_scale: f32,
        app_bundle_id: &str,
        left_context: &str,
        candidate_texts: &[String],
    ) -> f32 {
        let max_cand_chars = candidate_texts
            .iter()
            .map(|c| c.chars().count())
            .max()
            .unwrap_or(0);
        match self {
            Self::Always => {
                // Keep personal written-essay style gentle on 1-2 char spoken words so colloquial
                // disambiguation (`究竟有多大`) stays faithful to spoken priors!
                if max_cand_chars <= 2 && left_context.chars().count() <= 6 {
                    base_scale * 0.45
                } else {
                    base_scale
                }
            }
            Self::Developer => {
                let has_code_cue = left_context.bytes().any(|b| b.is_ascii_alphabetic())
                    || candidate_texts
                        .iter()
                        .any(|c| c.bytes().any(|b| b.is_ascii_alphabetic()));
                if has_code_cue {
                    base_scale
                } else if is_developer_bundle(app_bundle_id) {
                    base_scale * 0.25
                } else {
                    0.0
                }
            }
            Self::Classical => {
                // Smoothly gate classical Chinese adapter by archaic cues and 5/7-char poetic
                // cadence, while keeping `0.0` in everyday modern Mandarin!
                base_scale * classical_context_strength(left_context, candidate_texts)
            }
        }
    }
}

fn is_developer_bundle(bundle_id: &str) -> bool {
    let lower = bundle_id.to_ascii_lowercase();
    [
        "vscode",
        "zed",
        "ghostty",
        "iterm",
        "terminal",
        "warp",
        "alacritty",
        "kitty",
        "wezterm",
        "xcode",
        "jetbrains",
        "cursor",
        "sublime",
        "neovide",
    ]
    .iter()
    .any(|kw| lower.contains(kw))
}

fn classical_context_strength(left_context: &str, candidate_texts: &[String]) -> f32 {
    const HARD_CLASSICAL_CUES: &[char] = &[
        '之', '乎', '者', '也', '矣', '焉', '哉', '兮', '夫', '盖', '岂', '莫', '勿', '君', '吾',
        '汝', '尔', '樽', '阙', '霭', '婵', '娟', '潇', '湘', '沧', '浪', '阑', '珊', '琵', '琶',
        '霓', '裳', '戍', '烽', '燧', '渚', '猿', '槛', '帘',
    ];
    const MODERN_COLLOQUIAL_STOP: &[char] = &[
        '的', '了', '吗', '呢', '吧', '啊', '哦', '哈', '呀', '嘛', '咱', '啥', '挺', '特', '什',
        '么', '怎', '这', '那', '哪',
    ];

    if left_context
        .chars()
        .any(|c| HARD_CLASSICAL_CUES.contains(&c))
    {
        return 1.0;
    }
    if candidate_texts.iter().any(|c| {
        c.chars().count() >= 5
            && (c.contains('，') || c.chars().any(|ch| HARD_CLASSICAL_CUES.contains(&ch)))
    }) {
        return 1.0;
    }

    // Detect symmetric 5+5 or 7+7 regulated verse couplet cadence (e.g. `沧海月明珠有泪，` +
    // `蓝田日暖玉生烟`, `空山新雨后，` + `天气晚来秋`) even without archaic function particles!
    let trimmed = left_context.trim();
    if let Some(stripped) = trimmed.strip_suffix(['，', '。', '？', '！', '；']) {
        let last_clause = stripped
            .rsplit(['，', '。', '？', '！', '；', '\n'])
            .next()
            .unwrap_or(stripped);
        let cjk_len = last_clause
            .chars()
            .filter(|c| ('\u{4e00}'..='\u{9fa5}').contains(c))
            .count();
        let all_cjk = cjk_len == last_clause.chars().count();
        let has_colloquial = last_clause
            .chars()
            .any(|c| MODERN_COLLOQUIAL_STOP.contains(&c));
        let has_matching_couplet_cand = candidate_texts
            .iter()
            .any(|cand| cand.chars().count() == cjk_len);
        if all_cjk && (cjk_len == 5 || cjk_len == 7) && !has_colloquial && has_matching_couplet_cand
        {
            return 0.75;
        }
    }
    0.0
}

/// Specification for mounting a `.safetensors` `LoRA` adapter onto `MlxTransformer`.
#[derive(Debug, Clone)]
pub struct LoraMountSpec {
    /// Unique cartridge / adapter identifier (`"classics-poetry"`, `"tech-developer"`, …).
    pub id: String,
    /// Path to `lora.safetensors` (`FP16`/`F32` low-rank `lora_A` & `lora_B` tensors).
    pub path: PathBuf,
    /// Base `LoRA` scaling factor (`alpha / rank`, typically `1.0`..`2.0`).
    pub scale: f32,
    /// Dynamic app/context routing policy.
    pub routing: LoraRouting,
}

#[derive(Debug, Clone)]
struct MlxLoraPair {
    /// Pre-transposed `A^T` with shape `[in_features, rank]` cast to model activation dtype.
    a_t: Array,
    /// Pre-transposed `B^T` with shape `[rank, out_features]` cast to model activation dtype.
    b_t: Array,
    /// Low-rank dimension `r`.
    rank: usize,
}

/// Rank-concatenated fused `Multi-LoRA` projection (`A_cat^T: [d_in, sum_k r_k]`,
/// `B_cat^T: [sum_k r_k, d_out]`), reducing $K$ sequential adapter passes to 2 Metal GEMMs!
#[derive(Debug, Clone)]
struct MlxStackedLoraProj {
    a_cat_t: Array,
    b_cat_t: Array,
    slices: Vec<(usize, usize)>,
    total_rank: usize,
}

#[derive(Debug, Clone, Default)]
struct MlxStackedLoraLayer {
    q: Option<MlxStackedLoraProj>,
    k: Option<MlxStackedLoraProj>,
    v: Option<MlxStackedLoraProj>,
    o: Option<MlxStackedLoraProj>,
    gate: Option<MlxStackedLoraProj>,
    up: Option<MlxStackedLoraProj>,
    down: Option<MlxStackedLoraProj>,
}

#[derive(Debug, Clone, Default)]
struct MlxLoraLayer {
    q: Option<MlxLoraPair>,
    k: Option<MlxLoraPair>,
    v: Option<MlxLoraPair>,
    o: Option<MlxLoraPair>,
    gate: Option<MlxLoraPair>,
    up: Option<MlxLoraPair>,
    down: Option<MlxLoraPair>,
}

/// Hot-mounted low-rank adapter residing alongside the frozen 4-bit base model in Unified Memory.
#[derive(Debug, Clone)]
pub struct MlxLoraAdapter {
    /// Unique adapter ID.
    pub id: String,
    /// Base `alpha / rank` scale.
    pub base_scale: f32,
    /// Dynamic routing mode.
    pub routing: LoraRouting,
    layers: Vec<MlxLoraLayer>,
}

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
            } => quantized_matmul(
                x,
                weight,
                scales,
                Some(biases),
                QuantizedMatmulOptions {
                    transpose: true,
                    group_size: Some(*group_size),
                    bits: Some(*bits),
                    ..Default::default()
                },
            )
            .map_err(mlx_err),
        }
    }

    fn forward_with_stacked_lora(
        &self,
        x: &Array,
        stacked: Option<&MlxStackedLoraProj>,
        lora_scales: &[f32],
    ) -> Result<Array, NeuralError> {
        let mut out = self.forward(x)?;
        let Some(st) = stacked else {
            return Ok(out);
        };
        let mut any_active = false;
        let mut gate_vec = Vec::with_capacity(st.total_rank);
        for &(adapter_idx, r) in &st.slices {
            let s = lora_scales.get(adapter_idx).copied().unwrap_or(0.0);
            if s.abs() > 1e-4 {
                any_active = true;
            }
            gate_vec.extend(std::iter::repeat_n(s, r));
        }
        if !any_active {
            return Ok(out);
        }

        let target_dtype = out.dtype();
        let compute_dtype = st.a_cat_t.dtype();
        let x_cast = if x.dtype() == compute_dtype {
            x.clone()
        } else {
            x.as_dtype(compute_dtype).map_err(mlx_err)?
        };
        let total_r_i32 = i32::try_from(st.total_rank).unwrap_or(0);
        let gate_arr = Array::from_slice(&gate_vec, &[1, 1, total_r_i32])
            .as_dtype(compute_dtype)
            .map_err(mlx_err)?;
        let low = x_cast.matmul(&st.a_cat_t).map_err(mlx_err)?;
        let gated_low = low.multiply(&gate_arr).map_err(mlx_err)?;
        let delta = gated_low.matmul(&st.b_cat_t).map_err(mlx_err)?;
        let delta_cast = if delta.dtype() == target_dtype {
            delta
        } else {
            delta.as_dtype(target_dtype).map_err(mlx_err)?
        };
        out = out.add(&delta_cast).map_err(mlx_err)?;
        Ok(out)
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
        self.lookup_batch(step_tokens, 1, step_len, d)
    }

    fn lookup_batch(
        &self,
        batch_tokens: &[i32],
        batch_size: i32,
        step_len: i32,
        d: i32,
    ) -> Result<Array, NeuralError> {
        match self {
            Self::Dense(arr) => {
                let ids_arr = Array::from_slice(batch_tokens, &[batch_size, step_len]);
                arr.take_axis(&ids_arr, 0).map_err(mlx_err)
            }
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => {
                let ids_1d = Array::from_slice(batch_tokens, &[batch_size * step_len]);
                let w_rows = weight.take_axis(&ids_1d, 0).map_err(mlx_err)?;
                let s_rows = scales.take_axis(&ids_1d, 0).map_err(mlx_err)?;
                let b_rows = biases.take_axis(&ids_1d, 0).map_err(mlx_err)?;
                dequantize(
                    &w_rows,
                    &s_rows,
                    Some(&b_rows),
                    DequantizeOptions {
                        group_size: Some(*group_size),
                        bits: Some(*bits),
                        ..Default::default()
                    },
                )
                .map_err(mlx_err)?
                .reshape(&[batch_size, step_len, d])
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
    lora_sig: Vec<i32>,
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
            lora_sig: Vec::new(),
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

/// Metal-accelerated Qwen3 / Qwen2.5 Causal Transformer weights + Multi-LoRA sidecar in Apple
/// Silicon Unified Memory.
#[derive(Debug, Clone)]
pub struct MlxTransformer {
    config: ModelConfig,
    embed_tokens: MlxEmbedding,
    layers: Vec<MlxLayer>,
    final_norm: Array,
    lm_head: MlxLinear,
    loras: Vec<MlxLoraAdapter>,
    stacked_loras: Vec<MlxStackedLoraLayer>,
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
            loras: Vec::new(),
            stacked_loras: Vec::new(),
        })
    }

    /// Hot-mounts a `lora.safetensors` low-rank adapter (`model.layers.{i}.{proj}.{lora_A,lora_B}`)
    /// alongside the shared frozen base model in Unified Memory!
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if loading or transposing any `LoRA` matrix fails.
    pub fn mount_lora_safetensors(&mut self, spec: &LoraMountSpec) -> Result<(), NeuralError> {
        let mut raw = Array::load_safetensors(&spec.path).map_err(|e| {
            NeuralError::Safetensors(format!(
                "MLX load LoRA safetensors {} failed: {e}",
                spec.path.display()
            ))
        })?;

        let act_dtype = self.final_norm.dtype();
        let mut lora_layers = Vec::with_capacity(self.config.num_hidden_layers);
        let mut to_eval: Vec<Array> = Vec::new();
        let mut loaded_any = false;

        for idx in 0..self.config.num_hidden_layers {
            let p = format!("model.layers.{idx}");
            let mut take_pair = |proj: &str| -> Result<Option<MlxLoraPair>, NeuralError> {
                let ka1 = format!("{p}.{proj}.lora_A");
                let kb1 = format!("{p}.{proj}.lora_B");
                let ka2 = format!("{p}.{proj}.lora_A.weight");
                let kb2 = format!("{p}.{proj}.lora_B.weight");
                let a_opt = raw.remove(&ka1).or_else(|| raw.remove(&ka2));
                let b_opt = raw.remove(&kb1).or_else(|| raw.remove(&kb2));
                if let (Some(a), Some(b)) = (a_opt, b_opt) {
                    let rank =
                        usize::try_from(a.shape().first().copied().unwrap_or(0)).unwrap_or(0);
                    let a_t = transpose_2d(&a)?.as_dtype(act_dtype).map_err(mlx_err)?;
                    let b_t = transpose_2d(&b)?.as_dtype(act_dtype).map_err(mlx_err)?;
                    to_eval.push(a_t.clone());
                    to_eval.push(b_t.clone());
                    loaded_any = true;
                    Ok(Some(MlxLoraPair { a_t, b_t, rank }))
                } else {
                    Ok(None)
                }
            };

            lora_layers.push(MlxLoraLayer {
                q: take_pair("self_attn.q_proj")?,
                k: take_pair("self_attn.k_proj")?,
                v: take_pair("self_attn.v_proj")?,
                o: take_pair("self_attn.o_proj")?,
                gate: take_pair("mlp.gate_proj")?,
                up: take_pair("mlp.up_proj")?,
                down: take_pair("mlp.down_proj")?,
            });
        }

        if !loaded_any {
            return Err(NeuralError::Safetensors(format!(
                "no valid model.layers.* lora_A/lora_B tensors found in {}",
                spec.path.display()
            )));
        }

        eval(&to_eval).map_err(mlx_err)?;

        self.loras.retain(|existing| existing.id != spec.id);
        self.loras.push(MlxLoraAdapter {
            id: spec.id.clone(),
            base_scale: spec.scale,
            routing: spec.routing,
            layers: lora_layers,
        });
        self.rebuild_stacked_loras()?;
        Ok(())
    }

    fn rebuild_stacked_loras(&mut self) -> Result<(), NeuralError> {
        let mut stacked_layers = Vec::with_capacity(self.config.num_hidden_layers);
        let mut to_eval: Vec<Array> = Vec::new();

        for l_idx in 0..self.config.num_hidden_layers {
            let mut build_proj = |getter: fn(&MlxLoraLayer) -> Option<&MlxLoraPair>| -> Result<Option<MlxStackedLoraProj>, NeuralError> {
                let mut a_parts: Vec<&Array> = Vec::new();
                let mut b_parts: Vec<&Array> = Vec::new();
                let mut slices: Vec<(usize, usize)> = Vec::new();
                let mut total_rank = 0usize;
                for (a_idx, adapter) in self.loras.iter().enumerate() {
                    if let Some(ll) = adapter.layers.get(l_idx)
                        && let Some(pair) = getter(ll)
                        && pair.rank > 0
                    {
                        a_parts.push(&pair.a_t);
                        b_parts.push(&pair.b_t);
                        slices.push((a_idx, pair.rank));
                        total_rank += pair.rank;
                    }
                }
                if a_parts.is_empty() {
                    return Ok(None);
                }
                let a_cat_t = if a_parts.len() == 1 {
                    a_parts[0].clone()
                } else {
                    concatenate(&a_parts, 1).map_err(mlx_err)?
                };
                let b_cat_t = if b_parts.len() == 1 {
                    b_parts[0].clone()
                } else {
                    concatenate(&b_parts, 0).map_err(mlx_err)?
                };
                to_eval.push(a_cat_t.clone());
                to_eval.push(b_cat_t.clone());
                Ok(Some(MlxStackedLoraProj {
                    a_cat_t,
                    b_cat_t,
                    slices,
                    total_rank,
                }))
            };

            stacked_layers.push(MlxStackedLoraLayer {
                q: build_proj(|ll| ll.q.as_ref())?,
                k: build_proj(|ll| ll.k.as_ref())?,
                v: build_proj(|ll| ll.v.as_ref())?,
                o: build_proj(|ll| ll.o.as_ref())?,
                gate: build_proj(|ll| ll.gate.as_ref())?,
                up: build_proj(|ll| ll.up.as_ref())?,
                down: build_proj(|ll| ll.down.as_ref())?,
            });
        }

        if !to_eval.is_empty() {
            eval(&to_eval).map_err(mlx_err)?;
        }
        self.stacked_loras = stacked_layers;
        Ok(())
    }

    /// Number of active `Multi-LoRA` adapters mounted on this transformer.
    #[must_use]
    pub fn lora_count(&self) -> usize {
        self.loras.len()
    }

    /// Computes Simplex-bounded (`sum_k gamma_k <= 0.90`) dynamic per-adapter effective scales for
    /// `(app_bundle_id, left_context, cands)`, preventing multi-adapter register drift!
    #[must_use]
    pub fn compute_lora_scales(
        &self,
        app_bundle_id: &str,
        left_context: &str,
        candidate_texts: &[String],
    ) -> Vec<f32> {
        let mut raw: Vec<f32> = self
            .loras
            .iter()
            .map(|a| {
                a.routing
                    .effective_scale(a.base_scale, app_bundle_id, left_context, candidate_texts)
                    .clamp(0.0, 0.80)
            })
            .collect();
        let total: f32 = raw.iter().copied().sum();
        if total > 0.90 {
            let norm_factor = 0.90 / total;
            for s in &mut raw {
                *s *= norm_factor;
            }
        }
        raw
    }

    /// Returns model hyperparameters.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Incrementally prefills `prefix_ids` on Metal GPU, reusing cached KV states when both tokens
    /// and `Multi-LoRA` routing weights match!
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if an MLX Metal kernel fails.
    pub fn ensure_prefix_cached(
        &self,
        prefix_ids: &[u32],
        cache: &mut MlxPrefixCache,
    ) -> Result<(), NeuralError> {
        let default_scales: Vec<f32> = self.loras.iter().map(|a| a.base_scale).collect();
        self.ensure_prefix_cached_with_scales(prefix_ids, &default_scales, cache)
    }

    /// Incrementally prefills `prefix_ids` under explicit `lora_scales`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if an MLX Metal kernel fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn ensure_prefix_cached_with_scales(
        &self,
        prefix_ids: &[u32],
        lora_scales: &[f32],
        cache: &mut MlxPrefixCache,
    ) -> Result<(), NeuralError> {
        let target_i32: Vec<i32> = prefix_ids
            .iter()
            .map(|&id| i32::try_from(id).unwrap_or(0))
            .collect();
        if target_i32.is_empty() {
            return Ok(());
        }

        let sig: Vec<i32> = lora_scales
            .iter()
            .map(|&s| (s * 1000.0).round() as i32)
            .collect();
        let same_lora_sig = cache.lora_sig == sig;

        let common = if same_lora_sig {
            cache
                .cached_tokens
                .iter()
                .zip(target_i32.iter())
                .take_while(|(a, b)| a == b)
                .count()
        } else {
            0
        };

        if same_lora_sig
            && common == target_i32.len()
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
            self.forward_step(delta, keep, &cache.keys, &cache.values, lora_scales)?;
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
        cache.lora_sig = sig;
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
        let default_scales: Vec<f32> = self.loras.iter().map(|a| a.base_scale).collect();
        let joint = self.score_candidates_joint_lps_with_scales(cache, cands, &default_scales)?;
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
    /// using each mounted `LoRA`'s default `base_scale`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if an MLX Metal kernel fails during continuation pass.
    pub fn score_candidates_joint_lps(
        &self,
        cache: &MlxPrefixCache,
        cands: &[Vec<u32>],
    ) -> Result<Vec<f32>, NeuralError> {
        let default_scales: Vec<f32> = self.loras.iter().map(|a| a.base_scale).collect();
        self.score_candidates_joint_lps_with_scales(cache, cands, &default_scales)
    }

    /// Computes exact unnormalized joint log-likelihoods under explicit `lora_scales` using a
    /// **single batched `[B_cand, L_max]` Metal forward pass** over shared broadcast `past_kv` and
    /// on-GPU `take_along_axis` target log-probability gathering!
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::Safetensors`] if an MLX Metal kernel fails during continuation pass.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn score_candidates_joint_lps_with_scales(
        &self,
        cache: &MlxPrefixCache,
        cands: &[Vec<u32>],
        lora_scales: &[f32],
    ) -> Result<Vec<f32>, NeuralError> {
        if cands.is_empty() || cache.next_token_log_probs.is_empty() {
            return Ok(vec![0.0; cands.len()]);
        }

        let mut results = vec![0.0_f32; cands.len()];
        let mut multi_rows: Vec<usize> = Vec::with_capacity(cands.len());
        let mut max_cont_len = 0usize;

        for (c_idx, cand_ids) in cands.iter().enumerate() {
            if cand_ids.is_empty() {
                continue;
            }
            let first_idx = (cand_ids[0] as usize).min(cache.next_token_log_probs.len() - 1);
            results[c_idx] = cache.next_token_log_probs[first_idx];
            if cand_ids.len() > 1 {
                multi_rows.push(c_idx);
                max_cont_len = max_cont_len.max(cand_ids.len() - 1);
            }
        }

        if multi_rows.is_empty() {
            return Ok(results);
        }

        let vocab_max = i32::try_from(self.config.vocab_size.saturating_sub(1)).unwrap_or(0);
        let batch_size = multi_rows.len();
        let mut input_flat = vec![0_i32; batch_size * max_cont_len];
        let mut target_flat = vec![0_i32; batch_size * max_cont_len];

        for (row_idx, &c_idx) in multi_rows.iter().enumerate() {
            let cand_ids = &cands[c_idx];
            let steps = cand_ids.len() - 1;
            let row_base = row_idx * max_cont_len;
            for s in 0..steps {
                input_flat[row_base + s] =
                    i32::try_from(cand_ids[s]).unwrap_or(0).clamp(0, vocab_max);
                target_flat[row_base + s] = i32::try_from(cand_ids[s + 1])
                    .unwrap_or(0)
                    .clamp(0, vocab_max);
            }
        }

        let gathered = self.forward_batched_target_lps(
            &input_flat,
            &target_flat,
            batch_size as i32,
            max_cont_len as i32,
            cache.cached_tokens.len(),
            &cache.keys,
            &cache.values,
            lora_scales,
        )?;
        eval([&gathered]).map_err(mlx_err)?;

        let slice = gathered.as_slice::<f32>();
        for (row_idx, &c_idx) in multi_rows.iter().enumerate() {
            let steps = cands[c_idx].len() - 1;
            let row_base = row_idx * max_cont_len;
            for s in 0..steps {
                if let Some(&val) = slice.get(row_base + s) {
                    results[c_idx] += val;
                }
            }
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

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    fn forward_layer(
        &self,
        l_idx: usize,
        layer: &MlxLayer,
        hidden: &Array,
        batch_size: i32,
        step_len: i32,
        offset_i32: i32,
        past_kv: Option<(&Array, &Array)>,
        lora_scales: &[f32],
    ) -> Result<(Array, Array, Array), NeuralError> {
        let n_heads = self.config.num_attention_heads as i32;
        let n_kv = self.config.num_key_value_heads as i32;
        let head_dim = self.config.effective_head_dim() as i32;
        let q_out_dim = n_heads * head_dim;
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let normed =
            rms_norm(hidden, Some(&layer.input_norm), self.config.rms_norm_eps).map_err(mlx_err)?;

        let st_layer = self.stacked_loras.get(l_idx);

        let mut q_lin = layer.q_proj.forward_with_stacked_lora(
            &normed,
            st_layer.and_then(|sl| sl.q.as_ref()),
            lora_scales,
        )?;
        if let Some(b) = &layer.q_bias {
            q_lin = q_lin.add(b).map_err(mlx_err)?;
        }

        let mut k_lin = layer.k_proj.forward_with_stacked_lora(
            &normed,
            st_layer.and_then(|sl| sl.k.as_ref()),
            lora_scales,
        )?;
        if let Some(b) = &layer.k_bias {
            k_lin = k_lin.add(b).map_err(mlx_err)?;
        }

        let mut v_lin = layer.v_proj.forward_with_stacked_lora(
            &normed,
            st_layer.and_then(|sl| sl.v.as_ref()),
            lora_scales,
        )?;
        if let Some(b) = &layer.v_bias {
            v_lin = v_lin.add(b).map_err(mlx_err)?;
        }

        let mut q_reshaped = q_lin
            .reshape(&[batch_size, step_len, n_heads, head_dim])
            .map_err(mlx_err)?;
        if let Some(qn) = &layer.q_norm {
            q_reshaped =
                rms_norm(&q_reshaped, Some(qn), self.config.rms_norm_eps).map_err(mlx_err)?;
        }
        let q_heads = q_reshaped.transpose_axes(&[0, 2, 1, 3]).map_err(mlx_err)?;

        let mut k_reshaped = k_lin
            .reshape(&[batch_size, step_len, n_kv, head_dim])
            .map_err(mlx_err)?;
        if let Some(kn) = &layer.k_norm {
            k_reshaped =
                rms_norm(&k_reshaped, Some(kn), self.config.rms_norm_eps).map_err(mlx_err)?;
        }
        let k_heads = k_reshaped.transpose_axes(&[0, 2, 1, 3]).map_err(mlx_err)?;

        let v_heads = v_lin
            .reshape(&[batch_size, step_len, n_kv, head_dim])
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
            if batch_size > 1 && pk.shape().first().copied().unwrap_or(1) == 1 {
                let past_len = pk.shape().get(2).copied().unwrap_or(0);
                let past_keys_batch =
                    broadcast_to(pk, &[batch_size, n_kv, past_len, head_dim]).map_err(mlx_err)?;
                let past_vals_batch =
                    broadcast_to(pv, &[batch_size, n_kv, past_len, head_dim]).map_err(mlx_err)?;
                (
                    concatenate(&[&past_keys_batch, &k_rope], 2).map_err(mlx_err)?,
                    concatenate(&[&past_vals_batch, &v_heads], 2).map_err(mlx_err)?,
                )
            } else {
                (
                    concatenate(&[pk, &k_rope], 2).map_err(mlx_err)?,
                    concatenate(&[pv, &v_heads], 2).map_err(mlx_err)?,
                )
            }
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
            .reshape(&[batch_size, step_len, q_out_dim])
            .map_err(mlx_err)?;
        let attn_proj = layer.o_proj.forward_with_stacked_lora(
            &attn_flat,
            st_layer.and_then(|sl| sl.o.as_ref()),
            lora_scales,
        )?;
        let after_attn = hidden.add(&attn_proj).map_err(mlx_err)?;

        let post = rms_norm(
            &after_attn,
            Some(&layer.post_attn_norm),
            self.config.rms_norm_eps,
        )
        .map_err(mlx_err)?;
        let gate_raw = layer.gate_proj.forward_with_stacked_lora(
            &post,
            st_layer.and_then(|sl| sl.gate.as_ref()),
            lora_scales,
        )?;
        let gate = silu(&gate_raw).map_err(mlx_err)?;
        let up = layer.up_proj.forward_with_stacked_lora(
            &post,
            st_layer.and_then(|sl| sl.up.as_ref()),
            lora_scales,
        )?;
        let gated = gate.multiply(&up).map_err(mlx_err)?;
        let down = layer.down_proj.forward_with_stacked_lora(
            &gated,
            st_layer.and_then(|sl| sl.down.as_ref()),
            lora_scales,
        )?;
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
        lora_scales: &[f32],
    ) -> Result<(Array, Vec<Array>, Vec<Array>), NeuralError> {
        let step_len = step_tokens.len() as i32;
        let offset_i32 = pos_offset as i32;
        let d = self.config.hidden_size as i32;
        let mut hidden = self.embed_tokens.lookup(step_tokens, step_len, d)?;

        let mut next_keys = Vec::with_capacity(self.config.num_hidden_layers);
        let mut next_vals = Vec::with_capacity(self.config.num_hidden_layers);

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let past_kv = past_keys.get(l_idx).zip(past_vals.get(l_idx));
            let (h, k_full, v_full) = self.forward_layer(
                l_idx,
                layer,
                &hidden,
                1,
                step_len,
                offset_i32,
                past_kv,
                lora_scales,
            )?;
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

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_arguments
    )]
    fn forward_batched_target_lps(
        &self,
        input_flat: &[i32],
        target_flat: &[i32],
        batch_size: i32,
        step_len: i32,
        pos_offset: usize,
        past_keys: &[Array],
        past_vals: &[Array],
        lora_scales: &[f32],
    ) -> Result<Array, NeuralError> {
        let offset_i32 = pos_offset as i32;
        let d = self.config.hidden_size as i32;
        let mut hidden = self
            .embed_tokens
            .lookup_batch(input_flat, batch_size, step_len, d)?;

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let past_kv = past_keys.get(l_idx).zip(past_vals.get(l_idx));
            let (h, _, _) = self.forward_layer(
                l_idx,
                layer,
                &hidden,
                batch_size,
                step_len,
                offset_i32,
                past_kv,
                lora_scales,
            )?;
            hidden = h;
        }

        let out_norm =
            rms_norm(&hidden, Some(&self.final_norm), self.config.rms_norm_eps).map_err(mlx_err)?;
        let logits = self
            .lm_head
            .forward(&out_norm)?
            .as_dtype(MlxDtype::Float32)
            .map_err(mlx_err)?;
        let lps = log_softmax(&logits, -1).map_err(mlx_err)?;
        let target_arr = Array::from_slice(target_flat, &[batch_size, step_len, 1]);
        lps.take_along_axis(&target_arr, -1)
            .map_err(mlx_err)?
            .reshape(&[batch_size * step_len])
            .map_err(mlx_err)
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
