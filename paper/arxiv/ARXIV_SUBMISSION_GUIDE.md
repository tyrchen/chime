# arXiv 投稿一键发布指南（Chime Paper）

本目录已为您打包并本地编译验证好 **arXiv 官方首选的 LaTeX 源码归档包**（[chime-arxiv-source.tar.gz](./chime-arxiv-source.tar.gz)），可直接通过 arXiv 的 TeX Live (`pdflatex` + `CJKutf8`) 自动编译流水线并生成 arXiv HTML5 在线阅读版，无需人工 PDF 审核排队。

---

## 一、投稿文件选择（二选一，强烈推荐方案 A）

- **方案 A（官方首选 · 秒过 AutoTeX 且支持 arXiv HTML5）**：
  在 `https://arxiv.org/submit` 的 **Add Files** 步骤上传单个源码包：
  - `paper/arxiv/chime-arxiv-source.tar.gz`
  - 内含平铺的 5 个纯净文件（已剥离 macOS `._*` 隐藏属性）：`00README.json`、`main.tex`（首行含 `\pdfoutput=1`）、预编译 `main.bbl`、`fig1_architecture.pdf`、`fig2_scaling.pdf`。
- **方案 B（备选 · Typst 原生排版 PDF-Only）**：
  若希望直接展示 Typst 排版成品，可上传 `paper/chime-paper.pdf`（已通过 `pdffonts` 验证 13/13 矢量轮廓字体 100% 嵌入、零 Type-3 位图字体），并在 `Comments` 栏保留 `Typeset with Typst` 说明。

---

## 二、网页表单逐项粘贴内容（Metadata）

### 1. Category（学科分类）
- **Primary Category（主分类）**：`cs.CL`（Computation and Language）
- **Cross-List（交叉分类，可选推荐）**：`cs.LG`（Machine Learning）、`cs.HC`（Human-Computer Interaction）

### 2. Title（标题）
```text
Chime: Sub-3ms On-Device Neural Pinyin Input via Manifold-Aligned Distillation and Rank-Stacked Multi-LoRA Cartridges
```

### 3. Authors（作者）
```text
Tian Chen
```

### 4. Abstract（摘要 · 已严格控制在 1,689 字符 < arXiv 1,920 字符上限，纯 ASCII + MathJax）
```text
Applying Large Language Models (LLMs) to client-side Pinyin Input Method Editors (IMEs) faces a strict systems trilemma: human typing demands sub-10 ms keystroke latency without candidate-bar jitter, operating systems enforce tight resident memory budgets (< 100 MB) alongside zero-network privacy guarantees, and real-world input spans open-domain vocabulary from bilingual code identifiers to classical literature and idiosyncratic personal style. Existing generative Pinyin-to-Character (P2C) LLMs violate latency budgets via autoregressive decoding and suffer catastrophic representation collapse under aggressive layer pruning or out-of-vocabulary (OOV) blindness when relying solely on parameter-efficient adapters. We present Chime, a 100% offline neural IME for Apple Silicon that resolves this trilemma through lexical-neural co-design. First, we formalize Recall-Rerank Decoupling and Probability-Space Separation, pairing a < 0.8 ms zero-copy mmap dual-track lattice decoder (CHIMED02) with a batched prefix-cached causal language model reranker that isolates equal-span homophone scoring from variable-horizon lookahead completions. Second, we compress the 28-layer Qwen3-0.6B (1.19 GB FP16) into a 6-layer, 24,576-vocab 4-bit MLX student (64.17 MB, 18.6x smaller); our Manifold-Aligned Layer Distillation restores pre-head teacher cosine alignment from 0.022 to 0.853, matching the 28-layer teacher's 93.8% zero-shot stress accuracy and achieving 100% (44/44) end-to-end IME accuracy at 0.86 ms 6-candidate Metal reranking latency (6.15x faster). Third, we introduce pluggable .imepack cartridges co-mounting zero-copy lexical slices with Q4-Manifold Aligned 2.07 MB Rank-8 LoRA sidecars. Combining Rank-Stacked GEMM Fusion, Batched Broadcast KV-Gather, and a Couplet-Aware Simplex Router cuts 6-adapter reranking from 10.82 ms to 1.45 ms (7.46x speedup), eliminates multi-adapter register drift, and removes FP16-to-INT4 scale overshoot (-0.270 nats classical NLL gain).
```

### 5. Comments（备注）
```text
8 pages, 2 figures, 4 tables. Open-source Rust/Swift implementation and prebuilt Multi-LoRA .imepack cartridges available at https://github.com/tyrchen/chime
```

### 6. ACM Class（可选分类号）
```text
I.2.7; D.4.7; H.5.2
```

### 7. License（授权协议建议）
- 推荐选择 **`CC BY 4.0`**（Creative Commons Attribution 4.0 International）或 **`arXiv.org perpetual, non-exclusive license`**。

---

## 三、2 分钟提交流程

1. 登录 [https://arxiv.org/submit](https://arxiv.org/submit)，点击 **Start New Submission**。
2. 勾选协议与推荐的 **`CC BY 4.0`** 许可证，主分类选 `cs.CL`（交叉分类勾选 `cs.LG`、`cs.HC`）。
3. 在 **Add Files** 页面上传 `paper/arxiv/chime-arxiv-source.tar.gz`，点击 **Process Files** 并目检右侧自动生成的 PDF 预览。
4. 在 **Add Metadata** 页面粘贴上方的 `Title`、`Authors`、`Abstract` 与 `Comments`。
5. 在最终预览页确认无误后点击 **Submit Article** 完成发布（发布后拿到 `arXiv:2611.xxxxx` 编号即可填入 GitHub 仓库与主页）。
