#import "figures.typ": fig-architecture, fig-scaling-and-transfer

#set document(
  title: "Chime: Sub-3ms On-Device Neural Pinyin Input via Manifold-Aligned Distillation and Rank-Stacked Multi-LoRA Cartridges",
  author: "Tian Chen",
)

#set page(
  paper: "us-letter",
  margin: (top: 0.80in, bottom: 0.82in, left: 0.70in, right: 0.70in),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, fill: rgb("#71717a"))
      Chime: Sub-3ms On-Device Neural Pinyin Input via Manifold-Aligned Distillation and Multi-LoRA Cartridges
      #h(1fr)
      Preprint · 2026
    ]
  },
  numbering: "1",
)

#set text(
  font: ("New Computer Modern", "Noto Serif CJK SC", "Songti SC"),
  size: 9.6pt,
  lang: "en",
)

// Optically match CJK ideographs and fullwidth punctuation to New Computer Modern cap/x-height,
// and match inline ASCII monospace tokens without ever catching CJK inside raw blocks.
#show regex("[\p{scx=Han}，。！？；：、《》（）【】]+"): set text(
  font: ("Noto Serif CJK SC", "Songti SC"),
  size: 0.90em,
)
#show raw: set text(
  font: ("PT Mono", "JetBrains Mono NL", "DejaVu Sans Mono"),
  size: 0.92em,
)
#show "—": it => sym.wj + it

#set par(
  justify: true,
  leading: 0.55em,
  spacing: 0.72em,
)

#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

// Clear, unmistakable 3-tier academic heading hierarchy (never hyphenated or justified).
#show heading.where(level: 1): it => block(above: 1.42em, below: 0.60em, sticky: true)[
  #set par(justify: false, leading: 0.42em)
  #set text(size: 12.2pt, weight: "bold", hyphenate: false)
  #it
]
#show heading.where(level: 2): it => block(above: 1.16em, below: 0.46em, sticky: true)[
  #set par(justify: false, leading: 0.42em)
  #set text(size: 10.7pt, weight: "bold", hyphenate: false)
  #it
]
#show heading.where(level: 3): it => block(above: 0.92em, below: 0.38em, sticky: true)[
  #set par(justify: false, leading: 0.42em)
  #set text(size: 9.8pt, weight: "bold", style: "italic", hyphenate: false)
  #it
]

#show table: set text(size: 8.2pt)
#show table: set par(justify: false, leading: 0.45em)

#show figure.caption: it => context [
  #set text(size: 8.5pt)
  #set par(justify: true, leading: 0.48em)
  *#it.supplement #it.counter.display():* #it.body
]

#let toprule = table.hline(stroke: 0.85pt + black)
#let midrule = table.hline(stroke: 0.45pt + rgb("#3f3f46"))
#let bottomrule = table.hline(stroke: 0.85pt + black)

#let prop-box(label, body) = block(
  width: 100%,
  fill: rgb("#f8fafc"),
  stroke: (left: 2.2pt + rgb("#0f4c5c"), rest: 0.4pt + rgb("#cbd5e1")),
  inset: (x: 8pt, y: 6pt),
  radius: (right: 3pt),
)[
  *#label.* #emph[#body]
]

// ─── Title & Abstract (Full-Width Span) ───
#place(
  top + center,
  scope: "parent",
  float: true,
)[
  #align(center)[
    #block(width: 96%)[
      #set par(justify: false, leading: 0.44em)
      #set text(hyphenate: false)
      #text(size: 14.6pt, weight: "bold")[
        Chime: Sub-3ms On-Device Neural Pinyin Input\
        via Manifold-Aligned Distillation and Rank-Stacked Multi-LoRA Cartridges
      ]
      #v(7pt)
      #text(size: 10.4pt, weight: "semibold")[Tian Chen]
      #v(2pt)
      #text(size: 8.9pt, fill: rgb("#52525b"))[
        Independent Systems & Machine Learning Research · `tyr.chen@gmail.com` · #link("https://github.com/tyrchen/chime")[`github.com/tyrchen/chime`]
      ]
    ]
  ]
  #v(9pt)
  #align(center)[
    #block(
      width: 93%,
      fill: rgb("#fafafa"),
      stroke: 0.5pt + rgb("#e4e4e7"),
      radius: 4pt,
      inset: (x: 14pt, y: 9.5pt),
    )[
      #align(center)[#text(size: 9.6pt, weight: "bold")[Abstract]]
      #v(3pt)
      #set par(justify: true, leading: 0.50em)
      #set text(size: 8.85pt)
      Applying Large Language Models (LLMs) to client-side Pinyin Input Method Editors (IMEs) faces a strict systems trilemma: human typing demands sub-10 ms keystroke latency without candidate-bar jitter, operating systems enforce tight resident memory budgets ($< 100$ MB) alongside zero-network privacy guarantees, and real-world input spans open-domain vocabulary from bilingual code identifiers to classical literature and idiosyncratic personal style. Existing generative Pinyin-to-Character (P2C) LLMs violate latency budgets via autoregressive decoding and suffer catastrophic representation collapse under aggressive layer pruning or out-of-vocabulary (OOV) blindness when relying solely on parameter-efficient adapters. We present *Chime*, a 100% offline neural IME for Apple Silicon that resolves this trilemma through lexical-neural co-design. First, we formalize _Recall–Rerank Decoupling_ and _Probability-Space Separation_, pairing a $< 0.8$ ms zero-copy `mmap` dual-track lattice decoder (`CHIMED02`) with a batched prefix-cached causal language model reranker that isolates equal-span homophone scoring from variable-horizon lookahead completions. Second, we compress the 28-layer `Qwen3-0.6B` (1.19 GB FP16) into a 6-layer, 24,576-vocab 4-bit MLX student (*64.17 MB*, 18.6× smaller); whereas naive layer skipping collapses pre-head teacher alignment to $cos(bold(u)_S, bold(u)_T) = 0.022$ and halves homophone accuracy to 43.8%, our _Manifold-Aligned Layer Distillation_ restores $cos(bold(u)_S, bold(u)_T) = 0.853$, matching the 28-layer teacher's *93.8%* (15/16) zero-shot stress accuracy and achieving *100%* (44/44) end-to-end IME accuracy at *0.86 ms* 6-candidate Metal reranking latency (6.15× faster than 28-layer 4-bit). Third, we introduce pluggable _`.imepack` cartridges_ that co-mount zero-copy lexical slices (`pack.imedict`) with _Q4-Manifold Aligned_ 2.07 MB Rank-8 `LoRA` sidecars. Combining _Rank-Stacked GEMM Fusion_, _Batched Broadcast KV-Gather_, and a _Couplet-Aware Simplex Router_ cuts 6-adapter reranking latency from 10.82 ms to *1.45 ms* (*7.46× speedup*, $+39.6 thin mu"s"$/adapter), eliminates multi-adapter register drift, and removes FP16$arrow.r$INT4 scale overshoot ($-0.270$ nats classical NLL gain).
    ]
  ]
  #v(4pt)
]

#show: rest => columns(2, gutter: 0.25in, rest)

= Introduction

Every sentence written on a desktop or mobile device passes through an Input Method Editor (IME). For over a billion Chinese speakers, Pinyin-to-Character (P2C) conversion is a quintessential noisy-channel inference problem @chen2000new @jia2014joint: a user types a romanized phonetic keystroke stream—frequently interleaved with initial-letter abbreviations (`wsm` $arrow.r$ 为什么), regional phonetic mergers (`z`/`zh`, `in`/`ing`), motor slips on QWERTY layouts (`shaung` $arrow.r$ `shuang`), and unswitched English identifiers (`macOS`, `GitHub`)—expecting the intended text at slot \#1 within a single display frame.

Recent advances in Large Language Models (LLMs) have inspired a wave of generative IME prototypes @tan2022exploring @ding2023geneinput @gui2025attninput @shan2026huoziime that cast P2C as end-to-end conditional sequence generation. Despite strong perplexity on clean benchmarks, deploying LLMs inside a production OS input method exposes a fundamental *Latency–Memory–Recall Trilemma*:

1. *Perceptual Latency & Visual Jitter*: Human typists perceive interface lag above 10–15 ms and commit muscle-memory errors if slot \#1 shifts after $approx 60$ ms of visual fixation. Autoregressive LLM generation over candidate character sequences incurs 25–150 ms per keystroke even on modern accelerators, while cloud-offloaded IMEs introduce variable network tail latency and transmit every keystroke off-device.
2. *Resident Memory Ceiling & Collapse*: An OS-resident background service active across every window cannot consume gigabytes of RAM. Yet naively pruning a 28-layer, 1.19 GB model (`Qwen3-0.6B` @yang2025qwen3) down to 6 layers and 4-bit weights ($approx 64$ MB) destroys the alignment between intermediate transformer blocks and the frozen vocabulary embedding head, collapsing cosine similarity to $0.022$ and degrading homophone disambiguation to near-chance (43.8%).
3. *Lexical Recall vs. Multi-Domain Interference*: Neither a 64 MB base model nor Low-Rank Adaptation (`LoRA` @hu2022lora) alone can recall 260,000+ long-tail classical verses (e.g., 念去去，千里烟波) if Stage-1 beam search prunes them. Conversely, mounting multiple domain `LoRA`s simultaneously (e.g., classical literature, developer prose, and personal essays) triggers two failure modes on edge GPUs: _Register Superposition Drift_—where summing written-domain adapters overwhelms colloquial priors (flipping 究竟有多大 to 多达)—and _Sequential Kernel Launch Inflation_, which scales latency linearly with adapter count $K$.

In this paper, we design, implement, and empirically evaluate *Chime*, an open-source, zero-telemetry macOS IME written in Rust and Swift 6 on Apple Silicon Unified Memory @hannun2023mlx. Rather than forcing a compact LLM to hallucinate character mappings from raw keystrokes, Chime establishes a principled division of labor between deterministic zero-copy data structures and manifold-aligned low-rank neural rescoring (@fig-arch).

#figure(
  fig-architecture(),
  kind: image,
  scope: "parent",
  placement: top,
  caption: [
    End-to-end architecture of *Chime*. Stage 1 maps layered `CHIMED02` binary dictionaries via `mmap` ($< 0.8$ ms) and separates equal-span candidates $cal(C)_n$ from lookahead completions $cal(C)_"comp"$. Stage 2 runs a frozen 64.17 MB 4-bit `Qwen3-6L` student with batched broadcast `MlxPrefixCache`, on-GPU `take_along_axis`, and fused Rank-Stacked `.imepack` `LoRA` sidecars under a 60 ms anti-jitter contract.
  ],
) <fig-arch>

Our primary contributions are fourfold:
- *Recall–Rerank Co-Design & Measure Separation (§3)*: We prove that Stage-2 neural rescoring is upper-bounded by Stage-1 pool recall and must separate equal-span homophone hypotheses $cal(C)_n$ from lookahead completions $cal(C)_"comp"$ to avoid cross-horizon probability distortion, paired with sublinear length calibration $|c|^(-0.42)$ across heterogeneous BPE tokenizations.
- *Manifold-Aligned 6-Layer Student Distillation (§4)*: Combining vocabulary pruning ($"151,936" arrow.r "24,576"$ tokens), frozen tied embeddings, trajectory MSE, pre-head cosine alignment, and dual-temperature KL distillation, we compress `Qwen3-0.6B` by 18.6× into a 64.17 MB 4-bit MLX student that matches the 28-layer teacher's zero-shot disambiguation accuracy (93.8%) at *0.86 ms* 6-candidate batched reranking (6.15× faster).
- *Rank-Stacked `.imepack` Cartridges & Q4-Manifold Alignment (§5)*: We package zero-copy `pack.imedict` slices alongside 2.07 MB Rank-8 `LoRA`s into hot-swappable cartridges. Combining _Rank-Stacked GEMM Fusion_ with _Batched Broadcast KV-Gather_ (`take_along_axis`) cuts 6-adapter reranking from 10.82 ms to *1.45 ms* (*7.46× speedup*, $+39.6 thin mu"s"$ marginal cost/adapter), while _Q4-Manifold LoRA Training_ and our _Couplet-Aware Simplex Router_ eliminate multi-adapter register drift and INT4 transfer overshoot.
- *System Evaluation & Longitudinal Personalization (§6)*: Across classical Chinese (266K entries, 231K `Tab` chains), developer code-switching, and an air-gapped 815K-character pre-LLM longitudinal personal corpus (2013–2020), Chime achieves *44/44 (100%)* benchmark accuracy and 9× classical recall gain ($10.0% arrow.r 90.0%$).

= Related Work

*Statistical and Graph-Based Pinyin Decoding.* Classical P2C engines model conversion as Viterbi search over word lattices weighted by character/word #box[$n$-grams] @chen2000new and double-array tries @yoshinaga1992double. @jia2014joint introduced a joint graph model integrating QWERTY typo correction with P2C conversion. While sub-millisecond and memory-frugal, finite-context #box[$n$-grams] fail on context-sensitive homophones (e.g., 保障*权利* [safeguard rights] vs. 滥用*权力* [abuse power], especially across intervening modifiers).

*Neural and LLM-Based Input Methods.* @zhang2019open augmented neural P2C with online open-vocabulary sampling. @xu2022pert proposed `PERT`, a bidirectional encoder pre-trained directly on Pinyin sequences. With the rise of generative LLMs, @tan2022exploring adapted frozen Chinese GPT to P2C, observing severe degradation on abbreviated Pinyin due to combinatorial candidate explosion. `GeneInput` @ding2023geneinput unified full-mode key-to-character generation via prompting and RLHF, `AttnInput` @gui2025attninput attached a Pinyin side-network to linear RWKV models, and `HuoziIME` @shan2026huoziime explored on-device LLM personalization via hierarchical memory. Unlike these end-to-end generative designs—which struggle with sub-5 ms keystroke budgets, multi-syllable typo graphs, and exact casing for code-switching—Chime retains deterministic $O(log N)$ lattice recall in Stage 1 and deploys a prefix-cached causal SLM strictly as a constant-depth parallel rescorer in Stage 2.

*Edge Compression and Multi-LoRA Serving.* Knowledge distillation @hinton2015distilling with hidden-state alignment (`TinyBERT` @jiao2020tinybert) and group-wise weight quantization (`GPTQ` @frantar2023gptq, `AWQ` @lin2024awq) enable compact edge deployment. In cloud serving, `S-LoRA` @sheng2024slora batches heterogeneous requests across distinct per-tenant adapters using custom paging kernels, whereas personalized `LoRA` routing (`PLoRA` @zhang2024plora) blends adapters per user. To our knowledge, Chime is the first system to investigate _single-stream concurrent Multi-LoRA superposition_ on unified-memory client GPUs—identifying and solving both multi-adapter register interference and zero-shot FP16-to-INT4 scale overshoot.

= Problem Formulation & Hybrid Co-Design

== Joint Phonetic-Typo-Code Lattice Decoding

Given a raw ASCII keystroke sequence $bold(k) = k_(1:n)$, surrounding left document context $bold(c)_L$ (bounded to 96 characters), and active host application identifier $a$ (e.g., `com.mitchellh.ghostty` vs. `md.obsidian`), an IME seeks the highest-scoring surface string $bold(w)^*$:
$ bold(w)^* = arg max_(bold(w) in cal(G)(bold(k))) [& S_"lex" (bold(w), bold(k) | bold(c)_L, a) \ & + lambda dot S_"slm" (bold(w) | bold(c)_L, a)] $ <eq-joint>
where $cal(G)(bold(k))$ is the dual-track segmentation lattice constructed over byte spans $[i, j) subset.eq [0, n)$.

To achieve $< 0.8$ ms Stage-1 construction without memory bloat, Chime compiles static and cartridge lexicons into `CHIMED02`, a 48-byte-header immutable binary image mapped directly into virtual memory via `mmap` (`memmap2`). Rather than expanding fuzzy Pinyin permutations offline—which multiplies dictionary size by $2^6$—Stage 1 evaluates an algebraic equivalence relation $cal(F)(s)$ (`z`$harpoons.rtlb$`zh`, `an`$harpoons.rtlb$`ang`, `in`$harpoons.rtlb$`ing`, etc.) dynamically at span-lookup time, alongside four 1-edit motor-noise operators on raw slice $bold(k)_(i:j)$:
1. _Adjacent transposition_ (`shaung` $arrow.r$ `shuang`, `uq` $arrow.r$ `qu`, cost $delta_"swap" = 1.05$);
2. _QWERTY Euclidean neighbor substitution_ ($d^2 <= 1.85$ key pitches, e.g., `shuruga` $arrow.r$ `shurufa`, cost $0.85 + 0.45 d^2$);
3. _Accidental key-repeat deletion_ (`niian` $arrow.r$ `nian`, cost $delta_"rep" = 1.10$);
4. _Geminate boundary consonant elision_ (`chunuan` $arrow.r$ `chun`+`nuan` 春暖, `jiangun` $arrow.r$ `jiang`+`gun` 江滚 when a shared coda/onset `n` or `g` is pressed once before a vowel, cost $delta_"gem" = 0.85$).

Multi-syllable exact phrases, idioms, and merged classical clauses up to 16 syllables are traversed via depth-first span expansion gated at every step by an $O(log N)$ binary search (`has_longer_pinyin_prefix`) over sorted 16-byte `pinyin_groups`, pruning dead typo branches immediately at length 1–2. Simultaneously, a parallel code-switching track scores canonical-case English identifiers (`macos` $arrow.r$ `macOS`, `chatgpt` $arrow.r$ `ChatGPT`) and camelCase tokens (`useEffect`) with automatic Pangu boundary spacing.

== Recall–Rerank Decoupling & Measure Separation

Why not rely on `LoRA` fine-tuning alone to inject domain phrases or classical poetry? Let $cal(C)_M (bold(k)) subset cal(G)(bold(k))$ denote the top-$M$ candidates retained by Stage-1 beam search ($M=15$).

#prop-box("Proposition 1 (Stage-1 Recall Bottleneck)")[
  For any finite Stage-2 rescorer $S_"slm" : Sigma^* arrow.r RR$ with bounded dynamic range $|S_"slm"| <= B$, if a target multi-character entity or verse $bold(w)^*$ is absent from the mounted lexicon such that fragmented character paths push $bold(w)^*$ outside $cal(C)_M (bold(k))$, then $PP(bold(w)^* = bold(w)_1) = 0$ regardless of `LoRA` rank $r$ or Stage-2 capacity.
]

Proposition 1 explains why typing `mingyuejishiyou` without a classical lexicon yields 明月即使有 (_"even if the bright moon..."_): modern high-frequency bigrams (即使, $log P = -3.8$) dominate fragmented character beams (几 + 时, $log P = -8.9$), evicting 明月几时有 before Stage 2 executes.

Equally critical is _which_ candidates inside $cal(C)_M (bold(k))$ may be compared by $S_"slm"$. Modern IMEs surface both _full-span interpretations_ $cal(C)_n$ (consuming $bold(k)_(1:n)$) and _prefix lookahead completions_ $cal(C)_"comp"$ (e.g., typing 3 syllables `junbujian` surfaces the 10-syllable verse completion 君不见高堂明镜悲白发).

#prop-box("Proposition 2 (Measure Incompatibility across Horizons)")[
  Let $bold(w)_s in cal(V)^m$ be an $m$-character full-span candidate and $bold(w)_ell = bold(w)_s circle.tiny bold(w)_"rem" in cal(V)^(m+d)$ ($d >= 1$) be a lookahead completion extending $bold(w)_s$. Under any causal language model $P_theta$, by the chain rule:
  #set math.equation(numbering: none)
  $ log P_theta (bold(w)_ell | bold(c)_L) &= log P_theta (bold(w)_s | bold(c)_L) \ & quad + sum_(t=1)^d log P_theta (w_(m+t) | bold(c)_L, bold(w)_(<m+t)) \ &< log P_theta (bold(w)_s | bold(c)_L) $
  whenever $P_theta (w_(m+t) | dot) < 1$. Under sublinear length normalization $|bold(w)|^alpha$ ($alpha < 1$), $bold(w)_ell$ incurs a systematic horizon penalty of $Theta(d^(1-alpha))$ nats relative to $bold(w)_s$.
]

Guided by Proposition 2, Chime enforces _Probability-Space Separation_: Stage-2 zero-meaned neural rescoring operates strictly over the equal-input homophone subspace $cal(C)_n backslash cal(C)_"comp"$, while lookahead completions $cal(C)_"comp"$ are governed by prefix selectivity and deterministic $O(log C)$ continuation chains (`continuations_off` at byte offset 40 of `CHIMED02`), plus an anti-repeat penalty ($-5.5$ nats) if a #box[$>= 5$-character] verse already appears in $bold(c)_L$.

Within $cal(C)_n backslash cal(C)_"comp"$, candidates share identical character length ($|bold(w)|$) on homophones (权利 vs. 权力) yet often differ in BPE token count $T(bold(w))$: formal compounds like 多达 occupy 1 subword token, whereas colloquial phrases like 多大 split into 2 tokens (多 + 大). Dividing by $T(bold(w))$ artificially inflates 2-token candidates because splitting a joint probability $log P(w_1, w_2)$ by 2 halves its magnitude; instead, Chime sums the _exact unnormalized joint log-likelihood_ $ell(bold(w)) = sum_(t=1)^(T(bold(w))) log P_theta (z_t | bold(c)_L, z_(<t))$ and normalizes solely by _character length_ $|bold(w)|^(0.42)$:
$ S_"slm" (bold(w) | bold(c)_L, a) = lambda ( frac(ell(bold(w)), |bold(w)|^(0.42)) - frac(1, |cal(B)|) sum_(bold(w)' in cal(B)) frac(ell(bold(w)'), |bold(w)'|^(0.42)) ) $ <eq-rerank>
where $lambda = 0.65$ and $cal(B) subset.eq cal(C)_n backslash cal(C)_"comp"$ ($|cal(B)| <= 6$).

= Manifold-Aligned 6-Layer Distillation

== Vocabulary Pruning & Naive Skip Collapse

Our teacher is `Qwen/Qwen3-0.6B-Base` @yang2025qwen3 ($L_T = 28$ layers, $d = 1024$, $d_"ff" = 3072$, 16 query heads, 8 KV heads with `head_dim` $= 128$, QK-RMSNorm, tied embeddings, and vocabulary $V_T = "151,936"$, totaling 1,191.5 MB in FP16). Notice that in a 6-layer student with full vocabulary, the $"151,936" times 1024$ embedding table alone consumes 311 MB in FP16 (78 MB in 4-bit), dominating the transformer blocks. We prune the vocabulary to $V_S = "24,576"$ pieces covering all GB2312/common Unihan characters, high-frequency Chinese multi-character subwords, punctuation, and ASCII/developer subwords, shrinking the 4-bit embedding table by *83.8%* to just *12.6 MB*.

However, initializing a 6-layer student by uniformly selecting teacher layers $cal(I) = \{0, 5, 11, 16, 22, 27\}$ while freezing the pruned teacher embedding matrix $bold(E) in RR^(V_S times d)$ causes severe _Manifold Misalignment_. Because residual transformers refine representations incrementally across 28 blocks before projecting onto $bold(E)^top$, skipping 22 blocks rotates the pre-head hidden vector $bold(u)_S = "RMSNorm"(bold(h)_S^((6)))$ nearly orthogonal to the teacher's $bold(u)_T = "RMSNorm"(bold(h)_T^((28)))$: empirically, $cos(bold(u)_S, bold(u)_T)$ plummets to *0.0221* (@tab-pareto), causing colloquial 多大 to lose to 多达 and cutting homophone accuracy in half (43.8%).

== Composite Trajectory & Cosine-KL Objective

To realign the 6-layer student directly onto the frozen teacher embedding geometry $bold(E)$ (preventing embedding overfitting on finite distillation corpora), we freeze $bold(E)$ and optimize only the 6 transformer blocks $theta_S$ over 9.94M tokens of mixed conversational, encyclopedic, code-switching, and literary text using a three-term objective:
$ cal(L)(theta_S) &= alpha_"kl" cal(L)_"DKL" + alpha_"cos" (1 - cos(bold(u)_S, bold(u)_T)) \ & quad + frac(alpha_"hid", |cal(M)|) sum_((m, ell) in cal(M)) norm( hat(bold(h))_S^((m)) - hat(bold(h))_T^((ell)) )_2^2 $ <eq-distill>
where $hat(bold(h)) = bold(h) / norm(bold(h))_2$ aligns intermediate block trajectories across layer map $cal(M) = \{(1,5), (2,11), (3,16), (4,22), (5,27)\}$, and $cal(L)_"DKL"$ is a _Dual-Temperature Forward KL_ over the pruned 24,576-logit head $bold(z) = bold(u) bold(E)^top$:
$ cal(L)_"DKL" &= 0.7 tau_1^2 D_"KL" (sigma(bold(z)_T / tau_1) || sigma(bold(z)_S / tau_1)) \ & quad + 0.3 D_"KL" (sigma(bold(z)_T) || sigma(bold(z)_S)) $ <eq-dkl>
with $tau_1 = 2.0$, $alpha_"kl" = 1.0$, $alpha_"cos" = 2.5$, and $alpha_"hid" = 1.5$. Training for just 1,600 steps (batch 32, length 192, peak LR $3 times 10^(-4)$) on a single NVIDIA L4 GPU takes 14 minutes, driving $cos(bold(u)_S, bold(u)_T)$ from $0.0221 arrow.r 0.9310$ on validation sequences ($0.8531$ after group-64 4-bit MLX affine quantization into *64.17 MB*).

= Rank-Stacked Multi-LoRA Cartridges

== The `.imepack` Cartridge Abstraction

To let users extend Chime across specialized domains without bloating or retraining the 64.17 MB base model, we introduce `.imepack` v2 cartridges. Each cartridge bundles:
1. `pack.imedict`: A zero-copy `CHIMED02` slice containing domain words, canonical English terms, #box[$n$-grams], and $O(log C)$ `Tab` continuation records;
2. `lora.safetensors`: Rank-$r$ ($r=8$) FP16 low-rank matrices $(bold(A)_k^((ell, p)) in RR^(r times d_"in"), bold(B)_k^((ell, p)) in RR^(d_"out" times r))$ across all 7 linear projections $p in \{q, k, v, o, "gate", "up", "down"\}$ of all 6 layers ($"1,032,192"$ parameters $= 2.07$ MB per pack);
3. `manifest.json`: Declaring routing policy (`classical`, `developer`, or `always`), base scale $s_k$, and privacy flag `private_pack`.

== Rank-Stacked GEMM Fusion & Batched KV-Gather

When $K$ cartridges are mounted concurrently, a naive implementation evaluates each active adapter sequentially inside every linear layer:
$ bold(y) = "QuantMatmul"_"4bit" (bold(x), hat(bold(W))) + sum_(k=1)^K gamma_k (bold(x) bold(A)_k^top) bold(B)_k^top $ <eq-unfused>
Across 6 layers and 7 projections, @eq-unfused dispatches $42 times 2K$ tiny skinny-matrix multiplications (`[1, T, 1024] @ [1024, 8]`) plus $42 times 2K$ scalar/add kernels per forward pass. On Apple Silicon Metal GPUs, kernel launch and command-encoder overhead completely dominate arithmetic cost at $r=8$, causing 6-candidate reranking latency to grow linearly by $+1.58$ ms per adapter (reaching 10.82 ms at $K=6$, violating our 10 ms frame-0 synchronous budget).

We eliminate this bottleneck via _Rank-Stacked GEMM Fusion_. At cartridge mount time, Chime pre-transposes and concatenates all $K$ adapters along the rank dimension once in Unified Memory ($R = sum_(k=1)^K r_k = 8K$):
$ bold(A)_"cat"^top &= [ bold(A)_1^top thin dots thin bold(A)_K^top ] in RR^(d_"in" times R), \ bold(B)_"cat"^top &= [ bold(B)_1 thin dots thin bold(B)_K ]^top in RR^(R times d_"out") $ <eq-stacked>
For any dynamic routing vector $bold(gamma) = (gamma_1, dots, gamma_K) in RR^K$, let $bold(g) = "repeat"(bold(gamma), r) in RR^(1 times 1 times R)$ replicate each scalar $gamma_k$ across its $r$ rank channels. By block-matrix associativity, @eq-unfused reduces to _two_ GEMMs and one broadcast multiply regardless of $K$:
$ Delta bold(y) = ((bold(x) bold(A)_"cat"^top) circle.tiny bold(g)) bold(B)_"cat"^top $ <eq-fused>
When $bold(gamma) = bold(0)$, the LoRA branch is bypassed entirely (0 FLOPs). Furthermore, rather than evaluating multi-token candidates via $B$ sequential `[1, L, d]` forward passes and copying `[L, 24576]` log-softmax distributions ($approx 590$ KB) back to CPU, Chime packs all multi-token continuations into a _single right-padded_ `[B, L_max]` _batch tensor_, zero-copy broadcasts `MlxPrefixCache` along axis 0 (`[1, H_kv, T_ctx, D]` $arrow.r$ `[B, H_kv, T_ctx, D]`), applies lower-right causal masking (preventing valid positions $t < ell_b$ from attending to right-padding $t >= ell_b$), and extracts only target log-probabilities via on-GPU `take_along_axis` ($<= 128$ bytes GPU$arrow.r$CPU readout, a 4,600× bandwidth reduction).

#figure(
  fig-scaling-and-transfer(),
  kind: image,
  scope: "parent",
  placement: top,
  caption: [
    *(a)* Measured 6-candidate Metal reranking latency (p50, ms) across $K = 0, dots, 6$ Rank-8 cartridges: Rank-Stacked GEMMs flatten the $O(K)$ kernel ramp ($10.82 arrow.r 3.29$ ms), and Batched Broadcast KV-Gather cuts it further to *0.87 ms* ($K=0$) and *1.45 ms* ($K=6$, *7.46× speedup*). *(b)* Classical validation NLL vs. adapter scale $gamma$: zero-shot transfer of an FP16-trained LoRA overshoots at $gamma = 1.0$ (6.071), whereas *Q4-Manifold Aligned LoRA* training directly on dequantized 4-bit weights eliminates overshoot and deepens NLL reduction by 68% (5.978 $arrow.r$ *5.708*).
  ],
) <fig-scaling>

== Register Superposition & Q4-Manifold Alignment

Mounting multiple domain adapters simultaneously reveals two subtle statistical interactions that degrade naive additive LoRA composition:

1. *Register Superposition Drift*: Domain packs (`classics-poetry`, `tech-developer`, and long-form essay packs) are predominantly trained on _formal written corpora_. In written text, formal compounds such as 多达 (_"up to"_) vastly outnumber spoken interrogatives like 多大 (_"how large"_). Summing three adapters with uncalibrated scales ($sum_k gamma_k = 2.85$) constructively amplifies their shared written-register direction and pushes pre-norm activations into SwiGLU non-linear saturation, dropping classical accuracy to 5/8 and spiking NLL from 5.978 to 7.136 (@tab-routing).
2. *FP16$arrow.r$INT4 Manifold Shift & Q4-Aligned Training*: Community `LoRA`s trained against the clean FP16 student $bold(W)$ and mounted zero-shot onto the 4-bit quantized base $hat(bold(W)) = bold(W) + bold(E)_Q$ incur quantization cross-terms in expected NLL:
$ cal(L)(hat(bold(W)) + gamma Delta bold(W)) &approx cal(L)(bold(W)) + gamma chevron.l nabla cal(L), Delta bold(W) chevron.r \ & quad + frac(1, 2) norm(bold(E)_Q + gamma Delta bold(W))_(bold(H))^2 $ <eq-taylor>
Minimizing @eq-taylor gives $gamma^* = (- chevron.l nabla cal(L), Delta bold(W) chevron.r - chevron.l bold(E)_Q, Delta bold(W) chevron.r_(bold(H))) / norm(Delta bold(W))_(bold(H))^2$: full unit step $gamma = 1.0$ overshoots (6.071), whereas $gamma^* approx 0.50$ recovers $-0.161$ nats (@fig-scaling (b), blue). Crucially, training `LoRA` directly on the _exact dequantized 4-bit runtime manifold_ $hat(bold(W))$ allows $Delta bold(W) = bold(B) bold(A)$ to simultaneously adapt domain priors and compensate for group-64 quantization residuals $bold(E)_Q$, eliminating $gamma = 1.0$ overshoot above base and achieving *−0.270 nats* (5.708) at $gamma = 0.65$ (@fig-scaling (b), green).

Chime combines Q4-Manifold alignment with a *Couplet-Aware Simplex Router*:
- *Domain & Couplet Gating*: `Classical` routing activates on archaic particles (之乎者也矣焉哉兮) or symmetric $5+5$ / $7+7$ regulated-verse couplet cadence (空山新雨后，+ 天气晚来秋), while staying strictly $0$ on colloquial Mandarin; `Developer` activates on ASCII/code cues ($1.0 s_k$) or IDE bundle IDs ($0.25 s_k$); `Always` attenuates to $0.45 s_k$ on short $<= 2$-char colloquial inputs.
- *Sub-Simplex Gain Cap*: Individual scales are clamped to $gamma_k <= 0.80$ (guarding third-party FP16 `LoRA`s) and projected onto $sum_(k=1)^K gamma_k <= Gamma_"max" = 0.90$.

= Experiments & Analysis

== Experimental Setup & Datasets

All runtime and latency experiments are conducted natively on an Apple Silicon Mac running macOS 15 (`mlx-rs 0.34`, Metal GPU, single-threaded actor worker without locks). Layer distillation (1,600 steps) ran on an NVIDIA L4 24 GB GPU; all three Q4-aligned Rank-8 ($r=8$) `LoRA` adapters (320 steps, AdamW, cosine schedule, 2.07 MB FP16 each) were trained directly on Apple Silicon MPS:
- *`classics-poetry.imepack`*: Compiled from 37 canonical Tang/Song/Guwen masterpieces, 21 `chinese-poetry` collections (Tang/Song/Yuan/Shijing/Chuci/Four Books), and Xinhua idioms/xiehouyu, totaling *266,816* lexical entries (including merged short-pause Song Ci lead-in clauses such as 念去去，千里烟波) and *231,593* `Tab` continuation chains, plus 18,482 classical paragraphs for `LoRA` training.
- *`tech-developer.imepack`*: Curated bilingual systems/AI/cloud terminology + 1.20M characters (2,503 passages) filtered from `FineWeb-Edu-Chinese-V2.1` and `BelleGroup`.
- *Air-Gapped Longitudinal Personal Pack*: Trained strictly offline on *222 human-authored essays* (2013–2020; 6,533 paragraphs, 814,988 characters, strictly pre-dating generative AI tools to avoid synthetic stylistic contamination), scrubbed via Chime's PII/secret sanitizer and yielding 8,000 personal words + 4,727 bigrams.

#figure(
  table(
    columns: (2.26fr, 0.84fr, 0.66fr, 0.92fr, 1.02fr, 0.86fr, 0.86fr),
    align: (left, right, right, right, right, right, right),
    stroke: none,
    inset: (x: 4.5pt, y: 4.0pt),
    toprule,
    [*Model Checkpoint*], [*Size*], [*Compr.*], [*$cos(bold(u)_S, bold(u)_T)$*], [*Stress Acc.*], [*Margin $Delta ell$*], [*Rerank-6 p50*],
    midrule,
    [`Qwen3-0.6B` (28L, 152K, FP16)], [1,191.5 MB], [1.0×], [1.0000], [93.8% (15/16)], [+5.27 nats], [$approx 14.2$ ms],
    [`Qwen3-28L` (24K vocab, 4-bit)], [249.95 MB], [4.8×], [1.0000], [93.8% (15/16)], [+5.22 nats], [5.29 ms],
    [`Qwen3-6L-NaiveSkip` (4-bit)], [64.15 MB], [18.6×], [#text(fill: rgb("#be123c"))[0.0221]], [#text(fill: rgb("#be123c"))[43.8% (7/16)]], [+0.14 nats], [2.33 ms],
    [`Qwen3-6L-Distilled` (FP16)], [228.03 MB], [5.2×], [0.8125], [93.8% (15/16)], [+5.91 nats], [4.07 ms],
    [`Qwen3-6L-Distilled` (8-bit)], [121.15 MB], [9.8×], [0.8125], [93.8% (15/16)], [+5.91 nats], [3.44 ms],
    [`Qwen3-6L-Chime` (4-bit, Seq.)], [64.17 MB], [18.6×], [0.8531], [93.8% (15/16)], [+6.02 nats], [1.35 ms],
    [*`Qwen3-6L-Chime` (4-bit, Batched)*], [*64.17 MB*], [*18.6×*], [*0.8531*], [*93.8% (15/16)*], [*+6.02 nats*], [*0.86 ms*],
    bottomrule,
  ),
  scope: "parent",
  placement: top,
  caption: [
    *Teacher-to-Student Compression & Batched Metal Reranking Pareto Frontier* ($T = 16$ prefix tokens + 6 homophone candidates). Naive 6-layer skipping destroys representation alignment ($cos = 0.0221$), whereas manifold-aligned distillation + batched KV-gather recovers 100% of the 28-layer teacher's accuracy (15/16) in 64.17 MB at *0.86 ms* (6.15× faster than 28L 4-bit).
  ],
) <tab-pareto>

== Main Results & Ablation Studies

*1. Distillation & Batched Reranking Pareto Frontier (@tab-pareto).* Vocabulary pruning + 4-bit group-64 quantization compresses the 28-layer teacher to 249.95 MB (5.29 ms reranking). Skipping directly to 6 layers cuts size to 64.15 MB, but collapses pre-head cosine similarity to $0.0221$ and drops zero-shot minimal-pair accuracy from 15/16 to 7/16 (mean margin $+0.14$ nats). Our 1,600-step manifold-aligned distillation restores $cos(bold(u)_S, bold(u)_T) = 0.8531$, achieving *15/16 (93.8%)* zero-shot stress accuracy and *+6.02 nats* mean margin, while Batched Broadcast KV-Gather reduces 6-candidate reranking from 1.35 ms to *0.86 ms*—sub-millisecond on a laptop GPU. On the expanded 44-case Chime IME suite (covering full/abbreviated Pinyin, Xiaohe/Ziranma Shuangpin, fuzzy rhymes, neighbor/swap/repeat/geminate typos, code-switching, punctuation, and poetry chaining), Chime achieves *44/44 (100%)* Top-1 accuracy.

#figure(
  table(
    columns: (1.55fr, 1.12fr, 0.66fr, 0.67fr),
    align: (left, right, right, right),
    stroke: none,
    inset: (x: 3.4pt, y: 3.8pt),
    toprule,
    [*Configuration*], [*Verse Top-1*], [*Cls. NLL*], [*Pers. NLL*],
    midrule,
    [C0: Modern Base Only], [10.0% (3/30)], [5.978], [5.754],
    [C1: `LoRA` Sidecar Only], [10.0% (3/30)], [5.708], [5.287],
    [C2: `pack.imedict` Only], [90.0% (27/30)], [5.978], [5.754],
    [*C3: Full `.imepack`*], [*90.0% (27/30)*], [*5.708*], [*5.287*],
    bottomrule,
  ),
  placement: top,
  caption: [
    *Empirical Verification of Recall–Rerank Decoupling* on 30 held-out classical verses outside the core seed and held-out Q4 prose validation blocks. Q4-aligned `LoRA` reduces domain NLL ($-0.270$ classical, $-0.467$ personal) yet yields *0%* gain on OOV verses without `pack.imedict`.
  ],
) <tab-decouple>

*2. Lexical-Neural Decoupling (@tab-decouple).* Evaluating 30 held-out classical Tang/Song/Guwen verses against a modern-only base lexicon empirically confirms Proposition 1: mounting Q4-aligned `classics_lora.safetensors` alone (C1) lowers classical validation NLL by *−0.270 nats* ($5.978 arrow.r 5.708$, perplexity $394.7 arrow.r 301.2$) and personal `LoRA` lowers held-out personal essay NLL by *−0.467 nats* ($5.754 arrow.r 5.287$), yet verse Top-1 accuracy remains stuck at *3/30 (10.0%)* because Stage-1 never proposes OOV verses. Co-mounting the full `.imepack` (C3) boosts Top-1 conversion 9× to *27/30 (90.0%)* while retaining neural disambiguation and 231,593 `Tab` continuation chains.

#figure(
  table(
    columns: (1.44fr, 0.95fr, 0.95fr, 0.95fr, 0.60fr),
    align: (left, right, right, right, right),
    stroke: none,
    inset: (x: 2.2pt, y: 3.8pt),
    toprule,
    [*Routing ($K$=3)*], [*Colloq.*], [*Class.*], [*Dev.*], [*NLL*],
    midrule,
    [R0: Base ($bold(gamma) = bold(0)$)], [8/8 (+8.4)], [8/8 (+4.3)], [8/8 (+9.3)], [5.978],
    [R1: Uncalib. Sum], [#text(fill: rgb("#be123c"))[7/8 (+3.6)]], [#text(fill: rgb("#be123c"))[5/8 (+1.9)]], [8/8 (+6.8)], [#text(fill: rgb("#be123c"))[7.136]],
    [R2: Uniform $1/K$], [8/8 (+7.7)], [8/8 (+4.5)], [8/8 (+9.4)], [5.851],
    [*R3: Simplex-Gated*], [*8/8 (+9.0)*], [*8/8 (+4.4)*], [*8/8 (+9.7)*], [*5.781*],
    bottomrule,
  ),
  placement: top,
  caption: [
    *Multi-LoRA Interference vs. Simplex-Gated Routing* with 3 concurrent Q4-aligned packs (`classics` + `tech` + `personal`). Cells show Top-1 wins and mean margin (nats).
  ],
) <tab-routing>

*3. Multi-LoRA Interference & Routing (@tab-routing).* With three concurrent packs, naive summation (R1, $sum gamma_k = 2.85$) halves colloquial margin ($+8.4 arrow.r +3.6$ nats), drops classical accuracy to 5/8, and spikes NLL to 7.136. Uniform $1/K$ averaging (R2) dilutes colloquial margin ($+7.7$). Our Couplet-Aware Simplex Router (R3) achieves *8/8* across all three regimes with the highest margins in colloquial ($+9.00$ nats) and developer ($+9.73$ nats) probes and the lowest NLL ($5.781$).

#figure(
  table(
    columns: (0.42fr, 0.95fr, 0.95fr, 1.15fr, 0.85fr),
    align: (center, right, right, right, right),
    stroke: none,
    inset: (x: 4pt, y: 3.6pt),
    toprule,
    [*$K$*], [*Unfused*], [*Stacked*], [*Batched+Stk*], [*Speedup*],
    midrule,
    [0], [1.34 ms], [1.37 ms], [*0.87 ms*], [*1.55×*],
    [1], [2.90 ms], [2.95 ms], [*1.25 ms*], [*2.32×*],
    [2], [4.57 ms], [2.93 ms], [*1.38 ms*], [*3.31×*],
    [3], [6.12 ms], [3.15 ms], [*1.45 ms*], [*4.21×*],
    [4], [7.85 ms], [3.00 ms], [*1.41 ms*], [*5.57×*],
    [5], [9.37 ms], [3.11 ms], [*1.44 ms*], [*6.52×*],
    [6], [10.82 ms], [3.29 ms], [*1.45 ms*], [*7.46×*],
    bottomrule,
  ),
  placement: top,
  caption: [
    *Multi-LoRA Scaling on Metal* ($K = 0, dots, 6$ Rank-8 adapters, 6-candidate p50). Combining Rank-Stacked GEMMs with Batched KV-Gather reduces $K=6$ latency from 10.82 ms to *1.45 ms* ($+39.6 thin mu"s"$/adapter from $K = 1 arrow.r 6$, *7.46× speedup*).
  ],
) <tab-scaling>

*4. Kernel Fusion, Batched KV-Gather & Length Calibration (@tab-scaling & Exp. 5).* As mounted adapters grow from $K=1$ to $6$, Rank-Stacked GEMMs + Batched KV-Gather hold 6-candidate reranking at *1.25 ms $arrow.r$ 1.45 ms* ($+39.6 thin mu"s"$/adapter vs. $+"1,584" thin mu"s"$/adapter unfused—a *40.0× reduction* in marginal adapter cost and *7.46× total speedup*). Sweeping length-normalization exponent $alpha in [0, 1]$ in @eq-rerank confirms that character-count scaling preserves 15/16 stress accuracy across all $alpha$, while $alpha = 0.42$ paired with Probability-Space Separation uniquely achieves *44/44 (100%)* end-to-end accuracy.

== Qualitative Case Studies

- *Stateful Poetry Chaining & Anti-Repeat*: Typing `junbujianhuanghe` + `Tab` commits 君不见黄河之水天上来 and surfaces clause-by-clause `[Tab]` continuations (奔流到海不复回 $arrow.r$ 君不见高堂明镜悲白发) across all 24 clauses of Li Bai's _Qiang Jin Jiu_. Crucially, if the user manually types `junbujian` _after_ 奔流到海不复回, Chime's context-continuation boost ($+7.5$) and verbatim anti-repeat penalty ($-5.5$) promote 君不见高堂明镜悲白发 to \#1 instead of repeating the opening line.
- *Compound Motor Slip & Geminate Recovery*: On `nianquuqqianliyanbo` (containing transposed `uq` $arrow.r$ `qu` inside Liu Yong's Song Ci lead-in), Stage-1 recovers `qu` via $delta_"swap"$ and matches the merged rhythmic unit 念去去，千里烟波 at \#1 in 7.6 ms; on `chunuanhuakai`, $delta_"gem"$ recovers the elided boundary `n` into 春暖花开.

= Limitations & Ethical Considerations

*Limitations.* First, Chime's 64.17 MB student relies on Stage-1 lattice recall (`CHIMED02` + `SQLite` user store); truly novel neologisms absent from all mounted `.imepack`s and user history still require manual step-by-step prefix composition (今天 + `wo'men`) on their first occurrence before online bigram/phrase learning caches them. Second, our current context router uses interpretable lexical and bundle-ID features; learning an ultra-lightweight ($< 10$ KB) linear classifier over embedding pooled states could further refine soft blending in mixed classical-modern essays.

*Privacy & Broader Impact.* Input methods observe a user's most intimate thoughts, credentials, and unpublished code. Chime enforces zero network entitlement, bypasses `IsSecureEventInputEnabled()` password fields in hardware, and enables air-gapped personal `LoRA` training directly on local Apple Silicon GPUs after deterministic secret/PII redaction—demonstrating that deep LLM personalization need not come at the expense of keystroke surveillance.

= Conclusion

Chime demonstrates that LLMs are most effective in client-side input methods not as monolithic, latency-heavy sequence generators, but as _manifold-distilled, batched prefix-cached parallel rescorers_ co-designed with zero-copy lexical structures and _Q4-Aligned Rank-Stacked Multi-LoRA cartridges_. Compressing `Qwen3-0.6B` by 18.6× into 64.17 MB on Unified Memory while bounding multi-adapter mixtures on the simplex delivers sub-millisecond base reranking (0.86 ms), 1.45 ms 6-adapter inference, modular domain mastery, and strictly offline personal adaptation on everyday laptops.

#bibliography("refs.bib", title: "References", style: "association-for-computing-machinery")
