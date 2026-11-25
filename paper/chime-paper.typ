#import "figures.typ": fig-architecture, fig-scaling-and-transfer

#set document(
  title: "Chime: Sub-3ms On-Device Neural Pinyin Input via Manifold-Aligned Distillation and Rank-Stacked Multi-LoRA Cartridges",
  author: "Tyr Chen",
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

#set par(
  justify: true,
  leading: 0.55em,
  spacing: 0.74em,
)

#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#show heading.where(level: 1): it => block(above: 1.12em, below: 0.52em)[
  #set text(size: 11pt, weight: "bold")
  #it
]
#show heading.where(level: 2): it => block(above: 0.88em, below: 0.40em)[
  #set text(size: 10pt, weight: "bold")
  #it
]
#show figure.caption: it => [
  #set text(size: 8.6pt)
  #set par(justify: true, leading: 0.48em)
  #it
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
      #set par(justify: false, leading: 0.45em)
      #set text(hyphenate: false)
      #text(size: 14.6pt, weight: "bold")[
        Chime: Sub-3ms On-Device Neural Pinyin Input\
        via Manifold-Aligned Distillation and Rank-Stacked Multi-LoRA Cartridges
      ]
      #v(7pt)
      #text(size: 10.4pt, weight: "semibold")[Tyr Chen]
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
      #align(center)[#text(size: 9.4pt, weight: "bold")[Abstract]]
      #v(3pt)
      #set par(justify: true, leading: 0.50em)
      #set text(size: 8.85pt)
      Applying Large Language Models (LLMs) to client-side Pinyin Input Method Editors (IMEs) faces a strict systems trilemma: human typing demands sub-10 ms keystroke latency without candidate-bar jitter, operating systems enforce tight resident memory budgets ($< 100$ MB) alongside zero-network privacy guarantees, and real-world input spans open-domain vocabulary from bilingual code identifiers to classical literature and idiosyncratic personal style. Existing generative Pinyin-to-Character (P2C) LLMs violate latency budgets via autoregressive decoding and suffer catastrophic representation collapse under aggressive layer pruning or out-of-vocabulary (OOV) blindness when relying solely on parameter-efficient adapters. We present *Chime*, a 100% offline neural IME for Apple Silicon that resolves this trilemma through lexical-neural co-design. First, we formalize *Recall–Rerank Decoupling* and *Probability-Space Separation*, pairing a $< 0.8$ ms zero-copy `mmap` dual-track lattice decoder (`CHIMED02`) with a prefix-cached causal language model reranker that isolates equal-span homophone scoring from variable-horizon lookahead completions. Second, we compress the 28-layer `Qwen3-0.6B` (1.19 GB FP16) into a 6-layer, 24,576-vocab 4-bit MLX student (*64.17 MB*, $18.6 times$ smaller); whereas naive layer skipping collapses pre-head teacher alignment to $cos(bold(u)_S, bold(u)_T) = 0.019$ and halves homophone accuracy to 43.8%, our *Manifold-Aligned Layer Distillation* restores $cos(bold(u)_S, bold(u)_T) = 0.861$, matching the 28-layer teacher's 87.5% zero-shot stress accuracy and achieving 100% (42/42) end-to-end IME accuracy at *1.37 ms* 6-candidate Metal reranking latency ($3.91 times$ faster). Third, we introduce pluggable *`.imepack` cartridges* that co-mount zero-copy lexical slices (`pack.imedict`) with 2.07 MB Rank-8 `LoRA` sidecars. By concatenating adapter matrices into *Rank-Stacked GEMMs* and bounding weights via a *Simplex-Gated Context Router*, Chime reduces 6-adapter marginal latency overhead by $49.3 times$ ($3.38 times$ total speedup at $K=6$), eliminates multi-adapter register drift, and captures the optimal zero-shot FP16$arrow.r$INT4 transfer scale ($gamma^* approx 0.50$).
    ]
  ]
  #v(5pt)
]

#show: rest => columns(2, gutter: 0.25in, rest)

= Introduction

Every sentence written on a desktop or mobile device passes through an Input Method Editor (IME). For over a billion Chinese speakers, Pinyin-to-Character (P2C) conversion is a quintessential noisy-channel inference problem @chen2000new @jia2014joint: a user types a romanized phonetic keystroke stream—frequently interleaved with initial-letter abbreviations (`wsm` $arrow.r$ 为什么), regional phonetic mergers (`z`/`zh`, `in`/`ing`), motor slips on QWERTY layouts (`shaung` $arrow.r$ `shuang`), and unswitched English identifiers (`macOS`, `GitHub`)—expecting the intended text at slot \#1 within a single display frame.

Recent advances in Large Language Models (LLMs) have inspired a wave of generative IME prototypes @tan2022exploring @ding2023geneinput @gui2025attninput @shan2026huoziime that cast P2C as end-to-end conditional sequence generation. Despite strong perplexity on clean benchmarks, deploying LLMs inside a production OS input method exposes a fundamental *Latency–Memory–Recall Trilemma*:

1. *Perceptual Latency & Visual Jitter*: Human typists perceive interface lag above 10–15 ms and commit muscle-memory errors if slot \#1 shifts after $approx 60$ ms of visual fixation. Autoregressive LLM generation over candidate character sequences incurs 25–150 ms per keystroke even on modern accelerators, while cloud-offloaded IMEs introduce variable network tail latency and transmit every keystroke off-device.
2. *Resident Memory Ceiling & Collapse*: An OS-resident background service active across every window cannot consume gigabytes of RAM. Yet naively pruning a 28-layer, 1.19 GB model (`Qwen3-0.6B` @yang2025qwen3) down to 6 layers and 4-bit weights ($approx 64$ MB) destroys the alignment between intermediate transformer blocks and the frozen vocabulary embedding head, collapsing cosine similarity to $0.019$ and degrading homophone disambiguation to near-chance (43.8%).
3. *Lexical Recall vs. Multi-Domain Interference*: Neither a 64 MB base model nor Low-Rank Adaptation (`LoRA` @hu2022lora) alone can recall 260,000+ long-tail classical verses (e.g., 念去去，千里烟波) if Stage-1 beam search prunes them. Conversely, mounting multiple domain `LoRA`s simultaneously (e.g., classical literature, developer prose, and personal essays) triggers two undocumented failure modes on edge GPUs: *Register Superposition Drift*—where summing written-domain adapters overwhelms colloquial priors (flipping 究竟有多大 to 多达)—and *Sequential Kernel Launch Inflation*, which scales latency linearly with adapter count $K$.

In this paper, we design, implement, and empirically evaluate *Chime*, an open-source, zero-telemetry macOS IME written in Rust and Swift 6 on Apple Silicon Unified Memory @hannun2023mlx. Rather than forcing a compact LLM to hallucinate character mappings from raw keystrokes, Chime establishes a principled division of labor between deterministic zero-copy data structures and manifold-aligned low-rank neural rescoring.

#place(
  top + center,
  scope: "parent",
  float: true,
)[
  #figure(
    fig-architecture(),
    kind: image,
    caption: [
      End-to-end architecture of *Chime*. Stage 1 maps layered `CHIMED02` binary dictionaries via `mmap` ($< 0.8$ ms) and separates equal-span candidates $cal(C)_n$ from lookahead completions $cal(C)_"comp"$. Stage 2 runs a frozen 64.17 MB 4-bit `Qwen3-6L` student with persistent GPU `MlxPrefixCache` and fused Rank-Stacked `.imepack` `LoRA` sidecars under a 60 ms anti-jitter contract.
    ],
  ) <fig-arch>
]

Our primary contributions are fourfold:
- *Recall–Rerank Co-Design & Measure Separation (§3)*: We prove that Stage-2 neural rescoring is upper-bounded by Stage-1 pool recall and must separate equal-span homophone hypotheses $cal(C)_n$ from lookahead completions $cal(C)_"comp"$ to avoid cross-horizon probability distortion, paired with sublinear length calibration $|c|^(-0.42)$ across heterogeneous BPE tokenizations.
- *Manifold-Aligned 6-Layer Student Distillation (§4)*: Combining vocabulary pruning ($151,936 arrow.r 24,576$ tokens), frozen tied embeddings, trajectory MSE, pre-head cosine alignment, and dual-temperature KL distillation, we compress `Qwen3-0.6B` by $18.6 times$ into a 64.17 MB 4-bit MLX student that matches the 28-layer teacher's zero-shot disambiguation accuracy (87.5%) at $3.91 times$ lower latency (1.37 ms).
- *Rank-Stacked `.imepack` Cartridges & Simplex Gating (§5)*: We package zero-copy `pack.imedict` slices alongside 2.07 MB Rank-8 `LoRA`s into hot-swappable cartridges. Our *Rank-Stacked GEMM Fusion* collapses $2K$ low-rank dispatches per projection into 2 fused GEMMs ($3.38 times$ speedup at $K=6$, $+30.6 thin mu"s"$ marginal cost/adapter), while our *Simplex-Gated Context Router* prevents register superposition and hits the theoretical optimum ($gamma^* approx 0.50$) of zero-shot FP16$arrow.r$INT4 adapter transfer.
- *System Evaluation & Longitudinal Personalization (§6)*: Across classical Chinese (266K entries, 231K `Tab` chains), developer code-switching, and an air-gapped 815K-character pre-LLM longitudinal personal corpus (2013–2020), Chime achieves 42/42 (100%) benchmark accuracy and $9 times$ classical recall gain ($10.0% arrow.r 90.0%$).

= Related Work

*Statistical and Graph-Based Pinyin Decoding.* Classical P2C engines model conversion as Viterbi search over word lattices weighted by character/word $n$-grams @chen2000new and double-array tries @yoshinaga1992double. @jia2014joint introduced a joint graph model integrating QWERTY typo correction with P2C conversion. While sub-millisecond and memory-frugal, finite-context $n$-grams fail on long-range semantic homophones (e.g., 保障公民的*权利* vs. 滥用国家的*权力* across intervening clauses).

*Neural and LLM-Based Input Methods.* @zhang2019open augmented neural P2C with online open-vocabulary sampling. @xu2022pert proposed `PERT`, a bidirectional encoder pre-trained directly on Pinyin sequences. With the rise of generative LLMs, @tan2022exploring adapted frozen Chinese GPT to P2C, observing severe degradation on abbreviated Pinyin due to combinatorial candidate explosion. `GeneInput` @ding2023geneinput unified full-mode key-to-character generation via prompting and RLHF, `AttnInput` @gui2025attninput attached a Pinyin side-network to linear RWKV models, and `HuoziIME` @shan2026huoziime explored on-device LLM personalization via hierarchical memory. Unlike these end-to-end generative designs—which struggle with sub-5 ms keystroke budgets, multi-syllable typo graphs, and exact casing for code-switching—Chime retains deterministic $O(log N)$ lattice recall in Stage 1 and deploys a prefix-cached causal SLM strictly as a constant-depth parallel rescorer in Stage 2.

*Edge Compression and Multi-LoRA Serving.* Knowledge distillation @hinton2015distilling with hidden-state alignment (`TinyBERT` @jiao2020tinybert) and group-wise weight quantization (`GPTQ` @frantar2023gptq, `AWQ` @lin2024awq) enable compact edge deployment. In cloud serving, `S-LoRA` @sheng2024slora batches heterogeneous requests across distinct per-tenant adapters using custom paging kernels, whereas personalized `LoRA` routing (`PLoRA` @zhang2024plora) blends adapters per user. To our knowledge, Chime is the first system to investigate *single-stream concurrent Multi-LoRA superposition* on unified-memory client GPUs—identifying and solving both multi-adapter register interference and zero-shot FP16-to-INT4 scale overshoot.

= Problem Formulation & Hybrid Co-Design

== Joint Phonetic-Typo-Code Lattice Decoding

Given a raw ASCII keystroke sequence $bold(k) = k_(1:n)$, surrounding left document context $bold(c)_L$ (bounded to 96 characters), and active host application identifier $a$ (e.g., `com.mitchellh.ghostty` vs. `md.obsidian`), an IME seeks the highest-scoring surface string $bold(w)^*$:
$ bold(w)^* = arg max_(bold(w) in cal(G)(bold(k))) [& S_"lex" (bold(w), bold(k) | bold(c)_L, a) \ & + lambda dot S_"slm" (bold(w) | bold(c)_L, a)] $ <eq-joint>
where $cal(G)(bold(k))$ is the dual-track segmentation lattice constructed over byte spans $[i, j) subset.eq [0, n)$.

To achieve $< 0.8$ ms Stage-1 construction without memory bloat, Chime compiles static and cartridge lexicons into `CHIMED02`, a 48-byte-header immutable binary image mapped directly into virtual memory via `mmap` (`memmap2`). Rather than expanding fuzzy Pinyin permutations offline—which multiplies dictionary size by $2^6$—Stage 1 evaluates an algebraic equivalence relation $cal(F)(s)$ (`z`$harpoons.rtlb$`zh`, `an`$harpoons.rtlb$`ang`, `in`$harpoons.rtlb$`ing`, etc.) dynamically at span-lookup time, alongside three 1-edit motor-noise operators on raw slice $bold(k)_(i:j)$:
1. *Adjacent transposition* (`shaung` $arrow.r$ `shuang`, `uq` $arrow.r$ `qu`, cost $delta_"swap" = 1.05$);
2. *QWERTY Euclidean neighbor substitution* ($d^2 <= 1.85$ key pitches, e.g., `shuruga` $arrow.r$ `shurufa`, cost $0.85 + 0.45 d^2$);
3. *Accidental key-repeat deletion* (`niian` $arrow.r$ `nian`, cost $delta_"rep" = 1.10$).

Multi-syllable exact phrases, idioms, and merged classical clauses up to 16 syllables are traversed via depth-first span expansion gated at every step by an $O(log N)$ binary search (`has_longer_pinyin_prefix`) over sorted 16-byte `pinyin_groups`, pruning dead typo branches immediately at length 1–2. Simultaneously, a parallel code-switching track scores canonical-case English identifiers (`macos` $arrow.r$ `macOS`, `chatgpt` $arrow.r$ `ChatGPT`) and camelCase tokens (`useEffect`) with automatic Pangu boundary spacing.

== Recall–Rerank Decoupling & Measure Separation

Why not rely on `LoRA` fine-tuning alone to inject domain phrases or classical poetry? Let $cal(C)_M (bold(k)) subset cal(G)(bold(k))$ denote the top-$M$ candidates retained by Stage-1 beam search ($M=15$).

#prop-box("Proposition 1 (Stage-1 Recall Bottleneck)")[
  For any finite Stage-2 rescorer $S_"slm" : Sigma^* arrow.r RR$ with bounded dynamic range $|S_"slm"| <= B$, if a target multi-character entity or verse $bold(w)^*$ is absent from the mounted lexicon such that fragmented character paths push $bold(w)^*$ outside $cal(C)_M (bold(k))$, then $PP(bold(w)^* = bold(w)_1) = 0$ regardless of `LoRA` rank $r$ or Stage-2 capacity.
]

Proposition 1 explains why typing `mingyuejishiyou` without a classical lexicon yields 明月即使有 (#emph["even if the bright moon..."]): modern high-frequency bigrams (`即使`, $log P = -3.8$) dominate fragmented character beams (`几` + `时`, $log P = -8.9$), evicting 明月几时有 before Stage 2 executes.

Equally critical is *which* candidates inside $cal(C)_M (bold(k))$ may be compared by $S_"slm"$. Modern IMEs surface both *full-span interpretations* $cal(C)_n$ (consuming $bold(k)_(1:n)$) and *prefix lookahead completions* $cal(C)_"comp"$ (e.g., typing 3 syllables `junbujian` surfaces the 10-syllable verse completion 君不见高堂明镜悲白发).

#prop-box("Proposition 2 (Measure Incompatibility across Horizons)")[
  Let $bold(w)_s in cal(V)^m$ be an $m$-character full-span candidate and $bold(w)_ell = bold(w)_s circle.tiny bold(w)_"rem" in cal(V)^(m+d)$ ($d >= 1$) be a lookahead completion extending $bold(w)_s$. Under any causal language model $P_theta$, by the chain rule:
  #set math.equation(numbering: none)
  $ log P_theta (bold(w)_ell | bold(c)_L) &= log P_theta (bold(w)_s | bold(c)_L) \ & quad + sum_(t=1)^d log P_theta (w_(m+t) | bold(c)_L, bold(w)_(<m+t)) \ &< log P_theta (bold(w)_s | bold(c)_L) $
  whenever $P_theta (w_(m+t) | dot) < 1$. Under sublinear length normalization $|bold(w)|^alpha$ ($alpha < 1$), $bold(w)_ell$ incurs a systematic horizon penalty of $Theta(d^(1-alpha))$ nats relative to $bold(w)_s$.
]

Guided by Proposition 2, Chime enforces *Probability-Space Separation*: Stage-2 zero-meaned neural rescoring operates strictly over the equal-input homophone subspace $cal(C)_n backslash cal(C)_"comp"$, while lookahead completions $cal(C)_"comp"$ are governed by prefix selectivity and deterministic $O(log C)$ continuation chains (`continuations_off` at byte offset 40 of `CHIMED02`), plus an anti-repeat penalty ($-5.5$ nats) if a $>= 5$-character verse already appears in $bold(c)_L$.

Within $cal(C)_n backslash cal(C)_"comp"$, candidates share identical character length ($|bold(w)|$) on homophones (`权利` vs. `权力`) yet often differ in BPE token count $T(bold(w))$: formal compounds like `多达` occupy 1 subword token, whereas colloquial phrases like `多大` split into 2 tokens (`多` + `大`). Dividing by $T(bold(w))$ artificially inflates 2-token candidates because splitting a joint probability $log P(w_1, w_2)$ by 2 halves its magnitude; instead, Chime sums the *exact unnormalized joint log-likelihood* $ell(bold(w)) = sum_(t=1)^(T(bold(w))) log P_theta (z_t | bold(c)_L, z_(<t))$ and normalizes solely by *character length* $|bold(w)|^(0.42)$:
$ S_"slm" (bold(w) | bold(c)_L, a) = lambda ( frac(ell(bold(w)), |bold(w)|^(0.42)) - frac(1, |cal(B)|) sum_(bold(w)' in cal(B)) frac(ell(bold(w)'), |bold(w)'|^(0.42)) ) $ <eq-rerank>
where $lambda = 0.65$ and $cal(B) subset.eq cal(C)_n backslash cal(C)_"comp"$ ($|cal(B)| <= 6$).

= Manifold-Aligned 6-Layer Distillation

== Vocabulary Pruning & Naive Skip Collapse

Our teacher is `Qwen/Qwen3-0.6B-Base` @yang2025qwen3 ($L_T = 28$ layers, $d = 1024$, $d_"ff" = 3072$, 16 query heads, 8 KV heads with `head_dim` $= 128$, QK-RMSNorm, tied embeddings, and vocabulary $V_T = 151,936$, totaling 1,191.5 MB in FP16). Notice that in a 6-layer student with full vocabulary, the $151,936 times 1024$ embedding table alone consumes 311 MB in FP16 (78 MB in 4-bit), dominating the transformer blocks. We prune the vocabulary to $V_S = 24,576$ pieces covering all GB2312/common Unihan characters, high-frequency Chinese multi-character subwords, punctuation, and ASCII/developer subwords, shrinking the 4-bit embedding table by *83.8%* to just *12.6 MB*.

However, initializing a 6-layer student by uniformly selecting teacher layers $cal(I) = \{0, 5, 11, 16, 22, 27\}$ while freezing the pruned teacher embedding matrix $bold(E) in RR^(V_S times d)$ causes severe *Manifold Misalignment*. Because residual transformers refine representations incrementally across 28 blocks before projecting onto $bold(E)^top$, skipping 22 blocks rotates the pre-head hidden vector $bold(u)_S = "RMSNorm"(bold(h)_S^((6)))$ nearly orthogonal to the teacher's $bold(u)_T = "RMSNorm"(bold(h)_T^((28)))$: empirically, $cos(bold(u)_S, bold(u)_T)$ plummets to *0.0191* (@tab-pareto), causing colloquial `多大` to lose to `多达` and cutting homophone accuracy in half (43.8%).

== Composite Trajectory & Cosine-KL Objective

To realign the 6-layer student directly onto the frozen teacher embedding geometry $bold(E)$ (preventing embedding overfitting on finite distillation corpora), we freeze $bold(E)$ and optimize only the 6 transformer blocks $theta_S$ over 9.94M tokens of mixed conversational, encyclopedic, code-switching, and literary text using a three-term objective:
$ cal(L)(theta_S) &= alpha_"kl" cal(L)_"DKL" + alpha_"cos" (1 - cos(bold(u)_S, bold(u)_T)) \ & quad + frac(alpha_"hid", |cal(M)|) sum_((m, ell) in cal(M)) norm( hat(bold(h))_S^((m)) - hat(bold(h))_T^((ell)) )_2^2 $ <eq-distill>
where $hat(bold(h)) = bold(h) / norm(bold(h))_2$ aligns intermediate block trajectories across layer map $cal(M) = \{(1,5), (2,11), (3,16), (4,22), (5,27)\}$, and $cal(L)_"DKL"$ is a *Dual-Temperature Forward KL* over the pruned 24,576-logit head $bold(z) = bold(u) bold(E)^top$:
$ cal(L)_"DKL" &= 0.7 tau_1^2 D_"KL" (sigma(bold(z)_T / tau_1) || sigma(bold(z)_S / tau_1)) \ & quad + 0.3 D_"KL" (sigma(bold(z)_T) || sigma(bold(z)_S)) $ <eq-dkl>
with $tau_1 = 2.0$, $alpha_"kl" = 1.0$, $alpha_"cos" = 2.5$, and $alpha_"hid" = 1.5$. Training for just 1,600 steps (batch 32, length 192, peak LR $3 times 10^(-4)$) on a single NVIDIA L4 GPU takes 14 minutes, driving $cos(bold(u)_S, bold(u)_T)$ from $0.0191 arrow.r 0.9310$ on validation sequences ($0.8613$ after group-64 4-bit MLX affine quantization into *64.17 MB*).

= Rank-Stacked Multi-LoRA Cartridges

== The `.imepack` Cartridge Abstraction

To let users extend Chime across specialized domains without bloating or retraining the 64.17 MB base model, we introduce `.imepack` v2 cartridges. Each cartridge bundles:
1. `pack.imedict`: A zero-copy `CHIMED02` slice containing domain words, canonical English terms, $n$-grams, and $O(log C)$ `Tab` continuation records;
2. `lora.safetensors`: Rank-$r$ ($r=8$) FP16 low-rank matrices $(bold(A)_k^((ell, p)) in RR^(r times d_"in"), bold(B)_k^((ell, p)) in RR^(d_"out" times r))$ across all 7 linear projections $p in \{q, k, v, o, "gate", "up", "down"\}$ of all 6 layers ($1,032,192$ parameters $= 2.07$ MB per pack);
3. `manifest.json`: Declaring routing policy (`classical`, `developer`, or `always`), base scale $s_k$, and privacy flag `private_pack`.

== Rank-Stacked GEMM Fusion on Unified Memory

When $K$ cartridges are mounted concurrently, a naive implementation evaluates each active adapter sequentially inside every linear layer:
$ bold(y) = "QuantMatmul"_"4bit" (bold(x), hat(bold(W))) + sum_(k=1)^K gamma_k (bold(x) bold(A)_k^top) bold(B)_k^top $ <eq-unfused>
Across 6 layers and 7 projections, @eq-unfused dispatches $42 times 2K$ tiny skinny-matrix multiplications (`[1, T, 1024] @ [1024, 8]`) plus $42 times 2K$ scalar/add kernels per forward pass. On Apple Silicon Metal GPUs, kernel launch and command-encoder overhead completely dominate arithmetic cost at $r=8$, causing 6-candidate reranking latency to grow linearly by *+1.51 ms per adapter* (reaching 10.47 ms at $K=6$, violating our 10 ms frame-0 synchronous budget).

We eliminate this bottleneck via *Rank-Stacked GEMM Fusion*. At cartridge mount time, Chime pre-transposes and concatenates all $K$ adapters along the rank dimension once in Unified Memory ($R = sum_(k=1)^K r_k = 8K$):
$ bold(A)_"cat"^top &= [ bold(A)_1^top thin dots thin bold(A)_K^top ] in RR^(d_"in" times R), \ bold(B)_"cat"^top &= [ bold(B)_1 thin dots thin bold(B)_K ]^top in RR^(R times d_"out") $ <eq-stacked>
For any dynamic routing vector $bold(gamma) = (gamma_1, dots, gamma_K) in RR^K$, let $bold(g) = "repeat"(bold(gamma), r) in RR^(1 times 1 times R)$ replicate each scalar $gamma_k$ across its $r$ rank channels. By block-matrix associativity, @eq-unfused reduces to *two* GEMMs and one broadcast multiply regardless of $K$:
$ Delta bold(y) = ((bold(x) bold(A)_"cat"^top) circle.tiny bold(g)) bold(B)_"cat"^top $ <eq-fused>
When $bold(gamma) = bold(0)$, the LoRA branch is bypassed entirely (0 FLOPs). Furthermore, `MlxPrefixCache` tags cached KV tensors with the quantized signature $"round"(10^3 bold(gamma))$, reusing prefix KV states across consecutive keystrokes whenever routing weights remain unchanged and recomputing the 16–96 token prefix ($approx 1.5$ ms) only upon regime transitions.

#place(
  top + center,
  scope: "parent",
  float: true,
)[
  #figure(
    fig-scaling-and-transfer(),
    kind: image,
    caption: [
      *(a)* Measured 6-candidate Metal reranking latency (`p50`, ms) as mounted Rank-8 cartridges scale from $K=0$ to $6$: Rank-Stacked GEMM Fusion flattens the $O(K)$ kernel-launch ramp ($+1.51$ ms/adapter $arrow.r +0.03$ ms/adapter, $3.38 times$ speedup at $K=6$). *(b)* Zero-shot transfer of an FP16-trained `classics` LoRA onto the 4-bit quantized base model exhibits a convex U-shaped NLL curve achieving its global minimum at $gamma^* = 0.50$ ($-0.161$ nats) before quantization cross-terms cause overshoot at $gamma = 1.0$.
    ],
  ) <fig-scaling>
]

== Register Superposition & Zero-Shot INT4 Transfer

Mounting multiple domain adapters simultaneously reveals two subtle statistical interactions that degrade naive additive LoRA composition:

1. *Register Superposition Drift*: Domain packs (`classics-poetry`, `tech-developer`, and long-form essay packs) are predominantly trained on *formal written corpora*. In written text, formal homophones such as 多达 (#emph["up to"]) vastly outnumber spoken interrogatives like 多大 (#emph["how large"]). Summing three adapters with uncalibrated unit scales ($sum_k gamma_k = 2.85$) constructively amplifies their shared written-register direction and pushes pre-norm activations into SwiGLU non-linear saturation, dropping colloquial accuracy from 8/8 to 6/8 and classical NLL from 5.978 to 7.598 (@tab-routing).
2. *Zero-Shot FP16$arrow.r$INT4 Scale Shift*: Community and personal `LoRA`s are conveniently trained against the clean FP16 student $bold(W)$, yet deployed onto the 4-bit quantized base $hat(bold(W)) = bold(W) + bold(E)_Q$. Expanding expected token NLL $cal(L)(hat(bold(W)) + gamma Delta bold(W))$ to second order around $bold(W)$ yields:
$ cal(L)(hat(bold(W)) + gamma Delta bold(W)) &approx cal(L)(bold(W)) + gamma chevron.l nabla cal(L), Delta bold(W) chevron.r \ & quad + frac(1, 2) norm(bold(E)_Q + gamma Delta bold(W))_(bold(H))^2 $ <eq-taylor>
Minimizing with respect to $gamma$ gives $gamma^* = (- chevron.l nabla cal(L), Delta bold(W) chevron.r - chevron.l bold(E)_Q, Delta bold(W) chevron.r_(bold(H))) / norm(Delta bold(W))_(bold(H))^2$. Because quantization noise $bold(E)_Q$ perturbs intermediate feature covariance across 6 cascaded layers, full unit step $gamma = 1.0$ overshoots the quantized valley, whereas attenuated $gamma^* approx 0.45$–$0.55$ achieves significant NLL gains with *zero quantization-aware retraining* (@fig-scaling (b)).

Chime resolves both phenomena via a *Simplex-Bounded Context-Gated Router*:
- *Hard Domain Gating*: `Classical` routing sets $tilde(gamma)_k = 0$ unless classical lexical particles (`之乎者也矣焉哉兮`) or verse rhythms appear in $(bold(c)_L, cal(B))$; `Developer` sets $tilde(gamma)_k = s_k$ on code/ASCII cues, $0.25 s_k$ in terminal/IDE bundle IDs, and $0$ in chat/prose apps; `Always` (personal style) attenuates to $0.45 s_k$ on short $<= 2$-char colloquial inputs.
- *Simplex Gain Cap*: Active weights are projected onto the $ell_1$ ball $sum_(k=1)^K gamma_k <= Gamma_"max" = 0.90$ via $gamma_k = tilde(gamma)_k dot min(1, Gamma_"max" / sum_j tilde(gamma)_j)$, bounding total perturbation norm and naturally placing multi-pack mixtures inside the $[0.35, 0.55]$ INT4 transfer sweet spot.

= Experiments & Analysis

== Experimental Setup & Datasets

All runtime and latency experiments are conducted natively on an Apple Silicon Mac running macOS 15 (`mlx-rs 0.34`, Metal GPU, single-threaded actor worker without locks). Layer distillation (`1,600` steps) ran on an NVIDIA L4 24 GB GPU; all three Rank-8 ($r=8$) `LoRA` adapters (`300` steps, AdamW, cosine schedule, 2.07 MB FP16 each) were trained directly on Apple Silicon MPS:
- *`classics-poetry.imepack`*: Compiled from 37 canonical Tang/Song/Guwen masterpieces, 21 `chinese-poetry` collections (Tang/Song/Yuan/Shijing/Chuci/Four Books), and Xinhua idioms/xiehouyu, totaling *266,816* lexical entries (including merged short-pause Song Ci lead-in clauses such as 念去去，千里烟波) and *231,593* `Tab` continuation chains, plus 18,482 classical paragraphs for `LoRA` training.
- *`tech-developer.imepack`*: Curated bilingual systems/AI/cloud terminology + 1.20M characters (2,503 passages) filtered from `FineWeb-Edu-Chinese-V2.1` and `BelleGroup`.
- *Air-Gapped Longitudinal Personal Pack*: Trained strictly offline on *222 human-authored essays* (2013–2020; 6,533 paragraphs, 814,988 characters, strictly pre-dating generative AI tools to avoid synthetic stylistic contamination), scrubbed via Chime's PII/secret sanitizer and yielding 8,000 personal words + 4,727 bigrams.

#place(
  top + center,
  scope: "parent",
  float: true,
)[
  #figure(
    table(
      columns: (2.10fr, 0.82fr, 0.72fr, 0.86fr, 0.96fr, 0.86fr, 0.90fr),
      align: (left, right, right, right, right, right, right),
      stroke: none,
      inset: (x: 4.5pt, y: 4.2pt),
      toprule,
      [*Model Checkpoint*], [*Size*], [*Compr.*], [*$cos(bold(u)_S, bold(u)_T)$*], [*Stress Acc.*], [*Margin $Delta ell$*], [*Rerank-6 p50*],
      midrule,
      [`Qwen3-0.6B` (28L, 152K, FP16)], [1191.5 MB], [$1.0 times$], [1.0000], [87.5% (14/16)], [+5.48 nats], [$approx 14.2$ ms],
      [`Qwen3-28L` (24K vocab, 4-bit)], [249.95 MB], [$4.8 times$], [1.0000], [87.5% (14/16)], [+5.43 nats], [5.34 ms],
      [`Qwen3-6L-NaiveSkip` (4-bit)], [64.15 MB], [$18.6 times$], [#text(fill: rgb("#be123c"))[0.0191]], [#text(fill: rgb("#be123c"))[43.8% (7/16)]], [+0.10 nats], [1.49 ms],
      [`Qwen3-6L-Distilled` (FP16)], [228.03 MB], [$5.2 times$], [0.8287], [87.5% (14/16)], [+5.99 nats], [2.65 ms],
      [`Qwen3-6L-Distilled` (8-bit)], [121.15 MB], [$9.8 times$], [0.8287], [87.5% (14/16)], [+5.99 nats], [1.64 ms],
      [*`Qwen3-6L-Chime` (4-bit, Ours)*], [*64.17 MB*], [*$18.6 times$*], [*0.8613*], [*87.5% (14/16)*], [*+6.11 nats*], [*1.37 ms*],
      bottomrule,
    ),
    caption: [
      *Teacher-to-Student Compression & Manifold Alignment Pareto Frontier* on Apple Silicon Metal (`T=16` prefix tokens + 6 homophone candidates). Naive 6-layer skipping destroys representation alignment ($cos = 0.0191$), whereas our manifold-aligned distillation recovers 100% of the 28-layer teacher's disambiguation accuracy in 64.17 MB at $3.91 times$ faster reranking.
    ],
  ) <tab-pareto>
]

== Main Results & Ablation Studies

*1. Distillation & Quantization Pareto Frontier (@tab-pareto).* Vocabulary pruning + 4-bit group-64 quantization compresses the 28-layer teacher to 249.95 MB (5.34 ms reranking). Skipping directly to 6 layers cuts size to 64.15 MB and latency to 1.49 ms, but collapses pre-head cosine similarity to $0.0191$ and drops zero-shot minimal-pair accuracy from 14/16 to 7/16 (mean log-likelihood margin $+0.10$ nats). Our 1,600-step manifold-aligned distillation restores $cos(bold(u)_S, bold(u)_T) = 0.8613$, achieving *14/16 (87.5%)* zero-shot stress accuracy, *+6.11 nats* mean margin, *0.91 ms* 16-token prefill, *0.77 ms* 1-step continuation, and *1.37 ms* 6-candidate batch reranking—well inside the 10 ms frame-0 synchronous budget. On the full 42-case Chime IME suite (covering full/abbreviated Pinyin, Xiaohe/Ziranma Shuangpin, fuzzy rhymes, neighbor/swap/repeat typos, code-switching, punctuation, and poetry chaining), Chime achieves *42/42 (100%)* Top-1 accuracy.

#figure(
  table(
    columns: (1.58fr, 1.08fr, 0.82fr, 0.84fr),
    align: (left, right, right, right),
    stroke: none,
    inset: (x: 4pt, y: 4pt),
    toprule,
    [*Configuration*], [*Verse Top-1*], [*Cls. NLL*], [*Pers. NLL*],
    midrule,
    [C0: Modern Base Only], [10.0% (3/30)], [5.978], [5.628],
    [C1: `LoRA` Sidecar Only], [10.0% (3/30)], [5.817], [5.227],
    [C2: `pack.imedict` Only], [90.0% (27/30)], [5.978], [5.628],
    [*C3: Full `.imepack`*], [*90.0% (27/30)*], [*5.817*], [*5.227*],
    bottomrule,
  ),
  caption: [
    *Empirical Verification of Recall–Rerank Decoupling* on 30 held-out classical verses outside the core seed and held-out prose validation blocks. `LoRA` alone reduces domain NLL ($-0.161$ classical, $-0.401$ personal) yet yields *0%* gain on OOV verses without `pack.imedict`.
  ],
) <tab-decouple>

*2. Lexical-Neural Decoupling (@tab-decouple).* Evaluating 30 held-out classical Tang/Song/Guwen verses against a modern-only base lexicon empirically confirms Proposition 1: mounting `classics_lora.safetensors` alone (`C1`) lowers classical validation NLL from 5.978 to 5.817 (and personal `LoRA` lowers held-out personal essay NLL by *-0.401 nats*, from $5.628 arrow.r 5.227$), yet verse Top-1 accuracy remains stuck at *3/30 (10.0%)* because Stage-1 never proposes OOV verses. Co-mounting the full `.imepack` (`C3`) boosts Top-1 conversion $9 times$ to *27/30 (90.0%)* while retaining neural disambiguation and 231,593 `Tab` continuation chains.

#figure(
  table(
    columns: (1.42fr, 0.86fr, 0.86fr, 0.86fr, 0.72fr),
    align: (left, right, right, right, right),
    stroke: none,
    inset: (x: 3.5pt, y: 4pt),
    toprule,
    [*Routing ($K=3$)*], [*Colloq.*], [*Class.*], [*Dev.*], [*NLL*],
    midrule,
    [R0: Base ($gamma = bold(0)$)], [8/8 (+8.4)], [8/8 (+4.3)], [8/8 (+9.3)], [5.978],
    [R1: Uncalib. Sum], [#text(fill: rgb("#be123c"))[6/8 (+3.8)]], [#text(fill: rgb("#be123c"))[5/8 (+1.9)]], [8/8 (+6.7)], [#text(fill: rgb("#be123c"))[7.598]],
    [R2: Uniform $1/K$], [8/8 (+7.5)], [8/8 (+4.3)], [8/8 (+9.2)], [5.982],
    [*R3: Simplex-Gated*], [*8/8 (+8.8)*], [*8/8 (+4.3)*], [*8/8 (+9.4)*], [*5.937*],
    bottomrule,
  ),
  caption: [
    *Multi-LoRA Interference vs. Simplex-Gated Routing* with 3 concurrent packs (`classics` + `tech` + `personal`). Cells show Top-1 wins and mean margin (nats). Uncalibrated summation (`R1`) induces severe register drift and activation overshoot.
  ],
) <tab-routing>

*3. Multi-LoRA Interference & Routing (@tab-routing).* When all three packs are mounted concurrently, naive additive summation (`R1`, $sum gamma_k = 2.85$) degrades colloquial accuracy to 6/8 (margin halved from $+8.4 arrow.r +3.8$ nats), classical accuracy to 5/8, and explodes classical NLL to 7.598. Uniform $1/K$ averaging (`R2`) avoids saturation but dilutes colloquial margin ($+7.5$ vs. $+8.4$) and erases classical NLL gains ($5.982$). Our Simplex-Gated Router (`R3`) achieves *8/8* across all three regimes with the *highest margins* in colloquial ($+8.83$ nats) and developer ($+9.40$ nats) probes and the *lowest NLL* ($5.937$).

#figure(
  table(
    columns: (0.48fr, 1.05fr, 1.00fr, 1.00fr, 0.85fr),
    align: (center, right, right, right, right),
    stroke: none,
    inset: (x: 4pt, y: 3.8pt),
    toprule,
    [*$K$*], [*Unfused p50*], [*Fused p50*], [*Fused p95*], [*Speedup*],
    midrule,
    [0], [1.42 ms], [1.43 ms], [1.67 ms], [$1.00 times$],
    [1], [2.93 ms], [2.95 ms], [3.07 ms], [$0.99 times$],
    [2], [4.50 ms], [*3.01 ms*], [3.12 ms], [*$1.50 times$*],
    [3], [6.00 ms], [*3.01 ms*], [3.11 ms], [*$1.99 times$*],
    [4], [7.55 ms], [*3.03 ms*], [3.15 ms], [*$2.49 times$*],
    [5], [8.99 ms], [*3.09 ms*], [3.34 ms], [*$2.90 times$*],
    [6], [10.47 ms], [*3.10 ms*], [*3.20 ms*], [*$3.38 times$*],
    bottomrule,
  ),
  caption: [
    *Rank-Stacked GEMM Fusion Scaling* ($K=0 dots 6$ mounted Rank-8 adapters). Marginal latency from $K=1 arrow.r 6$ drops from $+1,508 thin mu"s"$/adapter (unfused) to *$+30.6 thin mu"s"$/adapter* (fused)—a *$49.3 times$ reduction* in per-adapter overhead.
  ],
) <tab-scaling>

*4. Kernel Fusion Scaling & Length Calibration (@tab-scaling & Exp. 5).* As mounted adapters grow from $K=1$ to $6$, Rank-Stacked GEMM Fusion keeps 6-candidate p50 reranking virtually flat (`2.95 ms` $arrow.r$ `3.10 ms`, p95 `3.20 ms`), cutting marginal per-adapter latency by *$49.3 times$* compared to sequential execution (`10.47 ms`). Finally, sweeping length-normalization exponent $alpha in [0, 1]$ in @eq-rerank confirms that character-count scaling preserves 14/16 stress accuracy across all $alpha$, while $alpha = 0.42$ paired with Probability-Space Separation (Proposition 2) uniquely achieves *42/42 (100%)* end-to-end accuracy—preventing both dilution on 9-character sentences ($alpha >= 0.9$, where `今天我们使用本模型` flips to `试用`) and horizon penalties on poem completions (`君不见` + context $arrow.r$ `君不见高堂明镜悲白发`).

== Qualitative Case Studies

- *Stateful Poetry Chaining & Anti-Repeat*: Typing `junbujianhuanghe` + `Tab` commits 君不见黄河之水天上来 and surfaces ghost continuation `[Tab] ，奔流到海不复回` $arrow.r$ `。君不见高堂明镜悲白发` across all 24 clauses of Li Bai's *Qiang Jin Jiu*. Crucially, if the user manually types `junbujian` *after* `...奔流到海不复回。`, Chime's context-continuation boost ($+7.5$) and verbatim anti-repeat penalty ($-5.5$) promote 君不见高堂明镜悲白发 to \#1 instead of repeating the opening line.
- *Compound Motor Slip + Ci Rhythm*: On `nianquuqqianliyanbo` (containing transposed `uq` $arrow.r$ `qu` inside Liu Yong's Song Ci lead-in), Stage-1 recovers `qu` via $delta_"swap"$ and matches the merged rhythmic unit 念去去，千里烟波 at \#1 in 7.6 ms.

= Limitations & Ethical Considerations

*Limitations.* First, Chime's 64.17 MB student relies on Stage-1 lattice recall (`CHIMED02` + `SQLite` user store); truly novel neologisms absent from all mounted `.imepack`s and user history still require manual step-by-step prefix composition (`今天 wo'men`) on their first occurrence before online bigram/phrase learning caches them. Second, our current context router uses interpretable lexical and bundle-ID features; learning an ultra-lightweight ($< 10$ KB) linear classifier over embedding pooled states could further refine soft blending in mixed classical-modern essays.

*Privacy & Broader Impact.* Input methods observe a user's most intimate thoughts, credentials, and unpublished code. Chime enforces zero network entitlement, bypasses `IsSecureEventInputEnabled()` password fields in hardware, and enables air-gapped personal `LoRA` training directly on local Apple Silicon GPUs after deterministic secret/PII redaction—demonstrating that deep LLM personalization need not come at the expense of keystroke surveillance.

= Conclusion

Chime demonstrates that LLMs are most effective in client-side input methods not as monolithic, latency-heavy sequence generators, but as *manifold-distilled, prefix-cached parallel rescorers* co-designed with zero-copy lexical structures and *Rank-Stacked Multi-LoRA cartridges*. Compressing `Qwen3-0.6B` by $18.6 times$ into 64.17 MB on Unified Memory while bounding multi-adapter mixtures on the simplex delivers sub-3 ms neural intelligence, modular domain mastery, and strictly offline personal adaptation on everyday laptops.

#bibliography("refs.bib", title: "References", style: "association-for-computing-machinery")
