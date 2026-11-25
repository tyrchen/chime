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
      Applying Large Language Models (LLMs) to client-side Pinyin Input Method Editors (IMEs) faces a strict systems trilemma: human typing demands sub-10 ms keystroke latency without candidate-bar jitter, operating systems enforce tight resident memory budgets ($< 100$ MB) alongside zero-network privacy guarantees, and real-world input spans open-domain vocabulary from bilingual code identifiers to classical literature and personal style. Existing generative Pinyin-to-Character (P2C) LLMs violate latency budgets via autoregressive decoding and suffer representation collapse under aggressive layer pruning or out-of-vocabulary (OOV) blindness when relying solely on parameter-efficient adapters. We present *Chime*, a 100% offline neural IME for Apple Silicon that resolves this trilemma through lexical-neural co-design. First, we formalize _Recall–Rerank Decoupling_, _Probability-Space Separation_, and _Pruned-BPE Partition Calibration_, pairing a $< 0.8$ ms zero-copy `mmap` prefix-diverse lattice decoder (`CHIMED02`) with a batched prefix-cached causal SLM rescorer. Second, we compress the 28-layer `Qwen3-0.6B` (1.19 GB FP16) into a 6-layer, 24,576-vocab 4-bit MLX student (*64.17 MB*, 18.6× smaller); whereas naive layer skipping collapses pre-head alignment to $cos(bold(u)_S, bold(u)_T) = 0.022$ and halves homophone accuracy to 43.8%, two-stage _Trajectory + Homophone-Subspace Distillation_ restores $cos(bold(u)_S, bold(u)_T) = 0.863$ and reduces within-cohort homophone KL by *5.2×*, matching the 28-layer teacher's *93.8%* (15/16) stress accuracy at *0.86 ms* 6-candidate Metal latency (6.20× faster than 28-layer 4-bit). Third, pluggable _`.imepack` cartridges_ co-mount zero-copy lexical slices (`pack.imedict`) with _Q4-Manifold Aligned_ 2.07 MB Rank-8 `LoRA` sidecars; _Rank-Stacked GEMM Fusion_, _Batched Broadcast KV-Gather_, and a _Couplet-Aware Simplex Router_ cut 6-adapter latency from 10.74 ms to *1.41 ms* (*7.60× speedup*, $+39.4 thin mu"s"$/adapter) and eliminate FP16$arrow.r$INT4 overshoot ($-0.301$ nats classical NLL gain). Across *2,268* strictly decontaminated queries without open-domain phrase hardcoding, Chime lifts Top-1 accuracy by *+18.44* points (*85.85%* Acc\@1, *93.00%* Hit\@5) on a 1,258-case 17-category benchmark and by *+28.52* points ($42.87% arrow.r bold("71.39%")$ Acc\@1, *77.13%* Hit\@5) on a blind 1,010-sentence holdout, alongside *56/56 (100%)* golden regression pass.
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
    End-to-end architecture of *Chime*. Stage 1 maps unified-Zipf `CHIMED02` binary dictionaries via `mmap` ($< 0.8$ ms), preserves syntactic diversity via prefix-diverse beam search, and separates equal-span candidates $cal(C)_n$ from lookahead completions $cal(C)_"comp"$. Stage 2 runs a homophone-distilled 64.17 MB 4-bit `Qwen3-6L` student with batched broadcast `MlxPrefixCache`, pruned-BPE partition calibration, and fused Rank-Stacked `.imepack` `LoRA` sidecars.
  ],
) <fig-arch>

Our primary contributions are fourfold:
- *Recall–Rerank Co-Design, Measure Separation & Partition Calibration (§3)*: We prove that Stage-2 rescoring is bounded by Stage-1 prefix-diverse recall, must separate equal-span hypotheses $cal(C)_n$ from lookahead completions $cal(C)_"comp"$, and must correct _pruned-vocabulary BPE partition inflation_ via a subword compression prior paired with relative length scaling.
- *Two-Stage Trajectory & Homophone-Subspace Distillation (§4)*: Combining vocabulary pruning ($"151,936" arrow.r "24,576"$), frozen tied embeddings, trajectory/cosine alignment, and phonetic cohort KL distillation, we compress `Qwen3-0.6B` by 18.6× into a 64.17 MB 4-bit MLX student ($cos(bold(u)_S, bold(u)_T) = 0.863$) that cuts within-cohort homophone KL by *5.2×* and matches the 28-layer teacher's stress accuracy (93.8%) at *0.86 ms* (6.20× faster).
- *Rank-Stacked `.imepack` Cartridges & Q4-Manifold Alignment (§5)*: We package zero-copy `pack.imedict` slices alongside 2.07 MB Rank-8 `LoRA`s into hot-swappable cartridges. _Rank-Stacked GEMM Fusion_ and _Batched Broadcast KV-Gather_ cut 6-adapter reranking from 10.74 ms to *1.41 ms* (*7.60× speedup*, $+39.4 thin mu"s"$/adapter), while _Q4-Manifold LoRA Training_ and our _Couplet-Aware Simplex Router_ eliminate multi-adapter drift and INT4 overshoot ($-0.301$ nats).
- *Decontaminated Thousand-Sentence Evaluation (§6)*: Across 2,268 queries strictly decontaminated at the 5-gram level—1,258 multi-category dev queries and 1,010 blind unseen whole-sentence holdout queries—plus a 266K-entry classical pack and an air-gapped 815K-char pre-LLM personal corpus (2013–2020), Chime improves blind whole-sentence Top-1 accuracy from $42.87% arrow.r bold("71.39%")$ (*+28.52* points) and passes *56/56 (100%)* golden regressions.

= Related Work

*Statistical and Graph-Based Pinyin Decoding.* Classical P2C engines model conversion as Viterbi search over word lattices weighted by character/word #box[$n$-grams] @chen2000new and double-array tries @yoshinaga1992double. @jia2014joint introduced a joint graph model integrating QWERTY typo correction with P2C conversion. While sub-millisecond and memory-frugal, finite-context #box[$n$-grams] fail on context-sensitive homophones (e.g., 保障*权利* [safeguard rights] vs. 滥用*权力* [abuse power]) and collapse on multi-clause syntactic dependencies when legacy newspaper priors collide with spoken Mandarin.

*Neural and LLM-Based Input Methods.* @zhang2019open augmented neural P2C with online open-vocabulary sampling. @xu2022pert proposed `PERT`, a bidirectional encoder pre-trained directly on Pinyin sequences. With the rise of generative LLMs, @tan2022exploring adapted frozen Chinese GPT to P2C, observing severe degradation on abbreviated Pinyin due to combinatorial candidate explosion. `GeneInput` @ding2023geneinput unified full-mode key-to-character generation via prompting and RLHF, `AttnInput` @gui2025attninput attached a Pinyin side-network to linear RWKV models, and `HuoziIME` @shan2026huoziime explored on-device LLM personalization via hierarchical memory. Unlike these end-to-end generative designs—which struggle with sub-5 ms keystroke budgets, multi-syllable typo graphs, and exact casing for code-switching—Chime retains deterministic $O(log N)$ lattice recall in Stage 1 and deploys a prefix-cached causal SLM strictly as a constant-depth parallel rescorer in Stage 2.

*Edge Compression and Multi-LoRA Serving.* Knowledge distillation @hinton2015distilling with hidden-state alignment (`TinyBERT` @jiao2020tinybert) and group-wise weight quantization (`GPTQ` @frantar2023gptq, `AWQ` @lin2024awq) enable compact edge deployment. In cloud serving, `S-LoRA` @sheng2024slora batches heterogeneous requests across distinct per-tenant adapters using custom paging kernels, whereas personalized `LoRA` routing (`PLoRA` @zhang2024plora) blends adapters per user. To our knowledge, Chime is the first system to investigate _single-stream concurrent Multi-LoRA superposition_ on unified-memory client GPUs—identifying and solving both multi-adapter register interference and zero-shot FP16-to-INT4 scale overshoot.

= Problem Formulation & Hybrid Co-Design

== Prefix-Diverse Phonetic-Typo-Code Lattice

Given a raw ASCII keystroke sequence $bold(k) = k_(1:n)$, surrounding left document context $bold(c)_L$ (bounded to 96 characters), and active host application identifier $a$ (e.g., `com.mitchellh.ghostty` vs. `md.obsidian`), an IME seeks the highest-scoring surface string $bold(w)^*$:
$ bold(w)^* = arg max_(bold(w) in cal(G)(bold(k))) [& S_"lex" (bold(w), bold(k) | bold(c)_L, a) \ & + S_"slm" (bold(w) | bold(c)_L, a)] $ <eq-joint>
where $cal(G)(bold(k))$ is the dual-track segmentation lattice constructed over byte spans $[i, j) subset.eq [0, n)$.

To achieve $< 0.8$ ms Stage-1 construction without memory bloat or open-domain phrase hardcoding, Chime separates _declarative closed-class orthography_ (compile-time YAML tables for polyphone readings, closed-class pronouns/classifiers/particles, and canonical English identifiers) from _corpus-calibrated open-domain priors_ compiled into `CHIMED02` (`mmap`). Crucially, fusing heterogeneous corpora requires _Affine Zipf Scale Alignment_: legacy 1990s newspaper dictionaries over-index formal bureaucracy (e.g., 届, 并, 试用) relative to modern web/conversational Mandarin (接, 饼, 使用). Chime maps modern multi-source frequencies (`wordfreq`), legacy segmenter dictionaries, and 28-layer `Qwen3` teacher log-probabilities onto a unified base-10 Zipf scale ($z(w) = log_10 P(w) + 9.0$) bounded below by a character-marginal floor so common spoken characters never starve on unseen inflections.

At span-lookup time, Stage 1 evaluates an algebraic fuzzy equivalence relation $cal(F)(s)$ (`z`$harpoons.rtlb$`zh`, `an`$harpoons.rtlb$`ang`, `in`$harpoons.rtlb$`ing`, etc.) alongside four 1-edit motor-noise operators on raw slice $bold(k)_(i:j)$: (1) _adjacent transposition_ (`shaung` $arrow.r$ `shuang`, cost $delta_"swap" = 1.05$); (2) _QWERTY neighbor substitution_ ($d^2 <= 1.85$, cost $0.85 + 0.45 d^2$); (3) _accidental repeat deletion_ (`niian` $arrow.r$ `nian`, cost $delta_"rep" = 1.10$); and (4) _geminate boundary elision_ (`chunuan` $arrow.r$ `chun`+`nuan` 春暖, cost $delta_"gem" = 0.85$). Rather than running standard Viterbi beam search—where open-class disyllabic homophones (`woshi` $arrow.r$ 我试, 我市, 卧室, 我司) monopolize all beam slots before closed-class copula-classifier skeletons (`woshiyigebing` $arrow.r$ 我是一个兵) reach Stage 2—Chime enforces _Prefix-Diverse Beam Search_ (capping identical 2-character prefixes per byte boundary) and _Prosodic Clitic Cohesion_ (zero word-boundary penalty across closed-class Mandarin clitics such as numeral+classifier and verb+了/着/过/的).

== Measure Separation & Pruned-BPE Partition Calibration

Why not rely on `LoRA` fine-tuning alone to inject domain phrases or classical poetry? Let $cal(C)_M (bold(k)) subset cal(G)(bold(k))$ denote the top-$M$ candidates retained by Stage-1 beam search.

#prop-box("Proposition 1 (Stage-1 Recall Bottleneck)")[
  For any finite Stage-2 rescorer $S_"slm" : Sigma^* arrow.r RR$ with bounded dynamic range $|S_"slm"| <= B$, if a target multi-character entity or verse $bold(w)^*$ is absent from the mounted lexicon such that fragmented character paths push $bold(w)^*$ outside $cal(C)_M (bold(k))$, then $PP(bold(w)^* = bold(w)_1) = 0$ regardless of `LoRA` rank $r$ or Stage-2 capacity.
]

Proposition 1 explains why typing `mingyuejishiyou` without a classical lexicon yields 明月即使有: modern high-frequency bigrams (即使, $log P = -3.8$) evict fragmented character paths (几 + 时) before Stage 2 executes. Equally critical is _which_ candidates inside $cal(C)_M (bold(k))$ may be compared by $S_"slm"$ when full-span interpretations $cal(C)_n$ coexist with prefix lookahead completions $cal(C)_"comp"$ (e.g., `junbujian` $arrow.r$ 君不见高堂明镜悲白发).

#prop-box("Proposition 2 (Measure Incompatibility across Horizons)")[
  Let $bold(w)_s in cal(V)^m$ be an $m$-character full-span candidate and $bold(w)_ell = bold(w)_s circle.tiny bold(w)_"rem" in cal(V)^(m+d)$ ($d >= 1$) be a lookahead completion. Under any causal LM $P_theta$, $log P_theta (bold(w)_ell | bold(c)_L) = log P_theta (bold(w)_s | bold(c)_L) + sum_(t=1)^d log P_theta (w_(m+t) | bold(c)_L, bold(w)_(<m+t)) < log P_theta (bold(w)_s | bold(c)_L)$ whenever $P_theta (w_(m+t) | dot) < 1$, incurring a systematic horizon penalty of $Theta(d^(1-alpha))$ nats under $abs(bold(w))^alpha$ ($alpha < 1$).
]

Guided by Proposition 2, Chime enforces _Probability-Space Separation_: Stage-2 rescoring operates strictly over the equal-input homophone subspace $cal(B) subset.eq cal(C)_n backslash cal(C)_"comp"$, while $cal(C)_"comp"$ is governed by deterministic $O(log C)$ continuation chains (`CHIMED02` offset 40) and verbatim anti-repeat penalties ($-5.5$ nats).

Evaluating $cal(B)$ under an aggressively vocabulary-pruned BPE model ($V_T = "151,936" arrow.r V_S = "24,576"$, pruning 83.8% of subwords) uncovers a second fundamental distortion that we formalize next:

#prop-box("Proposition 3 (Pruned-BPE Partition Inflation)")[
  Let $P_T$ be a causal LM over full vocabulary $V_T$ and $P_S (z | h_t) = exp(bold(u)_t^top bold(e)_z) / (sum_(v in V_S) exp(bold(u)_t^top bold(e)_v))$ be its projection onto pruned subset $V_S subset V_T$. At step $t$, pruning removes probability mass $p_"out" (h_t) = sum_(v in V_T backslash V_S) P_T (v | h_t)$, inflating every surviving token's log-probability:
  #set math.equation(numbering: none)
  $ log P_S (z_t | h_t) &= log P_T (z_t | h_t) - log(1 - p_"out" (h_t)) \ &> log P_T (z_t | h_t). $
  Consequently, among equal-character homophones ($abs(bold(w)_1) = abs(bold(w)_2) = m$), an out-of-$V_S$ compound $bold(w)_2$ that falls back to $Delta T(bold(w)_2) = T(bold(w)_2) - T_"min" >= 1$ extra single-character tokens accumulates $+Delta T(bold(w)_2) dot EE[-log(1 - p_"out")]$ nats of spurious partition inflation over an in-$V_S$ single-token word $bold(w)_1$.
]

Empirically, on `shiyongdeguochengzhong` after context 我在, common words 使用/适用/实用 remain single BPE tokens ($T = 3$ for the 6-char phrase), whereas rare 试用 was pruned from $V_S$ and splits into 试 + 用 ($T = 4$, $Delta T = 1$), leaking $+2.8$ nats from pruned 试$*$ compounds (试图, 试验) in the softmax denominator. Chime neutralizes Proposition 3 via a _Subword Compression Prior_ ($beta = 1.85$), anchors non-empty $bold(c)_L$ with “。” at RoPE index 0 to match standalone positional geometry, and normalizes by _relative_ character length $tilde(r)(bold(w)) = (abs(bold(w)) / overline(m))^(0.42)$ (preserving 1:1 natural-log scale when $abs(bold(w)) = overline(m)$):
$ tilde(ell)(bold(w)) = frac(ell(bold(w)) - beta max(0, T(bold(w)) - tau_"min" abs(bold(w))), (abs(bold(w)) / overline(m))^(0.42)), $ <eq-bpe>
$ S_"slm" (bold(w) | bold(c)_L, a) = lambda_(bold(c)_L) ( tilde(ell)(bold(w)) - frac(1, abs(cal(B))) sum_(bold(w)' in cal(B)) tilde(ell)(bold(w)') ) $ <eq-rerank>
where $tau_"min" = min_(bold(w)' in cal(B)) T(bold(w)') / abs(bold(w)')$, and prior-calibrated fusion sets $lambda_(bold(c)_L) = 1.18$ when $bold(c)_L != emptyset$ (letting contextual PMI override raw unigram skew) and $0.95$ when $bold(c)_L = emptyset$.

= Manifold-Aligned 6-Layer Distillation

== Vocabulary Pruning & Naive Skip Collapse

Our teacher is `Qwen/Qwen3-0.6B-Base` @yang2025qwen3 ($L_T = 28$ layers, $d = 1024$, $d_"ff" = 3072$, 16 query heads, 8 KV heads with `head_dim` $= 128$, QK-RMSNorm, tied embeddings, $V_T = "151,936"$, 1,191.5 MB FP16). Pruning to $V_S = "24,576"$ pieces covering GB2312/common Unihan characters, high-frequency Chinese compounds, punctuation, and developer ASCII subwords shrinks the 4-bit embedding table by *83.8%* to *12.6 MB*. However, uniformly selecting 6 teacher layers $cal(I) = \{0, 5, 11, 16, 22, 27\}$ against frozen pruned embeddings $bold(E) in RR^(V_S times d)$ rotates pre-head representations $bold(u)_S = "RMSNorm"(bold(h)_S^((6)))$ nearly orthogonal to $bold(u)_T$: $cos(bold(u)_S, bold(u)_T)$ collapses to *0.0221* (@tab-pareto), cutting homophone accuracy to 43.8%.

== Two-Stage Trajectory & Homophone-Subspace Distillation

We realign the 6 transformer blocks $theta_S$ onto frozen $bold(E)$ in two stages:

1. *Stage A: General Trajectory & Dual-Temperature KL (1,600 steps).* Over 9.94M tokens of mixed conversational, encyclopedic, code-switching, and literary text on a single NVIDIA L4 GPU (14 min), we minimize:
$ cal(L)_A (theta_S) &= alpha_"kl" cal(L)_"DKL" + alpha_"cos" (1 - cos(bold(u)_S, bold(u)_T)) \ & quad + frac(alpha_"hid", |cal(M)|) sum_((m, ell) in cal(M)) norm( hat(bold(h))_S^((m)) - hat(bold(h))_T^((ell)) )_2^2 $ <eq-distill>
where $hat(bold(h)) = bold(h) / norm(bold(h))_2$ aligns intermediate trajectories across $cal(M) = \{(1,5), (2,11), (3,16), (4,22), (5,27)\}$, and $cal(L)_"DKL"$ combines $tau_1 = 2.0$ (weight $0.7$) and $tau_2 = 1.0$ (weight $0.3$) forward KL over the 24,576-logit head ($alpha_"kl" = 1.0, alpha_"cos" = 2.5, alpha_"hid" = 1.5$), restoring 4-bit $cos(bold(u)_S, bold(u)_T)$ to $0.8531$.

2. *Stage B: Phonetic Homophone-Subspace KL (300 steps).* Full-vocabulary softmax spreads gradient capacity across 24,576 semantically unrelated tokens, whereas an IME rescorer only compares tokens sharing identical Pinyin readings. For each character position $t$ in 2,600 conversational passages (strictly decontaminated against all 12,455 5-grams in our dev and holdout suites; 0.00% overlap), let $cal(H)(z_t) subset V_S$ ($abs(cal(H)) >= 2$) denote the exact Pinyin homophone cohort of target character $z_t$. Stage B restricts KL divergence directly to the phonetic confusion subspace $sigma_(cal(H)) (bold(z))_v = exp(z_v) / (sum_(u in cal(H)(z_t)) exp(z_u))$:
$ cal(L)_B (theta_S) &= 0.55 D_"KL" (sigma_(cal(H)) (bold(z)_T) || sigma_(cal(H)) (bold(z)_S)) \ & quad + 0.30 cal(L)_"DKL" + 0.15 (1 - cos(bold(u)_S, bold(u)_T)). $ <eq-homo>
Just 300 steps on Apple Silicon MPS reduce within-cohort homophone KL by *5.2×* ($0.1389 arrow.r 0.0265$) and further lift 4-bit $cos(bold(u)_S, bold(u)_T)$ to *0.8631* in *64.17 MB*.

= Rank-Stacked Multi-LoRA Cartridges

== The `.imepack` Cartridge Abstraction

To extend Chime across specialized domains without bloating or retraining the 64.17 MB base model, `.imepack` v2 cartridges bundle: (1) `pack.imedict`, a zero-copy `CHIMED02` slice of domain words, canonical English terms, #box[$n$-grams], and $O(log C)$ `Tab` continuations; (2) `lora.safetensors`, Rank-8 FP16 matrices $(bold(A)_k^((ell, p)) in RR^(r times d_"in"), bold(B)_k^((ell, p)) in RR^(d_"out" times r))$ across all 7 projections $p in \{q, k, v, o, "gate", "up", "down"\}$ of all 6 layers ($"1,032,192"$ params $= 2.07$ MB); and (3) `manifest.json`, declaring routing policy (`classical`, `developer`, `always`), base scale $s_k$, and `private_pack`.

== Rank-Stacked GEMM Fusion & Batched KV-Gather

Evaluating $K$ active adapters sequentially inside every linear layer dispatches $42 times 2K$ tiny `[1, T, 1024] @ [1024, 8]` GEMMs per pass:
$ bold(y) = "QuantMatmul"_"4bit" (bold(x), hat(bold(W))) + sum_(k=1)^K gamma_k (bold(x) bold(A)_k^top) bold(B)_k^top $ <eq-unfused>
On Metal GPUs, kernel launch overhead dominates at $r=8$, inflating 6-candidate latency linearly by $+1.57$ ms/adapter to 10.74 ms at $K=6$ (@fig-scaling (a)).

At mount time, _Rank-Stacked GEMM Fusion_ pre-transposes and concatenates all $K$ adapters along rank $R = 8K$ once in Unified Memory ($bold(A)_"cat"^top in RR^(d_"in" times R), bold(B)_"cat"^top in RR^(R times d_"out")$). Replicating routing weights $bold(gamma) in RR^K$ into $bold(g) = "repeat"(bold(gamma), r) in RR^(1 times 1 times R)$ collapses @eq-unfused into _two_ GEMMs regardless of $K$:
$ Delta bold(y) = ((bold(x) bold(A)_"cat"^top) circle.tiny bold(g)) bold(B)_"cat"^top $ <eq-fused>
Simultaneously, _Batched Broadcast KV-Gather_ packs all multi-token continuations into a single right-padded `[B, L_max]` tensor, zero-copy broadcasts `MlxPrefixCache` along axis 0, applies lower-right causal masking, and extracts target log-probabilities via on-GPU `take_along_axis` ($<= 128$ bytes GPU$arrow.r$CPU readout, 4,600× bandwidth reduction).

#figure(
  fig-scaling-and-transfer(),
  kind: image,
  scope: "parent",
  placement: top,
  caption: [
    *(a)* Measured 6-candidate Metal reranking latency (p50, ms) across $K = 0, dots, 6$ Rank-8 cartridges: Rank-Stacked GEMMs flatten the $O(K)$ kernel ramp ($10.74 arrow.r 3.11$ ms), and Batched Broadcast KV-Gather cuts it to *0.85 ms* ($K=0$) and *1.41 ms* ($K=6$, *7.60× speedup*). *(b)* Classical validation NLL vs. adapter scale $gamma$: zero-shot transfer of an FP16-trained LoRA overshoots base at $gamma = 1.0$ (6.117), whereas *Q4-Manifold Aligned LoRA* training directly on dequantized 4-bit weights eliminates overshoot and deepens NLL reduction by 75% (6.041 $arrow.r$ *5.740*, $-0.301$ nats).
  ],
) <fig-scaling>

== Register Superposition & Q4-Manifold Alignment

Concurrent multi-adapter composition exhibits two statistical failure modes:

1. *Register Superposition Drift*: Domain packs trained on formal written corpora over-represent literary/technical compounds (多达) over spoken interrogatives (多大). Uncalibrated summation ($sum_k gamma_k = 2.85$) constructively amplifies their shared written-register direction into SwiGLU saturation, dropping classical accuracy to 6/8 and spiking NLL from 6.041 to 7.146 (@tab-routing).
2. *FP16$arrow.r$INT4 Manifold Shift & Q4-Aligned Training*: Zero-shot mounting of an FP16-trained `LoRA` onto quantized weights $hat(bold(W)) = bold(W) + bold(E)_Q$ incurs cross-term $(1/2) norm(bold(E)_Q + gamma Delta bold(W))_(bold(H))^2$, overshooting base NLL at $gamma = 1.0$ (6.117 vs. 6.041; @fig-scaling (b)). Training `LoRA` directly on dequantized 4-bit runtime weights $hat(bold(W))$ lets $Delta bold(W) = bold(B) bold(A)$ jointly adapt domain priors and compensate for quantization residuals $bold(E)_Q$, achieving *−0.301 nats* (5.740) without overshoot.

Chime pairs Q4-Manifold training with a *Couplet-Aware Simplex Router*: `Classical` activates on archaic particles (之乎者也矣焉哉兮) or symmetric $5+5$/$7+7$ couplet cadence while staying $0$ on colloquial chat; `Developer` activates on ASCII/code cues ($1.0 s_k$) or IDE/terminal bundle IDs ($0.25 s_k$); `Always` attenuates to $0.45 s_k$ on short $<= 2$-char colloquial inputs; and scales project onto $gamma_k <= 0.80, sum_k gamma_k <= 0.90$.

= Experiments & Analysis

== Experimental Setup & Decontaminated Suites

All runtime experiments run natively on an Apple Silicon Mac (`mlx-rs 0.34`, Metal GPU, lock-free single-threaded actor). Stage-A distillation ran on an NVIDIA L4 GPU; Stage-B homophone distillation and all Q4-aligned Rank-8 `LoRA`s (240–320 steps, 2.07 MB FP16, 5-gram decontaminated) ran on Apple Silicon MPS:
- *`classics-poetry.imepack`*: 37 canonical Tang/Song/Guwen masterpieces, 21 `chinese-poetry` collections, and Xinhua idioms/xiehouyu (*266,816* lexical entries, *231,593* `Tab` chains, 18,482 training paragraphs).
- *`tech-developer.imepack`*: Bilingual systems/AI/cloud lexicon + 1.20M characters (2,510 passages from `FineWeb-Edu-Chinese-V2.1` & `BelleGroup`, 0% 5-gram overlap with benchmarks).
- *Air-Gapped Personal Pack*: *222 human-authored essays* (2013–2020; 814,988 chars pre-dating generative AI), PII-scrubbed into 8,000 personal words + 4,727 bigrams.
- *Decontaminated Benchmark Hierarchy*: (i) *56-case Golden Regression* (`eval`), (ii) *Dev-1258* across 17 linguistic categories (spoken chat, copula+classifiers, 的/地/得, verbal complements, homophones, polyphones, hiatus, code-switching, Shuangpin, fuzzy rhymes, QWERTY typos), and (iii) *Blind Holdout-1010*, an unseen 1,010-case whole-sentence suite across 16 domains (0% sentence overlap with Dev-1258; all 12,455 benchmark 5-grams excluded from neural training).

#figure(
  table(
    columns: (2.32fr, 0.82fr, 0.64fr, 0.90fr, 1.00fr, 0.86fr, 0.86fr),
    align: (left, right, right, right, right, right, right),
    stroke: none,
    inset: (x: 4.2pt, y: 3.8pt),
    toprule,
    [*Model Checkpoint*], [*Size*], [*Compr.*], [*$cos(bold(u)_S, bold(u)_T)$*], [*Stress Acc.*], [*Margin $Delta ell$*], [*Rerank-6 p50*],
    midrule,
    [`Qwen3-0.6B` (28L, 152K, FP16)], [1,191.5 MB], [1.0×], [1.0000], [93.8% (15/16)], [+5.27 nats], [$approx 14.2$ ms],
    [`Qwen3-28L` (24K vocab, 4-bit)], [249.95 MB], [4.8×], [1.0000], [93.8% (15/16)], [+5.22 nats], [5.32 ms],
    [`Qwen3-6L-NaiveSkip` (4-bit)], [64.15 MB], [18.6×], [#text(fill: rgb("#be123c"))[0.0221]], [#text(fill: rgb("#be123c"))[43.8% (7/16)]], [+0.14 nats], [1.43 ms],
    [`Qwen3-6L-StageA` (FP16)], [228.03 MB], [5.2×], [0.8125], [93.8% (15/16)], [+5.91 nats], [2.64 ms],
    [`Qwen3-6L-StageA` (4-bit, Seq.)], [64.15 MB], [18.6×], [0.8531], [93.8% (15/16)], [+6.02 nats], [1.41 ms],
    [`Qwen3-6L-Chime` (4-bit, Seq.)], [64.17 MB], [18.6×], [0.8631], [93.8% (15/16)], [+6.04 nats], [1.43 ms],
    [*`Qwen3-6L-Chime` (4-bit, Batched)*], [*64.17 MB*], [*18.6×*], [*0.8631*], [*93.8% (15/16)*], [*+6.04 nats*], [*0.86 ms*],
    bottomrule,
  ),
  scope: "parent",
  placement: top,
  caption: [
    *Teacher-to-Student Compression & Batched Metal Reranking Pareto Frontier* ($T = 16$ prefix tokens + 6 homophone candidates). Two-stage trajectory + homophone-subspace distillation lifts 4-bit teacher cosine alignment to *0.8631* and matches 28-layer accuracy (15/16) in 64.17 MB at *0.86 ms* (6.20× faster than 28L 4-bit).
  ],
) <tab-pareto>

== Main Results & Ablation Studies

*1. Distillation & Batched Reranking Pareto Frontier (@tab-pareto).* Naive 6-layer skipping collapses $cos(bold(u)_S, bold(u)_T)$ to $0.0221$ and halves stress accuracy (7/16). Stage-A trajectory distillation restores $cos = 0.8531$ (15/16), and Stage-B homophone-subspace distillation further aligns the 4-bit representation ($cos = bold("0.8631")$, cohort KL $0.1389 arrow.r 0.0265$) while Batched Broadcast KV-Gather cuts 6-candidate latency from 1.43 ms to *0.86 ms* (6.20× faster than 28L 4-bit).

#figure(
  table(
    columns: (2.05fr, 0.78fr, 0.78fr, 0.72fr, 0.78fr, 0.78fr, 0.72fr),
    align: (left, right, right, right, right, right, right),
    stroke: none,
    inset: (x: 4.2pt, y: 3.8pt),
    toprule,
    [], table.cell(colspan: 3, align: center)[*Dev-1258 (17 Categories)*], table.cell(colspan: 3, align: center)[*Blind Holdout-1010 (16 Unseen Domains)*],
    [*System Configuration*], [*Acc\@1*], [*Hit\@5*], [*MRR*], [*Acc\@1*], [*Hit\@5*], [*MRR*],
    midrule,
    [S1: Stage-1 Lattice Only (`CHIMED02`, 0 MB SLM)], [67.41%], [86.01%], [0.7579], [42.87%], [64.46%], [0.5211],
    [S2: Stage-1 + `Chime-6L` (64.17 MB 4-bit MLX)], [85.45%], [92.45%], [0.8868], [70.59%], [76.83%], [0.7335],
    [*S3: Stage-1 + `Chime-6L` + `.imepack` ($K=2$)*], [*85.85%*], [*93.00%*], [*0.8913*], [*71.39%*], [*77.13%*], [*0.7392*],
    bottomrule,
  ),
  scope: "parent",
  placement: top,
  caption: [
    *Large-Scale End-to-End Evaluation across 2,268 Decontaminated Queries* (zero open-domain phrase hardcoding; all 12,455 benchmark 5-grams excluded from distillation and `LoRA` corpora). Stage-2 `Chime-6L` + `.imepack` cartridges lift Top-1 accuracy by *+18.44* points on Dev-1258 and by *+28.52* points (*49.9% relative error reduction*) on unseen whole-sentence Blind Holdout-1010.
  ],
) <tab-bench>

*2. Thousand-Sentence End-to-End Gain (@tab-bench).* Across 2,268 queries without hardcoded open-domain phrases, pure statistical lattice decoding (S1) achieves 67.41% Acc\@1 on Dev-1258 (strong on idioms 100%, motor typos 97.3%, fuzzy Pinyin 96.0%, and Shuangpin 95.8%) but drops to *42.87%* on unseen 8–14 character whole sentences in Blind Holdout-1010 due to combinatorial homophone ambiguity. Enabling Stage-2 `Chime-6L` rescoring with pruned-BPE partition calibration (S2) surges Acc\@1 by *+18.04* points on Dev-1258 (*85.45%*) and by *+27.72* points on Blind Holdout-1010 (*70.59%*, MRR $0.5211 arrow.r 0.7335$). Co-mounting `classics-poetry` and `tech-developer` cartridges (S3) further lifts Acc\@1 to *85.85%* (Hit\@5 *93.00%*) on Dev-1258 and *71.39%* (Hit\@5 *77.13%*, *+28.52* points total) on Blind Holdout-1010—reaching 100% on classical idioms, 92.5% on dialect mergers, 92.3% on minimal-pair homophones, 86.7% on Shuangpin sentences, 86.0% on polyphones, and *56/56 (100%)* on golden regression (`p50 = 8.76` ms).

#figure(
  table(
    columns: (1.55fr, 1.12fr, 0.66fr, 0.67fr),
    align: (left, right, right, right),
    stroke: none,
    inset: (x: 3.4pt, y: 3.6pt),
    toprule,
    [*Configuration*], [*Verse Top-1*], [*Cls. NLL*], [*Pers. NLL*],
    midrule,
    [C0: Modern Base Only], [10.0% (3/30)], [6.041], [5.628],
    [C1: `LoRA` Sidecar Only], [10.0% (3/30)], [5.740], [5.227],
    [C2: `pack.imedict` Only], [90.0% (27/30)], [6.041], [5.628],
    [*C3: Full `.imepack`*], [*90.0% (27/30)*], [*5.740*], [*5.227*],
    bottomrule,
  ),
  placement: top,
  caption: [
    *Verification of Recall–Rerank Decoupling* on 30 held-out OOV classical verses and held-out Q4 prose blocks. Q4-aligned `LoRA` cuts domain NLL ($-0.301$ classical, $-0.401$ personal) yet yields *0%* OOV verse gain without `pack.imedict`.
  ],
) <tab-decouple>

*3. Lexical-Neural Decoupling & Multi-LoRA Routing (@tab-decouple & @tab-routing).* Mounting `classics_lora` alone (C1) reduces classical NLL by *−0.301 nats* ($6.041 arrow.r 5.740$) and personal `LoRA` cuts held-out personal essay NLL by *−0.401 nats* ($5.628 arrow.r 5.227$), yet OOV verse conversion stays at 10.0% until `pack.imedict` is co-mounted (C3: *90.0%*, and *20/20* on canonical masterpieces). Under $K=3$ concurrent packs (@tab-routing), uncalibrated summation (R1) spikes NLL to 7.146 and drops classical accuracy to 6/8, whereas our Couplet-Aware Simplex Router (R3) preserves *8/8* across all regimes with top colloquial ($+9.00$) and developer ($+9.91$) margins.

#figure(
  table(
    columns: (1.44fr, 0.95fr, 0.95fr, 0.95fr, 0.60fr),
    align: (left, right, right, right, right),
    stroke: none,
    inset: (x: 2.2pt, y: 3.6pt),
    toprule,
    [*Routing ($K$=3)*], [*Colloq.*], [*Class.*], [*Dev.*], [*NLL*],
    midrule,
    [R0: Base ($bold(gamma) = bold(0)$)], [8/8 (+8.4)], [8/8 (+4.1)], [8/8 (+9.4)], [6.041],
    [R1: Uncalib. Sum], [8/8 (+3.8)], [#text(fill: rgb("#be123c"))[6/8 (+2.0)]], [8/8 (+6.9)], [#text(fill: rgb("#be123c"))[7.146]],
    [R2: Uniform $1/K$], [8/8 (+7.8)], [8/8 (+4.3)], [8/8 (+9.5)], [5.880],
    [*R3: Simplex-Gated*], [*8/8 (+9.0)*], [*8/8 (+4.3)*], [*8/8 (+9.9)*], [*5.812*],
    bottomrule,
  ),
  placement: top,
  caption: [
    *Multi-LoRA Interference vs. Simplex-Gated Routing* with 3 concurrent Q4-aligned packs (`classics` + `tech` + `personal`).
  ],
) <tab-routing>

#figure(
  table(
    columns: (0.42fr, 0.95fr, 0.95fr, 1.15fr, 0.85fr),
    align: (center, right, right, right, right),
    stroke: none,
    inset: (x: 4pt, y: 3.5pt),
    toprule,
    [*$K$*], [*Unfused*], [*Stacked*], [*Batched+Stk*], [*Speedup*],
    midrule,
    [0], [1.41 ms], [1.41 ms], [*0.85 ms*], [*1.66×*],
    [1], [2.91 ms], [2.95 ms], [*1.22 ms*], [*2.39×*],
    [2], [4.49 ms], [2.98 ms], [*1.34 ms*], [*3.35×*],
    [3], [5.97 ms], [3.05 ms], [*1.36 ms*], [*4.40×*],
    [4], [7.53 ms], [3.03 ms], [*1.38 ms*], [*5.47×*],
    [5], [9.14 ms], [3.11 ms], [*1.40 ms*], [*6.52×*],
    [6], [10.74 ms], [3.11 ms], [*1.41 ms*], [*7.60×*],
    bottomrule,
  ),
  placement: top,
  caption: [
    *Multi-LoRA Scaling on Metal* ($K = 0, dots, 6$ Rank-8 adapters, 6-candidate p50). Rank-Stacked GEMMs + Batched KV-Gather cut $K=6$ latency from 10.74 ms to *1.41 ms* ($+39.4 thin mu"s"$/adapter from $K = 1 arrow.r 6$, *7.60× speedup*).
  ],
) <tab-scaling>

*4. Kernel Fusion & Scaling (@tab-scaling).* From $K=1$ to $6$, Rank-Stacked GEMMs + Batched KV-Gather hold 6-candidate latency at *1.22 ms $arrow.r$ 1.41 ms* ($+39.4 thin mu"s"$/adapter vs. $+"1,567" thin mu"s"$/adapter unfused—a *39.8× reduction* in marginal adapter cost and *7.60× speedup*).

== Qualitative Case Studies

- *Syntactic Diversity & Classifier Selection*: On `woshiyigebing`, standard beam search fills top slots with ungrammatical open-class collisions (我试一个并, 卧室一个并); Chime's prefix-diverse beam retains the copula-classifier skeleton 我是一个 + `bing`, and `Chime-6L` selects human predicate *我是一个兵* (while selecting *吃了两个饼* on `chilelianggebing`).
- *Pruned-BPE Partition Calibration*: On `shiyongdeguochengzhong` after 我在, penalizing extra BPE splits ($Delta T = 1$ on OOV 试+用) prevents partition-function leakage from flipping *使用的过程中* (`-5.53`) to 试用的过程中 (`-6.36`) even under active developer `LoRA`.
- *Poetry Chaining & Motor-Slip Recovery*: Typing `junbujianhuanghe` + `Tab` chains all 24 clauses of _Qiang Jin Jiu_ while anti-repeat promotes 君不见高堂明镜悲白发 on manual `junbujian`; `nianquuqqianliyanbo` recovers transposed `uq` $arrow.r$ `qu` into *念去去，千里烟波*.

= Limitations & Ethical Considerations

*Limitations.* First, novel neologisms absent from all mounted `.imepack`s still require first-time prefix composition (今天 + `wo'men`) before online SQLite bigram learning caches them; while `Chime-6L` lifts blind whole-sentence accuracy by +28.52 points to 71.39%, closing the remaining gap on 12+ syllable technical sentences motivates compact phonetic-lattice reranking over wider $M > 20$ candidate graphs. Second, learning a $< 10$ KB linear gate over pooled prefix embeddings could further refine soft adapter blending in mixed classical-modern prose.

*Privacy & Broader Impact.* Chime enforces zero network entitlement, bypasses `IsSecureEventInputEnabled()` password fields in hardware, and enables air-gapped personal `LoRA` training directly on local Apple Silicon GPUs after deterministic PII/secret redaction—proving that LLM personalization need not require keystroke surveillance.

= Conclusion

Chime demonstrates that client-side IMEs benefit most from LLMs not as latency-heavy sequence generators, but as _trajectory- and homophone-distilled, partition-calibrated parallel rescorers_ co-designed with prefix-diverse zero-copy lattices and _Q4-Aligned Rank-Stacked Multi-LoRA cartridges_. Compressing `Qwen3-0.6B` by 18.6× into 64.17 MB delivers 0.86 ms base reranking, 1.41 ms 6-adapter inference, +28.52-point blind whole-sentence gains, and strictly offline personal adaptation on everyday laptops.

#bibliography("refs.bib", title: "References", style: "association-for-computing-machinery")
