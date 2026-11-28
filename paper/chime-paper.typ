#import "figures.typ": fig-architecture, fig-evolution-and-twotier, fig-scaling-and-transfer

#let h10k-t2-acc = "84.58"
#let h10k-t2-acc-num = 84.58
#let h10k-t2-hit3 = "90.32"
#let h10k-t2-hit5 = "91.05"
#let h10k-t2-mrr = "0.8762"
#let h10k-t2-p50 = "260.7"
#let h10k-t2-delta = "+3.03"
#let h10k-t2-esc = "33.5"

#set document(
  title: "Chime: High-Accuracy On-Device Neural Pinyin Input via Bayesian Pinyin-Constrained Beam Search, Block-Influence Compression, and Confidence-Gated Two-Tier Reranking",
  author: "Tian Chen",
)

#set page(
  paper: "us-letter",
  margin: (top: 0.78in, bottom: 0.80in, left: 0.68in, right: 0.68in),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 7.8pt, fill: rgb("#71717a"))
      Chime: On-Device Neural Pinyin Input via Bayesian Beam Search & Confidence-Gated Two-Tier Reranking
      #h(1fr)
      Preprint (v2) · 2026
    ]
  },
  numbering: "1",
)

#set text(
  font: ("New Computer Modern", "Noto Serif CJK SC", "Songti SC"),
  size: 9.5pt,
  lang: "en",
)

#show regex("[\p{scx=Han}，。！？；：、《》（）【】]+"): set text(
  font: ("Noto Serif CJK SC", "Songti SC"),
  size: 0.90em,
)
#show raw: set text(
  font: ("PT Mono", "JetBrains Mono NL", "DejaVu Sans Mono"),
  size: 0.91em,
)
#show "—": it => sym.wj + it

#set par(
  justify: true,
  leading: 0.54em,
  spacing: 0.70em,
)

#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#show heading.where(level: 1): it => block(above: 1.38em, below: 0.56em, sticky: true)[
  #set par(justify: false, leading: 0.42em)
  #set text(size: 12.0pt, weight: "bold", hyphenate: false)
  #it
]
#show heading.where(level: 2): it => block(above: 1.12em, below: 0.44em, sticky: true)[
  #set par(justify: false, leading: 0.42em)
  #set text(size: 10.5pt, weight: "bold", hyphenate: false)
  #it
]
#show heading.where(level: 3): it => block(above: 0.88em, below: 0.36em, sticky: true)[
  #set par(justify: false, leading: 0.42em)
  #set text(size: 9.6pt, weight: "bold", style: "italic", hyphenate: false)
  #it
]

#show table: set text(size: 8.0pt)
#show table: set par(justify: false, leading: 0.44em)

#show figure.caption: it => context [
  #set text(size: 8.3pt)
  #set par(justify: true, leading: 0.46em)
  *#it.supplement #it.counter.display():* #it.body
]

#let toprule = table.hline(stroke: 0.85pt + black)
#let midrule = table.hline(stroke: 0.45pt + rgb("#3f3f46"))
#let bottomrule = table.hline(stroke: 0.85pt + black)

#let prop-box(label, body) = block(
  width: 100%,
  fill: rgb("#f8fafc"),
  stroke: (left: 2.2pt + rgb("#0f4c5c"), rest: 0.4pt + rgb("#cbd5e1")),
  inset: (x: 8pt, y: 5.5pt),
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
    #block(width: 97%)[
      #set par(justify: false, leading: 0.42em)
      #set text(hyphenate: false)
      #text(size: 13.6pt, weight: "bold")[
        Chime: High-Accuracy On-Device Neural Pinyin Input via\
        Bayesian Pinyin-Constrained Beam Search, Block-Influence\
        Compression, and Confidence-Gated Two-Tier Reranking
      ]
      #v(6.5pt)
      #text(size: 10.2pt, weight: "semibold")[Tian Chen]
      #v(2pt)
      #text(size: 8.7pt, fill: rgb("#52525b"))[
        Independent Systems & Machine Learning Research · `tyr.chen@gmail.com` · #link("https://github.com/tyrchen/chime")[`github.com/tyrchen/chime`]
      ]
    ]
  ]
  #v(8pt)
  #align(center)[
    #block(
      width: 94%,
      fill: rgb("#fafafa"),
      stroke: 0.5pt + rgb("#e4e4e7"),
      radius: 4pt,
      inset: (x: 13pt, y: 9pt),
    )[
      #align(center)[#text(size: 9.4pt, weight: "bold")[Abstract]]
      #v(2.5pt)
      #set par(justify: true, leading: 0.49em)
      #set text(size: 8.7pt)
      Deploying Large Language Models (LLMs) inside client-side Chinese Pinyin Input Method Editors (IMEs) exposes a sharp systems trilemma across keystroke latency, resident memory ($<= 200$ MB default budget, 100% offline), and open-domain whole-sentence accuracy. Prior hybrid systems that limit neural models to reranking a statistical Stage-1 lattice hit a hard recall ceiling (71.39% blind holdout in our earlier 6-layer Chime v0.8), whereas unconstrained end-to-end Pinyin-to-Character (P2C) sequence generation underperforms at compact 0.3B parameter scales on typos, Shuangpin, and bilingual code-switching. We present *Chime* (`v0.16`), an open-source Rust and Swift 6 macOS IME running natively on Apple Silicon Metal (`mlx-rs`) that resolves this trilemma through four co-designed innovations. First, *Route A Bayesian Pinyin-Constrained Beam Search* directly maximizes $log P_"LM" (Y | C) + log P_"channel" (X | Y)$ over a deterministic `TokenPinyinIndex` trie with exact BPE marginalization (`logaddexp`), Dual-Proposal fusion with a $< 0.8$ ms `mmap` lattice, localized surprisal-spike typo rescue, and a *lossless transition-indexed incremental beam cache* (`RouteABeamCache`). Second, *Block-Influence 18-Layer Distillation (`BI-18L`)* compresses `Qwen3-0.6B` into a *190.3 MB* mixed-precision `e8_l4` student (8-bit 36,864-token tied embeddings, 4-bit transformer blocks) distilled from `Qwen3-14B-Base`, cutting model footprint by 26% and 10K-sentence `p50` latency by 3.01× relative to 28-layer 4-bit while lifting blind holdout accuracy to *92.28%* (*+20.89* points over Stage-2 reranking) and *81.55%* on a strictly blind 10,000-sentence benchmark (`Hit@5 91.05%`). Through controlled negative experiments—end-to-end P2C conditioning, 40M/65M multi-source distillation + Model Soup, and 4-bit/6-bit/8-bit quantization sweeps—we prove that 0.32B edge models hit a *parameter-capacity ceiling* rather than a quantization or teacher bottleneck. Third, pluggable *`.imepack` Multi-LoRA cartridges* combine Rank-Stacked GEMM Fusion (*7.60×* speedup at $K=6$) with Q4-Manifold alignment ($-0.301$ nats classical, $-0.401$ nats personal NLL). Fourth, *Optional Confidence-Gated Two-Tier Local Reranking* escalates only uncertain Tier-1 hypotheses ($s_1 - s_2 < 3.0$, *18.7%* of blind queries) to a single batched `[5, L]` Metal pass of a local 4-bit `Qwen3-4B-Base` model with linear fusion and a `{的, 地, 得}` grammatical guard—outperforming 100% standalone 4B replacement by *+2.20* points while lifting blind Holdout-1010 to *94.36%* (*+11.67* points on technical engineering, only *+8.3 ms* p50), Dev-1258 to *95.63%*, and blind Holdout-10K to *#h10k-t2-acc%* (`Hit@5` *#h10k-t2-hit5%*).
    ]
  ]
  #v(4pt)
]

#show: rest => columns(2, gutter: 0.24in, rest)

= Introduction

Every sentence composed on a desktop or mobile device passes through an Input Method Editor (IME). For Chinese Pinyin-to-Character (P2C) conversion, an IME must solve a real-time noisy-channel inference problem @chen2000new @jia2014joint under extreme domain diversity: full Pinyin, initial-letter abbreviations (`wsm` $arrow.r$ 为什么), three Shuangpin layouts (including omitted trailing/medial finals), regional phonetic mergers (`z`/`zh`, `in`/`ing`), QWERTY motor slips (`shaung` $arrow.r$ `shuang`), unswitched English code-switching (`macOS`, `Cloudflare`), and classical literature.

Bridging modern Large Language Models (LLMs) with OS-resident input methods exposes a fundamental *Latency–Memory–Accuracy Trilemma*:

1. *Stage-1 Recall Ceiling in Rerank-Only Designs*: Pairing a sub-millisecond `mmap` word lattice with a compact neural rescorer (our earlier Chime `v0.8` 6-layer 64 MB design) achieves sub-3 ms GPU reranking, yet saturates at *71.39%* on unseen whole sentences because Stage-1 #box[$n$-gram] beam search prunes long-range contextually correct character paths before the neural model ever sees them.
2. *Weak Sample Efficiency of End-to-End P2C Generation*: Recent generative IMEs @tan2022exploring @ding2023geneinput @gui2025attninput @shan2026huoziime feed raw Pinyin prefixes directly into an LLM to generate Chinese text. At client-side parameter scales ($approx 0.3$B), forcing a single model to jointly memorize phonetic mappings, Shuangpin layouts, QWERTY typo recovery, and open-domain semantics yields poor convergence and breaks zero-shot generalization whenever a user switches keyboard schemes or types novel bilingual terms.
3. *The 0.3B Parameter Capacity Ceiling*: Even with exact Bayesian Pinyin-constrained beam search and 14B-teacher Soft-KL distillation, a $<= 200$ MB on-device model (`BI-18L`, 0.32B parameters) inevitably plateaus on encyclopedic world knowledge, niche technical terms, and cross-clause coreference—yet running a 4B LLM autoregressively on every keystroke violates interactive latency and memory budgets.

In this paper, we present *Chime* (`v0.16`), an open-source, 100% offline macOS IME written in Rust and Swift 6 on Apple Silicon Unified Memory @hannun2023mlx (@fig-arch). Chime resolves all three tensions by combining *Bayesian Pinyin-Constrained Beam Search (Route A)*, *Block-Influence 18-Layer `e8_l4` Compression*, *Rank-Stacked Multi-LoRA Cartridges*, and *Optional Confidence-Gated Two-Tier Local Reranking*.

#figure(
  fig-architecture(),
  kind: image,
  scope: "parent",
  placement: top,
  caption: [
    End-to-end architecture of *Chime* (`v0.16`). A $< 0.8$ ms zero-copy `CHIMED02` Fast Path emits instant Frame-0 candidates and Stage-1 proposals $cal(P)_1$. In the background on Apple Silicon Metal, Tier-1 runs exact Bayesian Route A beam search on a *190.3 MB* `BI-18L e8_l4` student with a lossless transition-indexed `RouteABeamCache` and Rank-Stacked `.imepack` Multi-LoRA sidecars. When Tier-1's top-2 margin falls below $tau = 3.0$ (18.7% of blind queries), an optional local 4-bit `Qwen3-1.7B/4B-Base` Tier-2 reranker scores the top-5 aligned hypotheses in a single batched `[5, L]` Metal forward pass with a `{的, 地, 得}` grammatical guard.
  ],
) <fig-arch>

Our main contributions are fivefold:
- *First-Principles Bayesian Route A Beam Search & Lossless Incremental Caching (§3)*: We factorize P2C cleanly into $log P_"LM" (Y | C) + log P_"channel" (X | Y)$ over a deterministic `TokenPinyinIndex` trie with exact BPE marginalization (`logaddexp`), Dual-Proposal fusion, localized surprisal-spike typo rescue, and a transition-indexed `RouteABeamCache` that eliminates cross-boundary word loss discovered in naive bucket caching.
- *Block-Influence `BI-18L e8_l4` Compression & Capacity-Ceiling Proof (§4)*: Pruning 28-layer `Qwen3-0.6B` to 18 layers via Block Influence @men2024shortgpt with `Qwen3-14B-Base` Soft-KL distillation and `e8_l4` mixed precision (*190.3 MB*, $-26%$ vs. 28L) yields *3.01×* faster 10K `p50` latency while surpassing 28L accuracy. Controlled negative experiments (end-to-end P2C training, 40M/65M distillation + Model Soup, and 4-bit/6-bit/8-bit sweeps) establish that 0.32B students are *capacity-bound*, not quantization- or teacher-bound.
- *Rank-Stacked `.imepack` Multi-LoRA Cartridges (§5)*: Co-mounting zero-copy lexical slices (`pack.imedict`) with Q4-Manifold aligned Rank-8 `LoRA`s achieves *7.60×* speedup at $K=6$ ($+39.4 thin mu"s"$/adapter) and eliminates FP16$arrow.r$INT4 overshoot ($-0.301$ nats classical, $-0.401$ nats personal NLL).
- *Confidence-Gated Two-Tier Local Reranking with Grammatical Guard (§6)*: Escalating only uncertain Tier-1 beams ($s_1 - s_2 < 3.0$) to a single batched Metal pass of a local 4-bit `Qwen3-4B-Base` model—with pure-Rust BPE tokenization, shared `MlxPrefixCache`, linear interpolation, and `{的, 地, 得}` protection—beats 100% standalone 4B replacement by *+2.20* points while adding only *+8.3 ms* p50 latency on blind queries.
- *12,268-Query Strictly Decontaminated Evaluation (§7)*: Across Dev-1258, Blind Holdout-1010, and a strictly blind 10,000-sentence benchmark (Blind Holdout-10K), Chime improves Blind Holdout-1010 from `71.39%` (`v0.8`) $arrow.r$ `92.28%` (Tier-1 `v0.15.5`) $arrow.r$ *94.36%* (Two-Tier 4B), Dev-1258 to *95.63%*, and Blind Holdout-10K from `66.45%` (`v0.12` rerank) $arrow.r$ `81.55%` (Tier-1) $arrow.r$ *#h10k-t2-acc%* (Two-Tier 4B).

= Related Work

*Statistical and Graph-Based P2C Decoding.* Classical Pinyin engines model conversion via Viterbi search over word lattices backed by #box[$n$-grams] @chen2000new and double-array tries @yoshinaga1992double. @jia2014joint unified QWERTY typo recovery with lattice P2C. While sub-millisecond, finite-order #box[$n$-grams] fail on long-distance homophones and spoken-register syntax.

*Neural and Generative LLM Input Methods.* @zhang2019open introduced open-vocabulary neural P2C and @xu2022pert pre-trained `PERT` directly on Pinyin sequences. Generative LLM prototypes—including adapted Chinese GPT @tan2022exploring, `GeneInput` @ding2023geneinput, RWKV-based `AttnInput` @gui2025attninput, and personalized `HuoziIME` @shan2026huoziime—map Pinyin to characters via end-to-end conditional generation. Chime instead keeps phonetic/typo/Shuangpin structure inside a deterministic Bayesian channel and constrains a standard causal LLM (`Qwen3`) directly via syllable-synchronous beam search, preserving full zero-shot transfer across input schemes without retraining.

*Compression, Multi-LoRA Serving, and Model Cascades.* Layer pruning via Block Influence (`ShortGPT` @men2024shortgpt), knowledge distillation @hinton2015distilling @jiao2020tinybert, weight quantization @frantar2023gptq @lin2024awq @dettmers2023qlora, and weight averaging (`Model Soups` @wortsman2022modelsoups) reduce edge cost. Cloud multi-adapter systems (`S-LoRA` @sheng2024slora, `PLoRA` @zhang2024plora) serve per-tenant adapters across batches, whereas Chime fuses concurrent adapters for a single user stream on unified memory. Finally, LLM cascading (`FrugalGPT` @chen2023frugalgpt) routes uncertain queries to stronger models; Chime adapts confidence-gated cascading to client-side IMEs via *single-pass batched candidate scoring* rather than autoregressive generation, showing that gated fusion with a grammatical guard strictly dominates standalone 4B scoring.

= Bayesian Pinyin-Constrained Beam Search (Route A)

== Exact Noisy-Channel Factorization

Given an ASCII keystroke stream $X = k_(1:n)$, sliding left document context $C$ (up to 96 characters), and active host application $a$, Chime solves the exact noisy-channel MAP objective:
$ Y^* = arg max_(Y) [& log P_"LM" (Y | C, a) + log P_"channel" (X | Y) \ & + lambda_"cls" log P_"eos" (Y | C) ] $ <eq-bayes>
where $log P_"LM" (Y | C, a)$ is evaluated autoregressively by the Tier-1 causal Transformer, $log P_"channel" (X | Y) <= 0$ is a deterministic phonetic-keystroke channel cost (`0.0` for exact canonical full/Shuangpin matches; calibrated negative log-penalties for fuzzy mergers, omitted Shuangpin finals, initial-letter abbreviations, and 4 QWERTY motor-typo operators), and $lambda_"cls" = 0.25$ adds optional right-boundary utterance closure in `ClosedSentence` mode.

Crucially, this factorization decouples *language modeling* from *input mechanics*: supporting omitted-final Shuangpin across Xiaohe, Ziranma, and Microsoft layouts, adding bilingual developer terms, or refining QWERTY transposition costs requires zero model retraining—every change takes effect immediately through the deterministic `TokenPinyinIndex`.

== Syllable-Synchronous Trie, Exact BPE Marginalization & Dual-Proposal Fusion

At startup, Chime compiles the 36,864-token vocabulary into `TokenPinyinIndex`, mapping every Chinese character, multi-character BPE token, and curated English word to its valid Pinyin/Shuangpin byte-span transitions $e = (v, Delta b, c_"chan")$. Route A organizes beam hypotheses into byte-position buckets `buckets[0..=n]`:

1. *Lazy Per-Beam KV Advancement & Heterogeneous Step Batching*: Each hypothesis stores `(parent_kv, pending_token)` and advances its KV-cache lazily only if it survives bucket pruning ($B = 8$ offline, $B = 6$ live). Surviving beams at byte offset $p$ are batched into a single Metal forward pass via compact single-barrier sub-KV gathering, eliminating redundant GPU launches.
2. *Exact BPE Marginalization (`logaddexp`)*: Because a multi-character surface such as `为什么` can be tokenized either as one BPE piece or as `为 + 什么`, keeping only the Viterbi tokenization systematically penalizes multi-piece paths. Whenever two hypotheses at byte position $p$ emit the identical UTF-8 surface $Y$, Route A merges them via exact probability-space marginalization:
$ log P_"LM" (Y | C) arrow.l "logaddexp"( & log P_"LM"^((1)) (Y | C), \ & log P_"LM"^((2)) (Y | C) ) $ <eq-marginal>
while retaining the highest-scoring canonical tokenization's KV state for downstream continuation.
3. *Early-Sentence Funnel & Surprisal-Spike Rescue*: Ambiguity peaks in the first 6–8 keystrokes before syntactic constraints crystallize. Route A widens the beam by $3 times$ (capped at $18$) during early positions $p <= "clamp"(n/4, 6, 8)$. Moreover, each hypothesis tracks its worst single-step local surprisal spike $s_"spike" = max_(t > 0) (-log P_"LM" (y_t | C, y_(<t)))$. If the winning path exhibits $s_"spike" > 9.8$ nats on a canonical Chinese input (indicating a fluent sentence containing a hidden legal-syllable typo such as `hucihou` $arrow.r$ `huichou`), Chime triggers localized typo rescue and adaptive beam re-expansion around the spike span.
4. *Dual-Proposal Exact Scoring & Cache-LM Prior*: Stage-1's $< 0.8$ ms `CHIMED02` lattice (19.7 MB `mmap`) simultaneously produces Frame-0 instant candidates and Stage-1 proposals $cal(P)_1$. Any aligned full-span proposal from $cal(P)_1$ not already in Route A's beam is rescored under the *exact same* Bayesian objective (@eq-bayes) in a single batched pass—preventing unaligned non-exact proposals from hijacking the channel while rescuing rare multi-character entities. A lightweight left-context Cache-LM prior (`+1.10` per repeated non-function entity in $C$, capped at `+2.00`) preserves discourse consistency.

== Lossless Transition-Indexed Incremental Beam Cache

During interactive typing, a user appends or backspaces one keystroke at a time ($k_(1:m) arrow.r k_(1:m+1)$). Naively caching `buckets[1..=reuse_until]` and resuming search from `reuse_until` introduces a subtle, high-impact bug we uncovered during `v0.15.4` error forensics:

#prop-box("Proposition 1 (Cross-Boundary Edge Loss in Bucket Caching)")[
  Let a multi-character token edge $e = (v, Delta b)$ start at byte position $p < r = "reuse_until"$ and terminate at $q = p + Delta b >= r$. If incremental caching restores only `buckets[1..=r]` and skips expansion of positions $0..r$, every multi-character word spanning the prefix boundary $[p, q)$ is permanently dropped from the search space.
]

For example, when typing `feikuaidipaoxiang...` after a shared 6-byte prefix, `feikuai` (`0..7`) starts at `0 < 6` and ends at `7 >= 6`, causing a naive bucket cache to miss 飞快 entirely. Chime resolves Proposition 1 with a *lossless transition-indexed `RouteABeamCache`*: (i) caching activates strictly under monotonic append/backspace prefix relations; (ii) every `BeamHyp` records its originating byte offset `src_pos`; and (iii) only *pruned* active hypotheses ($<= B$ per bucket) archive their outgoing transitions into `saved_outgoing[src_pos]`. On prefix reuse up to $r$, Chime replays all `saved_outgoing[p]` with $p < r$ and destination $q >= r$ into `buckets[q]`—guaranteeing *bit-identical* beam search to cold decoding while avoiding unpruned clone overhead.

= Block-Influence Compression & Capacity-Ceiling Analysis

== From 6L Reranker to `BI-18L e8_l4`

Our `v0.13`–`v0.15` trajectory revealed how model architecture interacts with Route A beam search:
- *`v0.13` (`8L`, 87.8 MB, 36K vocab)*: Switching from Stage-2 reranking (`v0.12`, 75.84% blind holdout) to Route A immediately lifted Blind Holdout-1010 by *+7.82* points (*83.66%*) and Blind Holdout-10K by *+8.86* points (*75.31%*), confirming that Stage-1 lattice recall was the primary bottleneck.
- *`v0.14` (`28L`, 256.7 MB, distilled from `Qwen3-14B-Base`)*: Scaling to all 28 layers of `Qwen3-0.6B` with load-time fused `qkv_proj` / `gate_up_proj` projections (4.07× Metal speedup) pushed Blind Holdout-1010 to *90.00%* and Blind Holdout-10K to *79.70%*, but incurred a 256.7 MB footprint and 293.4 ms 10K `p50` latency.
- *`v0.15` (`BI-18L e8_l4`, 190.3 MB)*: Computing per-layer cosine Block Influence @men2024shortgpt on Chinese corpora identified heavy middle-layer redundancy (`layers 11..15` and `22..26`). Pruning to 18 layers $cal(I) = \{0..10, 16..21, 27\}$ while keeping full MLP width ($d_"ff" = 3072$) and lossless 36,864-token vocabulary, we distilled from `Qwen3-14B-Base` for 25,000 steps using exact tail-bucket Soft-KL divergence (zero weight decay on tied embeddings). Quantizing the shared embedding/LM head to *8-bit* and transformer blocks to *4-bit* (`group_size=64`, denoted `e8_l4`) preserves homophone output-head precision in *190.3 MB* ($-26\%$ vs. 28L), cutting 10K `p50` latency by *3.01×* (`293.4 ms` $arrow.r$ `97.6 ms`) while lifting Blind Holdout-10K to *80.93%* (`v0.15.0`) and *81.55%* after four zero-GPU decoder refinement rounds (`v0.15.4`/`v0.15.5`).

== Controlled Negative Results: Proving the 0.32B Capacity Ceiling

Before adopting Two-Tier reranking, we systematically tested three competing hypotheses for closing the remaining gap on `BI-18L` (0.32B). Every single one plateaued or failed (@tab-neg):

#figure(
  table(
    columns: (2.05fr, 0.72fr, 0.76fr, 0.78fr, 0.72fr),
    align: (left, right, right, right, right),
    stroke: none,
    inset: (x: 3.6pt, y: 3.4pt),
    toprule,
    [*Controlled Experiment (Fixed `v0.15.2` Decoder)*], [*Size*], [*Dev-1258*], [*Dev10k-Mini*], [*Dev p50*],
    midrule,
    [H1: End-to-end Pinyin$arrow.r$Char `BI-18L` (20M tok)], [190 MB], [#text(fill: rgb("#be123c"))[Stopped]], [#text(fill: rgb("#be123c"))[Regressed]], [—],
    [H2a: `BI-18L` baseline (65M Soft-KL, `e8_l4`)], [190 MB], [*94.44%*], [*81.10%*], [*81.8 ms*],
    [H2b: 6-source 40M 14B Soft-KL (`SUB_K=24`)], [190 MB], [94.28%], [80.40%], [81.9 ms],
    [H2c: FP16 Model Soup (65/35 blend @wortsman2022modelsoups)], [190 MB], [94.28%], [80.50%], [81.8 ms],
    [H3a: Finer 4-bit group (`e8_l4_g32`)], [207 MB], [93.72%], [80.70%], [82.5 ms],
    [H3b: 6-bit transformer blocks (`e8_l6`)], [258 MB], [94.12%], [80.10%], [91.7 ms],
    [H3c: Near-lossless 8-bit (`mlx8`)], [325 MB], [94.28%], [80.40%], [87.4 ms],
    bottomrule,
  ),
  scope: "parent",
  placement: top,
  caption: [
    *Controlled Negative & Quantization-Ceiling Experiments on `BI-18L` (0.32B).* Neither end-to-end P2C conditioning (H1), additional 14B multi-source distillation / FP16 Model Soup (H2), nor upgrading quantization from 4-bit to near-lossless 8-bit (`+135 MB`, H3) breaks past `94.4%` Dev / `81.1%` Dev10k-Mini under the same decoder—proving that `e8_l4` is Pareto-optimal and the residual bottleneck is *parameter capacity*.
  ],
) <tab-neg>

1. *H1 (End-to-End P2C Conditioning)*: Training `BI-18L` to read full Pinyin prefixes and emit characters directly converged slowly over 20M tokens and degraded severely on long sentences, code-switching, and motor slips relative to Bayesian Route A.
2. *H2 (Teacher & Corpus Scaling)*: On our exact 36K vocabulary, dense `Qwen3-14B-Base` outperformed MoE `Qwen3-30B-A3B-Base` as teacher (`93.20%` vs. `92.70%` oracle on `dev10k_mini`; `98.89%` vs. `94.44%` on contextual single characters). Yet re-distilling `BI-18L` on 40M 8-gram-decontaminated multi-source tokens (and blending 65/35 via FP16 Model Soup @wortsman2022modelsoups) stayed flat at `94.28%` / `80.50%`.
3. *H3 (Quantization Precision Sweep)*: Holding weights and decoder identical while lifting `BI-18L` from `e8_l4` (190 MB) to `e8_l4_g32` (207 MB), `e8_l6` (258 MB), and near-lossless `mlx8` (325 MB) delivered *+0.0% Acc\@1* at `+68`–`+135 MB` and `+6`–`+10 ms` latency.

Together, H1–H3 establish a clear systems conclusion: *`BI-18L e8_l4` has already saturated the 0.32B parameter capacity frontier.* Further accuracy gains require either zero-cost structural decoder fixes (which indeed delivered *+1.69* blind holdout points across `v0.15.1`–`v0.15.4`) or *selective escalation to a larger parameter budget*.

= Rank-Stacked Multi-LoRA `.imepack` Cartridges

To adapt Chime to specialized domains (classical poetry, developer prose, air-gapped personal writing) without retraining or duplicating the 190.3 MB base model, `.imepack` v2 cartridges co-mount: (i) `pack.imedict`, a zero-copy `CHIMED02` lexical slice with $O(log C)$ `Tab` continuation chains (resolving out-of-vocabulary recall); (ii) `lora.safetensors`, Rank-8 FP16 adapters across all 7 projections; and (iii) `manifest.json` declaring routing policy (`classical`, `developer`, `always`).

#figure(
  fig-scaling-and-transfer(),
  kind: image,
  scope: "parent",
  placement: top,
  caption: [
    *Multi-LoRA Kernel Fusion & Q4-Manifold Alignment.* *(a)* 6-candidate Metal latency (p50, ms) across $K=0..6$ Rank-8 adapters: Rank-Stacked GEMM Fusion + Batched KV-Gather flatten the $O(K)$ launch ramp from 10.74 ms to *1.41 ms* (*7.60× speedup*, $+39.4 thin mu"s"$/adapter). *(b)* Classical validation NLL vs. scale $gamma$: zero-shot mounting of an FP16-trained LoRA onto 4-bit weights overshoots base at $gamma=1.0$ (6.117 vs. 6.041), whereas *Q4-Manifold Aligned LoRA* training directly on dequantized 4-bit weights eliminates overshoot and achieves *−0.301 nats* (5.740).
  ],
) <fig-scaling>

1. *Rank-Stacked GEMM Fusion & Batched KV-Gather*: Pre-transposing and concatenating $K$ mounted adapters along rank $R = 8K$ at mount time collapses $42 times 2K$ sequential tiny Metal GEMMs into *two* fused GEMMs per projection:
$ Delta bold(y) = ((bold(x) bold(A)_"cat"^top) circle.tiny bold(g)(C, a)) bold(B)_"cat"^top $ <eq-lora>
cutting $K=6$ adapter latency from 10.74 ms to *1.41 ms* (*7.60× speedup*, @fig-scaling (a)).
2. *Q4-Manifold Alignment & Couplet-Aware Simplex Routing*: Training `LoRA` directly on dequantized 4-bit runtime weights $hat(bold(W)) = bold(W) + bold(E)_Q$ lets $Delta bold(W) = bold(B) bold(A)$ compensate for quantization residuals $bold(E)_Q$, eliminating zero-shot FP16$arrow.r$INT4 overshoot (@fig-scaling (b)) and reducing held-out NLL by *−0.301 nats* on classical literature and *−0.401 nats* on an air-gapped 815K-char personal essay corpus (2013–2020). Our *Couplet-Aware Simplex Router* activates `Classical` on archaic particles or symmetric $5+5$/$7+7$ couplets, `Developer` on ASCII/IDE cues, and projects scales onto $gamma_k <= 0.80, sum_k gamma_k <= 0.90$, preventing written-register superposition drift.

= Confidence-Gated Two-Tier Local Reranking

== Why Single-Pass Batched Reranking Breaks the 0.32B Ceiling

While running a 4B LLM autoregressively inside beam search is prohibitively slow on laptops, our 10K blind error analysis reveals a striking structural property of `BI-18L`: *Tier-1's `Hit@5` is already 91.05% on Blind Holdout-10K and 96.34% on Blind Holdout-1010*—meaning that in over 90% of queries, the ground-truth sentence is *already aligned and present inside Tier-1's top-5 candidates*, requiring only stronger world knowledge to rank \#1.

Chime `v0.16` exploits this via *Optional Confidence-Gated Two-Tier Local Reranking* implemented in pure Rust + `mlx-rs` (zero Python runtime, zero default bundle bloat):

1. *Confidence Margin Gate*: After Tier-1 Route A produces ranked candidates $c_1, dots, c_M$ with Bayesian scores $s_1 >= s_2 >= dots$, Tier-2 triggers *if and only if*:
$ Delta s = s_1 - s_2 < tau quad (tau = 3.0 "default") $ <eq-gate>
On high-confidence daily typing ($Delta s >= 3.0$, representing *81.3%* of Blind Holdout-1010 and *78.5%* of Dev-1258), Tier-2 performs *zero GPU work*.
2. *Single Batched Metal Pass over Shared Prefix Cache*: When triggered, Chime tokenizes left context $C$ and top-$K$ ($K=5$) candidates using a zero-dependency pure-Rust HuggingFace byte-level BPE tokenizer (`vocab.json` + `merges.txt`, exact 6-branch Qwen3 pre-tokenizer), prefills $C$ once into a cached `MlxPrefixCache` (reused across keystrokes whenever $C$ is unchanged), and evaluates all $K=5$ candidate sequences in a *single batched `[K, L_max]` Metal forward pass* on a local 4-bit MLX `Qwen3-1.7B-Base` (984 MB) or `Qwen3-4B-Base` (2.27 GB) model.
3. *Linear Log-Probability Fusion & `{的, 地, 得}` Grammatical Guard*: Rather than discarding Tier-1's calibrated phonetic/syntactic priors, Chime fuses the two tiers linearly:
$ S_"final" (c_i) &= (1 - alpha) ell_"T1" (c_i) + alpha ell_"T2" (c_i) \ & quad + c_"chan" (c_i) + c_"extra" (c_i) $ <eq-blend>
with $alpha = 0.65$, where $c_"chan"$ is Tier-1's exact acoustic/typo channel score and $c_"extra"$ preserves length-matched utterance closure.

Crucially, unconstrained general-purpose base LLMs trained on noisy web text frequently confuse colloquial Chinese structural particles `{的, 地, 得}`—causing raw 4B reranking to regress `de_di_de_zai` from `96.0%` down to `93.0%`. Chime therefore enforces a lightweight *`{的, 地, 得}` Grammatical Guard*: (i) if Tier-1's top-2 candidates differ *solely* on positions in `{的, 地, 得}`, escalation is skipped entirely; and (ii) if Tier-2's winner differs from Tier-1's top-1 solely in `{的, 地, 得}`, Tier-1's grammatical choice is preserved.

== Gated Ensemble Synergy Over Standalone 4B Replacement

On `dev10k_mini` (1,000 stratified sentences), we discovered a strongly counter-intuitive synergy: *replacing Tier-1 with 100% standalone `Qwen3-4B-Base` scoring (`alpha = 1.0`, all queries) achieves only 83.00% Acc\@1, whereas confidence-gated Two-Tier fusion (`alpha = 0.65`, escalating only 33.8% of queries) achieves 84.90% (+1.90 pts; up to 85.20% at $tau=3.5$, +2.20 pts)*. This proves that domain-distilled Tier-1 and open-domain Tier-2 are genuinely complementary: Tier-1 contributes IME-calibrated grammar and channel structure, while Tier-2 resolves world-knowledge and long-range semantic ambiguity.

= Experiments & Analysis

== Strictly Decontaminated 12,268-Query Evaluation Suite

All measurements run natively in Rust on an Apple M5 Pro (`mlx-rs 0.34`, Metal GPU, single-threaded GPU actor):
- *Dev-1258 (`dev_suite.tsv`)*: 1,258 curated queries across 17 linguistic categories (spoken chat, copula+classifiers, 的/地/得, verbal complements, minimal-pair homophones, polyphones, code-switching, Shuangpin, fuzzy rhymes, QWERTY typos, idioms/poetry).
- *Blind Holdout-1010 (`test_holdout.tsv`)*: 1,010 unseen whole sentences across 16 domains (0% sentence overlap with Dev-1258; all benchmark 5/8-grams excluded from neural training; individual rows never inspected).
- *Blind Holdout-10K (`test_holdout_10k.tsv`)*: 10,000 strictly blind open-domain sentences spanning standalone prose, contextual dialogue, contextual single characters, polyphone heteronyms, and motor typos (plus `dev10k_mini`, a 1,000-row stratified dev split used for fast ablation).

#figure(
  fig-evolution-and-twotier(holdout10k_t2: h10k-t2-acc-num),
  kind: image,
  scope: "parent",
  placement: top,
  caption: [
    *(a)* Longitudinal Top-1 accuracy across six Chime generations (`v0.8` $arrow.r$ `v0.16`): adopting Route A beam search (`v0.13`), `BI-18L e8_l4` compression (`v0.15`), zero-GPU decoder refinements (`v0.15.5`), and optional Two-Tier 4B reranking (`v0.16`) raises Blind Holdout-1010 by *+22.97* points (`71.39%` $arrow.r$ *94.36%*) and Blind Holdout-10K from `66.45%` to *#h10k-t2-acc%*. *(b)* Pure-Rust Two-Tier 4B category gains on Blind Holdout-1010 (`18.7%` escalation, `+8.3 ms` p50), led by *+11.67%* on technical engineering and *+5.00%* on contextual dialogue.
  ],
) <fig-evol>

== Longitudinal & End-to-End Pure-Rust Results

@tab-main summarizes the full architectural trajectory across all three benchmarks, all measured with our native Rust `chime bench` harness (`beam_width = 8`):

#figure(
  table(
    columns: (2.08fr, 0.72fr, 0.68fr, 0.68fr, 0.68fr, 0.70fr, 0.68fr),
    align: (left, right, right, right, right, right, right),
    stroke: none,
    inset: (x: 3.4pt, y: 3.5pt),
    toprule,
    [], [], table.cell(align: center)[*Dev-1258*], table.cell(colspan: 2, align: center)[*Blind Holdout-1010*], table.cell(colspan: 2, align: center)[*Blind Holdout-10K*],
    [*System / Generation*], [*Size*], [*Acc\@1*], [*Acc\@1*], [*Hit\@5*], [*Acc\@1*], [*Hit\@5*],
    midrule,
    [`v0.8` · Stage-1 Lattice Only], [0 MB], [67.41%], [42.87%], [64.46%], [—], [—],
    [`v0.8` · 6L Stage-2 Rerank + Packs], [64.2 MB], [85.85%], [71.39%], [77.13%], [—], [—],
    [`v0.12` · 8L 36K Stage-2 Rerank], [87.8 MB], [89.51%], [75.84%], [—], [66.45%], [—],
    [`v0.13` · 8L 36K *Route A Beam*], [87.8 MB], [91.49%], [83.66%], [92.38%], [75.31%], [87.58%],
    [`v0.14` · 28L 14B-Distill Route A], [256.7 MB], [93.16%], [90.00%], [92.38%], [79.70%], [87.36%],
    [`v0.15.0` · `BI-18L e8_l4` Dual-Prop.], [190.3 MB], [93.80%], [90.59%], [95.54%], [80.93%], [90.58%],
    [`v0.15.5` · `BI-18L` + Decoder Fixes], [190.3 MB], [95.31%], [92.28%], [96.34%], [81.55%], [91.05%],
    [`v0.16` · +Two-Tier `Qwen3-1.7B 4b`], [+984 MB], [*95.63%*], [93.96%], [96.34%], [—], [—],
    [*`v0.16` · +Two-Tier `Qwen3-4B 4b`*], [*+2.27 GB*], [*95.63%*], [*94.36%*], [*96.34%*], [*#h10k-t2-acc%*], [*#h10k-t2-hit5%*],
    bottomrule,
  ),
  scope: "parent",
  placement: top,
  caption: [
    *Longitudinal End-to-End Evaluation across 12,268 Decontaminated Queries* (`chime bench --route-a --beam-width 8`, pure Rust + Apple Silicon Metal). Compared against our earlier Stage-2 reranking baseline (`v0.8`), Route A + `BI-18L e8_l4` + lossless caching delivers *+20.89* points on Blind Holdout-1010 (`92.28%`) in 190.3 MB; optional Confidence-Gated Two-Tier 4B adds a further *+2.08* points on Blind Holdout-1010 (*94.36%*, *80.3% relative error reduction* over `v0.8`) and *#h10k-t2-delta* points on Blind Holdout-10K (*#h10k-t2-acc%*, MRR *#h10k-t2-mrr*).
  ],
) <tab-main>

Key findings from @tab-main and @tab-t2:

1. *Zero-GPU Decoder Refinements (`v0.15.0` $arrow.r$ `v0.15.5`)*: Without changing a single weight byte in `BI-18L`, fixing `en_multi` KV advancement (`v0.15.1`), contiguous English split penalties & speaker-tag pronoun resolution (`v0.15.3`), and our lossless transition-indexed `RouteABeamCache` + curated EN split alignment + Stage-1 ghost guard (`v0.15.4`) lifted Dev-1258 by *+1.51* points (`93.80%` $arrow.r$ `95.31%`), Blind Holdout-1010 by *+1.69* points (`90.59%` $arrow.r$ `92.28%`), and Blind Holdout-10K by *+0.62* points (`80.93%` $arrow.r$ `81.55%`)—matching an entire model generation at `$0` GPU cost.
2. *Pure-Rust Two-Tier Gains with Negligible Median Latency (@tab-t2)*: With $tau=3.0, alpha=0.65$ and `guard_de_di_de = true`, Two-Tier `Qwen3-4B-Base` (4-bit) escalates only *18.7%* of Blind Holdout-1010 queries (`189/1,010`), adding just *+8.3 ms* to median end-to-end latency (`138.0 ms` $arrow.r$ `146.3 ms`) while delivering *+2.08* points overall (*94.36%*, `MRR 0.9531`), *+11.67* points on `tech_engineering` (`68.33%` $arrow.r$ `80.00%`), *+5.00* points on `contextual_dialogue` (`90.00%` $arrow.r$ `95.00%`), and *+4.00* points on `prose_essay` (`86.67%` $arrow.r$ `90.67%`). On Blind Holdout-10K, Two-Tier 4B escalates *#h10k-t2-esc%* of queries and lifts Acc\@1 from `81.55%` to *#h10k-t2-acc%* (#h10k-t2-delta, `Hit@3` #h10k-t2-hit3%, `Hit@5` #h10k-t2-hit5%, `MRR` #h10k-t2-mrr, `p50` #h10k-t2-p50 ms).
3. *Effect of the `{的, 地, 得}` Grammatical Guard*: Without the guard, raw 4B reranking dropped Dev-1258 `de_di_de_zai` from `96.00%` to `93.00%` (overall Dev `95.15%`). With `guard_de_di_de = true`, Dev `de_di_de_zai` jumps to *98.00%*, Blind `de_di_de_grammar` climbs to *96.00%*, and Dev-1258 hits *95.63%* with zero category regressions.
4. *1.7B vs. 4B Tier-2 Tradeoff*: Even the compact `Qwen3-1.7B-Base` 4-bit (984 MB) captures *81%–88%* of 4B's gain (*93.96%* Blind Holdout-1010, *84.50%* `dev10k_mini`, *95.63%* Dev-1258), making it an attractive mid-RAM tier for 16 GB Apple Silicon machines.

#figure(
  table(
    columns: (1.58fr, 0.58fr, 0.68fr, 0.70fr, 0.86fr),
    align: (left, right, right, right, right),
    stroke: none,
    inset: (x: 2.8pt, y: 3.4pt),
    toprule,
    [*Tier-2 Ablation*], [*Esc.%*], [*Mini*], [*Blind*], [*p50*],
    midrule,
    [Tier-1 Only (`BI-18L`, 190 MB)], [0.0%], [81.50%], [92.28%], [138.0 ms],
    [Standalone 4B (`alpha=1.0`, All)], [100%], [83.00%], [—], [—],
    [Two-Tier 1.7B ($tau=3.0$, Guard)], [18.7%], [84.50%], [93.96%], [155.3 ms],
    [Two-Tier 4B (No `{的/地/得}` Guard)], [18.7%], [84.90%], [94.36%], [146.3 ms],
    [*Two-Tier 4B ($tau=3.0$, +Guard)*], [*18.7%*], [*84.90%*], [*94.36%*], [*146.3 ms*],
    bottomrule,
  ),
  placement: top,
  caption: [
    *Pure-Rust Two-Tier Ablation* (`Esc.%`, `Blind`, and `p50` on Blind Holdout-1010; `Mini` on stratified `dev10k_mini`). Confidence-gated fusion beats 100% standalone 4B replacement by *+1.90* points, while the `{的, 地, 得}` guard lifts Dev `de_di_de_zai` from `93.0%` to *98.0%* without losing any blind gain.
  ],
) <tab-t2>

= Discussion, Limitations & Future Work

*Tier-2 Footprint Compression.* Standard HuggingFace MLX 4-bit bundles carry full 151,936-token multilingual vocabularies (`~175 MB` in 1.7B; `~219 MB` in 4B). Zero-training vocabulary pruning to our lossless 36,864 Chinese-English subset paired with `e4_l3` mixed quantization (4-bit embeddings + 3-bit blocks) can shrink `Qwen3-1.7B-Base` from 984 MB down to *$approx$ 650 MB* ($-34%$) and `Qwen3-4B-Base` from 2.27 GB to *$approx$ 1.60 GB* without retraining; adding idle-timeout weight eviction further lowers background RAM overhead to 0 MB when not typing.

*Long-Sentence Tail Latency & Residual Errors.* While Fast Path guarantees $< 0.8$ ms Frame-0 display and anti-jitter slot-\#1 locking prevents visual shifts, 10K-suite offline `p95` beam latency remains dominated by long 18+ syllable inputs under widened early funnels. Adaptive funnel narrowing driven by step-0 entropy and prefix-shared multi-keystroke GPU batching are promising avenues to halve tail latency. Remaining Blind Holdout-10K errors concentrate in unseen named entities and ambiguous standalone single characters where no context exists; lightweight on-device RAG over user contacts/notes or optional Tier-2 left-context window expansion could address them while preserving strict zero-network privacy.

*Privacy & Ethical Considerations.* Chime enforces zero network entitlement in the IME runtime, bypasses `IsSecureEventInputEnabled()` password fields in hardware, stores habits strictly in local SQLite WAL, and runs both Tier-1, Tier-2, and personal `LoRA` fine-tuning 100% offline on Apple Silicon.

= Conclusion

Chime demonstrates that high-accuracy client-side neural Pinyin input does not require cloud round-trips or brute-force autoregressive generation over multi-billion-parameter LLMs. By combining *Bayesian Pinyin-Constrained Beam Search (Route A)* with a *lossless transition-indexed beam cache*, *Block-Influence 18-layer `e8_l4` distillation* (190.3 MB), *Rank-Stacked Q4-Manifold Multi-LoRA cartridges*, and *Optional Confidence-Gated Two-Tier Local Reranking*, Chime achieves *95.63%* on Dev-1258, *94.36%* on Blind Holdout-1010 (*+22.97* points over our earlier Stage-2 reranker), and *#h10k-t2-acc%* on Blind Holdout-10K—all 100% offline on everyday Apple Silicon laptops.

#bibliography("refs.bib", title: "References", style: "association-for-computing-machinery")
