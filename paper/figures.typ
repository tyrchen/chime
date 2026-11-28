// Pure-vector publication figures for Chime Paper v2 (zero raster bitmaps).

#let ink = rgb("#18181b")
#let muted = rgb("#52525b")
#let faint = rgb("#a1a1aa")
#let border-col = rgb("#d4d4d8")
#let bg-soft = rgb("#f8fafc")
#let teal = rgb("#0f4c5c")
#let teal-soft = rgb("#e6f2f5")
#let blue = rgb("#1d4ed8")
#let blue-soft = rgb("#eff6ff")
#let amber = rgb("#b45309")
#let amber-soft = rgb("#fef3c7")
#let crimson = rgb("#be123c")
#let crimson-soft = rgb("#ffe4e6")
#let emerald = rgb("#047857")
#let emerald-soft = rgb("#ecfdf5")
#let violet = rgb("#6d28d9")
#let violet-soft = rgb("#f5f3ff")

#let arch-box(title, subtitle, body, fill-c, stroke-c) = box(
  width: 100%,
  fill: fill-c,
  stroke: 0.8pt + stroke-c,
  radius: 5pt,
  inset: (x: 6.5pt, y: 5.5pt),
)[
  #set par(justify: false, leading: 0.42em)
  #set text(hyphenate: false)
  #text(size: 7.8pt, weight: "bold", fill: stroke-c)[#title]
  #if subtitle != "" [
    #h(2.5pt)
    #text(size: 6.4pt, fill: muted)[(#subtitle)]
  ]
  #v(2.5pt)
  #text(size: 7.0pt, fill: ink)[#body]
]

#let fig-architecture() = block(
  width: 100%,
  stroke: 0.6pt + border-col,
  radius: 6pt,
  inset: 9pt,
  fill: white,
)[
  #set par(justify: false, leading: 0.42em)
  #show raw: set text(font: ("PT Mono", "JetBrains Mono NL", "DejaVu Sans Mono"), size: 0.90em)
  #grid(
    columns: (1.02fr, 11pt, 1.48fr, 11pt, 1.40fr, 11pt, 0.96fr),
    align: (left + horizon, center + horizon, left + horizon, center + horizon, left + horizon, center + horizon, left + horizon),
    arch-box(
      "1. Fast Path",
      "< 0.8 ms · mmap",
      [
        • `CHIMED02` lattice (19.7 MB)\
        • Full / Shuangpin / fuzzy\
        • 4 motor-typo operators\
        • Instant Frame-0 candidates\
        • Stage-1 proposals $cal(P)_1$
      ],
      teal-soft,
      teal,
    ),
    text(size: 10.5pt, weight: "bold", fill: teal)[$arrow.r$],
    arch-box(
      "2. Tier-1 Route A Beam Search",
      "190.3 MB · BI-18L e8_l4",
      [
        • Syllable-sync `TokenPinyinIndex` (36K)\
        • Exact Bayesian $log P_"LM" + log P_"chan"$\
        • Exact BPE `logaddexp` + Dual-Proposal\
        • Lossless `RouteABeamCache` (`src_pos`)\
        • Surprisal-spike rescue + Multi-LoRA
      ],
      blue-soft,
      blue,
    ),
    text(size: 10.5pt, weight: "bold", fill: blue)[$arrow.r$],
    arch-box(
      "3. Confidence-Gated Tier-2",
      "Opt-in · 18.7% Escalation",
      [
        • Gate: $s_1 - s_2 < tau$ ($tau = 3.0$)\
        • Local 4-bit `Qwen3-1.7B/4B-Base`\
        • Single batched `[5, L]` Metal pass\
        • Shared `MlxPrefixCache` + linear blend\
        • `{的, 地, 得}` grammatical guard
      ],
      violet-soft,
      violet,
    ),
    text(size: 10.5pt, weight: "bold", fill: violet)[$arrow.r$],
    arch-box(
      "4. Zero-Jitter UI",
      "Swift 6 IMK",
      [
        • Session anchor lock\
        • 60 ms slot-\#1 freeze\
        • SQLite WAL habits\
        • Offline Emoji / `Tab`\
        • 0-byte telemetry
      ],
      emerald-soft,
      emerald,
    ),
  )
  #v(5.5pt)
  #block(
    width: 100%,
    fill: amber-soft,
    stroke: (thickness: 0.6pt, paint: amber, dash: "dashed"),
    radius: 4pt,
    inset: (x: 8pt, y: 4.2pt),
  )[
    #set par(justify: true, leading: 0.42em)
    #text(size: 7.0pt, fill: ink)[
      *Dual Co-Design & Zero Default Bloat*: Pluggable `.imepack` cartridges (`classics-poetry`, `tech-developer`, air-gapped `personal-style`) co-mount `pack.imedict` into Fast Path and 18L Rank-8 `LoRA` into Tier-1 via Rank-Stacked GEMMs. Tier-2 is strictly opt-in (zero footprint in the 192 MB base `.pkg`) and runs natively in pure Rust + `mlx-rs` with live hot-mount on the dedicated Metal GPU actor thread.
    ]
  ]
]

#let fig-evolution-and-twotier(holdout10k_t2: 85.00) = {
  set par(justify: false, leading: 0.42em)
  show raw: set text(font: ("PT Mono", "JetBrains Mono NL", "DejaVu Sans Mono"), size: 0.90em)
  grid(
    columns: (1.06fr, 0.94fr),
    gutter: 11pt,

    // Panel (a): Multi-Generation Accuracy Trajectory (v0.8 -> v0.16)
    block(
      width: 100%,
      stroke: 0.6pt + border-col,
      radius: 5pt,
      inset: 8pt,
      fill: white,
    )[
      #align(center)[
        #text(size: 7.6pt, weight: "bold", fill: ink)[(a) End-to-End Top-1 Accuracy Across Generations (%)]
      ]
      #v(4pt)
      #let w = 216pt
      #let h = 118pt
      #let ymin = 62.0
      #let ymax = 98.0
      #align(center)[
        #box(width: w + 28pt, height: h + 26pt)[
          #for (val, lbl) in ((65.0, "65"), (70.0, "70"), (75.0, "75"), (80.0, "80"), (85.0, "85"), (90.0, "90"), (95.0, "95")) {
            let y = h - ((val - ymin) / (ymax - ymin)) * h
            place(dx: 24pt, dy: y, line(length: w, stroke: 0.4pt + rgb("#e4e4e7")))
            place(dx: 0pt, dy: y - 4pt, box(width: 20pt, align(right, text(size: 6.3pt, fill: muted)[#lbl])))
          }
          // Vertical divider: Stage-2 Rerank -> Route A Beam Search
          #let x_routea = 24pt + (1.5 / 6.0) * w
          #place(dx: x_routea, dy: 0pt, line(end: (0pt, h), stroke: (thickness: 0.7pt, paint: amber, dash: "dashed")))
          #place(dx: x_routea + 3pt, dy: 3pt, text(size: 5.8pt, fill: amber, weight: "bold")[Route A paradigm])

          // Axes
          #place(dx: 24pt, dy: 0pt, line(end: (0pt, h), stroke: 0.7pt + ink))
          #place(dx: 24pt, dy: h, line(length: w, stroke: 0.7pt + ink))

          #let labels = ("v0.8\n6L", "v0.12\n8L", "v0.13\n8L-A", "v0.14\n28L", "v0.15.0\nBI-18L", "v0.15.5\nDec.", "v0.16\n2-Tier")
          #for i in range(0, 7) {
            let x = 24pt + (i / 6.0) * w
            place(dx: x - 14pt, dy: h + 3pt, box(width: 28pt, align(center, text(size: 5.7pt, fill: muted)[#labels.at(i)])))
          }

          // Dev-1258 (blue)
          #let dev = ((0, 85.85), (1, 89.51), (2, 91.49), (3, 93.16), (4, 93.80), (5, 95.31), (6, 95.63))
          // Blind Holdout-1010 (emerald)
          #let h1k = ((0, 71.39), (1, 75.84), (2, 83.66), (3, 90.00), (4, 90.59), (5, 92.28), (6, 94.36))
          // Blind Holdout-10K (violet, from v0.12 onward)
          #let h10k = ((1, 66.45), (2, 75.31), (3, 79.70), (4, 80.93), (5, 81.55), (6, holdout10k_t2))

          #for i in range(0, dev.len() - 1) {
            let (x1i, v1) = dev.at(i)
            let (x2i, v2) = dev.at(i + 1)
            place(line(
              start: (24pt + (x1i / 6.0) * w, h - ((v1 - ymin) / (ymax - ymin)) * h),
              end: (24pt + (x2i / 6.0) * w, h - ((v2 - ymin) / (ymax - ymin)) * h),
              stroke: (thickness: 1.15pt, paint: blue, dash: "densely-dashed"),
            ))
          }
          #for (xi, v) in dev {
            place(dx: 24pt + (xi / 6.0) * w - 1.9pt, dy: h - ((v - ymin) / (ymax - ymin)) * h - 1.9pt, rect(width: 3.8pt, height: 3.8pt, fill: blue))
          }

          #for i in range(0, h1k.len() - 1) {
            let (x1i, v1) = h1k.at(i)
            let (x2i, v2) = h1k.at(i + 1)
            place(line(
              start: (24pt + (x1i / 6.0) * w, h - ((v1 - ymin) / (ymax - ymin)) * h),
              end: (24pt + (x2i / 6.0) * w, h - ((v2 - ymin) / (ymax - ymin)) * h),
              stroke: 1.55pt + emerald,
            ))
          }
          #for (xi, v) in h1k {
            place(dx: 24pt + (xi / 6.0) * w - 2.1pt, dy: h - ((v - ymin) / (ymax - ymin)) * h - 2.1pt, circle(radius: 2.1pt, fill: emerald))
          }

          #for i in range(0, h10k.len() - 1) {
            let (x1i, v1) = h10k.at(i)
            let (x2i, v2) = h10k.at(i + 1)
            place(line(
              start: (24pt + (x1i / 6.0) * w, h - ((v1 - ymin) / (ymax - ymin)) * h),
              end: (24pt + (x2i / 6.0) * w, h - ((v2 - ymin) / (ymax - ymin)) * h),
              stroke: 1.45pt + violet,
            ))
          }
          #for (xi, v) in h10k {
            place(dx: 24pt + (xi / 6.0) * w - 2.0pt, dy: h - ((v - ymin) / (ymax - ymin)) * h - 2.0pt, circle(radius: 2.0pt, fill: violet))
          }

          // Clean labels
          #place(dx: 148pt, dy: h - ((97.2 - ymin) / (ymax - ymin)) * h - 4pt, text(size: 6.0pt, weight: "bold", fill: blue)[Dev-1258 (95.63%)])
          #place(dx: 122pt, dy: h - ((88.5 - ymin) / (ymax - ymin)) * h, text(size: 6.0pt, weight: "bold", fill: emerald)[Blind-1010 (94.36%, +23.0)])
          #place(dx: 122pt, dy: h - ((76.5 - ymin) / (ymax - ymin)) * h, text(size: 6.0pt, weight: "bold", fill: violet)[Blind-10K (#str(holdout10k_t2)%)])
        ]
      ]
    ],

    // Panel (b): Two-Tier Domain Boosts on Blind Holdout-1010 & Synergy
    block(
      width: 100%,
      stroke: 0.6pt + border-col,
      radius: 5pt,
      inset: 8pt,
      fill: white,
    )[
      #align(center)[
        #text(size: 7.6pt, weight: "bold", fill: ink)[(b) Two-Tier 4B Blind-1010 Category Gains (%)]
      ]
      #v(4pt)
      #let rows = (
        ("Tech & Eng.", 68.33, 80.00, "+11.67"),
        ("Dialogue", 90.00, 95.00, "+5.00"),
        ("Prose Essay", 86.67, 90.67, "+4.00"),
        ("Everyday Chat", 89.90, 93.03, "+3.13"),
        ("的/地/得 Guard", 94.00, 96.00, "+2.00"),
        ("Code-Switch", 89.83, 91.53, "+1.70"),
      )
      #let barw = 104pt
      #box(width: 100%, height: 144pt)[
        #v(2pt)
        #for (name, base, t2, delta) in rows {
          grid(
            columns: (58pt, barw + 4pt, 30pt),
            align: (right + horizon, left + horizon, right + horizon),
            column-gutter: 4pt,
            text(size: 6.3pt, fill: ink)[#name],
            box(width: barw, height: 11.5pt, fill: bg-soft, radius: 2pt)[
              // Tier-2 full bar (violet-emerald)
              #place(dx: 0pt, dy: 1.5pt, rect(width: (t2 / 100.0) * barw, height: 8.5pt, radius: 1.5pt, fill: violet))
              // Tier-1 baseline bar overlay
              #place(dx: 0pt, dy: 1.5pt, rect(width: (base / 100.0) * barw, height: 8.5pt, radius: 1.5pt, fill: rgb("#94a3b8")))
            ],
            text(size: 6.3pt, weight: "bold", fill: emerald)[#delta],
          )
          v(4.8pt)
        }
        #v(2pt)
        #align(center)[
          #box(fill: bg-soft, stroke: 0.4pt + border-col, radius: 3pt, inset: (x: 6pt, y: 3.5pt))[
            #text(size: 6.0pt, fill: ink)[
              #box(width: 6pt, height: 6pt, fill: rgb("#94a3b8"), radius: 1pt) #h(2pt) Tier-1 (`BI-18L`) #h(7pt)
              #box(width: 6pt, height: 6pt, fill: violet, radius: 1pt) #h(2pt) +Two-Tier 4B (18.7% gate, *+8.3 ms* p50)
            ]
          ]
        ]
      ]
    ],
  )
}

// Retained pure-vector Multi-LoRA Scaling + Q4-Manifold Transfer chart for Section 5.
#let fig-scaling-and-transfer() = {
  set par(justify: false, leading: 0.42em)
  show raw: set text(font: ("PT Mono", "JetBrains Mono NL", "DejaVu Sans Mono"), size: 0.90em)
  grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    // Panel (a): K = 0..6 Adapter Scaling Latency
    block(
      width: 100%,
      stroke: 0.6pt + border-col,
      radius: 5pt,
      inset: 8pt,
      fill: white,
    )[
      #align(center)[
        #text(size: 7.6pt, weight: "bold", fill: ink)[(a) 6-Candidate Latency (ms) vs. Mounted LoRAs ($K$)]
      ]
      #v(4pt)
      #let w = 205pt
      #let h = 112pt
      #align(center)[
        #box(width: w + 28pt, height: h + 22pt)[
          #for (val, lbl) in ((0.0, "0"), (3.0, "3"), (6.0, "6"), (9.0, "9"), (12.0, "12")) {
            let y = h - (val / 12.0) * h
            place(dx: 24pt, dy: y, line(length: w, stroke: 0.4pt + rgb("#e4e4e7")))
            place(dx: 0pt, dy: y - 4pt, box(width: 20pt, align(right, text(size: 6.3pt, fill: muted)[#lbl])))
          }
          #let y10 = h - (10.0 / 12.0) * h
          #place(dx: 24pt, dy: y10, line(length: w, stroke: (thickness: 0.7pt, paint: amber, dash: "dashed")))
          #place(dx: 28pt, dy: y10 - 9pt, text(size: 6.0pt, fill: amber, weight: "bold")[10 ms perceptual budget])

          #place(dx: 24pt, dy: 0pt, line(end: (0pt, h), stroke: 0.7pt + ink))
          #place(dx: 24pt, dy: h, line(length: w, stroke: 0.7pt + ink))

          #let unfused = ((0, 1.411), (1, 2.907), (2, 4.494), (3, 5.973), (4, 7.525), (5, 9.143), (6, 10.741))
          #let fused   = ((0, 1.405), (1, 2.948), (2, 2.981), (3, 3.050), (4, 3.030), (5, 3.106), (6, 3.105))
          #let batched = ((0, 0.846), (1, 1.216), (2, 1.340), (3, 1.357), (4, 1.375), (5, 1.403), (6, 1.413))

          #for k in range(0, 7) {
            let x = 24pt + (k / 6.0) * w
            place(dx: x - 3pt, dy: h + 4pt, text(size: 6.3pt, fill: muted)[#str(k)])
          }
          #place(dx: 24pt, dy: h + 13pt, box(width: w, align(center, text(size: 6.6pt, fill: ink)[Mounted Rank-8 Adapters ($K$)])))

          #for i in range(0, 6) {
            let (k1, v1) = unfused.at(i)
            let (k2, v2) = unfused.at(i + 1)
            place(line(start: (24pt + (k1 / 6.0) * w, h - (v1 / 12.0) * h), end: (24pt + (k2 / 6.0) * w, h - (v2 / 12.0) * h), stroke: 1.3pt + crimson))
          }
          #for (k, v) in unfused {
            place(dx: 24pt + (k / 6.0) * w - 2.0pt, dy: h - (v / 12.0) * h - 2.0pt, circle(radius: 2.0pt, fill: crimson))
          }

          #for i in range(0, 6) {
            let (k1, v1) = fused.at(i)
            let (k2, v2) = fused.at(i + 1)
            place(line(start: (24pt + (k1 / 6.0) * w, h - (v1 / 12.0) * h), end: (24pt + (k2 / 6.0) * w, h - (v2 / 12.0) * h), stroke: (thickness: 1.1pt, paint: blue, dash: "densely-dashed")))
          }
          #for (k, v) in fused {
            place(dx: 24pt + (k / 6.0) * w - 1.9pt, dy: h - (v / 12.0) * h - 1.9pt, rect(width: 3.8pt, height: 3.8pt, fill: blue))
          }

          #for i in range(0, 6) {
            let (k1, v1) = batched.at(i)
            let (k2, v2) = batched.at(i + 1)
            place(line(start: (24pt + (k1 / 6.0) * w, h - (v1 / 12.0) * h), end: (24pt + (k2 / 6.0) * w, h - (v2 / 12.0) * h), stroke: 1.5pt + emerald))
          }
          #for (k, v) in batched {
            place(dx: 24pt + (k / 6.0) * w - 2.1pt, dy: h - (v / 12.0) * h - 2.1pt, circle(radius: 2.1pt, fill: emerald))
          }

          #place(dx: 48pt, dy: 24pt, text(size: 6.0pt, weight: "bold", fill: crimson)[Unfused Sequential (10.74 ms)])
          #place(dx: 88pt, dy: h - (3.11 / 12.0) * h - 10.5pt, text(size: 6.0pt, weight: "bold", fill: blue)[Rank-Stacked Only (3.11 ms)])
          #place(dx: 56pt, dy: h - (1.41 / 12.0) * h + 4.5pt, text(size: 6.0pt, weight: "bold", fill: emerald)[Batched KV + Rank-Stacked (1.41 ms, 7.60×)])
        ]
      ]
    ],

    // Panel (b): FP16 Zero-Shot vs Q4-Manifold Aligned LoRA Scale Curves
    block(
      width: 100%,
      stroke: 0.6pt + border-col,
      radius: 5pt,
      inset: 8pt,
      fill: white,
    )[
      #align(center)[
        #text(size: 7.6pt, weight: "bold", fill: ink)[(b) Classical Validation NLL (nats) vs. LoRA Scale ($gamma$)]
      ]
      #v(4pt)
      #let w = 205pt
      #let h = 112pt
      #let ymin = 5.66
      #let ymax = 6.16
      #align(center)[
        #box(width: w + 30pt, height: h + 22pt)[
          #let sx1 = 26pt + 0.40 * w
          #let sx2 = 26pt + 0.75 * w
          #place(dx: sx1, dy: 0pt, rect(width: sx2 - sx1, height: h, fill: emerald-soft))
          #place(dx: sx1 + 4pt, dy: 4pt, text(size: 5.9pt, weight: "bold", fill: emerald)[Calibrated router zone])

          #for (val, lbl) in ((5.70, "5.70"), (5.80, "5.80"), (5.90, "5.90"), (6.00, "6.00"), (6.10, "6.10")) {
            let y = h - ((val - ymin) / (ymax - ymin)) * h
            place(dx: 26pt, dy: y, line(length: w, stroke: 0.4pt + rgb("#e4e4e7")))
            place(dx: 0pt, dy: y - 4pt, box(width: 22pt, align(right, text(size: 6.2pt, fill: muted)[#lbl])))
          }

          #let ybase = h - ((6.041 - ymin) / (ymax - ymin)) * h
          #place(dx: 26pt, dy: ybase, line(length: w, stroke: (thickness: 0.6pt, paint: muted, dash: "dotted")))
          #place(dx: 30pt, dy: ybase - 9pt, text(size: 5.9pt, fill: muted)[4-bit Base ($gamma = 0$: 6.041)])

          #place(dx: 26pt, dy: 0pt, line(end: (0pt, h), stroke: 0.7pt + ink))
          #place(dx: 26pt, dy: h, line(length: w, stroke: 0.7pt + ink))

          #for (g, lbl) in ((0.0, "0.0"), (0.2, "0.2"), (0.4, "0.4"), (0.6, "0.6"), (0.8, "0.8"), (1.0, "1.0")) {
            let x = 26pt + g * w
            place(dx: x - 5pt, dy: h + 4pt, text(size: 6.3pt, fill: muted)[#lbl])
          }
          #place(dx: 26pt, dy: h + 13pt, box(width: w, align(center, text(size: 6.6pt, fill: ink)[Effective Adapter Scale ($gamma$)])))

          #let pts_fp16 = ((0.0, 6.0410), (0.2, 5.8973), (0.35, 5.8719), (0.5, 5.8694), (0.65, 5.8911), (0.8, 5.9523), (1.0, 6.1165))
          #let pts_q4   = ((0.0, 6.0410), (0.2, 5.8545), (0.35, 5.7972), (0.5, 5.7595), (0.65, 5.7399), (0.8, 5.7521), (1.0, 5.8427))

          #for i in range(0, pts_fp16.len() - 1) {
            let (g1, n1) = pts_fp16.at(i)
            let (g2, n2) = pts_fp16.at(i + 1)
            place(line(start: (26pt + g1 * w, h - ((n1 - ymin) / (ymax - ymin)) * h), end: (26pt + g2 * w, h - ((n2 - ymin) / (ymax - ymin)) * h), stroke: (thickness: 1.2pt, paint: blue, dash: "densely-dashed")))
          }
          #for (g, n) in pts_fp16 {
            place(dx: 26pt + g * w - 1.9pt, dy: h - ((n - ymin) / (ymax - ymin)) * h - 1.9pt, rect(width: 3.8pt, height: 3.8pt, fill: blue))
          }

          #for i in range(0, pts_q4.len() - 1) {
            let (g1, n1) = pts_q4.at(i)
            let (g2, n2) = pts_q4.at(i + 1)
            place(line(start: (26pt + g1 * w, h - ((n1 - ymin) / (ymax - ymin)) * h), end: (26pt + g2 * w, h - ((n2 - ymin) / (ymax - ymin)) * h), stroke: 1.5pt + emerald))
          }
          #for (g, n) in pts_q4 {
            place(dx: 26pt + g * w - 2.1pt, dy: h - ((n - ymin) / (ymax - ymin)) * h - 2.1pt, circle(radius: 2.1pt, fill: emerald))
          }

          #place(dx: 26pt + 0.76 * w, dy: 9pt, text(size: 5.8pt, weight: "bold", fill: crimson)[FP16 overshoot\ (6.117 at $gamma=1$)])
          #place(dx: 26pt + 0.27 * w, dy: h - ((5.8694 - ymin) / (ymax - ymin)) * h - 11pt, text(size: 5.9pt, weight: "bold", fill: blue)[FP16 Zero-Shot (5.869)])
          #place(dx: 26pt + 0.34 * w, dy: h - ((5.7399 - ymin) / (ymax - ymin)) * h + 5.5pt, text(size: 6.0pt, weight: "bold", fill: emerald)[Q4-Aligned Min: 5.740 ($-0.301$)])
        ]
      ]
    ],
  )
}
