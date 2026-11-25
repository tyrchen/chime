// Pure-vector publication figures for the Chime arXiv preprint (zero raster bitmaps).

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

#let arch-box(title, subtitle, body, fill-c, stroke-c) = box(
  width: 100%,
  fill: fill-c,
  stroke: 0.8pt + stroke-c,
  radius: 5pt,
  inset: (x: 7pt, y: 6pt),
)[
  #set par(justify: false, leading: 0.44em)
  #set text(hyphenate: false)
  #text(size: 8pt, weight: "bold", fill: stroke-c)[#title]
  #if subtitle != "" [
    #h(3pt)
    #text(size: 6.7pt, fill: muted)[(#subtitle)]
  ]
  #v(2.5pt)
  #text(size: 7.2pt, fill: ink)[#body]
]

#let fig-architecture() = block(
  width: 100%,
  stroke: 0.6pt + border-col,
  radius: 6pt,
  inset: 9.5pt,
  fill: white,
)[
  #set par(justify: false, leading: 0.44em)
  #show raw: set text(font: ("PT Mono", "JetBrains Mono NL", "DejaVu Sans Mono"), size: 0.92em)
  #grid(
    columns: (1.06fr, 13pt, 1.38fr, 13pt, 1.56fr, 13pt, 1.02fr),
    align: (left + horizon, center + horizon, left + horizon, center + horizon, left + horizon, center + horizon, left + horizon),
    arch-box(
      "1. Keystroke & Context",
      "IMK Swift 6",
      [
        • Raw ASCII keys $bold(k)_(1:n)$\
        • Sliding left window $bold(c)_L$\
        • Host bundle ID $a$\
        • Secure-input guard
      ],
      bg-soft,
      muted,
    ),
    text(size: 11pt, weight: "bold", fill: muted)[$arrow.r$],
    arch-box(
      "2. Stage-1 Zero-Copy Lattice",
      "< 0.8 ms · mmap",
      [
        • Unified-Zipf `CHIMED02` `.imedict` stack\
        • Prefix-diverse beam & clitic cohesion\
        • Dynamic fuzzy + 4 motor-typo operators\
        • Modeless EN + $O(log C)$ `Tab` chains\
        • *Measure split*: Exact $cal(C)_n$ vs. lookahead $cal(C)_"comp"$
      ],
      teal-soft,
      teal,
    ),
    text(size: 11pt, weight: "bold", fill: teal)[$arrow.r$],
    arch-box(
      "3. Stage-2 Metal SLM + Multi-LoRA",
      "0.86–1.41 ms · Unified Memory",
      [
        • Homophone-distilled 4-bit `Qwen3-6L` (*64.17 MB*, $V = "24,576"$)\
        • Batched `[B, L]` broadcast `MlxPrefixCache` + BPE partition prior\
        • *Q4-Aligned Rank-Stacked LoRA* ($r=8$, *2.07 MB*):\
          $Delta bold(y) = ((bold(x) bold(A)_"cat"^top) circle.tiny bold(g)(bold(c)_L, a)) bold(B)_"cat"^top$\
        • Couplet-aware simplex router: $sum_k gamma_k <= 0.90$
      ],
      blue-soft,
      blue,
    ),
    text(size: 11pt, weight: "bold", fill: blue)[$arrow.r$],
    arch-box(
      "4. Anti-Jitter & UI",
      "60 ms Freeze",
      [
        • Frame-0 sync budget ($<= 12$ ms)\
        • Slot-\#1 freeze if $> 60$ ms\
        • SQLite WAL habit & backspace decay\
        • `Tab` verse chaining
      ],
      emerald-soft,
      emerald,
    ),
  )
  #v(6pt)
  #block(
    width: 100%,
    fill: amber-soft,
    stroke: (thickness: 0.6pt, paint: amber, dash: "dashed"),
    radius: 4pt,
    inset: (x: 8pt, y: 4.5pt),
  )[
    #set par(justify: true, leading: 0.44em)
    #text(size: 7.2pt, fill: ink)[
      *Pluggable `.imepack` Cartridge Co-Mounting*: Installing a cartridge (`classics-poetry`, `tech-developer`, or air-gapped `personal-style`) _simultaneously_ mounts `pack.imedict` into Stage-1 (resolving lexical OOV & `Tab` chains) and concatenates `lora.safetensors` into Stage-2's rank-stacked matrices $bold(A)_"cat"^top, bold(B)_"cat"^top$ without duplicating the 64.17 MB base model.
    ]
  ]
]

// Pure-Typst vector line chart helper using `place` + `line` + `circle`.
#let fig-scaling-and-transfer() = {
  set par(justify: false, leading: 0.42em)
  show raw: set text(font: ("PT Mono", "JetBrains Mono NL", "DejaVu Sans Mono"), size: 0.92em)
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
        #text(size: 7.8pt, weight: "bold", fill: ink)[(a) 6-Candidate Latency (ms) vs. Mounted LoRAs ($K$)]
      ]
      #v(4pt)
      #let w = 205pt
      #let h = 114pt
      #align(center)[
        #box(width: w + 28pt, height: h + 22pt)[
          // Y-axis gridlines (0, 3, 6, 9, 12 ms)
          #for (val, lbl) in ((0.0, "0"), (3.0, "3"), (6.0, "6"), (9.0, "9"), (12.0, "12")) {
            let y = h - (val / 12.0) * h
            place(dx: 24pt, dy: y, line(length: w, stroke: 0.4pt + rgb("#e4e4e7")))
            place(dx: 0pt, dy: y - 4pt, box(width: 20pt, align(right, text(size: 6.5pt, fill: muted)[#lbl])))
          }
          // 10 ms perceptual budget line
          #let y10 = h - (10.0 / 12.0) * h
          #place(dx: 24pt, dy: y10, line(length: w, stroke: (thickness: 0.7pt, paint: amber, dash: "dashed")))
          #place(dx: 28pt, dy: y10 - 9pt, text(size: 6.2pt, fill: amber, weight: "bold")[10 ms perceptual budget])

          // Axes
          #place(dx: 24pt, dy: 0pt, line(end: (0pt, h), stroke: 0.7pt + ink))
          #place(dx: 24pt, dy: h, line(length: w, stroke: 0.7pt + ink))

          // X-axis ticks K = 0..6 (exact v0.8.0 Apple Silicon Metal measurements)
          #let unfused = ((0, 1.411), (1, 2.907), (2, 4.494), (3, 5.973), (4, 7.525), (5, 9.143), (6, 10.741))
          #let fused   = ((0, 1.405), (1, 2.948), (2, 2.981), (3, 3.050), (4, 3.030), (5, 3.106), (6, 3.105))
          #let batched = ((0, 0.846), (1, 1.216), (2, 1.340), (3, 1.357), (4, 1.375), (5, 1.403), (6, 1.413))

          #for k in range(0, 7) {
            let x = 24pt + (k / 6.0) * w
            place(dx: x - 3pt, dy: h + 4pt, text(size: 6.5pt, fill: muted)[#str(k)])
          }
          #place(dx: 24pt, dy: h + 13pt, box(width: w, align(center, text(size: 6.8pt, fill: ink)[Mounted Rank-8 Adapters ($K$)])))

          // Unfused polyline (crimson)
          #for i in range(0, 6) {
            let (k1, v1) = unfused.at(i)
            let (k2, v2) = unfused.at(i + 1)
            let x1 = 24pt + (k1 / 6.0) * w
            let y1 = h - (v1 / 12.0) * h
            let x2 = 24pt + (k2 / 6.0) * w
            let y2 = h - (v2 / 12.0) * h
            place(dx: 0pt, dy: 0pt, line(start: (x1, y1), end: (x2, y2), stroke: 1.3pt + crimson))
          }
          #for (k, v) in unfused {
            let x = 24pt + (k / 6.0) * w
            let y = h - (v / 12.0) * h
            place(dx: x - 2.1pt, dy: y - 2.1pt, circle(radius: 2.1pt, fill: crimson))
          }

          // Rank-Stacked Only polyline (blue, dashed)
          #for i in range(0, 6) {
            let (k1, v1) = fused.at(i)
            let (k2, v2) = fused.at(i + 1)
            let x1 = 24pt + (k1 / 6.0) * w
            let y1 = h - (v1 / 12.0) * h
            let x2 = 24pt + (k2 / 6.0) * w
            let y2 = h - (v2 / 12.0) * h
            place(dx: 0pt, dy: 0pt, line(start: (x1, y1), end: (x2, y2), stroke: (thickness: 1.1pt, paint: blue, dash: "densely-dashed")))
          }
          #for (k, v) in fused {
            let x = 24pt + (k / 6.0) * w
            let y = h - (v / 12.0) * h
            place(dx: x - 2.0pt, dy: y - 2.0pt, rect(width: 4.0pt, height: 4.0pt, fill: blue))
          }

          // Batched KV-Gather + Rank-Stacked polyline (emerald, solid)
          #for i in range(0, 6) {
            let (k1, v1) = batched.at(i)
            let (k2, v2) = batched.at(i + 1)
            let x1 = 24pt + (k1 / 6.0) * w
            let y1 = h - (v1 / 12.0) * h
            let x2 = 24pt + (k2 / 6.0) * w
            let y2 = h - (v2 / 12.0) * h
            place(dx: 0pt, dy: 0pt, line(start: (x1, y1), end: (x2, y2), stroke: 1.5pt + emerald))
          }
          #for (k, v) in batched {
            let x = 24pt + (k / 6.0) * w
            let y = h - (v / 12.0) * h
            place(dx: x - 2.2pt, dy: y - 2.2pt, circle(radius: 2.2pt, fill: emerald))
          }

          // Annotations
          #place(dx: 48pt, dy: 25pt, text(size: 6.2pt, weight: "bold", fill: crimson)[Unfused Sequential (10.74 ms)])
          #place(dx: 88pt, dy: h - (3.11 / 12.0) * h - 10.5pt, text(size: 6.1pt, weight: "bold", fill: blue)[Rank-Stacked Only (3.11 ms)])
          #place(dx: 56pt, dy: h - (1.41 / 12.0) * h + 4.5pt, text(size: 6.2pt, weight: "bold", fill: emerald)[Batched KV + Rank-Stacked (1.41 ms, 7.60×)])
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
        #text(size: 7.8pt, weight: "bold", fill: ink)[(b) Classical Validation NLL (nats) vs. LoRA Scale ($gamma$)]
      ]
      #v(4pt)
      #let w = 205pt
      #let h = 114pt
      #let ymin = 5.66
      #let ymax = 6.16
      #align(center)[
        #box(width: w + 30pt, height: h + 22pt)[
          // Shaded Simplex sweet spot [0.40, 0.75]
          #let sx1 = 26pt + 0.40 * w
          #let sx2 = 26pt + 0.75 * w
          #place(dx: sx1, dy: 0pt, rect(width: sx2 - sx1, height: h, fill: emerald-soft))
          #place(dx: sx1 + 4pt, dy: 4pt, text(size: 6.0pt, weight: "bold", fill: emerald)[Calibrated router zone])

          #for (val, lbl) in ((5.70, "5.70"), (5.80, "5.80"), (5.90, "5.90"), (6.00, "6.00"), (6.10, "6.10")) {
            let y = h - ((val - ymin) / (ymax - ymin)) * h
            place(dx: 26pt, dy: y, line(length: w, stroke: 0.4pt + rgb("#e4e4e7")))
            place(dx: 0pt, dy: y - 4pt, box(width: 22pt, align(right, text(size: 6.3pt, fill: muted)[#lbl])))
          }

          // Base NLL horizontal reference (6.041)
          #let ybase = h - ((6.041 - ymin) / (ymax - ymin)) * h
          #place(dx: 26pt, dy: ybase, line(length: w, stroke: (thickness: 0.6pt, paint: muted, dash: "dotted")))
          #place(dx: 30pt, dy: ybase - 9pt, text(size: 6.0pt, fill: muted)[4-bit Base ($gamma = 0$: 6.041)])

          // Axes
          #place(dx: 26pt, dy: 0pt, line(end: (0pt, h), stroke: 0.7pt + ink))
          #place(dx: 26pt, dy: h, line(length: w, stroke: 0.7pt + ink))

          #for (g, lbl) in ((0.0, "0.0"), (0.2, "0.2"), (0.4, "0.4"), (0.6, "0.6"), (0.8, "0.8"), (1.0, "1.0")) {
            let x = 26pt + g * w
            place(dx: x - 5pt, dy: h + 4pt, text(size: 6.5pt, fill: muted)[#lbl])
          }
          #place(dx: 26pt, dy: h + 13pt, box(width: w, align(center, text(size: 6.8pt, fill: ink)[Effective Adapter Scale ($gamma$)])))

          #let pts_fp16 = (
            (0.0, 6.0410),
            (0.2, 5.8973),
            (0.35, 5.8719),
            (0.5, 5.8694),
            (0.65, 5.8911),
            (0.8, 5.9523),
            (1.0, 6.1165),
          )
          #let pts_q4 = (
            (0.0, 6.0410),
            (0.2, 5.8545),
            (0.35, 5.7972),
            (0.5, 5.7595),
            (0.65, 5.7399),
            (0.8, 5.7521),
            (1.0, 5.8427),
          )
          #for i in range(0, pts_fp16.len() - 1) {
            let (g1, n1) = pts_fp16.at(i)
            let (g2, n2) = pts_fp16.at(i + 1)
            let x1 = 26pt + g1 * w
            let y1 = h - ((n1 - ymin) / (ymax - ymin)) * h
            let x2 = 26pt + g2 * w
            let y2 = h - ((n2 - ymin) / (ymax - ymin)) * h
            place(dx: 0pt, dy: 0pt, line(start: (x1, y1), end: (x2, y2), stroke: (thickness: 1.2pt, paint: blue, dash: "densely-dashed")))
          }
          #for (g, n) in pts_fp16 {
            let x = 26pt + g * w
            let y = h - ((n - ymin) / (ymax - ymin)) * h
            place(dx: x - 2.0pt, dy: y - 2.0pt, rect(width: 4.0pt, height: 4.0pt, fill: blue))
          }

          #for i in range(0, pts_q4.len() - 1) {
            let (g1, n1) = pts_q4.at(i)
            let (g2, n2) = pts_q4.at(i + 1)
            let x1 = 26pt + g1 * w
            let y1 = h - ((n1 - ymin) / (ymax - ymin)) * h
            let x2 = 26pt + g2 * w
            let y2 = h - ((n2 - ymin) / (ymax - ymin)) * h
            place(dx: 0pt, dy: 0pt, line(start: (x1, y1), end: (x2, y2), stroke: 1.5pt + emerald))
          }
          #for (g, n) in pts_q4 {
            let x = 26pt + g * w
            let y = h - ((n - ymin) / (ymax - ymin)) * h
            place(dx: x - 2.2pt, dy: y - 2.2pt, circle(radius: 2.2pt, fill: emerald))
          }

          #place(dx: 26pt + 0.76 * w, dy: 9pt, text(size: 5.9pt, weight: "bold", fill: crimson)[FP16 overshoot\ (6.117 at $gamma=1$)])
          #place(dx: 26pt + 0.27 * w, dy: h - ((5.8694 - ymin) / (ymax - ymin)) * h - 11pt, text(size: 6.0pt, weight: "bold", fill: blue)[FP16 Zero-Shot (5.869)])
          #place(dx: 26pt + 0.34 * w, dy: h - ((5.7399 - ymin) / (ymax - ymin)) * h + 5.5pt, text(size: 6.1pt, weight: "bold", fill: emerald)[Q4-Aligned Min: 5.740 ($-0.301$)])
        ]
      ]
    ],
  )
}
