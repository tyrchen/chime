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
  #text(size: 8pt, weight: "bold", fill: stroke-c)[#title]
  #if subtitle != "" [
    #h(4pt)
    #text(size: 6.8pt, fill: muted)[(#subtitle)]
  ]
  #v(2.5pt)
  #text(size: 7.2pt, fill: ink)[#body]
]

#let fig-architecture() = block(
  width: 100%,
  stroke: 0.6pt + border-col,
  radius: 6pt,
  inset: 10pt,
  fill: white,
)[
  #grid(
    columns: (1.05fr, 14pt, 1.35fr, 14pt, 1.55fr, 14pt, 1.05fr),
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
        • Layered `CHIMED02` `.imedict` stack\
        • Dynamic fuzzy syllable algebra\
        • Spatial, swap & repeat typo edges\
        • Modeless EN + $O(log C)$ `Tab` chains\
        • *Measure split*: Exact pool $cal(C)_n$ vs. lookahead pool $cal(C)_"comp"$
      ],
      teal-soft,
      teal,
    ),
    text(size: 11pt, weight: "bold", fill: teal)[$arrow.r$],
    arch-box(
      "3. Stage-2 Metal SLM + Multi-LoRA",
      "1.4–3.0 ms · Unified Memory",
      [
        • Frozen 4-bit `Qwen3-6L` (*64.17 MB*, $V=24,576$)\
        • Persistent GPU `MlxPrefixCache` on $bold(c)_L$\
        • *Rank-Stacked Sidecar* ($r=8$, *2.07 MB* per pack):\
          $Delta bold(y) = ((bold(x) bold(A)_"cat"^top) circle.tiny bold(g)(bold(c)_L, a)) bold(B)_"cat"^top$\
        • Simplex-gated router: $sum_k gamma_k <= 0.90$
      ],
      blue-soft,
      blue,
    ),
    text(size: 11pt, weight: "bold", fill: blue)[$arrow.r$],
    arch-box(
      "4. Anti-Jitter & UI",
      "60 ms Freeze",
      [
        • Frame-0 sync budget ($<= 10$ ms)\
        • Slot-#1 freeze if $> 60$ ms\
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
    stroke: ( thickness: 0.6pt, paint: amber, dash: "dashed" ),
    radius: 4pt,
    inset: (x: 8pt, y: 4.5pt),
  )[
    #text(size: 7.2pt, fill: ink)[
      *Pluggable `.imepack` Cartridge Co-Mounting*: Installing a cartridge (`classics-poetry`, `tech-developer`, or air-gapped `personal-style`) *simultaneously* mounts `pack.imedict` into Stage-1 (resolving lexical OOV & `Tab` chains) and concatenates `lora.safetensors` into Stage-2's rank-stacked matrices $bold(A)_"cat"^top, bold(B)_"cat"^top$ without duplicating the 64.17 MB base model.
    ]
  ]
]

// Pure-Typst vector line chart helper using `place` + `line` + `circle`.
#let fig-scaling-and-transfer() = grid(
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
      #text(size: 7.8pt, weight: "bold", fill: ink)[(a) 6-Candidate Rerank Latency vs. Mounted LoRAs ($K$)]
    ]
    #v(4pt)
    #let w = 205pt
    #let h = 112pt
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
        #place(dx: 28pt, dy: y10 - 9pt, text(size: 6.2pt, fill: amber, weight: "bold")[10 ms frame-0 budget])

        // Axes
        #place(dx: 24pt, dy: 0pt, line(end: (0pt, h), stroke: 0.7pt + ink))
        #place(dx: 24pt, dy: h, line(length: w, stroke: 0.7pt + ink))

        // X-axis ticks K = 0..6
        #let unfused = ((0, 1.424), (1, 2.928), (2, 4.504), (3, 6.002), (4, 7.550), (5, 8.986), (6, 10.468))
        #let fused   = ((0, 1.428), (1, 2.948), (2, 3.008), (3, 3.014), (4, 3.030), (5, 3.094), (6, 3.101))

        #for k in range(0, 7) {
          let x = 24pt + (k / 6.0) * w
          place(dx: x - 4pt, dy: h + 4pt, text(size: 6.5pt, fill: muted)[#str(k)])
        }
        #place(dx: 24pt + w / 2 - 42pt, dy: h + 13pt, text(size: 6.8pt, fill: ink)[Mounted Rank-8 Adapters ($K$)])

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
          place(dx: x - 2.2pt, dy: y - 2.2pt, circle(radius: 2.2pt, fill: crimson))
        }

        // Fused polyline (teal)
        #for i in range(0, 6) {
          let (k1, v1) = fused.at(i)
          let (k2, v2) = fused.at(i + 1)
          let x1 = 24pt + (k1 / 6.0) * w
          let y1 = h - (v1 / 12.0) * h
          let x2 = 24pt + (k2 / 6.0) * w
          let y2 = h - (v2 / 12.0) * h
          place(dx: 0pt, dy: 0pt, line(start: (x1, y1), end: (x2, y2), stroke: 1.4pt + teal))
        }
        #for (k, v) in fused {
          let x = 24pt + (k / 6.0) * w
          let y = h - (v / 12.0) * h
          place(dx: x - 2.2pt, dy: y - 2.2pt, rect(width: 4.4pt, height: 4.4pt, fill: teal))
        }

        // Annotations
        #place(dx: 92pt, dy: 24pt, text(size: 6.4pt, weight: "bold", fill: crimson)[Unfused Sequential (10.47 ms)])
        #place(dx: 88pt, dy: h - (3.10 / 12.0) * h + 5pt, text(size: 6.4pt, weight: "bold", fill: teal)[Rank-Stacked Fused (3.10 ms, 3.38×)])
      ]
    ]
  ],

  // Panel (b): Zero-Shot FP16 -> INT4 LoRA Transfer Scale Curve
  block(
    width: 100%,
    stroke: 0.6pt + border-col,
    radius: 5pt,
    inset: 8pt,
    fill: white,
  )[
    #align(center)[
      #text(size: 7.8pt, weight: "bold", fill: ink)[(b) Zero-Shot FP16$arrow.r$INT4 LoRA Scale Curve (NLL)]
    ]
    #v(4pt)
    #let w = 205pt
    #let h = 112pt
    #let ymin = 5.78
    #let ymax = 6.10
    #align(center)[
      #box(width: w + 30pt, height: h + 22pt)[
        // Shaded Simplex sweet spot [0.35, 0.55]
        #let sx1 = 26pt + 0.35 * w
        #let sx2 = 26pt + 0.55 * w
        #place(dx: sx1, dy: 0pt, rect(width: sx2 - sx1, height: h, fill: emerald-soft))
        #place(dx: sx1 + 2pt, dy: 5pt, text(size: 6.0pt, weight: "bold", fill: emerald)[Simplex\ sweet spot])

        #for (val, lbl) in ((5.80, "5.80"), (5.90, "5.90"), (6.00, "6.00"), (6.10, "6.10")) {
          let y = h - ((val - ymin) / (ymax - ymin)) * h
          place(dx: 26pt, dy: y, line(length: w, stroke: 0.4pt + rgb("#e4e4e7")))
          place(dx: 0pt, dy: y - 4pt, box(width: 22pt, align(right, text(size: 6.3pt, fill: muted)[#lbl])))
        }

        // Base NLL horizontal reference (5.978)
        #let ybase = h - ((5.978 - ymin) / (ymax - ymin)) * h
        #place(dx: 26pt, dy: ybase, line(length: w, stroke: (thickness: 0.6pt, paint: muted, dash: "dotted")))
        #place(dx: 30pt, dy: ybase - 9pt, text(size: 6.1pt, fill: muted)[4-bit Base ($gamma = 0$: 5.978)])

        // Axes
        #place(dx: 26pt, dy: 0pt, line(end: (0pt, h), stroke: 0.7pt + ink))
        #place(dx: 26pt, dy: h, line(length: w, stroke: 0.7pt + ink))

        #for (g, lbl) in ((0.0, "0.0"), (0.2, "0.2"), (0.4, "0.4"), (0.6, "0.6"), (0.8, "0.8"), (1.0, "1.0")) {
          let x = 26pt + g * w
          place(dx: x - 5pt, dy: h + 4pt, text(size: 6.5pt, fill: muted)[#lbl])
        }
        #place(dx: 26pt + w / 2 - 42pt, dy: h + 13pt, text(size: 6.8pt, fill: ink)[Effective Adapter Scale ($gamma$)])

        #let pts = (
          (0.0, 5.9780),
          (0.2, 5.8416),
          (0.35, 5.8178),
          (0.5, 5.8174),
          (0.65, 5.8407),
          (0.8, 5.9042),
          (1.0, 6.0707),
        )
        #for i in range(0, pts.len() - 1) {
          let (g1, n1) = pts.at(i)
          let (g2, n2) = pts.at(i + 1)
          let x1 = 26pt + g1 * w
          let y1 = h - ((n1 - ymin) / (ymax - ymin)) * h
          let x2 = 26pt + g2 * w
          let y2 = h - ((n2 - ymin) / (ymax - ymin)) * h
          place(dx: 0pt, dy: 0pt, line(start: (x1, y1), end: (x2, y2), stroke: 1.4pt + blue))
        }
        #for (g, n) in pts {
          let x = 26pt + g * w
          let y = h - ((n - ymin) / (ymax - ymin)) * h
          place(dx: x - 2.3pt, dy: y - 2.3pt, circle(radius: 2.3pt, fill: blue))
        }
        #let xmin_opt = 26pt + 0.5 * w
        #let ymin_opt = h - ((5.8174 - ymin) / (ymax - ymin)) * h
        #place(dx: xmin_opt - 24pt, dy: ymin_opt + 5pt, text(size: 6.3pt, weight: "bold", fill: blue)[Min 5.817 ($gamma = 0.50$)])
        #place(dx: 26pt + 0.72 * w, dy: 12pt, text(size: 6.1pt, fill: crimson)[INT4 overshoot\ ($gamma = 1.0$: 6.071)])
      ]
    ]
  ],
)
