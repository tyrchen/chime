# Chime

A privacy-first, sub-10ms local neural Chinese Pinyin Input Method Engine for **macOS 15+ (Apple Silicon)**, built with a **Rust core** and a **Swift 6 UI**.

## Core Highlights

- **Full Pinyin, Super-Abbreviation, Shuangpin & Zero-Bloat Fuzzy Pinyin**: Supports Full Pinyin, initial skeletons (`zgr` -> `中国人`), Xiaohe / Ziranma / Microsoft Shuangpin, and configurable fuzzy syllable algebra (`z/zh`, `c/ch`, `s/sh`, `n/l`, `f/h`, `an/ang`, `en/eng`, `in/ing`, `ian/iang`, `uan/uang`) without inflating lexicon size.
- **Modeless Chinese-English Code-Switching**: Dual-track lattice simultaneously decodes Chinese syllables and English/code tokens (`macOS`, `GitHub`, `TypeScript`, `useEffect`, `PR`, `v0.1`) with per-app Pangu spacing rules.
- **Joint Typo Auto-Correction**: Models MacBook QWERTY key spatial distance and rapid two-hand keystroke transpositions (`shaung` -> `shuang`, `nohao` -> `nihao`) directly in the phonetic lattice with conservative gating.
- **Sub-10ms On-Device Reranking & Colab Personalization**: Designed for an 80M-parameter 24k-vocab distilled Causal Transformer (`mlx-rs` / macOS 15 `CoreML MLState` stateful KV-cache) with a 60ms anti-jitter freeze window, paired with local SQLite negative-feedback adaptation and a Google Colab LoRA training pipeline.
- **Permissive & Clean**: Engine licensed under `MIT OR Apache-2.0`, strictly separated from GPL dictionaries while natively importing Rime `.dict.yaml`, Sogou `.scel`, and TSV lexicons.

## Workspace Layout

```text
chime/
├── crates/
│   ├── phonetics/   # Syllable algebra, fuzzy expansion, Shuangpin, QWERTY typo & code-switch detector
│   ├── lexicon/     # Core dictionary, canonical English casing, .scel & Rime YAML importers, user store
│   ├── decoder/     # Dual-track error-tolerant lattice, Viterbi/Beam search, NeuralReranker trait
│   ├── core/        # Bounded LRU session pool, modifier state, Pangu spacing, TOML config
│   └── ffi/         # Narrow 8-function C-ABI staticlib bridge for Swift 6
├── apps/
│   └── cli/         # `chime` CLI for interactive decoding, dictionary inspection, and accuracy/latency eval
├── macos/           # macOS 15+ Swift 6 InputMethodKit host & custom non-activating NSPanel
└── colab/           # Google Colab LoRA personal model fine-tuning blueprint
```

## Quick Start

```bash
# Run unit & integration tests
make test

# Run built-in accuracy & latency benchmark suite
make eval

# Build Rust FFI staticlib + Swift 6 macOS 15 InputMethodKit bundle (dist/Chime.app)
make macos-bundle
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
