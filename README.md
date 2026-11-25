# Chime

A 100% offline, zero-telemetry, sub-3ms local neural Chinese Pinyin Input Method Engine for **macOS 15+ (Apple Silicon)**, built with a **Rust core**, **Swift 6 UI**, **4-bit MLX Metal `Qwen3-6L` base model (`64.2 MB`)**, and **hot-swappable `.imepack` (`Multi-LoRA` + zero-copy `.imedict`) cartridges**.

## Highlights (`v0.6.0`)

1. **Pinyin, Shuangpin, Super-Abbreviation, Typo Recovery & Zero-Bloat Fuzzy Pinyin**:
   - Full Pinyin, initial skeletons (`wsm` -> `为什么`, `rgzn` -> `人工智能`), and **Xiaohe / Ziranma / Microsoft Shuangpin**.
   - Dynamic fuzzy Pinyin (`z/zh`, `c/ch`, `s/sh`, `n/l`, `f/h`, `r/l`, `an/ang`, `en/eng`, `in/ing`, `ian/iang`, `uan/uang`) plus QWERTY neighbor, transposition (`shaungpin -> 双拼`, `nianquuqqianliyanbo -> 念去去，千里烟波`), and repeated-keystroke recovery (`mohuyign -> 模糊音`).
   - Clean single-syllable Top-5 curation (`gen -> 1. 跟 2. 根 3. 根据 4. 根本 5. 更`) with rare Unihan characters tucked into the `5x5` expandable grid (`↓`).
2. **Modeless Chinese-English Code-Switching**:
   - Simultaneous Mandarin + English/code identifier decoding (`macOS`, `GitHub`, `TypeScript`, `useEffect`, `PR`, `Kubernetes` prefix completion) with automatic Pangu spacing.
3. **Zero-Copy `mmap` Lexicon (`CHIMED02`) & Full-Poem `Tab` Continuation Chains**:
   - Compiles 620,000+ Mandarin & English entries, teacher-scored Bigram/Trigram transitions, and $O(\log C)$ `Tab` verse continuation chains into `core.imedict`, memory-mapped in `< 1 ms`.
   - Supports end-to-end classical poetry & prose `Tab` chaining (e.g., 《将进酒》《水调歌头》《雨霖铃》《滕王阁序》《岳阳楼记》《赤壁赋》) plus context-aware next-verse promotion and anti-repeat scoring (`君不见黄河之水天上来，奔流到海不复回。` + `junbujian` -> `#1 君不见高堂明镜悲白发`).
   - Native pure-Rust importers for **Sogou `.scel`**, **Baidu `.bdict`**, **Rime `.dict.yaml`**, **TSV**, and **macOS System Text Replacements** (`TextReplacements.db`), backed by a `SQLite` (`WAL`) user habit store.
4. **Frozen 64 MB 4-bit MLX `Qwen3-6L` Base + Pluggable `Multi-LoRA` `.imepack` Cartridges**:
   - 6-layer layer-distilled **`Qwen/Qwen3-0.6B-Base`** student (`24,576` compact BPE vocab, `head_dim=128`, QK-Norm, 4-bit affine quantized `64.17 MB`) running natively on Apple Silicon Metal via **`mlx-rs 0.34`** (`0.72 ms` warm 1-step Metal latency, `p50 ~ 2.95 ms` end-to-end).
   - **Multi-LoRA Sidecar**: Mount multiple `~2.06 MB` Rank-8 FP16 `lora.safetensors` + `pack.imedict` `.imepack` cartridges (`classics-poetry`, `tech-developer`, or strictly private personal style packs) with dynamic per-app and context routing (`< 0.04 ms` overhead) while sharing a single frozen base model in Unified Memory.
5. **Release Installer & GitHub Pages Hub (`site/`)**:
   - `make release-artifacts` builds `Chime-v0.6.0-macos-arm64.pkg`, `Chime-v0.6.0-macos-arm64.zip`, `classics-poetry.imepack.zip`, `tech-developer.imepack.zip`, and `SHA256SUMS.txt`.
   - Static landing page & `.imepack` download hub in [`site/`](site/index.html), paired with `.github/workflows/pages.yml` and `.github/workflows/release.yml`.

## Quick Start

```bash
# Run formatting, strict clippy pedantic lints, and all unit/doc tests
make lint && make test

# Build core.imedict and run the 42-case accuracy + latency benchmark suite on Metal GPU
make eval

# Build, sign, and install ~/Library/Input Methods/Chime.app
make install-macos

# Build macOS .pkg/.zip installer and public .imepack.zip cartridges
make release-artifacts

# Install or list .imepack cartridges via CLI
cargo run --release -p chime-cli -- pack install dist/releases/classics-poetry.imepack.zip
cargo run --release -p chime-cli -- pack list
```

## License

Dual-licensed under `MIT OR Apache-2.0` ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)). Third-party open lexical attributions are listed in [NOTICE](NOTICE).
