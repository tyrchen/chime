# Chime (Chime 拼音)

<p>
  <a href="README.md">简体中文</a> · <strong>English</strong> · <a href="https://tyrchen.github.io/chime/">Website &amp; Live Demo</a> · <a href="https://tyrchen.github.io/chime/chime-paper.pdf">Paper (PDF)</a> · <a href="https://github.com/tyrchen/chime/releases/latest">Releases</a>
</p>

[![Release](https://img.shields.io/github/v/release/tyrchen/chime?label=Release&color=0f4c5c)](https://github.com/tyrchen/chime/releases/latest)
[![Platform](https://img.shields.io/badge/macOS-15%2B%20Apple%20Silicon-111827)](https://tyrchen.github.io/chime/)
[![Network](https://img.shields.io/badge/Network%20%26%20Telemetry-0%20Bytes%20(100%25%20Offline)-047857)](#privacy--security)
[![Paper](https://img.shields.io/badge/Paper-PDF%20Preprint-be123c)](https://tyrchen.github.io/chime/chime-paper.pdf)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

**Chime** is a 100% offline, zero-telemetry, sub-millisecond neural Chinese Pinyin Input Method Editor (IME) for **macOS 15+ on Apple Silicon**, written in **Rust** and **Swift 6**.

It combines a **$< 0.8\text{ ms}$ zero-copy `mmap` dual-track lattice decoder (`CHIMED02`)** with a **manifold-distilled `64.17 MB` 6-layer 4-bit MLX `Qwen3` parallel rescorer** and hot-swappable **Multi-LoRA `.imepack` cartridges** in Apple Silicon Unified Memory.

- **Website & Interactive Live Demo**: [https://tyrchen.github.io/chime/](https://tyrchen.github.io/chime/)
- **Research Paper (PDF)**: [*Chime: Sub-3ms On-Device Neural Pinyin Input via Manifold-Aligned Distillation and Rank-Stacked Multi-LoRA Cartridges*](https://tyrchen.github.io/chime/chime-paper.pdf) ([Typst & arXiv LaTeX sources in `paper/`](paper/))
- **Latest macOS Installer & Cartridges (`v0.8.0`)**: [GitHub Releases](https://github.com/tyrchen/chime/releases/latest)

---

## Download & Installation (`v0.8.0`)

Requires **macOS 15.0 (Sequoia) or newer** on **Apple Silicon (`M1` / `M2` / `M3` / `M4`)**.

| Artifact | Direct Download | Size | Notes |
| :--- | :--- | ---: | :--- |
| **macOS Installer (`.pkg`)** | [**`Chime-v0.8.0-macos-arm64.pkg`**](https://github.com/tyrchen/chime/releases/latest/download/Chime-macos-arm64.pkg) | `~75 MB` | Installs to `/Library/Input Methods/Chime.app`, registers TIS without logout & opens Input Sources |
| **Portable App Bundle (`.zip`)** | [**`Chime-v0.8.0-macos-arm64.zip`**](https://github.com/tyrchen/chime/releases/latest/download/Chime-macos-arm64.zip) | `~75 MB` | Unzip, drag `Chime.app` into `~/Library/Input Methods/`, and open once |
| **Classical Poetry & Guwen Pack** | [**`classics-poetry.imepack.zip`**](https://github.com/tyrchen/chime/releases/latest/download/classics-poetry.imepack.zip) | `16.8 MB` | `266,816` classical entries + `231,593` `Tab` chains + Q4 Rank-8 LoRA |
| **Developer Bilingual Pack** | [**`tech-developer.imepack.zip`**](https://github.com/tyrchen/chime/releases/latest/download/tech-developer.imepack.zip) | `1.9 MB` | Systems/AI/Cloud bilingual lexicon + exact casing + Q4 Rank-8 LoRA |
| **SHA-256 Checksums** | [**`SHA256SUMS.txt`**](https://github.com/tyrchen/chime/releases/latest/download/SHA256SUMS.txt) | `377 B` | Release integrity checksums |

> **Apple Signed & Notarized (`Notarized Developer ID`)**: Both the `.pkg` installer and `.zip` app bundle are signed with **Apple Developer ID (Hardened Runtime + Secure Timestamp), notarized by Apple Notary Service, and offline-stapled**—double-click to install with zero macOS Gatekeeper warnings. After installation, open **System Settings -> Keyboard -> Input Sources -> Edit -> `+` -> Simplified Chinese** to confirm **Chime 拼音** is enabled.

### Installing `.imepack` Cartridges

You can hot-mount downloaded `.imepack.zip` cartridges with **zero restart**:
- **Via Menu Bar UI**: Click `[中]` in the macOS menu bar -> **Settings (`偏好设置`)** -> **Import Dictionary or `.imepack`**, and select `classics-poetry.imepack.zip` or `tech-developer.imepack.zip`.
- **Via CLI**:
  ```bash
  chime pack install ~/Downloads/classics-poetry.imepack.zip
  chime pack install ~/Downloads/tech-developer.imepack.zip
  chime pack list
  ```
  Cartridges are installed into `~/Library/Application Support/Chime/packs/` and shared across the single `64.17 MB` base model in Unified Memory.

---

## Key Capabilities

1. **Full Pinyin, Shuangpin, Skeletons & 4-Operator Motor-Slip Recovery**:
   - Full Pinyin, initial abbreviations (`wsm` -> `为什么`, `rgzn` -> `人工智能`), and **Xiaohe / Ziranma / Microsoft Shuangpin**.
   - Zero-bloat algebraic fuzzy Pinyin (`z/zh`, `c/ch`, `s/sh`, `n/l`, `f/h`, `r/l`, `an/ang`, `en/eng`, `in/ing`, `ian/iang`, `uan/uang`) evaluated dynamically without dictionary explosion.
   - Four 1-edit motor-slip operators: **adjacent swap** (`shaungpin -> 双拼`, `nianquuqqianliyanbo -> 念去去，千里烟波`), **QWERTY Euclidean neighbor** (`shuruga -> 输入法`), **accidental key-repeat** (`mohuyign -> 模糊音`), and **geminate boundary consonant elision** (`chunuanhuakai -> 春暖花开`, `bujinchangjiangungunlai -> 不尽长江滚滚来`).
   - Clean single-syllable Top-5 curation (`gen -> 1. 跟 2. 根 3. 根据 4. 根本 5. 更`) with `35+` single-character fallbacks accessible via the expandable `5x5` grid (`↓`).
2. **Mainstream IME Ergonomics, Configurable `Shift` Toggle & Auto-Spacing**:
   - **Modeless Code-Switching + Configurable Inline `Shift` Mode**: Supports both seamless modeless mixed decoding (`woshiyongmacoshegithub` -> `我使用 macOS 和 GitHub`, `bangworeviewzhegepr` -> `帮我 review 这个 PR`, leading-uppercase direct English `Mac` -> `Mac` / `MacBook`, `kubern` -> `Kubernetes`) and Sogou/Doubao-compatible internal Chinese/English toggling (**Left/Right `Shift` by default**, configurable to Left Shift, Right Shift, `Caps Lock`, or off; configurable preedit `Shift` commit behavior and a subtle cursor-side `中 / 英` HUD toast).
   - **Configurable Symmetric Auto-Spacing (Default ON, User-Toggleable)**: Automatically inserts half-width spaces across Chinese↔English and Chinese↔number boundaries (`我有 100 个`, `使用 Rust 开发`), and can be toggled off anytime in Settings or the menu bar.
   - **Smart Punctuation & Paging Customization**: `⌃.` full/half-width punctuation toggle with automatic numeric decimal (`3.14`) and time colon (`12:30`) exemptions, customizable `- / =`, `[ / ]`, and `, / .` page keys, optional `; / '` 2nd/3rd candidate quick-select, and adjustable candidate font size (`14–19 pt`).
3. **Zero-Copy `CHIMED02` Lexicon & Full-Poem `Tab` Chaining**:
   - `660,000+` core Mandarin & English entries + sorted 16-byte FNV-1a continuation table mapped via `mmap` in `< 0.8 ms`.
   - Stateful `Tab` verse chaining across Tang poetry, Song Ci, and classical prose (e.g., typing `junbujianhuanghe` + `Tab` chains through all 24 clauses of Li Bai's *Qiang Jin Jiu* 《将进酒》, Su Shi's 《水调歌头》, Liu Yong's 《雨霖铃》), with context-guided next-verse promotion and verbatim anti-repeat scoring (`...奔流到海不复回。` + `junbujian` -> `#1 君不见高堂明镜悲白发`).
   - Native pure-Rust importers for **Sogou `.scel`**, **Baidu `.bdict`**, **Rime `.dict.yaml`**, **TSV**, and **macOS System Text Replacements**, backed by a `SQLite` (`WAL`) online habit & backspace negative-feedback store.
4. **Sub-Millisecond Batched Metal SLM + Q4-Aligned Multi-LoRA Sidecar**:
   - **Manifold-Aligned 6-Layer Student (`64.17 MB`)**: Distilled from 28-layer `Qwen/Qwen3-0.6B-Base` (`1.19 GB` FP16 -> `64.17 MB` 4-bit MLX, $18.6\times$ smaller) via dual-temperature KL, layer trajectory alignment, and pre-head cosine alignment ($\cos(\mathbf{u}_S, \mathbf{u}_T) = 0.853$), disambiguating semantic homophones (`依法保障` + `quanli` -> `权利` vs. `严禁滥用` + `quanli` -> `权力`; `4/5究竟有` + `duoda` -> `多大`).
   - **Batched Broadcast KV-Gather + Rank-Stacked GEMM Fusion**: Evaluates all multi-token candidates in a single right-padded `[B, L_max]` Metal pass over zero-copy broadcast `MlxPrefixCache` with on-GPU `take_along_axis` target gathering (`<= 128 B` GPU->CPU readout) and rank-concatenated LoRA projections $\Delta\mathbf{y} = ((\mathbf{x}\mathbf{A}_{\text{cat}}^\top)\odot\mathbf{g})\mathbf{B}_{\text{cat}}^\top$.
   - **Q4-Manifold Aligned LoRA & Couplet-Aware Simplex Router**: Adapters (`2.07 MB` Rank-8 FP16) are trained directly on the dequantized 4-bit runtime manifold and dynamically gated by host bundle ID, lexical cues, and symmetric `5+5` / `7+7` regulated-verse couplet cadence under $\sum_k \gamma_k \le 0.90$.

---

## Architecture & Measured Performance

```text
Keystrokes + Sliding 96-Char Context + Host Bundle ID
                       │
                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 1 · Zero-Copy Dual-Track Lattice (< 0.8 ms · mmap CHIMED02)     │
│  • Base core.imedict + mounted pack.imedict slices + SQLite WAL store │
│  • Dynamic fuzzy algebra + swap / neighbor / repeat / geminate edges  │
│  • Probability-Space Separation: Exact pool C_n vs. Completions C_cmp │
└──────────────────────┬────────────────────────────────────────────────┘
                       │ Top-6 equal-span homophone candidates
                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 2 · Batched Metal SLM + Multi-LoRA (0.86–1.45 ms · mlx-rs 0.34) │
│  • Frozen 6-layer 4-bit Qwen3-6L-Chime (64.17 MB, 24,576 BPE vocab)   │
│  • [B, L_max] broadcast MlxPrefixCache + on-GPU take_along_axis       │
│  • Rank-Stacked Q4-Aligned LoRA sidecars + Couplet Simplex Router     │
└──────────────────────┬────────────────────────────────────────────────┘
                       ▼
         60 ms Anti-Jitter Freeze Window & Swift 6 IMK Panel
```

| Configuration (Apple Silicon Metal · `T=16` Prefix + 6 Candidates) | Model / Sidecar Size | Stress Probes | Chime 44-Case Suite | Rerank-6 `p50` | Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Qwen3-0.6B-Base` (28L, 152K vocab, FP16) | `1,191.5 MB` | `93.8% (15/16)` | — | `~14.2 ms` | `1.0x` |
| `Qwen3-28L` (24K vocab, 4-bit MLX) | `249.95 MB` | `93.8% (15/16)` | `100% (44/44)` | `5.29 ms` | `2.7x` |
| `Qwen3-6L-NaiveSkip` (4-bit, unaligned) | `64.15 MB` | `43.8% (7/16)` | — | `2.33 ms` | — |
| **`Qwen3-6L-Chime` (4-bit, Batched KV-Gather, `K=0`)** | **`64.17 MB`** | **`93.8% (15/16)`** | **`100% (44/44)`** | **`0.86 ms`** | **`16.5x`** |
| **+ `K=3` Concurrent Rank-8 `.imepack` LoRAs (Fused + Batched)** | **`+2.07 MB` / pack** | **`93.8% (15/16)`** | **`100% (44/44)`** | **`1.45 ms`** | **`4.21x` vs unfused** |
| **+ `K=6` Concurrent Rank-8 `.imepack` LoRAs (Fused + Batched)** | **`+2.07 MB` / pack** | **`93.8% (15/16)`** | **`100% (44/44)`** | **`1.45 ms`** | **`7.46x` vs unfused** |

---

## Train an Air-Gapped Private Personal Style `.imepack`

You can fine-tune a strictly local personal style cartridge (`private_pack: true`) from your own historical Markdown essays or notes **100% locally on your Mac's Apple Silicon GPU (`mps`)** without uploading a single byte anywhere:

```bash
# 1. Extract sanitized paragraphs, personal words, and bigrams into an ephemeral local folder
#    (automatically strips URLs, emails, API tokens, code blocks, and YAML frontmatter)
uv run --python 3.12 --no-project --with jieba --with pypinyin \
  python tools/prepare_pack_corpora.py \
  --personal-posts ~/Notes/essays \
  --private-out /tmp/chime_private_personal

# 2. Train a 2.07 MB Q4-Manifold Aligned Rank-8 LoRA directly on Apple Silicon Metal (~45s)
uv run --python 3.12 --no-project --with torch --with safetensors --with transformers \
  --with datasets --with huggingface_hub --with mlx --with numpy \
  python tools/colab_train_loras.py \
  --q4-model-dir dist/base_neural \
  --vocab dist/base_neural/vocab.json \
  --private-txt /tmp/chime_private_personal/personal_lora_corpus.txt \
  --private-out-lora /tmp/chime_private_personal/personal_lora.safetensors \
  --steps 320

# 3. Package & install into ~/Library/Application Support/Chime/packs/, then scrub /tmp
cargo run --release -p chime-cli -- pack build /tmp/chime_private_personal/personal-style.imepack \
  --id personal-style \
  --name "Personal Writing Style" \
  --routing always \
  --lora-scale 0.75 \
  --private \
  --tsv /tmp/chime_private_personal/personal_lexicon.tsv \
  --ngrams /tmp/chime_private_personal/personal_ngrams.tsv \
  --lora /tmp/chime_private_personal/personal_lora.safetensors

cargo run --release -p chime-cli -- pack install /tmp/chime_private_personal/personal-style.imepack
rm -rf /tmp/chime_private_personal
```

---

## Workspace Structure & Building from Source

| Path | Description |
| :--- | :--- |
| `crates/phonetics` | Mandarin syllable algebra, Xiaohe/Ziranma/MS Shuangpin, dynamic fuzzy expansion, 4-operator motor-slip model, and code-switch phonotactics |
| `crates/lexicon` | Zero-copy `mmap` `CHIMED02` binary dictionary, $O(\log C)$ FNV-1a verse continuations, `.scel`/`.bdict`/Rime importers, and `SQLite` (`WAL`) user habit store |
| `crates/decoder` | Dual-track beam lattice decoder, probability-space separation, context next-verse promotion, and 60 ms anti-jitter freeze window |
| `crates/neural` | Apple Silicon Metal `mlx-rs 0.34` 4-bit `Qwen3` transformer, batched `[B, L_max]` `MlxPrefixCache` + `take_along_axis`, Rank-Stacked Multi-LoRA, and Couplet-Aware Simplex Router |
| `crates/core` | 32-slot LRU session state machine, smart punctuation, PII/secret sanitizer, and `.imepack` v2 cartridge manager |
| `crates/ffi` | Stateless C ABI boundary bridging Rust engine and Swift 6 host |
| `apps/cli` | `chime` CLI (`decode`, `dict-build-base`, `dict-compile`, `pack`, `corpus-export`, and 44-case `eval` suite) |
| `macos/` | Swift 6 `@MainActor` `InputMethodKit` app bundle (`Chime.app`), `5x5` candidate grid, and Settings UI |
| `paper/` | Research paper in Typst (`chime-paper.typ`, `figures.typ`), reproducible Metal ablation results (`experiment_results.json`), and arXiv LaTeX bundle (`paper/arxiv/`) |
| `site/` | Static GitHub Pages landing page, interactive `Tab` poetry demo, and `.imepack` download hub |

```bash
# Format check, strict clippy pedantic lints (-D warnings), and all unit + doc tests
make fmt && make lint && make test

# Run the 44-case end-to-end accuracy & Metal GPU latency benchmark suite
cargo run --release -p chime-cli -- eval --dict dist/core.imedict --neural dist/base_neural

# Build, ad-hoc sign, and install ~/Library/Input Methods/Chime.app
make install-macos

# Build release .pkg, .zip, public .imepack.zip cartridges, and SHA256SUMS.txt
make release-artifacts

# Compile research paper PDF (Typst) and arXiv pdflatex tarball
make paper-arxiv
```

---

## Privacy & Security

- **Zero Network Access**: `Chime.app` requests zero network entitlements and links zero HTTP/telemetry SDKs. Every keystroke, `SQLite` user habit, and `.imepack` adapter stays strictly on your Mac.
- **Hardware Password Bypass**: Automatically checks macOS `IsSecureEventInputEnabled()` and yields immediately whenever a password field, SSH passphrase prompt, or 1Password/Keychain dialog is focused.
- **Memory Safety**: `#![forbid(unsafe_code)]` enforced across all pure-logic Rust crates (`phonetics`, `decoder`, `neural`, `core`, `cli`), isolating `unsafe` solely to a read-only `mmap` wrapper in `lexicon` and the thin C FFI shim in `ffi`.

---

## Citation

If you find Chime, its manifold-aligned SLM distillation, or its Rank-Stacked Multi-LoRA `.imepack` design useful in your research or systems work, please cite:

```bibtex
@article{chen2026chime,
  title   = {Chime: Sub-3ms On-Device Neural Pinyin Input via Manifold-Aligned Distillation and Rank-Stacked Multi-LoRA Cartridges},
  author  = {Chen, Tian},
  journal = {Preprint},
  year    = {2026},
  url     = {https://github.com/tyrchen/chime}
}
```

## License

Dual-licensed under `MIT OR Apache-2.0` ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)). Third-party open lexical attributions are listed in [NOTICE](NOTICE).
