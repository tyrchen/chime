# Chime (Chime 拼音)

<p>
  <a href="README.md">简体中文</a> · <strong>English</strong> · <a href="https://tyrchen.github.io/chime/">Website &amp; Live Demo</a> · <a href="https://tyrchen.github.io/chime/chime-paper.pdf">Paper (PDF)</a> · <a href="https://github.com/tyrchen/chime/releases/latest">Releases</a>
</p>

[![Release](https://img.shields.io/github/v/release/tyrchen/chime?label=Release&color=0f4c5c)](https://github.com/tyrchen/chime/releases/latest)
[![Platform](https://img.shields.io/badge/macOS-15%2B%20Apple%20Silicon-111827)](https://tyrchen.github.io/chime/)
[![Network](https://img.shields.io/badge/Network%20%26%20Telemetry-0%20Bytes%20(100%25%20Offline)-047857)](#privacy--security)
[![Paper](https://img.shields.io/badge/Paper-PDF%20Preprint-be123c)](https://tyrchen.github.io/chime/chime-paper.pdf)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

**Chime** is a 100% offline, zero-telemetry neural Chinese Pinyin Input Method Editor (IME) for **macOS 15+ on Apple Silicon**, written in **Rust** and **Swift 6**.

In **`v0.14.0`**, Chime combines a **$< 0.8\text{ ms}$ zero-copy `mmap` lattice fast path (`CHIMED02`)** with **asynchronous Route A Pinyin-constrained autoregressive beam search** over an intact **28-layer 4-bit MLX `Qwen3-0.6B` softly KL-distilled from `Qwen3-14B-Base`** (`256.7 MB`, lossless `36,864` full-CJK vocab), accelerated **4.07×** on Metal with fused projections and paired with hot-swappable **28L Rank-8 Multi-LoRA `.imepack` cartridges** — achieving **`90.00%`** Acc@1 on the strictly blind 1,010-case holdout and **`79.70%`** on the strictly blind 10,000-sentence holdout.

- **Website & Interactive Live Demo**: [https://tyrchen.github.io/chime/](https://tyrchen.github.io/chime/)
- **Research Paper (PDF)**: [*Chime: Sub-3ms On-Device Neural Pinyin Input via Manifold-Aligned Distillation and Rank-Stacked Multi-LoRA Cartridges*](https://tyrchen.github.io/chime/chime-paper.pdf) ([Typst & arXiv LaTeX sources in `paper/`](paper/))
- **Latest macOS Installer & Cartridges (`v0.14.1`)**: [GitHub Releases](https://github.com/tyrchen/chime/releases/latest)

---

## Download & Installation (`v0.14.1`)

Requires **macOS 15.0 (Sequoia) or newer** on **Apple Silicon (`M1` / `M2` / `M3` / `M4`)**.

| Artifact | Direct Download | Size | Notes |
| :--- | :--- | ---: | :--- |
| **macOS Installer (`.pkg`)** | [**`Chime-v0.14.1-macos-arm64.pkg`**](https://github.com/tyrchen/chime/releases/latest/download/Chime-macos-arm64.pkg) | `257 MB` | Installs & migrates Chime into user-writable `~/Library/Input Methods/Chime.app` (cleaning legacy `/Library` copy) so all future in-app updates run **silently with zero passwords** |
| **Portable App Bundle (`.zip`)** | [**`Chime-v0.14.1-macos-arm64.zip`**](https://github.com/tyrchen/chime/releases/latest/download/Chime-macos-arm64.zip) | `257 MB` | Unzip, drag `Chime.app` into `~/Library/Input Methods/`, and open once |
| **Classical Poetry & Guwen Pack** | [**`classics-poetry.imepack.zip`**](https://github.com/tyrchen/chime/releases/latest/download/classics-poetry.imepack.zip) | `24 MB` | `266,816` classical entries + `231,593` `Tab` chains + 28L Rank-8 LoRA (`9.66 MB`) |
| **Developer Bilingual Pack** | [**`tech-developer.imepack.zip`**](https://github.com/tyrchen/chime/releases/latest/download/tech-developer.imepack.zip) | `8.9 MB` | Systems/AI/Cloud bilingual lexicon + exact casing + 28L Rank-8 LoRA (`9.66 MB`) |
| **SHA-256 Checksums** | [**`SHA256SUMS.txt`**](https://github.com/tyrchen/chime/releases/latest/download/SHA256SUMS.txt) | `555 B` | Release integrity checksums |

> **Apple Notarized + Seamless Passwordless Future Upgrades (`v0.14.0`)**:
> - Both `.pkg` and `.zip` are signed with **Apple Developer ID (Hardened Runtime + Secure Timestamp), notarized by Apple Notary Service, and offline-stapled**, paired with a built-in zero-telemetry updater (`SHA-256` + `Developer ID` verification, toggleable in Preferences).
> - Upgrading from legacy `/Library/Input Methods/Chime.app` launches macOS native `Installer.app` (**Touch ID / Apple Watch** supported), migrates Chime to `~/Library/Input Methods/Chime.app`, and skips redundant `TISEnableInputSource` calls — eliminating both future password prompts and macOS 15+ `"Allow Chime 拼音 to enable Chime 拼音?"` popups.

### Installing `.imepack` Cartridges

You can hot-mount `.imepack` cartridges with **zero restart**:
- **1-Click Built-In Store (Recommended)**: Click `[中]` in the macOS menu bar -> **Settings (`偏好设置…`)** -> **Lexicon & AI (`词库与 AI`)**, and click **Install / Update** next to `classics-poetry` or `tech-developer`.
- **Local File Import**: Click **Import Dictionary or `.imepack`…** in Settings to import local `.imepack.zip`, `.scel`, or `.dict.yaml` files.
- **Via CLI**:
  ```bash
  chime pack install ~/Downloads/classics-poetry.imepack.zip
  chime pack install ~/Downloads/tech-developer.imepack.zip
  chime pack list
  ```
  Cartridges live in `~/Library/Application Support/Chime/packs/` and share the single `256.7 MB` 28L base model in Unified Memory.

---

## Key Capabilities

1. **Intact 28-Layer `Qwen3-14B` Soft-KL Base + Route A Pinyin-Constrained Neural Beam Search**:
   - Retains all 28 layers of `Qwen/Qwen3-0.6B-Base` gently distilled from `Qwen/Qwen3-14B-Base` on 65M zero-contamination Chinese tokens (`256.7 MB` 4-bit MLX, lossless `36,864` full-CJK vocab).
   - Route A advances an incremental autoregressive beam (`RouteABeamCache`) over a deterministic `TokenPinyinIndex` Trie, natively handling long-range homophone disambiguation, out-of-vocabulary compounds, and contextual single-character selection.
   - **4.07× Fused Metal Speedup**: Fuses `q/k/v -> qkv_proj` and `gate/up -> gate_up_proj` at load time (`-84` Metal GEMM launches/step on 28L, Multi-LoRA preserved) and materializes compact child KV caches inside a single `eval()` barrier to sever cross-step lazy DAG blowup; mid-syllable fast path responds in `0 ms` while Route A refines asynchronously without blocking typing.
2. **Zero-Jitter Swift 6 Native UX (`v0.14.0` Overhaul)**:
   - **Rock-Solid Candidate Bar**: Composition-session vertical anchor & above/below orientation lock, constant compact row height, disabled implicit `CATransaction` animations, and position preservation during async neural refreshes completely eliminate vertical panel jumping.
   - **Zero Duplicate `Shift` Toasts**: `deviceIndependentFlagsMask` deduplication + `150 ms` toggle cooldown + post-`setMarkedText` validated caret geometry ensure the `中 / En` indicator appears once at the true caret; includes hover highlights and 4-edge multi-monitor clamping.
   - **Minimal Compact Bar + Offline Emoji & Kaomoji (`v0.14.1`)**: Decluttered compact bar keeps only a clean `⌄` expand control; 100% offline Unicode Emoji & Kaomoji suggestions (`笑哭 -> 😂`, `开心 -> 😄 / ヽ(✿ﾟ▽ﾟ)ノ`, browse via `biaoqing` / `yanwenzi`) work across all schemes, never steal `#1` or pollute LM context, and offer independent Preferences toggles (including optional IDE quiet mode).
   - **Full Pinyin, Shuangpin, Skeletons & 4-Operator Motor-Slip Recovery**: Full Pinyin, initial abbreviations (`wsm -> 为什么`), **Xiaohe / Ziranma / Microsoft Shuangpin**, zero-bloat algebraic fuzzy Pinyin, and 4 single-step error operators (adjacent swap `shaungpin -> 双拼`, QWERTY neighbor `shuruga -> 输入法`, key repeat `mohuyign -> 模糊音`, geminate boundary elision `chunuanhuakai -> 春暖花开`).
3. **Mainstream IME Ergonomics, Dual-Mode English Auto-Correction & Auto-Spacing**:
   - **Modeless Code-Switching + Configurable Inline `Shift` Mode**: Seamless mixed decoding (`woshiyongmacoshegithub -> 我使用 macOS 和 GitHub`, `bangworeviewzhegepr -> 帮我 review 这个 PR`, leading-uppercase English) plus Sogou/Doubao-compatible Left/Right `Shift` English toggling (`En` menu bar indicator).
   - **Dual-Mode English Spelling Auto-Correction (1-Key `Backspace` Undo)**: Zero-RAM `mmap` binary-search recovery across Chinese mixed mode and inline `En` mode (`reivew -> review`, `kubernets -> Kubernetes`, `definately -> definitely`), exempting terminals and camelCase identifiers.
   - **Smart Spacing & Punctuation**: Configurable Chinese↔English/number auto-spacing, `⌃.` full/half-width punctuation with decimal (`3.14`) and time (`12:30`) exemptions, customizable `- / =`, `[ / ]`, `, / .` paging, and `; / '` 2nd/3rd candidate quick-select.
4. **Zero-Copy `CHIMED02` Lexicon, Full-Poem `Tab` Chaining & Multi-LoRA Cartridges**:
   - `660,000+` core entries + sorted 16-byte FNV-1a continuation table mapped via `mmap` in `< 0.8 ms`, enabling stateful `Tab` verse chaining across classical poetry and prose with anti-repeat disambiguation.
   - Native pure-Rust importers for **Sogou `.scel`**, **Baidu `.bdict`**, **Rime `.dict.yaml`**, **TSV**, and **macOS Text Replacements**, backed by `SQLite` (`WAL`) online habit learning.
   - **Rank-Stacked Multi-LoRA + Couplet-Aware Simplex Router**: Dynamically gates 28L Rank-8 FP16 LoRAs by host app Bundle ID, lexical cues, and regulated-verse cadence.

---

## Architecture & Official Benchmarks

```text
Keystrokes + Sliding 96-Char Context + Host Bundle ID
                       │
                       ├────────────────────────────────────────────────┐
                       ▼                                                ▼
┌─────────────────────────────────────────────┐  ┌─────────────────────────────────────────────┐
│ Fast Path · Zero-Copy Lattice (< 0.8 ms)    │  │ Async Route A · Constrained Neural Beam     │
│  • mmap CHIMED02 core + mounted pack dicts  │  │  • 28L 4-bit MLX Qwen3 (14B Distilled, 36K) │
│  • Fuzzy algebra + 4-op typo + SQLite store │  │  • Fused QKV/Gate+Up + 1-Barrier KV (4.07×) │
│  • Instant non-blocking keystroke feedback  │  │  • Deterministic TokenPinyinIndex + Cache   │
└──────────────────────┬──────────────────────┘  │  • Rank-Stacked 28L Rank-8 Multi-LoRA       │
                       │                         └──────────────────────┬──────────────────────┘
                       └───────────────────────┬────────────────────────┘
                                               ▼
                     Session-Anchored Zero-Jitter Swift 6 IMK Panel
```

### Measured End-to-End Accuracy Progression (5 Official Suites · 22,328 Prompts)

> Strict zero-leakage protocol: `test_holdout.tsv` (1,010 cases) and `test_holdout_10k.tsv` (10,000 open-domain sentences) are strictly blind holdouts — individual rows have never been inspected or used in training or hyperparameter tuning.

| Release & Architecture | Model Size | `contextual_open` (60) | `dev_suite` (1,258) | Blind `test_holdout` (1,010) | `dev_suite_10k` (10,000) | Blind `test_holdout_10k` (10,000) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v0.10.0` · 6L `Qwen3-0.6B` Rerank (24K) | `64.2 MB` | — | `86.01%` | `70.89%` | — | — |
| `v0.12.2` · 8L `Qwen3-4B` Distill Rerank (36K) | `87.8 MB` | — | `89.51%` | `75.84%` | — | `66.45%` |
| `v0.13.0` · 8L `Qwen3-4B` Distill + Route A | `87.8 MB` | — | `91.49%` | `83.66%` | `77.00%` | `75.31%` |
| **`v0.14.0` · 28L `Qwen3-14B` Distill + 4× Fused Route A** | **`256.7 MB`** | **`93.33%`** (`Hit@3 98.3%`) | **`93.16%`** (`Hit@5 97.9%`) | **`90.00%`** (`+6.34%`) | **`81.34%`** (`+4.34%`) | **`79.70%`** (`+4.39%`) |

- **`v0.14.0` Highlights**:
  - `test_holdout.tsv` (1,010 blind): Science/Medical/Legal `100%`, Finance/Business `100%`, Idioms/Classics `100%`, all 3 Shuangpin schemes `100%`, Fuzzy/Dialect `97.50%`, Contextual Dialogue `95.00%`, `MRR 0.9100`.
  - `test_holdout_10k.tsv` (10,000 blind): Contextual Sentences `89.53%`, Tech/Developer Prose `85.58%`, Contextual Single-Char `85.14%`, `Hit@5 87.36%`, `MRR 0.8303`.
  - **4.07× Metal Inference Speedup**: Same 28L model on `dev_suite` dropped from `351.8 s -> 86.5 s` (`p50 278 ms -> 71.1 ms`); `contextual_open` `p50` dropped from `21.2 ms -> 6.44 ms`.

---

## Train an Air-Gapped Private Personal Style `.imepack`

Fine-tune a strictly local 28L Rank-8 personal style cartridge (`private_pack: true`) from your Markdown essays or notes **100% locally on Apple Silicon GPU (`mps`)** without uploading a single byte:

```bash
# 1. Extract sanitized paragraphs, personal words, and bigrams into an ephemeral folder
#    (automatically strips URLs, emails, API tokens, code blocks, and YAML frontmatter)
uv run --python 3.12 --no-project --with jieba --with pypinyin \
  python tools/prepare_pack_corpora.py \
  --personal-posts ~/Notes/essays \
  --private-out /tmp/chime_private_personal

# 2. Train a 9.66 MB 28L Q4-Manifold Aligned Rank-8 LoRA on Apple Silicon Metal
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

## Voice Input (Hold-to-Talk with a Local / Self-Hosted Model)

Enable it under **Preferences… → 语音输入**. **Hold Right `Option`** (or Right `⌘` / Right `⌃` / `Fn`) for about 0.3 s and a small voice HUD appears: recognized text shows up at the caret while you speak, releasing the key commits it, and `Esc` cancels. Grant microphone access on first use. The installer includes a bilingual **Streaming Paraformer INT8** model: no runtime download, server, or API key is needed. The default backend keeps audio on device and exits its helper process after dictation. This model does not insert punctuation.

To use an existing endpoint, explicitly select the custom HTTP backend (`backend = "http"` under `[voice]`). Only this backend sends audio to the configured URL; the optional API key stays in Keychain. Any OpenAI-compatible transcription server works (`POST multipart/form-data` with `file`, `model`, `language`, `prompt`; response `{"text": …}`), for example whisper.cpp:

```bash
brew install whisper-cpp
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
whisper-server -m ggml-large-v3-turbo.bin --host 127.0.0.1 --port 8178 -l zh
# Preferences endpoint: http://127.0.0.1:8178/inference
```

For speaches / faster-whisper-server, vLLM or LocalAI, use their `/v1/audio/transcriptions` URL and model name. `cargo run -p chime-voice --example dictate_wav -- speech.wav <endpoint>` checks a server without the IME.

---

## Workspace Structure & Building from Source

| Path | Description |
| :--- | :--- |
| `crates/phonetics` | Mandarin syllable algebra, Xiaohe/Ziranma/MS Shuangpin, dynamic fuzzy expansion, 4-operator motor-slip model, and code-switch phonotactics |
| `crates/lexicon` | Zero-copy `mmap` `CHIMED02` binary dictionary, $O(\log C)$ FNV-1a verse continuations, `.scel`/`.bdict`/Rime importers, and `SQLite` (`WAL`) user habit store |
| `crates/decoder` | Dual-track beam lattice decoder, probability-space separation, context next-verse promotion, and anti-jitter freeze window |
| `crates/neural` | Apple Silicon Metal `mlx-rs 0.34` 4-bit 28L `Qwen3` transformer, fused `QKV`/`Gate+Up`, Route A constrained neural beam search, Rank-Stacked Multi-LoRA, and Couplet Simplex Router |
| `crates/core` | 32-slot LRU session state machine, async Route A scheduler, smart punctuation, PII/secret sanitizer, and `.imepack` v2 cartridge manager |
| `crates/voice` | Hold-to-talk dictation: energy VAD segmentation, live partial transcripts, and a client for local / self-hosted OpenAI-compatible ASR servers |
| `crates/ffi` | Stateless C ABI boundary bridging Rust engine and Swift 6 host |
| `apps/cli` | `chime` CLI (`decode`, `dict-build-base`, `dict-compile`, `pack`, `corpus-export`, `eval`, and `bench` suites) |
| `macos/` | Swift 6 `@MainActor` `InputMethodKit` app bundle (`Chime.app`), session-anchored candidate bar, `5x5` grid, verified zero-telemetry updater, and Settings UI |
| `tools/` | `Qwen3-14B` Soft-KL distillation (`modal_distill_qwen3.py`), 28L Multi-LoRA training (`modal_train_loras.py` / `colab_train_loras.py`), and 4-bit MLX quantization |
| `paper/` | Research paper in Typst (`chime-paper.typ`, `figures.typ`), reproducible Metal ablation results, and arXiv LaTeX bundle (`paper/arxiv/`) |
| `site/` | Static GitHub Pages landing page, interactive `Tab` poetry demo, benchmark table, and `.imepack` download hub |

```bash
# Format check, strict clippy pedantic lints (-D warnings), and all unit + doc tests
make fmt && make lint && make test

# Run official Route A accuracy & Metal latency benchmarks
make bench-dev
make bench-test

# Build, sign, and install ~/Library/Input Methods/Chime.app
make install-macos

# Build release .pkg, .zip, public 28L .imepack.zip cartridges, and SHA256SUMS.txt
make release-artifacts

# Compile research paper PDF (Typst) and arXiv pdflatex tarball
make paper-arxiv
```

---

## Privacy & Security

- **Zero Keystroke Network Access**: Typing never triggers any network call, device fingerprint, or telemetry upload; every keystroke, `SQLite` user habit, and `.imepack` adapter stays strictly on your Mac. (The optional 24-hour update check performs an ephemeral cookie-free read-only GET of `releases-manifest.json` and can be disabled anytime in Settings.)
- **Hardware Password Bypass**: Automatically checks macOS `IsSecureEventInputEnabled()` and yields immediately whenever a password field, SSH passphrase prompt, or 1Password/Keychain dialog is focused.
- **Memory Safety**: `#![forbid(unsafe_code)]` enforced across all pure-logic Rust crates (`phonetics`, `decoder`, `neural`, `core`, `cli`), isolating `unsafe` solely to a read-only `mmap` wrapper in `lexicon` and the thin C FFI shim in `ffi`.

---

## Citation

If you find Chime, its Route A Pinyin-constrained neural decoder, distillation pipeline, or Rank-Stacked Multi-LoRA `.imepack` design useful in your research or systems work, please cite:

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
