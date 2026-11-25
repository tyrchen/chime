# Chime

A privacy-first, sub-10ms local neural Chinese Pinyin Input Method Engine for **macOS 15+ (Apple Silicon)**, built with a **Rust core** and a **Swift 6 UI**.

## Complete Feature Matrix

1. **Pinyin, Shuangpin, Super-Abbreviation & Zero-Bloat Fuzzy Pinyin**:
   - Full Pinyin, initial skeletons (`wsm` -> `为什么`, `rgzn` -> `人工智能`), and **Xiaohe / Ziranma / Microsoft Shuangpin**.
   - Configurable fuzzy syllable algebra (`z/zh`, `c/ch`, `s/sh`, `n/l`, `f/h`, `r/l`, `an/ang`, `en/eng`, `in/ing`, `ian/iang`, `uan/uang`) evaluated dynamically without inflating dictionary storage.
   - Full-sentence decoding plus step-by-step prefix word selection (`今天 wo'men`), formatted preedit (`jin'tian'wo'men`), smart Chinese full-width punctuation (paired quotes `“”`/`‘’` & decimal `3.14` exemption), candidate paging (`-`/`=`, `PageUp`/`PageDown`, Arrows), and `Tab` one-key typo rollback.
2. **Modeless Chinese-English Code-Switching**:
   - Dual-track lattice simultaneously decodes Mandarin syllables and English/code identifiers (`macOS`, `GitHub`, `TypeScript`, `useEffect`, `PR`, `Kubernetes` prefix completion) with per-app Pangu auto-spacing.
3. **Zero-Copy `mmap` Lexicon (`CHIMED02`), Dense Bigram/Trigram LM, Third-Party Importers & `SQLite` WAL User Store**:
   - Compiles ~330,000+ permissive Mandarin & English entries (`pinyin` + `jieba` MIT frequencies + polyphone phrase table + `/usr/share/dict/words`) plus ~2,800 teacher-scored Bigram & Trigram transitions into `core.imedict`, memory-mapped in `< 1ms` with particle-transparent Skip-Bigram decoding (`公民的权利` vs `国家的权力`, `如实地反映` vs `化学反应`).
   - Native pure-Rust importers for **Sogou `.scel`**, **Baidu `.bdict`**, **Rime `.dict.yaml`**, **TSV**, and **macOS System Text Replacements** (`TextReplacements.db`).
   - Durable `SQLite` (`WAL` mode) `UserStore` tracking exponential time-decayed habits, per-app affinity, custom user phrases (`Cedar` trie), and backspace negative-feedback overrides.
4. **On-Device Apple Silicon Metal Neural Reranking (`mlx-rs`) & Colab `LoRA` Personalization (`Train Your Own Brain`)**:
   - Ships an out-of-the-box 4-layer KL-distilled **`Qwen/Qwen2.5-0.5B`** Causal SLM (`FP16` Safetensors, `16,384` Mandarin+English BPE vocabulary, tied embeddings) running natively on Apple Silicon Metal GPU via **`mlx-rs`** with persistent **`MlxPrefixCache`**, single-dispatch lazy continuation batching, actor-isolated background worker (`p50 ~ 3.2ms`), **60ms Anti-Jitter Freeze Window**, plus optional macOS 15 **`CoreML MLState`** ANE runner.
   - Local high-recall secret/PII sanitizer + noisy reverse-G2P training pair synthesizer (`chime corpus-export`) paired with [colab/train_personal_brain.ipynb](colab/train_personal_brain.ipynb) to fine-tune and hot-load `.imepack` bundles.
5. **Native macOS 15 Swift 6 Host (`Chime 拼音` / `Chime.app`)**:
   - Localized input source (`Chime 拼音`) with crisp vector template menu bar & app icons, zero-property `ChimeInputController` (immune to `IMKServer` controller leaks), `< 250ms` clean `Left Shift` inline English toggle, `IsSecureEventInputEnabled()` password bypass, Chromium `CGRectZero` fallback, interactive vibrancy `CandidatePanel` (horizontal/vertical + mouse click), menu bar actions, and native **SwiftUI Preferences Window**.

## Quick Start

```bash
# Run all unit & doc tests
make test

# Distill Qwen2.5-0.5B base SLM, build ~330k-entry CHIMED02 core.imedict, and run the 25-case benchmark suite
make eval

# Build and sign dist/Chime.app (bundles core.imedict + Qwen2.5 Metal neural model + icons)
make macos-bundle

# Install and register "Chime 拼音" at ~/Library/Input Methods/Chime.app
make install-macos

# Export sanitized personal notes into noisy G2P JSONL for Google Colab LoRA training
cargo run --release -p chime-cli -- corpus-export ~/Notes/*.md -o personal_corpus.jsonl
```

## License

Dual-licensed under `MIT OR Apache-2.0` ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)). Third-party open lexical attributions are listed in [NOTICE](NOTICE).
