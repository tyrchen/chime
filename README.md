# Chime 拼音

<p>
  <strong>简体中文</strong> · <a href="README.en.md">English</a> · <a href="https://tyrchen.github.io/chime/">官网与在线演示</a> · <a href="https://tyrchen.github.io/chime/chime-paper.pdf">技术论文 (PDF)</a> · <a href="https://github.com/tyrchen/chime/releases/latest">下载安装包</a>
</p>

[![Release](https://img.shields.io/github/v/release/tyrchen/chime?label=Release&color=0f4c5c)](https://github.com/tyrchen/chime/releases/latest)
[![Platform](https://img.shields.io/badge/macOS-15%2B%20Apple%20Silicon-111827)](https://tyrchen.github.io/chime/)
[![Network](https://img.shields.io/badge/联网与遥测-0%20Bytes%20(100%25%20离线)-047857)](#隐私与安全设计)
[![Paper](https://img.shields.io/badge/Paper-PDF%20Preprint-be123c)](https://tyrchen.github.io/chime/chime-paper.pdf)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

**Chime** 是一款专为 **macOS 15+（Apple Silicon）** 打造的 **100% 本地离线、零遥测**中文神经拼音输入法，使用 **Rust** 编写核心解码与推理引擎、**Swift 6** 编写原生 `InputMethodKit` 界面。

在 **`v0.14.0`** 中，Chime 采用**零延迟快路径 + Route A 拼音约束自回归神经束搜索**架构：敲击任意按键时由 `< 0.8 ms` 的零拷贝 `mmap` 双轨词图（`CHIMED02`）瞬间出字不卡手；后台异步线程在 Apple Silicon 统一内存上运行经 **`Qwen3-14B-Base` 柔和 KL 蒸馏**的完整 **28 层 4-bit MLX `Qwen3-0.6B`（`256.7 MB`，无损 `36,864` 全中文词表）**，配合 **4.07× Metal 融合算子提速**与可热插拔的 **28L Rank-8 Multi-LoRA `.imepack` 卡带**，将严格盲测整句首项准确率推升至 **`90.00%`**（万条开放域盲测 **`79.70%`**）。

- **在线交互体验与下载中心**：[https://tyrchen.github.io/chime/](https://tyrchen.github.io/chime/)
- **学术论文预印本（PDF）**：[*Chime: Sub-3ms On-Device Neural Pinyin Input via Manifold-Aligned Distillation and Rank-Stacked Multi-LoRA Cartridges*](https://tyrchen.github.io/chime/chime-paper.pdf)（Typst 与 arXiv LaTeX 源码位于 [`paper/`](paper/)）
- **最新发布版本（`v0.14.1`）**：[GitHub Releases](https://github.com/tyrchen/chime/releases/latest)

---

## 下载与安装（`v0.14.1`）

系统要求：**macOS 15.0（Sequoia）及以上**，搭载 **Apple Silicon（`M1` / `M2` / `M3` / `M4`）** 的 Mac。

| 发布产物 | 下载直链 | 体积 | 说明 |
| :--- | :--- | ---: | :--- |
| **macOS 安装包（`.pkg`）** | [**`Chime-v0.14.1-macos-arm64.pkg`**](https://github.com/tyrchen/chime/releases/latest/download/Chime-macos-arm64.pkg) | `257 MB` | 自动安装并迁移至用户态 `~/Library/Input Methods/Chime.app`（清理旧系统目录副本），此后所有应用内升级**无需密码/指纹、静默完成** |
| **便携版应用包（`.zip`）** | [**`Chime-v0.14.1-macos-arm64.zip`**](https://github.com/tyrchen/chime/releases/latest/download/Chime-macos-arm64.zip) | `257 MB` | 解压后将 `Chime.app` 拖入 `~/Library/Input Methods/` 双击即可自动注册 |
| **古诗词曲赋与古文卡带** | [**`classics-poetry.imepack.zip`**](https://github.com/tyrchen/chime/releases/latest/download/classics-poetry.imepack.zip) | `24 MB` | `266,816` 古典词条 + `231,593` 条全篇 `Tab` 续写链 + 28L Rank-8 古文 LoRA（`9.66 MB`） |
| **开发者中英双语卡带** | [**`tech-developer.imepack.zip`**](https://github.com/tyrchen/chime/releases/latest/download/tech-developer.imepack.zip) | `8.9 MB` | 系统/AI/云原生中英术语 + 精确驼峰大小写 + 28L Rank-8 技术 LoRA（`9.66 MB`） |
| **SHA-256 校验清单** | [**`SHA256SUMS.txt`**](https://github.com/tyrchen/chime/releases/latest/download/SHA256SUMS.txt) | `555 B` | 安装包与卡带完整性哈希校验值 |

> **Apple 官方公证 + 零密码静默升级体验（v0.14.0 新增）**：
> - `.pkg` 与 `.zip` 均已通过 **Apple Developer ID 正式签名与 Apple Notary Service 公证装订**；内置零遥测自动更新器（双重校验 `SHA-256` 与 `Developer ID` 证书链，支持随时关闭）。
> - 从旧版本（位于 `/Library/Input Methods/Chime.app`）升级时，应用内更新会直接调起原生 macOS 安装器（原生支持 **Touch ID / Apple Watch**），一次迁移至 `~/Library/Input Methods/Chime.app` 后，**未来所有版本更新均免密静默完成**，且已启用 Chime 的设备升级时**不再重复弹出 macOS 15+ 系统「允许启用 Chime 拼音」确认框**。

### 挂载 `.imepack` 领域扩展卡带（免重启热加载）

- **偏好设置一键在线安装（推荐）**：点击右上角菜单栏 `[中]` -> **偏好设置…** -> **词库与 AI**，直接点击 **「一键安装」** 或 **「更新」** 即可自动完成下载、`SHA-256` 校验与免重启热挂载（应用升级后已安装的官方卡带也会自动同步到对应 28L 版本）。
- **本地文件导入**：在偏好设置中点击 **导入词库或 `.imepack`…**，选择本地 `.imepack.zip` 或搜狗 `.scel` / Rime `.dict.yaml`。
- **命令行导入**：
  ```bash
  chime pack install ~/Downloads/classics-poetry.imepack.zip
  chime pack install ~/Downloads/tech-developer.imepack.zip
  chime pack list
  ```
  所有扩展包存放在 `~/Library/Application Support/Chime/packs/` 下，与常驻统一内存的同一份 `256.7 MB` 28L 基础模型共享权重，不重复占用内存。

---

## 核心特性

1. **完整 28 层 `Qwen3-14B` 柔和蒸馏底座 + Route A 拼音约束自回归束搜索**：
   - 保留 `Qwen/Qwen3-0.6B-Base` 全部 28 层深度预训练世界知识，以 `Qwen/Qwen3-14B-Base` 作为教师在 6,500 万 Token 零污染中文语料上进行柔和 Soft-KL 蒸馏（`36,864` 无损全中文词表，4-bit MLX `256.7 MB`）。
   - Route A 引擎直接在确定性 `TokenPinyinIndex` 前缀树约束下执行增量束搜索（`RouteABeamCache`），天然具备跨多音节长程语义理解、未登录新词造词与复杂同音句消歧能力。
   - **4.07× Metal 推理提速**：加载期将每层 `q/k/v` 融合为单次 `qkv_proj`、将 `gate/up` 融合为 `gate_up_proj`（每步减少 84 次 Metal GEMM 启动），并在单次 `eval()` 屏障内紧致物化子束 KV 缓存以切断跨步惰性 DAG 膨胀；交互态零阻塞异步渲染，音节输入中途 0 ms 瞬时响应。
2. **原生 Swift 6 极致顺滑交互（v0.14.0 全面打磨）**：
   - **候选栏零跳动**：引入会话级垂直锚点与上下方向锁、紧凑单行等高布局，并在异步神经候选刷新时锁定窗口位置、关闭隐式 `CATransaction` 动画，彻底解决长句输入时候选栏上下乱跳问题。
   - **中英切换零重复浮窗**：使用 `deviceIndependentFlagsMask` 精确去重与 150 ms 防抖冷却，配合 `setMarkedText` 后的光标矩形有效性校验，根治 `Shift` 切换时 `中 / En` 徽章在屏幕两个位置重复闪现的问题；支持鼠标悬停高亮与四边多屏安全夹紧。
   - **清爽零冗余单行栏 + 离线 Emoji / 颜文字联想（v0.14.1 新增）**：紧凑横排候选栏仅保留极简 `⌄` 展开按钮；内置 100% 离线 Unicode Emoji 与颜文字联想（如 `笑哭 -> 😂`、`开心 -> 😄 / ヽ(✿ﾟ▽ﾟ)ノ`、`赞 -> 👍`，输入 `biaoqing` / `yanwenzi` 浏览全部），绝不抢占 `#1` 首选、不污染语言模型上文，偏好设置支持独立开关及「终端/IDE 静默」策略。
   - **全拼、双拼、简拼与四类盲打容错**：支持全拼、首字母简拼（`wsm -> 为什么`）、**小鹤 / 自然码 / 微软双拼**、零膨胀动态模糊音，以及邻键乱序（`shaungpin -> 双拼`）、QWERTY 邻键误触（`shuruga -> 输入法`）、连击重键剔除（`mohuyign -> 模糊音`）与跨音节吞音恢复（`chunuanhuakai -> 春暖花开`）。
3. **对齐主流输入法习惯的中英混输、双模式英文纠错与智能空格**：
   - **免切换同轨混输 + 可配置 `Shift` 内联切换**：既支持免切换同轨解码（`woshiyongmacoshegithub -> 我使用 macOS 和 GitHub`、`bangworeviewzhegepr -> 帮我 review 这个 PR`、大写开头直输英文），也支持左/右 `Shift` 切换内联英文模式（菜单栏实时显示 **`En`**）。
   - **双模式英文拼写自动纠错（退格键 `⌫` 一键撤回）**：零额外内存 `mmap` 二分探测覆盖中文混输与 `En` 直输模式（`reivew -> review`、`kubernets -> Kubernetes`、`definately -> definitely`），终端与驼峰标识符自动豁免。
   - **智能排版与标点**：支持中英文/数字前后自动加空格（默认开启可随时关闭）、`⌃.` 切换全半角标点（自动豁免 `3.14` 与 `12:30`）、自定义 `- / =` / `[ / ]` / `, / .` 翻页与 `; / '` 二三候选快选。
4. **零拷贝 `CHIMED02` 词库、全篇「一路 `Tab`」诗词续写与多 LoRA 卡带**：
   - `core.imedict` 内置 `660,000+` 中英词条与 16 字节有序 FNV-1a 续写哈希表，`< 0.8 ms` `mmap` 冷启动；支持唐诗宋词古文名篇逐句 `Tab` 续写、重首句消歧与防复读。
   - 原生纯 Rust 导入 **搜狗 `.scel`**、**百度 `.bdict`**、**Rime `.dict.yaml`**、**TSV** 与 **macOS 系统文本替换**，配合本地 `SQLite`（`WAL`）实时学习个人词频与退格负反馈。
   - **秩堆叠 Multi-LoRA 旁路 + 律绝对仗单纯形路由**：多个 28L Rank-8 卡带预拼接为两次矩阵乘，按宿主应用 Bundle ID、文言虚词与五七言对仗韵律平滑调度。

---

## 整体架构与官方评测基准

```text
原始按键流 + 光标前 96 字滑动上下文 + 当前宿主应用 Bundle ID
                       │
                       ├────────────────────────────────────────────────┐
                       ▼                                                ▼
┌─────────────────────────────────────────────┐  ┌─────────────────────────────────────────────┐
│ 快路径 · 零拷贝双轨网格 (< 0.8 ms · 0 阻塞) │  │ 异步精排 · Route A 拼音约束自回归神经束搜索  │
│  • mmap CHIMED02 核心词图 + 挂载卡带词典    │  │  • 28L 4-bit MLX Qwen3 (14B 柔和蒸馏, 36K)  │
│  • 模糊音代数 + 4 类错键容错 + SQLite 习惯  │  │  • 融合 QKV / Gate+Up + 单屏障紧凑 KV (4.07×) │
│  • 按键瞬间即刻上屏候选，永不卡手           │  │  • 确定性 TokenPinyinIndex + 增量 Beam Cache │
└──────────────────────┬──────────────────────┘  │  • 秩堆叠 28L Rank-8 Multi-LoRA 动态路由     │
                       │                         └──────────────────────┬──────────────────────┘
                       └───────────────────────┬────────────────────────┘
                                               ▼
                        会话锚点锁定 & Swift 6 零跳动原生候选面板
```

### 跨版本端到端准确率演进（5 套官方基准 · 共 22,328 条用例）

> 严格遵守零数据泄露纪律：`test_holdout.tsv`（1,010 条）与 `test_holdout_10k.tsv`（10,000 条）为严格盲测集，全程从未查看单条内容，也未参与任何蒸馏、LoRA 微调或超参搜索。

| 版本与模型配置 | 模型体积 | `contextual_open` (60) | `dev_suite` (1,258) | 盲测 `test_holdout` (1,010) | `dev_suite_10k` (10,000) | 盲测 `test_holdout_10k` (10,000) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v0.10.0` · 6L `Qwen3-0.6B` Rerank (24K) | `64.2 MB` | — | `86.01%` | `70.89%` | — | — |
| `v0.12.2` · 8L `Qwen3-4B` 蒸馏 Rerank (36K) | `87.8 MB` | — | `89.51%` | `75.84%` | — | `66.45%` |
| `v0.13.0` · 8L `Qwen3-4B` 蒸馏 + Route A | `87.8 MB` | — | `91.49%` | `83.66%` | `77.00%` | `75.31%` |
| **`v0.14.0` · 28L `Qwen3-14B` 蒸馏 + 4× 融合 Route A** | **`256.7 MB`** | **`93.33%`** (`Hit@3 98.3%`) | **`93.16%`** (`Hit@5 97.9%`) | **`90.00%`** (`+6.34%`) | **`81.34%`** (`+4.34%`) | **`79.70%`** (`+4.39%`) |

- **`v0.14.0` 细分维度亮点**：
  - `test_holdout.tsv`（1,010 盲测）：科学医疗法律 `100%`、金融商业 `100%`、成语古典 `100%`、三大双拼方案 `100%`、模糊音方言 `97.50%`、语境对话 `95.00%`，`MRR 0.9100`。
  - `test_holdout_10k.tsv`（10,000 盲测）：长句语境 `89.53%`、技术开发语境 `85.58%`、语境单字 `85.14%`、`Hit@5 87.36%`、`MRR 0.8303`。
  - **4.07× Metal 推理提速（Apple Silicon）**：优化前后同模型（28L）`dev_suite` 总耗时 `351.8 s -> 86.5 s`，单句 `p50 278 ms -> 71.1 ms`，`contextual_open` 语境探针 `p50 21.2 ms -> 6.44 ms`。

---

## 训练只属于本机的私有文风 `.imepack` 卡带

你可以用自己过去的 Markdown 博客或笔记，**100% 在本机 Apple Silicon GPU（`mps`）上断网微调**出专属的 28L Rank-8 私有文风卡带（`private_pack: true`），全程零数据出本机：

```bash
# 1. 在本地扫描并清洗历史 Markdown 文章（自动剔除链接、邮箱、API Key、代码块与 Frontmatter）
uv run --python 3.12 --no-project --with jieba --with pypinyin \
  python tools/prepare_pack_corpora.py \
  --personal-posts ~/Notes/essays \
  --private-out /tmp/chime_private_personal

# 2. 直接在本机 Apple Silicon Metal 上基于 28L 4-bit 流形训练 9.66 MB Rank-8 LoRA
uv run --python 3.12 --no-project --with torch --with safetensors --with transformers \
  --with datasets --with huggingface_hub --with mlx --with numpy \
  python tools/colab_train_loras.py \
  --q4-model-dir dist/base_neural \
  --vocab dist/base_neural/vocab.json \
  --private-txt /tmp/chime_private_personal/personal_lora_corpus.txt \
  --private-out-lora /tmp/chime_private_personal/personal_lora.safetensors \
  --steps 320

# 3. 打包并安装至 ~/Library/Application Support/Chime/packs/，随后清理临时目录
cargo run --release -p chime-cli -- pack build /tmp/chime_private_personal/personal-style.imepack \
  --id personal-style \
  --name "个人文风私有包" \
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

## 语音输入（按住说话 · 本地 / 私有化识别模型）

在 **偏好设置… → 语音输入** 中开启后，**按住右 `Option`（可改为右 `⌘` / 右 `⌃` / `Fn`）约 0.3 秒**即弹出语音浮窗：说话时识别结果实时出现在光标处，松开上屏，按 `Esc` 取消。首次使用需允许麦克风权限。默认使用安装包内置的 **Streaming Paraformer INT8** 中英双语模型：不需要下载模型、启动服务或填写 API Key，音频不离开本机。每次按住使用一个独立识别进程，结束即退出。模型不会自动补标点。

开发者可用 `make macos-installer` 构建含模型的 `.pkg`。构建时从固定版本下载并校验权重和原生运行库，缓存位于 `dist/voice-cache/`（可通过 `VOICE_CACHE` 指定）。只有构建需要这些下载，安装和听写均不需要 Python。

如需使用自己的模型，可将「模型来源」改为「自定义 HTTP 服务」（`[voice].backend = "http"`）；原有服务地址、模型名和钥匙串中的 API Key 可继续使用。仅此模式会发送音频，内置模式出错不会转发。识别服务需兼容 OpenAI 转写接口（`POST multipart/form-data`：`file`、`model`、`language`、`prompt`，返回 `{"text": …}`），例如 whisper.cpp：

```bash
brew install whisper-cpp
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
whisper-server -m ggml-large-v3-turbo.bin --host 127.0.0.1 --port 8178 -l zh
# 偏好设置中填写服务地址：http://127.0.0.1:8178/inference
```

speaches / faster-whisper-server、vLLM、LocalAI 等服务填写其 `/v1/audio/transcriptions` 地址与模型名即可。不打开输入法也能用 `cargo run -p chime-voice --example dictate_wav -- speech.wav <服务地址>` 检查服务端效果。

---

## 仓库目录结构与从源码构建

| 目录 | 模块职责 |
| :--- | :--- |
| `crates/phonetics` | 汉语拼音音节代数、小鹤/自然码/微软双拼、动态模糊音、四类错键算子与中英混输音系判定 |
| `crates/lexicon` | 零拷贝 `mmap` `CHIMED02` 二进制词典、$O(\log C)$ FNV-1a 诗词续写表、第三方词库导入与 `SQLite`（`WAL`）个人习惯库 |
| `crates/decoder` | 双轨束搜索网格解码器、概率空间分离、上下文诗词续写晋升与防抖冻结窗口 |
| `crates/neural` | Apple Silicon Metal `mlx-rs 0.34` 4-bit 28L `Qwen3` 引擎、融合 `QKV`/`Gate+Up`、Route A 拼音约束自回归束搜索、秩堆叠多 LoRA 与律绝单纯形路由器 |
| `crates/core` | 32 槽位 LRU 会话状态机、异步 Route A 调度器、智能标点、PII/密钥脱敏清洗器与 `.imepack` v2 卡带管理 |
| `crates/voice` | 按住说话语音输入：能量 VAD 分句、边说边出字的实时预览，以及对接本地/私有化 ASR 服务（OpenAI 兼容转写接口）的客户端 |
| `crates/ffi` | 无状态窄 C ABI 接口层，衔接 Rust 核心与 Swift 6 宿主 |
| `apps/cli` | `chime` 命令行工具集（`decode`、`dict-build-base`、`dict-compile`、`pack`、`corpus-export`、`eval` 与全套 `bench` 评测） |
| `macos/` | Swift 6 `@MainActor` 原生 `InputMethodKit` 应用包（`Chime.app`）、会话锚定候选栏、`5×5` 网格、零遥测自动更新器与偏好设置窗口 |
| `tools/` | `Qwen3-14B` 柔和 KL 蒸馏（`modal_distill_qwen3.py`）、28L 多 LoRA 训练（`modal_train_loras.py` / `colab_train_loras.py`）与 4-bit MLX 量化工具 |
| `paper/` | Typst 学术论文源码（`chime-paper.typ`、`figures.typ`）、可复现实验结果及 arXiv LaTeX 投稿包（`paper/arxiv/`） |
| `site/` | GitHub Pages 静态落地页、交互式诗词 `Tab` 续写演示、官方基准面板与 `.imepack` 下载中心 |

```bash
# 代码格式检查、严格 Clippy Pedantic 静态分析（零告警）及全套单元/文档测试
make fmt && make lint && make test

# 运行端到端准确率与 Metal GPU 延迟基准评测
make bench-dev
make bench-test

# 从源码编译、签名并安装至 ~/Library/Input Methods/Chime.app
make install-macos

# 构建 .pkg 安装器、.zip 归档、公开 28L .imepack.zip 卡带与 SHA256SUMS.txt
make release-artifacts

# 编译 Typst 学术论文 PDF 与 arXiv pdflatex 源码包
make paper-arxiv
```

---

## 隐私与安全设计

- **语音输入默认离线**：语音输入默认关闭；开启后只在按住快捷键期间录音，安全输入模式下不启动录音。只有选择自定义 HTTP 服务才会将音频发往配置的地址。
- **彻底零按键联网**：日常输入过程零网络请求、零设备指纹、零遥测 SDK；你的每一次按键、个人高频词与 `.imepack` 权重严格只留在你的 Mac 本地磁盘中。（可选的自动更新检查默认每 24 小时仅只读拉取静态 `releases-manifest.json`，可在偏好设置随时一键关闭。）
- **硬件级安全输入旁路**：每次按键自动检查 macOS `IsSecureEventInputEnabled()`，一旦焦点进入系统密码框、终端 `sudo`/SSH 口令提示或 1Password/钥匙串弹窗，立即原样放行并停止记录上下文。
- **内存安全边界**：全部纯逻辑 Rust Crate（`phonetics`、`decoder`、`neural`、`core`、`cli`）开启 `#![forbid(unsafe_code)]`，仅在只读词典 `mmap` 映射与极薄的 C FFI 边界保留最小受控封装。

---

## 引用（Citation）

如果 Chime 的拼音约束端侧神经解码、流形/柔和 KL 蒸馏或秩堆叠多 LoRA `.imepack` 架构对你的研究或工程有所帮助，欢迎引用：

```bibtex
@article{chen2026chime,
  title   = {Chime: Sub-3ms On-Device Neural Pinyin Input via Manifold-Aligned Distillation and Rank-Stacked Multi-LoRA Cartridges},
  author  = {Chen, Tian},
  journal = {Preprint},
  year    = {2026},
  url     = {https://github.com/tyrchen/chime}
}
```

## 开源许可（License）

本项目采用 `MIT OR Apache-2.0` 双重开源许可（见 [LICENSE-MIT](LICENSE-MIT) 与 [LICENSE-APACHE](LICENSE-APACHE)）。第三方开源词库署名详见 [NOTICE](NOTICE)。
