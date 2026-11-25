# Chime 拼音

<p>
  <strong>简体中文</strong> · <a href="README.en.md">English</a> · <a href="https://tyrchen.github.io/chime/">官网与在线演示</a> · <a href="https://tyrchen.github.io/chime/chime-paper.pdf">技术论文 (PDF)</a> · <a href="https://github.com/tyrchen/chime/releases/latest">下载安装包</a>
</p>

[![Release](https://img.shields.io/github/v/release/tyrchen/chime?label=Release&color=0f4c5c)](https://github.com/tyrchen/chime/releases/latest)
[![Platform](https://img.shields.io/badge/macOS-15%2B%20Apple%20Silicon-111827)](https://tyrchen.github.io/chime/)
[![Network](https://img.shields.io/badge/联网与遥测-0%20Bytes%20(100%25%20离线)-047857)](#隐私与安全设计)
[![Paper](https://img.shields.io/badge/Paper-PDF%20Preprint-be123c)](https://tyrchen.github.io/chime/chime-paper.pdf)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

**Chime** 是一款专为 **macOS 15+（Apple Silicon）** 打造的 **100% 本地离线、零遥测、亚毫秒级**中文拼音输入法，使用 **Rust** 编写核心解码与推理引擎、**Swift 6** 编写原生 `InputMethodKit` 界面。

它不把你的每一次按键发往云端大模型，也不在本地跑高延迟的自回归生成，而是采用**词法—神经两阶段协同架构**：一阶段用 `< 0.8 ms` 的零拷贝 `mmap` 双轨网格解码器（`CHIMED02`）完成候选召回与按键容错，二阶段在 Apple Silicon 统一内存上运行经流形对齐蒸馏压缩至 **`64.17 MB`** 的 6 层 4-bit MLX `Qwen3` 小模型做并行语义重排，并支持像插拔游戏卡带一样热挂载 **`2.07 MB` 的领域与个人文风 Multi-LoRA `.imepack` 扩展包**。

- **在线交互体验与下载中心**：[https://tyrchen.github.io/chime/](https://tyrchen.github.io/chime/)
- **学术论文预印本（PDF）**：[*Chime: Sub-3ms On-Device Neural Pinyin Input via Manifold-Aligned Distillation and Rank-Stacked Multi-LoRA Cartridges*](https://tyrchen.github.io/chime/chime-paper.pdf)（Typst 与 arXiv LaTeX 源码位于 [`paper/`](paper/)）
- **最新发布版本（`v0.8.0`）**：[GitHub Releases](https://github.com/tyrchen/chime/releases/latest)

---

## 下载与安装（`v0.8.0`）

系统要求：**macOS 15.0（Sequoia）及以上**，搭载 **Apple Silicon（`M1` / `M2` / `M3` / `M4`）** 的 Mac。

| 发布产物 | 下载直链 | 体积 | 说明 |
| :--- | :--- | ---: | :--- |
| **macOS 安装包（`.pkg`）** | [**`Chime-v0.8.0-macos-arm64.pkg`**](https://github.com/tyrchen/chime/releases/latest/download/Chime-macos-arm64.pkg) | `~75 MB` | 双击安装至 `/Library/Input Methods/Chime.app` 并自动注册输入源、打开键盘设置页 |
| **便携版应用包（`.zip`）** | [**`Chime-v0.8.0-macos-arm64.zip`**](https://github.com/tyrchen/chime/releases/latest/download/Chime-macos-arm64.zip) | `~75 MB` | 解压后将 `Chime.app` 拖入 `~/Library/Input Methods/` 双击即可自动注册 |
| **古诗词曲赋与古文卡带** | [**`classics-poetry.imepack.zip`**](https://github.com/tyrchen/chime/releases/latest/download/classics-poetry.imepack.zip) | `16.8 MB` | `266,816` 古典词条 + `231,593` 条全篇 `Tab` 续写链 + Q4 Rank-8 LoRA |
| **开发者中英双语卡带** | [**`tech-developer.imepack.zip`**](https://github.com/tyrchen/chime/releases/latest/download/tech-developer.imepack.zip) | `1.9 MB` | 系统/AI/云原生中英术语 + 精确驼峰大小写 + Q4 Rank-8 LoRA |
| **SHA-256 校验清单** | [**`SHA256SUMS.txt`**](https://github.com/tyrchen/chime/releases/latest/download/SHA256SUMS.txt) | `377 B` | 安装包与卡带完整性哈希校验值 |

> **Apple 官方签名与公证（Notarized Developer ID）**：`.pkg` 安装器与 `.zip` 应用包均已通过 **Apple Developer ID 正式签名、Hardened Runtime 硬化运行时与 Apple Notary Service 官方公证并离线钉票（Stapled）**，下载后可直接双击安装，不会触发 macOS Gatekeeper 恶意软件拦截弹窗。安装后前往 **系统设置 -> 键盘 -> 输入法 -> 编辑 -> `+` -> 简体中文**，确认勾选启用 **Chime 拼音**。

### 挂载 `.imepack` 领域扩展卡带（免重启热加载）

- **图形界面导入**：点击右上角菜单栏 `[中]` 图标 -> **偏好设置** -> **导入词库或 `.imepack` 扩展包**，直接选择下载好的 `.imepack.zip` 即可。
- **命令行导入**：
  ```bash
  chime pack install ~/Downloads/classics-poetry.imepack.zip
  chime pack install ~/Downloads/tech-developer.imepack.zip
  chime pack list
  ```
  所有扩展包存放在 `~/Library/Application Support/Chime/packs/` 下，与常驻统一内存的同一份 `64.17 MB` 基础模型共享权重，不重复占用内存。

---

## 核心特性

1. **全拼、双拼、简拼与四类盲打容错算子**：
   - 原生支持全拼、首字母简拼（`wsm -> 为什么`、`rgzn -> 人工智能`）以及 **小鹤 / 自然码 / 微软双拼**。
   - 零体积膨胀的代数模糊音展开（`z/zh`、`c/ch`、`s/sh`、`n/l`、`f/h`、`r/l`、`an/ang`、`en/eng`、`in/ing`、`ian/iang`、`uan/uang`）。
   - 内置四类单步按键噪声恢复：**邻键乱序交换**（`shaungpin -> 双拼`、`nianquuqqianliyanbo -> 念去去，千里烟波`）、**QWERTY 欧氏距离邻键误触**（`shuruga -> 输入法`）、**连击重键剔除**（`mohuyign -> 模糊音`）以及 **跨音节双辅音吞音恢复**（`chunuanhuakai -> 春暖花开`、`bujinchangjiangungunlai -> 不尽长江滚滚来`）。
   - 单音节候选清爽克制（如 `gen -> 1. 跟 2. 根 3. 根据 4. 根本 5. 更`），按 `↓` 键即可展开 `5×5` 网格面板查看后置的 35+ 同音单字兜底。
2. **对齐主流输入法习惯的中英切换、智能空格与偏好设置**：
   - **免切换同轨混输 + 可配置 `Shift` 内联切换**：既支持免切换同轨解码（`woshiyongmacoshegithub -> 我使用 macOS 和 GitHub`、`bangworeviewzhegepr -> 帮我 review 这个 PR`、大写字母开头直接输英文 `Mac -> Mac / MacBook`），也完整对齐搜狗 / 豆包输入法的内部中英切换习惯（默认 **左 / 右 `Shift` 均可切换**，可配置仅左 `Shift`、仅右 `Shift`、`Caps Lock` 或关闭；支持配置拼音输入中按 `Shift`「上屏英文并切换」或「仅上屏英文保持中文」，并在光标旁弹出轻量 `中 / 英` 状态浮窗）。
   - **中英文与数字前后自动插入空格（默认开启，可一键关闭）**：全覆盖中文↔英文词、中文↔阿拉伯数字（如 `我有 100 个`、`使用 Rust 开发`）四向边界，可在偏好设置或菜单栏一键开启/关闭。
   - **智能全半角标点与自定义翻页键**：支持 `⌃.` 快速切换中英标点，自动豁免数字小数点（`3.14`）与时间冒号（`12:30`）；支持自定义 `- / =`、`[ / ]`、`, / .` 翻页键，`; / '` 二三候选快选，以及候选栏字号（`14–19 pt`）。
3. **零拷贝 `CHIMED02` 词库与全篇「一路 `Tab`」诗词续写**：
   - `core.imedict` 内置 `660,000+` 中英词条、教师模型打分的二元/三元语法及 16 字节有序 FNV-1a 续写哈希表，通过 `mmap` 在 `< 0.8 ms` 内完成冷启动映射。
   - 支持唐诗、宋词、古文名篇逐句 `Tab` 续写（如敲 `junbujianhuanghe` 上屏后连按 `Tab` 续写《将进酒》全篇，或续写《水调歌头》《雨霖铃》《滕王阁序》《岳阳楼记》《赤壁赋》），并具备重首句上下文消歧与防复读机制（上文已有「君不见黄河之水天上来，奔流到海不复回。」时再敲 `junbujian`，自动将下一联 **`#1 君不见高堂明镜悲白发`** 置顶）。
   - 纯 Rust 原生支持导入 **搜狗 `.scel`**、**百度 `.bdict`**、**Rime `.dict.yaml`**、**TSV** 与 **macOS 系统文本替换**，配合本地 `SQLite`（`WAL` 模式）实时记忆个人词频与退格负反馈。
4. **亚毫秒批处理 Metal 小模型重排 + Q4 流形对齐多 LoRA 旁路**：
   - **流形对齐 6 层蒸馏底座（`64.17 MB`）**：从 28 层 `Qwen/Qwen3-0.6B-Base`（`1.19 GB` FP16）经词表裁剪（`24,576` 高频子词）、层间隐状态轨迹对齐、输出头前余弦对齐（$\cos(\mathbf{u}_S, \mathbf{u}_T) = 0.853$）与双温度 KL 蒸馏压缩为 `64.17 MB` 4-bit MLX 模型（缩小 $18.6\times$），精准消歧长程语境同音词（如「依法保障**权利**」vs.「严禁滥用**权力**」、「4/5 究竟有**多大**」vs.「**多达**」）。
   - **单次批处理共享前缀重排 + 秩堆叠算子融合**：将多词元候选打包为单次 `[B, L_max]` 张量前向，零拷贝广播 `MlxPrefixCache` 并在 Metal GPU 上通过 `take_along_axis` 直接抽取目标对数似然（GPU$\to$CPU 拷回量 `<= 128 B`），同时将 $K$ 个挂载 LoRA 预拼接为两次矩阵乘 $\Delta\mathbf{y} = ((\mathbf{x}\mathbf{A}_{\text{cat}}^\top)\odot\mathbf{g})\mathbf{B}_{\text{cat}}^\top$。
   - **Q4 量化流形对齐训练 + 律绝对仗单纯形门控路由**：直接在反量化的运行时 4-bit 权重流形上训练 `2.07 MB` Rank-8 LoRA，消除零样本迁移过冲；运行时结合宿主应用 Bundle ID、文言虚词及 **五言/七言上下联字数对称韵律**（如「空山新雨后，」+「天气晚来秋」）在单纯形总预算 $\sum_k \gamma_k \le 0.90$ 下平滑路由。

---

## 整体架构与实测性能

```text
原始按键流 + 光标前 96 字滑动上下文 + 当前宿主应用 Bundle ID
                       │
                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 一阶段 · 零拷贝双轨网格解码 (< 0.8 ms · mmap CHIMED02)                │
│  • 基础 core.imedict + 热挂载 pack.imedict 切片 + SQLite WAL 个人库   │
│  • 动态模糊音代数 + 邻键 / 乱序 / 重键 / 跨音节辅音吞音恢复           │
│  • 概率空间分离：等跨度同音池 C_n 与跨音节预判池 C_cmp 独立度量       │
└──────────────────────┬────────────────────────────────────────────────┘
                       │ Top-6 等长同音候选集
                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 二阶段 · 批处理 Metal 小模型 + 多 LoRA 旁路 (0.86–1.45 ms · mlx-rs)   │
│  • 冻结 6 层 4-bit Qwen3-6L-Chime (64.17 MB, 24,576 紧凑 BPE 词表)    │
│  • [B, L_max] 广播 MlxPrefixCache + GPU 侧 take_along_axis 采集       │
│  • 秩堆叠 Q4 对齐 LoRA 旁路 + 律绝对仗感知单纯形门控路由              │
└──────────────────────┬────────────────────────────────────────────────┘
                       ▼
          60 ms 首选防抖冻结窗口 & Swift 6 原生候选面板
```

| 模型与挂载配置（Apple Silicon Metal · `T=16` 前缀 + 6 候选） | 模型 / 卡带体积 | 同音压力集准确率 | 44 项端到端基准 | 6 候选重排 `p50` | 相对加速比 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Qwen3-0.6B-Base`（28 层，152K 词表，FP16） | `1,191.5 MB` | `93.8% (15/16)` | — | `~14.2 ms` | `1.0x` |
| `Qwen3-28L`（24K 裁剪词表，4-bit MLX） | `249.95 MB` | `93.8% (15/16)` | `100% (44/44)` | `5.29 ms` | `2.7x` |
| `Qwen3-6L-NaiveSkip`（4-bit 直接跳层，未对齐） | `64.15 MB` | `43.8% (7/16)` | — | `2.33 ms` | — |
| **`Qwen3-6L-Chime`（4-bit 批处理重排，`K=0`）** | **`64.17 MB`** | **`93.8% (15/16)`** | **`100% (44/44)`** | **`0.86 ms`** | **`16.5x`** |
| **+ 同时挂载 `K=3` 个 Rank-8 `.imepack` LoRA** | **`+2.07 MB` / 包** | **`93.8% (15/16)`** | **`100% (44/44)`** | **`1.45 ms`** | **较未融合快 `4.21x`** |
| **+ 同时挂载 `K=6` 个 Rank-8 `.imepack` LoRA** | **`+2.07 MB` / 包** | **`93.8% (15/16)`** | **`100% (44/44)`** | **`1.45 ms`** | **较未融合快 `7.46x`** |

---

## 训练只属于本机的私有文风 `.imepack` 卡带

你可以用自己过去的 Markdown 博客或笔记，**100% 在本机 Apple Silicon GPU（`mps`）上断网微调**出专属的私有文风卡带（`private_pack: true`），全程零数据出本机：

```bash
# 1. 在本地扫描并清洗历史 Markdown 文章（自动剔除链接、邮箱、API Key、代码块与 Frontmatter）
uv run --python 3.12 --no-project --with jieba --with pypinyin \
  python tools/prepare_pack_corpora.py \
  --personal-posts ~/Notes/essays \
  --private-out /tmp/chime_private_personal

# 2. 直接在本机 Apple Silicon Metal 上基于 4-bit 运行时流形训练 2.07 MB Rank-8 LoRA（约 45 秒）
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

## 仓库目录结构与从源码构建

| 目录 | 模块职责 |
| :--- | :--- |
| `crates/phonetics` | 汉语拼音音节代数、小鹤/自然码/微软双拼、动态模糊音、四类错键算子与中英混输音系判定 |
| `crates/lexicon` | 零拷贝 `mmap` `CHIMED02` 二进制词典、$O(\log C)$ FNV-1a 诗词续写表、第三方词库导入与 `SQLite`（`WAL`）个人习惯库 |
| `crates/decoder` | 双轨束搜索网格解码器、概率空间分离、上下文诗词续写晋升与 60 ms 防抖冻结窗口 |
| `crates/neural` | Apple Silicon Metal `mlx-rs 0.34` 4-bit `Qwen3` 推理器、`[B, L_max]` 批量 `MlxPrefixCache` + `take_along_axis`、秩堆叠多 LoRA 与律绝单纯形路由器 |
| `crates/core` | 32 槽位 LRU 会话状态机、智能标点、PII/密钥脱敏清洗器与 `.imepack` v2 卡带管理 |
| `crates/ffi` | 无状态窄 C ABI 接口层，衔接 Rust 核心与 Swift 6 宿主 |
| `apps/cli` | `chime` 命令行工具集（`decode`、`dict-build-base`、`dict-compile`、`pack`、`corpus-export` 及 44 项 `eval` 基准评测） |
| `macos/` | Swift 6 `@MainActor` 原生 `InputMethodKit` 应用包（`Chime.app`）、`5×5` 候选网格与偏好设置窗口 |
| `paper/` | Typst 学术论文源码（`chime-paper.typ`、`figures.typ`）、可复现实验结果（`experiment_results.json`）及 arXiv LaTeX 投稿包（`paper/arxiv/`） |
| `site/` | GitHub Pages 静态落地页、交互式诗词 `Tab` 续写演示与 `.imepack` 下载中心 |

```bash
# 代码格式检查、严格 Clippy Pedantic 静态分析（零告警）及全套单元/文档测试
make fmt && make lint && make test

# 运行 44 项端到端准确率与 Metal GPU 延迟基准套件
cargo run --release -p chime-cli -- eval --dict dist/core.imedict --neural dist/base_neural

# 从源码编译、签名并安装至 ~/Library/Input Methods/Chime.app
make install-macos

# 构建 .pkg 安装器、.zip 归档、公开 .imepack.zip 卡带与 SHA256SUMS.txt
make release-artifacts

# 编译 Typst 学术论文 PDF 与 arXiv pdflatex 源码包
make paper-arxiv
```

---

## 隐私与安全设计

- **彻底零联网**：`Chime.app` 不申请任何网络权限，不链接任何 HTTP 或遥测 SDK；你的每一次按键、个人高频词与 `.imepack` 权重严格只留在你的 Mac 本地磁盘中。
- **硬件级安全输入旁路**：每次按键自动检查 macOS `IsSecureEventInputEnabled()`，一旦焦点进入系统密码框、终端 `sudo`/SSH 口令提示或 1Password/钥匙串弹窗，立即原样放行并停止记录上下文。
- **内存安全边界**：全部纯逻辑 Rust Crate（`phonetics`、`decoder`、`neural`、`core`、`cli`）开启 `#![forbid(unsafe_code)]`，仅在只读词典 `mmap` 映射与极薄的 C FFI 边界保留最小受控封装。

---

## 引用（Citation）

如果 Chime 的词法—神经协同设计、流形对齐小模型蒸馏或秩堆叠多 LoRA `.imepack` 架构对你的研究或工程有所帮助，欢迎引用：

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
