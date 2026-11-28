# Chime 本地 ASR 对照实验（不是 IMK 端到端测试）

Apple M5 Max，64 GiB，macOS 26.6；CPU provider，2 个 ONNX 线程，greedy 解码。两模型串行测试，推理时只读本地权重。`para.jsonl`、`sense.jsonl` 是每个样例的逐次预览、最终文本、参考文本、实际输入 PCM 哈希、延迟和 CPU 时间。`*-concurrent.jsonl`、`keys-summary.json` 是另一次拼音并发测试。**没有 M1、真人麦克风或宿主应用数据。**

## 本轮结果

| 后端 | 首字中位数 | 松键中位数 | 长句 CPU 秒 | 回改字符（7 段合计） | 中英混说编辑距离 / 参考字符 |
|---|---:|---:|---:|---:|---:|
| Streaming Paraformer | 684 ms | 45 ms | 3.34 | 0 | 10 / 60 |
| SenseVoice-Small | 277 ms | 79 ms | 14.79 | 29 | 9 / 60 |

两者长句都是 0/131 个非标点字符错误，但 Paraformer 没有输出句读。纯键盘、与 Paraformer 并发、与 SenseVoice 并发的同步按键 P95 分别为 30.0、29.6、26.2 ms；单轮没有显示 ASR 加剧延迟，也不能据此断言输入法无卡顿。建议先试 Paraformer 流式后端，在 M1 真人录音上验证标点、英文术语和尾延迟，再选产品默认值。

**卸载探针**：分别运行 `$P benchmarks/voice/unload.py "$D" sense` 和 `para`（每个新进程），输入 2 s 零值 PCM 后比较 `ps` 当前 RSS，而不是峰值。SenseVoice：35 → 874 → 669 MiB；Paraformer：35 → 744 → 547 MiB（初始 → 解码后 → 销毁 recognizer 和 stream、GC 后 1 s）。这不证明泄漏，但说明释放对象不等于立刻返还全部常驻内存；若要求空闲时彻底卸载，值得单测可退出的本地语音子进程。

## 复现

```bash
D=/tmp/chime-asr-compare
mkdir -p "$D"/{para,sense}
uv venv --python 3.12 "$D/venv"
uv pip install --python "$D/venv/bin/python" sherpa-onnx==2.0.3 numpy==2.5.5 opencc-python-reimplemented==0.1.7
for f in encoder.int8.onnx decoder.int8.onnx tokens.txt; do curl -fL -o "$D/para/$f" "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en/resolve/main/$f"; done
for f in model.int8.onnx tokens.txt; do curl -fL -o "$D/sense/$f" "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/$f"; done
# 从仓库根目录运行；prepare 使用 macOS say (Tingting) 和 ffmpeg。
P="$D/venv/bin/python"
$P benchmarks/voice/compare.py "$D" prepare
for m in para sense; do $P benchmarks/voice/compare.py "$D" "$m" > "$D/$m.jsonl"; done
```

实测权重 SHA-256：SenseVoice `c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51`；Paraformer encoder `81a70226a8934e6ed92aa1d4fc486b428b5398e2f2619ed4897b7294cab90e9a`，decoder `f3cca9f77bb9d93c8fcbfb63ae617b6b1ee96818df3aa3b151c40658fe38594f`。

## 测量口径

- 一轮 7 个固定样例，3 个中文短句、3 个夹英文的短句、1 个 30.55 秒的长句，均为同一合成音色、无背景噪声。每段裁剪至最后一个明显非静音采样后最多 120 ms，以 16 kHz 单声道每 50 ms 按墙钟回放。脚本模拟后端，不安装输入法，不访问麦克风、用户词库或训练数据。
- SenseVoice 固定中文、开启 ITN；以能量门限、约 700 ms 预览、650 ms 停顿和 24 s 硬切模拟 Chime 的分句/重算策略，不是直接调用 Rust VAD。Paraformer 每段一个持久流，结束时一次性补 0.8 s 静音作为右侧上下文，不需要等 0.8 s 墙钟时间。未加标点模型、热词或二次纠错。
- `first_ms` 是从音频开始至非空词字出现，不要求第一个字正确；`tail_ms` 是最后一块音频应到达至得到最终文本，包含推理积压，不包括客户端提交。两者都不含热键 280 ms 防误触和 IMK 刷新。`load_ms` 是创建 recognizer 耗时，非清空系统文件缓存的冷启动。
- `errors/chars` 是归一化后编辑距离除以参考字符数：简繁统一、大小写折叠、剔除空格和标点；数字没有做语义归一化（`三` → `3` 算一处差异）。英文按字母计。计数反映这组样本，而非生产准确率。`deleted` 是相邻预览最长共同前缀之后被撤回的旧字符数之和（含最终修订）；只数追加不算回改。
- `cpu_s` 是整个识别进程的 CPU 秒数，跨线程求和，非实际录音时长。`peak_mib` 是整段测试截至该样例的进程峰值 RSS，包括 Python、ONNX Runtime、模型和临时张量，不是模型净内存，也不是卸载后残留内存。

## 快语速最终稿对照

沿用上面的 `$P`、`$D`：`$P benchmarks/voice/fast_speech.py "$D"`。固定 6 句普通话、2 种 macOS 合成音色、3 个 `say -r` 档位。对每段相同 PCM 比较 Paraformer 流式收尾和 SenseVoice 整句离线解码；不模拟键盘、录音或 UI。`fast-synthetic.jsonl` 保存逐条原始结果。最快档 12 段共 220 个参考字，两个后端分别有 29 和 16 个字符编辑错误（忽略标点）。个别音色两者都错，不能外推到真人快语音。`sense_ms_loaded` 是模型已加载时的整句处理耗时，不是产品松键延迟。

若有自愿提供的本地录音，可用 `$P benchmarks/voice/fast_speech.py "$D" --audio /path/to/speech.m4a --reference '实际说的原句'`。不会开麦或上传；结果打到标准输出，私人录音和转写不要提交到仓库。

## 安装包内置后端冒烟测试

构建 `make macos-installer` 后，用包内原生 helper、动态库和包外模型目录验证 Rust 会话，不启动 IMK 或麦克风。使用上面 `prepare` 生成的语音：

```bash
ffmpeg -y -i "$D/short3.wav" -f s16le "$D/short3.pcm"
CHIME_TEST_APP="$PWD/dist/Chime.app" CHIME_TEST_MODEL_DIR="$D/para" CHIME_TEST_PCM="$D/short3.pcm" \
  CHIME_TEST_TEXT='下午三点开会记得带上电脑' \
  cargo test -p chime-voice --test streaming -- --ignored --nocapture
```

要验证实际 `.pkg`，先用 `pkgutil --expand-full` 解包，将 `CHIME_TEST_APP` 指向解出的 `Chime.app`。测试覆盖取消后重新开始、开关实时预览、松键后最终文本和录音时限。冷启动加载时间计入首字等待，不等于上表已加载模型的推理延迟。

## 键盘并发

使用 Chime v0.14.1 发布包的 `core.imedict` 和 `neural/`，只解压资源，不安装或运行下载的可执行文件。链接当前工作区 release Rust FFI。下面的命令将并发结果写到脚本同目录；在自己的验证目录复制脚本后运行可保留原结果。

```bash
curl -fL -o "$D/release.zip" https://github.com/tyrchen/chime/releases/download/v0.14.1/Chime-macos-arm64.zip
unzip -o "$D/release.zip" '*/Contents/Resources/neural/*' '*/Contents/Resources/core.imedict' -d "$D"
export MLX_RS_METAL_PATH=/tmp/chime-mlx-metallib MACOSX_DEPLOYMENT_TARGET=14.0 CMAKE_OSX_DEPLOYMENT_TARGET=14.0
mkdir -p "$MLX_RS_METAL_PATH"
cargo build --release -p chime-ffi
clang -O2 -Icrates/ffi/include benchmarks/voice/keys.c target/release/libchime_ffi.a -lc++ -lsqlite3 -framework Metal -framework Foundation -framework Accelerate -framework Security -framework CoreFoundation -framework IOKit -o "$D/keys"
$P benchmarks/voice/contention.py "$D" "$D/Chime.app/Contents/Resources" > "$D/keys-summary.json"
```

键盘进程用临时 HOME 和 SQLite，确认 `neural_loaded` 后循环输入三组拼音；每次同步 FFI 调用结束后间隔 60 ms 再取一次异步候选。预热后测 30 s 纯键盘基线，再分别和两种模型的长句推理重叠。仅按同一单调时钟取实际重叠的按键，保留原始样本于 `$D/keys.csv`。测的是 `chime_process_key` 调用延迟，不含 macOS 事件分发、候选窗绘制或 Final ASR 提交；独立进程能测资源争用，不能模拟主线程集成错误。
