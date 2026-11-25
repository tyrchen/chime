#!/usr/bin/env python3
"""
Prepares lexical TSVs, N-grams, and clean LoRA training corpora for:
  1. Public `tech-developer.imepack` (stored in `dist/`)
  2. Optional local private `personal-*.imepack` (stored ONLY in `/tmp/chime_private_personal/`
     from local Markdown essays, never touching the `chime` repository)
"""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

import jieba
from pypinyin import Style, lazy_pinyin

PUBLIC_TECH_TERMS: list[tuple[str, str, float, bool]] = [
    # Chinese Systems / AI / Cloud / Rust / macOS technical terms
    ("端侧推理", "duan ce tui li", -3.05, True),
    ("端侧模型", "duan ce mo xing", -3.00, True),
    ("零拷贝", "ling kao bei", -3.10, True),
    ("统一内存", "tong yi nei cun", -3.15, True),
    ("低秩适配", "di zhi shi pei", -3.30, True),
    ("低秩矩阵", "di zhi ju zhen", -3.35, True),
    ("键值缓存", "jian zhi huan cun", -3.20, True),
    ("前缀缓存", "qian zhui huan cun", -3.25, True),
    ("四比特量化", "si bi te liang hua", -3.30, True),
    ("量化感知训练", "liang hua gan zhi xun lian", -3.35, True),
    ("知识蒸馏", "zhi shi zheng liu", -3.15, True),
    ("交叉熵损失", "jiao cha shang sun shi", -3.35, True),
    ("余弦退火", "yu xian tui huo", -3.40, True),
    ("注意力机制", "zhu yi li ji zhi", -3.10, True),
    ("分组查询注意力", "fen zu cha xun zhu yi li", -3.45, True),
    ("旋转位置编码", "xuan zhuan wei zhi bian ma", -3.40, True),
    ("词表裁剪", "ci biao cai jian", -3.35, True),
    ("束搜索", "shu sou suo", -3.25, True),
    ("网格解码", "wang ge jie ma", -3.30, True),
    ("双数组字典树", "shuang shu zu zi dian shu", -3.45, True),
    ("内存映射", "nei cun ying she", -3.15, True),
    ("无锁队列", "wu suo dui lie", -3.25, True),
    ("并发安全", "bing fa an quan", -3.15, True),
    ("所有权模型", "suo you quan mo xing", -3.25, True),
    ("生命周期标注", "sheng ming zhou qi biao zhu", -3.35, True),
    ("借用检查器", "jie yong jian cha qi", -3.25, True),
    ("零成本抽象", "ling cheng ben chou xiang", -3.20, True),
    ("代数数据类型", "dai shu shu ju lei xing", -3.40, True),
    ("模式匹配", "mo shi pi pei", -3.15, True),
    ("异步运行时", "yi bu yun xing shi", -3.20, True),
    ("尾延迟", "wei yan chi", -3.20, True),
    ("冷启动", "leng qi dong", -3.10, True),
    ("热更新", "re geng xin", -3.15, True),
    ("热加载", "re jia zai", -3.15, True),
    ("灰度发布", "hui du fa bu", -3.15, True),
    ("金丝雀发布", "jin si que fa bu", -3.30, True),
    ("幂等性", "mi deng xing", -3.20, True),
    ("背压控制", "bei ya kong zhi", -3.35, True),
    ("熔断降级", "rong duan jiang ji", -3.30, True),
    ("限流器", "xian liu qi", -3.25, True),
    ("可观测性", "ke guan ce xing", -3.15, True),
    ("链路追踪", "lian lu zhui zong", -3.20, True),
    ("火焰图", "huo yan tu", -3.20, True),
    ("内存泄漏", "nei cun xie lou", -3.05, True),
    ("死锁检测", "si suo jian ce", -3.30, True),
    ("反向代理", "fan xiang dai li", -3.15, True),
    ("负载均衡", "fu zai jun heng", -3.10, True),
    ("服务网格", "fu wu wang ge", -3.25, True),
    ("基础设施即代码", "ji chu she shi ji dai ma", -3.35, True),
    ("声明式配置", "sheng ming shi pei zhi", -3.30, True),
    ("预写式日志", "yu xie shi ri zhi", -3.35, True),
    ("快照隔离", "kuai zhao ge li", -3.35, True),
    ("向量数据库", "xiang liang shu ju ku", -3.15, True),
    ("检索增强生成", "jian suo zeng qiang sheng cheng", -3.25, True),
    ("智能体工作流", "zhi neng ti gong zuo liu", -3.25, True),
    ("代码审查", "dai ma shen cha", -3.10, True),
    ("单元测试", "dan yuan ce shi", -3.00, True),
    ("集成测试", "ji cheng ce shi", -3.10, True),
    ("基准测试", "ji zhun ce shi", -3.15, True),
    ("回归测试", "hui gui ce shi", -3.15, True),
]

PUBLIC_TECH_ENGLISH: list[tuple[str, float]] = [
    ("Rust", -2.90),
    ("Cargo", -3.20),
    ("Tokio", -3.25),
    ("Axum", -3.45),
    ("Serde", -3.40),
    ("Rayon", -3.50),
    ("WebAssembly", -3.30),
    ("WASI", -3.55),
    ("Swift", -3.05),
    ("SwiftUI", -3.15),
    ("AppKit", -3.35),
    ("CoreML", -3.30),
    ("Metal", -3.15),
    ("MLX", -3.20),
    ("SafeTensors", -3.35),
    ("PyTorch", -3.05),
    ("HuggingFace", -3.20),
    ("Qwen", -3.25),
    ("LoRA", -3.05),
    ("QLoRA", -3.40),
    ("vLLM", -3.35),
    ("GGUF", -3.45),
    ("llama.cpp", -3.40),
    ("Transformer", -3.10),
    ("Kubernetes", -3.05),
    ("Docker", -3.00),
    ("PostgreSQL", -3.05),
    ("SQLite", -3.10),
    ("DuckDB", -3.35),
    ("ClickHouse", -3.35),
    ("Redis", -3.05),
    ("Kafka", -3.15),
    ("gRPC", -3.15),
    ("Protobuf", -3.25),
    ("GraphQL", -3.25),
    ("OpenTelemetry", -3.35),
    ("Prometheus", -3.30),
    ("Grafana", -3.30),
    ("TypeScript", -2.95),
    ("JavaScript", -2.95),
    ("React", -2.95),
    ("Next.js", -3.15),
    ("Tailwind", -3.25),
    ("Vite", -3.25),
    ("Playwright", -3.30),
    ("GitHub", -2.85),
    ("GitLab", -3.25),
    ("macOS", -2.85),
    ("iOS", -2.90),
    ("Linux", -2.90),
    ("Neovim", -3.30),
    ("Ghostty", -3.35),
    ("VSCode", -3.05),
    ("Zed", -3.25),
    ("Cursor", -3.15),
    ("Claude", -3.10),
    ("OpenAI", -3.00),
    ("ChatGPT", -2.90),
]

PUBLIC_TECH_CORPUS_SEED: list[str] = [
    "在 Apple Silicon 的统一内存架构上，使用 Rust 配合 MLX Metal 跑 4-bit 量化的端侧模型，单步推理延迟不到一毫秒。",
    "帮我 review 一下这个 GitHub PR，重点看看异步运行时里有没有死锁或者内存泄漏的隐患。",
    "通过 mmap 零拷贝映射二进制词库，进程冷启动时不需要反序列化任何哈希表，驻留内存非常克制。",
    "在分布式微服务里，接口的幂等性、背压控制和熔断降级往往比单纯压榨吞吐量更重要。",
    "我们把 SQLite 切到 WAL 模式之后，读写并发的尾延迟下降了一个数量级。",
    "利用低秩适配 LoRA 技术，可以在完全冻结基础模型权重的前提下，按需挂载不同领域的增量矩阵。",
    "写 Rust 的时候，与其到处加 Arc 和 Mutex，不如用 actor 模式加有界 channel 把非线程安全的状态隔离在单线程里。",
    "升级完 TypeScript 和 Vite 依赖之后，记得跑一遍端到端的 Playwright 回归测试和 lint 检查。",
    "在 Kubernetes 集群上做灰度发布时，先看 OpenTelemetry 和 Prometheus 面板里的错误率与 p99 延迟。",
    "通过前缀键值缓存 KV-Cache 复用上下文状态，打字重排只需要计算候选词新增的几个 token。",
    "借用检查器和所有权模型虽然前期上手有门槛，但把大量数据竞争直接拦在了编译期。",
    "设计 API 边界时保持沙漏型窄接口，核心状态机全部收拢在纯函数式的 Rust 模块里，外层 SwiftUI 只负责薄视图渲染。",
]

VALID_SYL_RE = re.compile(r"^[a-z]+$")
CJK_WORD_RE = re.compile(r"^[\u4e00-\u9fa5]{2,8}$")
SECRET_OR_URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
    r"sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,})"
)


def word_to_pinyin(word: str) -> str | None:
    syls = lazy_pinyin(word, style=Style.NORMAL, errors="ignore")
    if len(syls) != len(word):
        return None
    out = []
    for s in syls:
        s = s.lower().replace("ü", "v").replace("u:", "v")
        if s in ("jv", "qv", "xv", "yv"):
            s = s[0] + "u"
        elif s.startswith(("jve", "qve", "xve", "yve")):
            s = s[0] + "ue"
        if not VALID_SYL_RE.match(s):
            return None
        out.append(s)
    return " ".join(out)


def clean_markdown_to_paragraphs(md_text: str) -> list[str]:
    # 1. Strip YAML frontmatter
    if md_text.startswith("---"):
        parts = md_text.split("---", 2)
        if len(parts) >= 3:
            md_text = parts[2]
    # 2. Strip fenced code blocks and HTML blocks
    md_text = re.sub(r"```[\s\S]*?```", "\n", md_text)
    md_text = re.sub(r"~~~[\s\S]*?~~~", "\n", md_text)
    md_text = re.sub(r"<[^>]+>", "", md_text)
    # 3. Convert markdown links `[label](url)` -> `label`, drop images `![alt](url)`
    md_text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md_text)
    md_text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md_text)
    # 4. Scrub URLs, emails, keys
    md_text = SECRET_OR_URL_RE.sub("", md_text)

    paragraphs: list[str] = []
    for raw_line in md_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ">", "|", "- ", "* ", "1.", "```")):
            # Still keep clean bullet/blockquote prose if long enough!
            line = re.sub(r"^[#>*\-\d.\s]+", "", line).strip()
        line = re.sub(r"[`*_~]+", "", line).strip()
        if len(line) < 18:
            continue
        cjk_count = sum(1 for c in line if "\u4e00" <= c <= "\u9fa5")
        if cjk_count < 10:
            continue
        paragraphs.append(line)
    return paragraphs


def prepare_tech_pack(dist_dir: Path) -> None:
    dist_dir.mkdir(parents=True, exist_ok=True)
    lex_lines = []
    for surf, py, lp, is_tech in PUBLIC_TECH_TERMS:
        lex_lines.append(f"{surf}\t{py}\t{lp:.3f}\t{'1' if is_tech else '0'}")
    for eng, lp in PUBLIC_TECH_ENGLISH:
        lex_lines.append(f"EN\t{eng}\t{lp:.3f}\t1")

    (dist_dir / "tech_pack_lexicon.tsv").write_text("\n".join(lex_lines) + "\n", encoding="utf-8")
    (dist_dir / "tech_lora_corpus.txt").write_text(
        "\n".join(PUBLIC_TECH_CORPUS_SEED * 25) + "\n",
        encoding="utf-8",
    )
    print(f"[tech-developer] Wrote {len(lex_lines)} lexical entries to {dist_dir / 'tech_pack_lexicon.tsv'}")


def prepare_private_personal(posts_root: Path, private_out_dir: Path) -> None:
    private_out_dir.mkdir(parents=True, exist_ok=True)
    pre_2023_years = [str(y) for y in range(2013, 2023)]
    md_files: list[Path] = []
    for yr in pre_2023_years:
        yr_dir = posts_root / yr
        if yr_dir.is_dir():
            md_files.extend(sorted(yr_dir.rglob("*.md")))
    if not md_files:
        md_files = sorted(posts_root.rglob("*.md"))

    all_paragraphs: list[str] = []
    word_counts: collections.Counter[str] = collections.Counter()
    bigram_counts: collections.Counter[tuple[str, str]] = collections.Counter()

    for path in md_files:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        paras = clean_markdown_to_paragraphs(raw)
        all_paragraphs.extend(paras)
        for p in paras:
            tokens = [t.strip() for t in jieba.cut(p, cut_all=False) if t.strip()]
            prev_cjk = None
            for tok in tokens:
                if CJK_WORD_RE.match(tok):
                    word_counts[tok] += 1
                    if prev_cjk is not None:
                        bigram_counts[(prev_cjk, tok)] += 1
                    prev_cjk = tok
                elif not tok.isalnum():
                    prev_cjk = None

    total_words = max(1, sum(word_counts.values()))
    lex_rows: list[str] = []
    for word, cnt in word_counts.most_common(8000):
        if cnt < 3:
            break
        py = word_to_pinyin(word)
        if not py:
            continue
        # Smooth personal frequency into [-4.2, -2.7]
        import math

        lp = max(-4.25, min(-2.75, math.log(cnt / total_words) + 4.15))
        lex_rows.append(f"{word}\t{py}\t{lp:.3f}\t0")

    ngram_rows: list[str] = []
    for (w1, w2), cnt in bigram_counts.most_common(6000):
        if cnt < 3:
            break
        import math

        bonus = min(1.85, 0.35 + 0.28 * math.log1p(cnt))
        ngram_rows.append(f"2\t{w1}\t{w2}\t{bonus:.3f}")

    (private_out_dir / "personal_lexicon.tsv").write_text("\n".join(lex_rows) + "\n", encoding="utf-8")
    (private_out_dir / "personal_ngrams.tsv").write_text("\n".join(ngram_rows) + "\n", encoding="utf-8")
    (private_out_dir / "personal_lora_corpus.txt").write_text(
        "\n".join(all_paragraphs) + "\n",
        encoding="utf-8",
    )
    total_chars = sum(len(p) for p in all_paragraphs)
    print(
        f"[private-personal] Processed {len(md_files)} hand-written essays "
        f"({len(all_paragraphs):,} paragraphs, {total_chars:,} chars) -> "
        f"{len(lex_rows):,} personal words, {len(ngram_rows):,} personal bigrams in {private_out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--personal-posts",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--private-out",
        type=Path,
        default=Path("/tmp/chime_private_personal"),
    )
    args = parser.parse_args()

    prepare_tech_pack(args.dist_dir)
    if args.personal_posts is not None and args.personal_posts.exists():
        prepare_private_personal(args.personal_posts, args.private_out)


if __name__ == "__main__":
    main()
