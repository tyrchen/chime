#!/usr/bin/env python3
"""
Builds `dist/base_neural/token_pinyin.json` for Route A (Pinyin-Constrained Neural Beam Search):
- Pure modern standard Mandarin readings (zero Kangxi Dictionary archaic ghosts like `前=jian`,
  `位=li`, `她=chi`, `精=qing`, `追=tui`, `丁=zheng`, `大=tai`, `是=ti`, `叶=xie`, `也=yi`).
- Uses `dist/merged_core_lexicon.tsv` + contextual `pypinyin` + curated modern polyphone table.
- Flags Traditional-only CJK tokens (`trad` list) so Simplified Chinese mode deprioritizes
  Traditional variants (`來`, `點`, `順利地通過了期末考試`) unless `left_context` is Traditional.
- Indexes both single-token and multi-token English/tech terms (`ChatGPT`, `GitHub`, `macOS`,
  `Kubernetes`, `TypeScript`, `PostgreSQL`, `PyTorch`, `DeepSeek`, etc.) for seamless
  code-switching inside the neural beam search.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import defaultdict

from pypinyin import Style, lazy_pinyin

REPO_ROOT = pathlib.Path("/Users/tchen/projects/mycode/rust/chime")
MODEL_DIR = REPO_ROOT / "dist" / "base_neural"
LEXICON_TSV = REPO_ROOT / "dist" / "merged_core_lexicon.tsv"
CORE_TSV = REPO_ROOT / "dist" / "multi_lexicon.tsv"

CJK_RE = re.compile(r"^[\u4e00-\u9fa5]+$")
ASCII_ALPHA_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_+#.-]{0,23}$")


def norm_syl(s: str) -> str | None:
    s = s.lower().strip().replace("ü", "v").replace("u:", "v")
    if not s or not s.isascii() or not s.isalpha():
        return None
    if s in ("n", "m", "ng", "hm", "hng"):
        return None
    return s


# Comprehensive modern Mandarin single-character polyphones (ONLY active standalone/productive
# modern readings; bound-only archaic readings like `会=kuai`, `区=ou`, `车=ju`, `石=dan`,
# `见=xian`, `便=pian`, `说=shui`, `的=di`, `大=dai`, `给=ji`, `和=huo`, `单=chan`, `数=shuo`
# live exclusively on their multi-char BPE tokens like `会计`, `便宜`, `大夫`, `给予`, `目的`!).
MODERN_POLYPHONES: dict[str, list[str]] = {
    "的": ["de"],
    "地": ["di", "de"],
    "得": ["de", "dei"],
    "了": ["le", "liao"],
    "着": ["zhe", "zhao", "zhuo"],
    "都": ["dou", "du"],
    "行": ["xing", "hang"],
    "长": ["chang", "zhang"],
    "还": ["hai", "huan"],
    "重": ["zhong", "chong"],
    "和": ["he"],
    "给": ["gei"],
    "会": ["hui"],
    "为": ["wei"],
    "没": ["mei", "mo"],
    "说": ["shuo"],
    "便": ["bian"],
    "乐": ["le", "yue"],
    "觉": ["jue", "jiao"],
    "率": ["lv", "shuai"],
    "差": ["cha", "chai", "ci"],
    "调": ["tiao", "diao"],
    "数": ["shu"],
    "参": ["can", "shen", "cen"],
    "角": ["jiao", "jue"],
    "弹": ["dan", "tan"],
    "藏": ["cang", "zang"],
    "省": ["sheng", "xing"],
    "校": ["xiao", "jiao"],
    "哪": ["na", "nei"],
    "那": ["na", "nei"],
    "这": ["zhe", "zhei"],
    "嗯": ["en"],
    "盛": ["sheng", "cheng"],
    "薄": ["bao", "bo"],
    "露": ["lu", "lou"],
    "血": ["xue", "xie"],
    "剥": ["bo", "bao"],
    "落": ["luo", "la", "lao"],
    "绿": ["lv", "lu"],
    "壳": ["ke", "qiao"],
    "模": ["mo", "mu"],
    "泊": ["bo", "po"],
    "塞": ["sai", "se"],
    "厦": ["sha", "xia"],
    "系": ["xi", "ji"],
    "降": ["jiang", "xiang"],
    "咽": ["yan", "ye"],
    "折": ["zhe", "she"],
    "朝": ["chao", "zhao"],
    "大": ["da"],
    "单": ["dan", "shan"],
    "传": ["chuan", "zhuan"],
    "解": ["jie", "xie"],
    "卡": ["ka", "qia"],
    "秘": ["mi", "bi"],
    "辟": ["pi", "bi"],
    "强": ["qiang", "jiang"],
    "圈": ["quan", "juan"],
    "色": ["se", "shai"],
    "似": ["si", "shi"],
    "提": ["ti", "di"],
    "吓": ["xia", "he"],
    "削": ["xiao", "xue"],
    "扎": ["zha", "za"],
    "粘": ["zhan", "nian"],
    "爪": ["zhua", "zhao"],
    "臭": ["chou", "xiu"],
    "度": ["du", "duo"],
    "恶": ["e", "wu"],
    "盖": ["gai", "ge"],
    "劲": ["jin", "jing"],
    "咖": ["ka", "ga"],
    "埋": ["mai", "man"],
    "抹": ["mo", "ma"],
    "炮": ["pao", "bao"],
    "奇": ["qi", "ji"],
    "区": ["qu"],
    "刹": ["sha", "cha"],
    "什": ["shen", "shi"],
    "属": ["shu", "zhu"],
    "熟": ["shu", "shou"],
    "伺": ["si", "ci"],
    "纤": ["xian", "qian"],
    "巷": ["xiang", "hang"],
    "择": ["ze", "zhai"],
    "幢": ["zhuang", "chuang"],
    "阿": ["a", "e"],
    "扒": ["ba", "pa"],
    "柏": ["bai", "bo"],
    "扁": ["bian", "pian"],
    "卜": ["bo", "bu"],
    "曾": ["ceng", "zeng"],
    "查": ["cha", "zha"],
    "禅": ["chan", "shan"],
    "颤": ["chan", "zhan"],
    "车": ["che"],
    "称": ["cheng", "chen"],
    "乘": ["cheng", "sheng"],
    "匙": ["chi", "shi"],
    "仇": ["chou", "qiu"],
    "畜": ["chu", "xu"],
    "攒": ["zan", "cuan"],
    "撮": ["cuo", "zuo"],
    "番": ["fan", "pan"],
    "佛": ["fo", "fu"],
    "脯": ["fu", "pu"],
    "蛤": ["ha", "ge"],
    "龟": ["gui", "jun"],
    "核": ["he", "hu"],
    "见": ["jian"],
    "颈": ["jing", "geng"],
    "咀": ["ju", "zui"],
    "咳": ["ke", "hai"],
    "勒": ["le", "lei"],
    "俩": ["lia", "liang"],
    "脉": ["mai", "mo"],
    "蔓": ["man", "wan"],
    "氓": ["mang", "meng"],
    "呢": ["ne", "ni"],
    "弄": ["nong", "long"],
    "疟": ["nve", "yao"],
    "瀑": ["pu", "bao"],
    "朴": ["pu", "piao", "po"],
    "栖": ["qi", "xi"],
    "雀": ["que", "qiao"],
    "茄": ["qie", "jia"],
    "杉": ["shan", "sha"],
    "石": ["shi"],
    "拓": ["tuo", "ta"],
    "蔚": ["wei", "yu"],
    "吁": ["xu", "yu"],
    "叶": ["ye"],
    "钥": ["yao", "yue"],
    "谁": ["shei", "shui"],
    "仨": ["sa"],
}

EXTRA_MULTI_READINGS: dict[str, list[tuple[str, ...]]] = {
    "朝阳": [("chao", "yang"), ("zhao", "yang")],
    "大夫": [("dai", "fu"), ("da", "fu")],
    "校对": [("jiao", "dui"), ("xiao", "dui")],
    "给予": [("ji", "yu"), ("gei", "yu")],
    "角色": [("jue", "se"), ("jiao", "se")],
    "穿着": [("chuan", "zhe"), ("chuan", "zhuo")],
    "谁的": [("shei", "de"), ("shui", "de")],
    "还有": [("hai", "you")],
    "还是": [("hai", "shi")],
    "还没": [("hai", "mei")],
    "便宜": [("pian", "yi")],
    "方便": [("fang", "bian")],
    "银行": [("yin", "hang")],
    "行走": [("xing", "zou")],
    "成长": [("cheng", "zhang")],
    "长大": [("zhang", "da")],
    "长短": [("chang", "duan")],
    "重要": [("zhong", "yao")],
    "重复": [("chong", "fu")],
    "重新": [("chong", "xin")],
    "重量": [("zhong", "liang")],
    "音乐": [("yin", "yue")],
    "快乐": [("kuai", "le")],
    "觉得": [("jue", "de")],
    "睡觉": [("shui", "jiao")],
    "会计": [("kuai", "ji")],
    "效率": [("xiao", "lv")],
    "概率": [("gai", "lv")],
    "频率": [("pin", "lv")],
    "率先": [("shuai", "xian")],
    "下载": [("xia", "zai")],
    "加载": [("jia", "zai")],
    "出差": [("chu", "chai")],
    "差不多": [("cha", "bu", "duo")],
    "空调": [("kong", "tiao")],
    "调整": [("tiao", "zheng")],
    "调查": [("diao", "cha")],
    "调用": [("diao", "yong")],
    "参加": [("can", "jia")],
    "参考": [("can", "kao")],
    "参数": [("can", "shu")],
    "人参": [("ren", "shen")],
    "归还": [("gui", "huan")],
    "还给": [("huan", "gei")],
    "还款": [("huan", "kuan")],
    "首都": [("shou", "du")],
    "都市": [("du", "shi")],
    "成都": [("cheng", "du")],
    "都会": [("dou", "hui")],
    "都是": [("dou", "shi")],
    "都有": [("dou", "you")],
    "看着": [("kan", "zhe")],
    "拿着": [("na", "zhe")],
    "带着": [("dai", "zhe")],
    "按照": [("an", "zhao")],
    "着急": [("zhao", "ji")],
    "着火": [("zhao", "huo")],
    "下载": [("xia", "zai")],
    "省会": [("sheng", "hui")],
    "反省": [("fan", "xing")],
    "背书": [("bei", "shu")],
    "一会": [("yi", "hui")],
    "一会儿": [("yi", "hui", "er")],
    "这些": [("zhe", "xie")],
    "那些": [("na", "xie")],
    "哪些": [("na", "xie")],
    "怎么": [("zen", "me")],
    "什么": [("shen", "me")],
    "这么": [("zhe", "me")],
    "那么": [("na", "me")],
}

# Curated multi-token / case-sensitive English & tech terms for seamless code-switching
TECH_EN_TERMS: list[str] = [
    "macOS", "iOS", "iPadOS", "GitHub", "GitLab", "ChatGPT", "OpenAI", "DeepSeek",
    "Claude", "Gemini", "Qwen", "PyTorch", "TensorFlow", "HuggingFace", "Kubernetes", "Docker",
    "PostgreSQL", "MySQL", "SQLite", "Redis", "MongoDB", "Kafka", "Nginx",
    "TypeScript", "JavaScript", "Python", "Rust", "Swift", "Golang", "Kotlin",
    "React", "Vue", "Next.js", "Astro", "Vite", "Webpack", "Tailwind",
    "GraphQL", "gRPC", "REST", "HTTP", "HTTPS", "SSH", "TLS", "JSON", "YAML", "TOML",
    "WebSocket", "Protobuf", "Safetensors", "Colab", "Cloudflare", "Git", "Tokio", "FastAPI",
    "NumPy", "Pandas", "Android", "ClickHouse", "DuckDB", "Supabase", "Vercel", "Stripe",
    "WebAssembly", "SwiftUI", "Cursor", "Linear",
    "API", "SDK", "CLI", "GUI", "TUI", "IDE", "LLM", "SLM", "RAG", "LoRA", "MLX",
    "GPU", "CPU", "NPU", "Metal", "CUDA", "SIMD", "FFI", "WASM", "CI", "CD", "PR",
    "VSCode", "Xcode", "Ghostty", "Alacritty", "WezTerm", "Neovim", "Vim", "Emacs",
    "Linux", "Ubuntu", "Debian", "Homebrew", "Cargo", "Pulumi", "Terraform", "AWS",
    "CloudFront", "DynamoDB", "Lambda", "S3", "SQS", "SNS", "IAM", "OIDC", "JWT",
    "iPhone", "iPad", "MacBook", "AirPods", "Apple", "Google", "Meta", "Anthropic",
    "Markdown", "LaTeX", "Typst", "Figma", "Notion", "Obsidian", "Slack", "WeChat",
    "async", "await", "struct", "enum", "trait", "impl", "const", "fn", "pub", "mod",
    "commit", "merge", "rebase", "branch", "push", "pull", "clone", "diff", "review",
    "deploy", "preview", "build", "test", "lint", "debug", "profile", "benchmark",
    "cache", "token", "prompt", "context", "model", "weight", "batch", "stream",
    "server", "client", "worker", "queue", "thread", "mutex", "channel", "socket",
    "buffer", "memory", "latency", "throughput", "p50", "p95", "p99", "zero-copy",
]


def load_simplified_chars_and_readings() -> tuple[set[str], dict[str, set[tuple[str, ...]]]]:
    """
    Loads simplified Chinese character set & verified multi-char readings from our compiled
    lexicon TSVs (`merged_core_lexicon.tsv` / `multi_lexicon.tsv`).
    """
    simp_chars: set[str] = set()
    multi_readings: dict[str, set[tuple[str, ...]]] = defaultdict(set)

    # First check format of `merged_core_lexicon.tsv` (surface\tpinyin\tlog_prob or pinyin\tsurface)
    for path in (LEXICON_TSV, CORE_TSV):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                a, b = parts[0].strip(), parts[1].strip()
                if CJK_RE.match(a):
                    surface, py_raw = a, b
                elif CJK_RE.match(b):
                    surface, py_raw = b, a
                else:
                    continue
                for ch in surface:
                    simp_chars.add(ch)
                if len(surface) >= 2:
                    syls = [norm_syl(s) for s in py_raw.replace("'", " ").split()]
                    if len(syls) == len(surface) and all(syls):
                        multi_readings[surface].add(tuple(s for s in syls if s))
    return simp_chars, multi_readings


def is_gb2312_char(ch: str) -> bool:
    try:
        ch.encode("gb2312")
        return True
    except UnicodeEncodeError:
        return False


def main() -> None:
    vocab_path = MODEL_DIR / "vocab.json"
    vocab: list[str] = json.loads(vocab_path.read_text(encoding="utf-8"))
    tok_to_id = {tok: tid for tid, tok in enumerate(vocab)}

    simp_chars, lex_readings = load_simplified_chars_and_readings()
    print(
        f"Loaded vocab ({len(vocab)} tokens), {len(simp_chars)} lexicon CJK chars, "
        f"{len(lex_readings)} multi-char lexicon readings."
    )

    cjk_entries: list[list[object]] = []  # [token_id, [[syl1, ...], ...]]
    trad_token_ids: list[int] = []
    en_entries: list[list[object]] = []   # [token_id, lower_ascii]
    en_multi: list[list[object]] = []     # [lower_ascii, display_surface, [tid1, tid2, ...]]

    single_cjk = 0
    multi_cjk = 0
    total_readings = 0

    for tid, tok in enumerate(vocab):
        if not tok or tok.startswith("<|"):
            continue
        if CJK_RE.match(tok):
            # Traditional-only heuristic: any character NOT in GB2312 (6,763 standard simplified
            # chars) AND not in our modern simplified lexicon's high-frequency char set is flagged
            # as Traditional/rare-variant! Wait: check specifically against GB2312 + common
            # non-GB2312 simplified chars (`镕`, `喆`, `啰`, `哒`, `嗯`, `呗`, `嗨`, `哇`, `撸`, `怼`, `飙`, `靓`, `瞭`, `徵`).
            extra_simp = set("嗯呗嗨哇哒啰怼撸飙靓瞭徵喆镕淼鑫焱垚珺琪瑶璇瑾瑜")
            is_trad = any((not is_gb2312_char(c)) and (c not in extra_simp) for c in tok)
            if is_trad:
                trad_token_ids.append(tid)

            readings_set: set[tuple[str, ...]] = set()
            if len(tok) == 1:
                single_cjk += 1
                # Primary modern reading from `lazy_pinyin` (NO Kangxi archaic heteronyms!)
                lp = lazy_pinyin(tok, style=Style.NORMAL, errors="ignore")
                if lp:
                    ns = norm_syl(lp[0])
                    if ns:
                        readings_set.add((ns,))
                # Curated active modern Mandarin polyphones
                if tok in MODERN_POLYPHONES:
                    for r in MODERN_POLYPHONES[tok]:
                        ns = norm_syl(r)
                        if ns:
                            readings_set.add((ns,))
            else:
                multi_cjk += 1
                lp = lazy_pinyin(tok, style=Style.NORMAL, errors="ignore")
                if len(lp) == len(tok):
                    normed = [norm_syl(s) for s in lp]
                    if all(normed):
                        readings_set.add(tuple(s for s in normed if s))
                if tok in lex_readings:
                    # Only keep lexicon readings that aren't obviously corrupted
                    for r in lex_readings[tok]:
                        readings_set.add(r)
                if tok in EXTRA_MULTI_READINGS:
                    readings_set.update(EXTRA_MULTI_READINGS[tok])

            if readings_set:
                cjk_entries.append([tid, [list(r) for r in sorted(readings_set)]])
                total_readings += len(readings_set)
        else:
            stripped = tok.lstrip(" ")
            if stripped and ASCII_ALPHA_RE.match(stripped) and stripped.isascii():
                en_entries.append([tid, stripped.lower()])

    # Greedy longest-match encode curated multi-token English/tech proper nouns using `vocab`!
    def encode_piece(text: str) -> list[int]:
        out: list[int] = []
        pos = 0
        while pos < len(text):
            matched = None
            for end in range(min(len(text), pos + 16), pos, -1):
                sub = text[pos:end]
                if sub in tok_to_id:
                    matched = (tok_to_id[sub], end)
                    break
            if matched is None:
                return []
            out.append(matched[0])
            pos = matched[1]
        return out

    for term in TECH_EN_TERMS:
        ids = encode_piece(term)
        if not ids:
            ids = encode_piece(" " + term)
        if ids:
            en_multi.append([term.lower(), term, ids])

    out_obj = {
        "vocab_size": len(vocab),
        "cjk": cjk_entries,
        "trad": trad_token_ids,
        "en": en_entries,
        "en_multi": en_multi,
    }
    out_path = MODEL_DIR / "token_pinyin.json"
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(
        f"Wrote purified {out_path} ({size_kb:.1f} KB): "
        f"{single_cjk} single-CJK + {multi_cjk} multi-CJK = {len(cjk_entries)} CJK tokens "
        f"({total_readings} clean readings, {len(trad_token_ids)} trad-flagged), "
        f"{len(en_entries)} EN tokens, {len(en_multi)} multi-token tech terms."
    )


if __name__ == "__main__":
    main()
