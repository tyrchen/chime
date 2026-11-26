#!/usr/bin/env python3
"""
Builds 10,000-case development (`benchmarks/dev_suite_10k.tsv`) and 10,000-case strictly blind
holdout (`benchmarks/test_holdout_10k.tsv`) Chinese & mixed-English IME benchmarks across 13+
real-world scenarios with 0% overlap between dev and holdout and 0% overlap with distillation
corpora (`FineWeb-Edu-Chinese`, `COIG-CQIA/zhihu`, `COIG-CQIA/coig_pc`, `BelleGroup`, `wikipedia`).

STRICT BLIND RULE:
Never inspect individual rows or failures of `benchmarks/test_holdout_10k.tsv` during development.
Use `benchmarks/dev_suite_10k.tsv` exclusively for error analysis and tuning.
"""

from __future__ import annotations

import hashlib
import pathlib
import random
import re
from collections import Counter, defaultdict

from datasets import load_dataset
from pypinyin import Style, lazy_pinyin

REPO_ROOT = pathlib.Path("/Users/tchen/projects/mycode/rust/chime")
BENCH_DIR = REPO_ROOT / "benchmarks"

CJK_RE = re.compile(r"^[\u4e00-\u9fa5]+$")
MIXED_RE = re.compile(r"^[\u4e00-\u9fa5A-Za-z0-9 ]+$")
CLAUSE_SPLIT_RE = re.compile(r"[。！？；!?\n]+")
SUBCLAUSE_SPLIT_RE = re.compile(r"[，,、：:]+")

DE_DI_ZAI_TA_CHARS = set("的地得在再他她它做作那哪")
POLYPHONE_CHARS = set("行长重还发为乐都觉数传量调处省差解角参着应当更少干空假背率便")
HIATUS_PATTERNS = (
    "xian", "fangan", "pingan", "fanyi", "changan", "tianan", "jiaoao",
    "haiou", "jiange", "nengai", "kuai", "dianan", "shian", "jianan",
    "minge", "qie", "pie", "die", "xie", "liange",
)

XIAOHE_INITIAL = {"zh": "v", "ch": "i", "sh": "u"}
XIAOHE_FINAL = {
    "a": "a", "o": "o", "e": "e", "i": "i", "u": "u", "v": "v",
    "ai": "d", "ei": "w", "ui": "v", "ao": "c", "ou": "z", "iu": "q",
    "ie": "p", "ve": "t", "ue": "t", "er": "e",
    "an": "j", "en": "f", "in": "b", "un": "y", "vn": "y",
    "ang": "h", "eng": "g", "ing": "k", "ong": "s", "iong": "s",
    "ia": "x", "iao": "n", "ian": "m", "iang": "l",
    "ua": "x", "uo": "o", "uai": "k", "uan": "r", "uang": "l",
}
ZIRANMA_FINAL = {
    "a": "a", "o": "o", "e": "e", "i": "i", "u": "u", "v": "v",
    "ai": "l", "ei": "z", "ui": "v", "ao": "k", "ou": "b", "iu": "q",
    "ie": "x", "ve": "t", "ue": "t", "er": "e",
    "an": "j", "en": "f", "in": "n", "un": "p", "vn": "p",
    "ang": "h", "eng": "g", "ing": "y", "ong": "s", "iong": "s",
    "ia": "w", "iao": "c", "ian": "m", "iang": "d",
    "ua": "w", "uo": "o", "uai": "y", "uan": "r", "uang": "d",
}
MS_FINAL = {
    "a": "a", "o": "o", "e": "e", "i": "i", "u": "u", "v": "y",
    "ai": "l", "ei": "z", "ui": "v", "ao": "k", "ou": "b", "iu": "q",
    "ie": "x", "ve": "t", "ue": "t", "er": "r",
    "an": "j", "en": "f", "in": "n", "un": "p", "vn": "p",
    "ang": "h", "eng": "g", "ing": ";", "ong": "s", "iong": "s",
    "ia": "w", "iao": "c", "ian": "m", "iang": "d",
    "ua": "w", "uo": "o", "uai": "y", "uan": "r", "uang": "d",
}

QWERTY_NEIGHBORS = {
    "a": "s", "b": "v", "c": "x", "d": "f", "e": "r", "f": "g", "g": "h",
    "h": "j", "i": "o", "j": "k", "k": "l", "l": "k", "m": "n", "n": "m",
    "o": "p", "p": "o", "q": "w", "r": "t", "s": "d", "t": "y", "u": "i",
    "v": "b", "w": "e", "x": "c", "y": "u", "z": "x",
}

FUZZY_SWAPS = [
    ("zh", "z"), ("ch", "c"), ("sh", "s"),
    ("ang", "an"), ("eng", "en"), ("ing", "in"),
]


def to_syllables(text: str) -> list[str] | None:
    syls = lazy_pinyin(text, style=Style.NORMAL, errors="default")
    if len(syls) != len(text):
        return None
    out: list[str] = []
    for s in syls:
        norm = s.lower().replace("ü", "v").replace("u:", "v")
        if not norm.isalpha() or not norm.isascii():
            return None
        out.append(norm)
    return out


def to_mixed_keys(text: str) -> str | None:
    """Converts mixed CJK + ASCII English sentence into raw lowercase keystrokes."""
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == " ":
            i += 1
            continue
        if "\u4e00" <= ch <= "\u9fa5":
            j = i
            while j < len(text) and "\u4e00" <= text[j] <= "\u9fa5":
                j += 1
            syls = to_syllables(text[i:j])
            if not syls:
                return None
            out.extend(syls)
            i = j
        elif ch.isascii() and ch.isalpha():
            j = i
            while j < len(text) and text[j].isascii() and text[j].isalpha():
                j += 1
            out.append(text[i:j].lower())
            i = j
        else:
            return None
    return "".join(out) if out else None


def split_init_final(syl: str) -> tuple[str, str]:
    for retro in ("zh", "ch", "sh"):
        if syl.startswith(retro):
            return retro, syl[2:]
    if syl in ("a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "er"):
        return "", syl
    return syl[0], syl[1:]


def encode_shuangpin(syls: list[str], scheme: str) -> str | None:
    fmap = {"xiaohe": XIAOHE_FINAL, "ziranma": ZIRANMA_FINAL, "microsoft": MS_FINAL}[scheme]
    out: list[str] = []
    for syl in syls:
        init, fin = split_init_final(syl)
        if not init:
            if scheme == "microsoft":
                if len(syl) == 1:
                    out.append("o" + syl)
                elif len(syl) == 2:
                    out.append("o" + fmap.get(syl, ""))
                    if len(out[-1]) != 2:
                        return None
                elif syl in ("ang", "eng"):
                    out.append("o" + fmap[syl])
                else:
                    return None
            else:
                if len(syl) == 1:
                    out.append(syl * 2)
                elif len(syl) == 2:
                    out.append(syl)
                elif scheme == "xiaohe" and syl in ("ang", "eng"):
                    out.append(syl[0] + ("h" if syl == "ang" else "g"))
                else:
                    return None
            continue
        if init in ("j", "q", "x", "y") and fin == "ue":
            fin = "ve"
        ic = XIAOHE_INITIAL.get(init, init)
        fc = fmap.get(fin)
        if not fc:
            return None
        out.append(ic + fc)
    return "".join(out)


def apply_fuzzy(syls: list[str], rng: random.Random) -> str | None:
    indices = list(range(len(syls)))
    rng.shuffle(indices)
    for idx in indices:
        s = syls[idx]
        for a, b in FUZZY_SWAPS:
            if s.startswith(a) and len(a) == 2:
                mut = list(syls)
                mut[idx] = b + s[2:]
                return "".join(mut)
            if s.endswith(a) and len(a) == 3:
                mut = list(syls)
                mut[idx] = s[:-3] + b
                return "".join(mut)
            if s.endswith(b) and len(b) == 2 and not s.endswith(a):
                mut = list(syls)
                mut[idx] = s[:-2] + a
                return "".join(mut)
    return None


def apply_typo(keys: str, rng: random.Random) -> str | None:
    if len(keys) < 6:
        return None
    if rng.random() < 0.65:
        # Adjacent transposition on internal non-identical letters
        cands = [i for i in range(1, len(keys) - 2) if keys[i] != keys[i + 1]]
        if not cands:
            return None
        i = rng.choice(cands)
        return keys[:i] + keys[i + 1] + keys[i] + keys[i + 2:]
    # Single QWERTY neighbor slip on an internal letter
    i = rng.randint(1, len(keys) - 2)
    nb = QWERTY_NEIGHBORS.get(keys[i])
    if not nb:
        return None
    return keys[:i] + nb + keys[i + 1:]


def doc_split(doc_text: str) -> str:
    h = hashlib.blake2b(doc_text.encode("utf-8"), digest_size=8).digest()
    return "dev" if (h[0] & 1) == 0 else "holdout"


def load_existing_tsv(path: pathlib.Path) -> list[tuple[str, str, str, str, str, str]]:
    rows = []
    if not path.exists():
        return rows
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if idx == 0 or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 6:
            rows.append((parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]))
    return rows


def clean_text(text: str) -> str:
    text = text.replace("\t", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_cjk_clauses(passage: str) -> list[tuple[str, str]]:
    """Returns `(left_context, clause)` pairs from a passage."""
    out: list[tuple[str, str]] = []
    # Split on sentence-ending punctuation while keeping running left context
    pos = 0
    for m in re.finditer(r"[^。！？；!?\n，,、：:]+", passage):
        seg = m.group(0).strip()
        start = m.start()
        if CJK_RE.match(seg) and 2 <= len(seg) <= 16:
            raw_left = passage[max(0, start - 96):start].strip()
            # Keep clean left context without tabs/newlines
            left = clean_text(raw_left).replace("\n", "")
            if len(left) > 90:
                left = left[-90:]
            out.append((left, seg))
        pos = m.end()
    _ = pos
    return out


def extract_mixed_clauses(passage: str) -> list[tuple[str, str]]:
    """Extracts realistic Chinese+English mixed clauses (e.g. `用 Python 处理数据`)."""
    out: list[tuple[str, str]] = []
    for m in re.finditer(r"[^。！？；!?\n，,：:（）()\[\]\"'`]+", passage):
        seg = re.sub(r"\s+", " ", m.group(0)).strip()
        if not (5 <= len(seg) <= 22):
            continue
        if not MIXED_RE.match(seg):
            continue
        cjk_cnt = sum(1 for c in seg if "\u4e00" <= c <= "\u9fa5")
        en_words = re.findall(r"[A-Za-z]{2,12}", seg)
        if cjk_cnt < 3 or len(en_words) != 1:
            continue
        if any(c.isdigit() for c in seg):
            continue
        # Normalize Pangu spacing around the single English word
        w = en_words[0]
        if w.lower() in {"http", "https", "www", "com", "org", "nbsp"}:
            continue
        norm = re.sub(r"([\u4e00-\u9fa5])([A-Za-z])", r"\1 \2", seg)
        norm = re.sub(r"([A-Za-z])([\u4e00-\u9fa5])", r"\1 \2", norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        out.append(("", norm))
    return out


def stream_disjoint_passages() -> dict[str, list[tuple[str, str]]]:
    """
    Streams public Chinese corpora completely disjoint from our Colab distillation sources.
    Returns `{"dev": [(domain, text), ...], "holdout": [(domain, text), ...]}`.
    """
    buckets: dict[str, list[tuple[str, str]]] = {"dev": [], "holdout": []}
    seen_doc_hashes: set[bytes] = set()

    sources = [
        ("m-a-p/COIG-CQIA", "douban", "train", ["instruction", "output"], "spoken", 4000),
        ("m-a-p/COIG-CQIA", "segmentfault", "train", ["instruction", "output"], "tech", 3500),
        ("m-a-p/COIG-CQIA", "finance", "train", ["instruction", "output"], "prose", 2500),
        ("m-a-p/COIG-CQIA", "ruozhiba", "train", ["instruction", "output"], "spoken", 1500),
        ("clue/clue", "c3", "train", ["context", "question"], "spoken", 4000),
        ("clue/clue", "cmrc2018", "train", ["context", "question"], "prose", 3000),
        ("clue/clue", "afqmc", "train", ["sentence1", "sentence2"], "spoken", 5000),
        ("clue/clue", "ocnli", "train", ["sentence1", "sentence2"], "spoken", 5000),
        ("clue/clue", "tnews", "train", ["sentence"], "prose", 4000),
        ("clue/clue", "iflytek", "train", ["sentence"], "tech", 3000),
    ]

    for repo, sub, split, fields, domain, limit in sources:
        print(f"Streaming {repo}/{sub} ({domain}, limit={limit})...", flush=True)
        try:
            ds = load_dataset(repo, sub, split=split, streaming=True)
            cnt = 0
            for row in ds:
                parts: list[str] = []
                for f in fields:
                    val = row.get(f)
                    if isinstance(val, list):
                        parts.extend(str(x) for x in val if x)
                    elif val:
                        parts.append(str(val))
                text = clean_text(" ".join(parts))
                if len(text) < 12:
                    continue
                h = hashlib.blake2b(text[:200].encode("utf-8"), digest_size=8).digest()
                if h in seen_doc_hashes:
                    continue
                seen_doc_hashes.add(h)
                sp = doc_split(text[:200])
                buckets[sp].append((domain, text[:600]))
                cnt += 1
                if cnt >= limit:
                    break
            print(f"  -> collected {cnt} docs", flush=True)
        except Exception as exc:
            print(f"  -> skipped {repo}/{sub}: {exc}", flush=True)

    return buckets


def build_split(
    split_name: str,
    existing_rows: list[tuple[str, str, str, str, str, str]],
    docs: list[tuple[str, str]],
    forbidden_targets: set[str],
    target_total: int = 10_000,
) -> tuple[list[tuple[str, str, str, str, str, str]], set[str]]:
    rng = random.Random(20261126 if split_name == "dev" else 20261127)
    rng.shuffle(docs)

    rows: list[tuple[str, str, str, str, str, str]] = []
    used_targets: set[str] = set()
    used_pairs: set[tuple[str, str, str]] = set()

    def try_add(cat: str, scheme: str, app: str, ctx: str, keys: str, exp: str, unique_exp: bool = True) -> bool:
        if not keys or not exp:
            return False
        if "\t" in ctx or "\t" in keys or "\t" in exp or "\n" in ctx or "\n" in exp:
            return False
        if unique_exp and (exp in forbidden_targets or exp in used_targets):
            return False
        pair = (ctx[-16:], keys, exp)
        if pair in used_pairs:
            return False
        if len(exp) >= 2 and exp in forbidden_targets:
            return False
        rows.append((cat, scheme, app, ctx, keys, exp))
        used_pairs.add(pair)
        if len(exp) >= 2:
            used_targets.add(exp)
        return True

    # 1. Seed with existing curated rows first
    for cat, scheme, app, ctx, keys, exp in existing_rows:
        try_add(cat, scheme, app, ctx, keys, exp, unique_exp=False)

    remaining = max(0, target_total - len(rows))
    quotas = {
        "standalone_sentence": int(remaining * 0.19),
        "contextual_sentence": int(remaining * 0.19),
        "contextual_short_phrase": int(remaining * 0.13),
        "contextual_single_char": int(remaining * 0.10),
        "de_di_de_zai_ta": int(remaining * 0.10),
        "polyphone_heteronym": int(remaining * 0.08),
        "tech_developer_prose": int(remaining * 0.07),
        "fuzzy_pinyin": int(remaining * 0.05),
        "motor_typos": int(remaining * 0.04),
        "shuangpin_all3": int(remaining * 0.03),
        "hiatus_segmentation": int(remaining * 0.02),
    }
    # Fill rounding remainder into contextual_sentence
    quotas["contextual_sentence"] += remaining - sum(quotas.values())
    counts: Counter[str] = Counter()

    def need(cat: str) -> bool:
        return counts[cat] < quotas[cat]

    sp_schemes = ["xiaohe", "ziranma", "microsoft"]

    for domain, passage in docs:
        if len(rows) >= target_total:
            break

        # Also harvest code-switching mixed sentences from tech docs into tech_developer_prose
        if domain == "tech" and need("tech_developer_prose"):
            for _, mseg in extract_mixed_clauses(passage):
                if not need("tech_developer_prose"):
                    break
                mkeys = to_mixed_keys(mseg)
                if mkeys and 6 <= len(mkeys) <= 36:
                    if try_add("tech_developer_prose", "full", "com.microsoft.VSCode", "", mkeys, mseg):
                        counts["tech_developer_prose"] += 1

        clauses = extract_cjk_clauses(passage)
        rng.shuffle(clauses)
        for left, seg in clauses:
            if len(rows) >= target_total:
                break
            syls = to_syllables(seg)
            if not syls:
                continue
            keys = "".join(syls)
            n_chars = len(seg)
            app = "com.microsoft.VSCode" if domain == "tech" else "com.apple.Notes"

            # Contextual single-char continuation from real clause interior (only when left_ctx >= 10)
            if need("contextual_single_char") and n_chars >= 6 and len(left) >= 6 and rng.random() < 0.22:
                cut = rng.randint(3, n_chars - 2)
                c_left = (left + "，" + seg[:cut])[-90:]
                c_exp = seg[cut]
                c_key = syls[cut]
                # Require non-trivial syllable with homophones (avoid unique single-char syllables)
                if len(c_key) >= 2 and try_add("contextual_single_char", "full", app, c_left, c_key, c_exp, unique_exp=False):
                    counts["contextual_single_char"] += 1
                    continue

            # Contextual 2-3 char short phrase continuation
            if need("contextual_short_phrase") and n_chars >= 6 and len(left) >= 6 and rng.random() < 0.22:
                span = rng.choice([2, 3])
                cut = rng.randint(2, n_chars - span)
                c_left = (left + "，" + seg[:cut])[-90:]
                c_exp = seg[cut:cut + span]
                c_key = "".join(syls[cut:cut + span])
                if try_add("contextual_short_phrase", "full", app, c_left, c_key, c_exp):
                    counts["contextual_short_phrase"] += 1
                    continue

            if n_chars < 4:
                continue

            # Hiatus segmentation
            if need("hiatus_segmentation") and any(p in keys for p in HIATUS_PATTERNS):
                if try_add("hiatus_segmentation", "full", app, left if len(left) >= 12 else "", keys, seg):
                    counts["hiatus_segmentation"] += 1
                    continue

            # Tech / developer prose
            if domain == "tech" and need("tech_developer_prose") and 5 <= n_chars <= 14:
                if try_add("tech_developer_prose", "full", "com.mitchellh.ghostty", left if len(left) >= 12 else "", keys, seg):
                    counts["tech_developer_prose"] += 1
                    continue

            # 的/地/得/在/再/他/她/它/做/作/那/哪
            if need("de_di_de_zai_ta") and 5 <= n_chars <= 14 and any(c in DE_DI_ZAI_TA_CHARS for c in seg):
                if try_add("de_di_de_zai_ta", "full", app, left if len(left) >= 12 and rng.random() < 0.5 else "", keys, seg):
                    counts["de_di_de_zai_ta"] += 1
                    continue

            # Polyphone / heteronym
            if need("polyphone_heteronym") and 5 <= n_chars <= 14 and any(c in POLYPHONE_CHARS for c in seg):
                if try_add("polyphone_heteronym", "full", app, left if len(left) >= 12 and rng.random() < 0.5 else "", keys, seg):
                    counts["polyphone_heteronym"] += 1
                    continue

            # Fuzzy pinyin
            if need("fuzzy_pinyin") and 5 <= n_chars <= 12:
                fkeys = apply_fuzzy(syls, rng)
                if fkeys and fkeys != keys:
                    if try_add("fuzzy_pinyin", "full", app, "", fkeys, seg):
                        counts["fuzzy_pinyin"] += 1
                        continue

            # Motor typos
            if need("motor_typos") and 5 <= n_chars <= 12:
                tkeys = apply_typo(keys, rng)
                if tkeys and tkeys != keys:
                    if try_add("motor_typos", "full", app, "", tkeys, seg):
                        counts["motor_typos"] += 1
                        continue

            # Shuangpin (Xiaohe / Ziranma / Microsoft)
            if need("shuangpin_all3") and 5 <= n_chars <= 12:
                sch = sp_schemes[counts["shuangpin_all3"] % 3]
                sp_keys = encode_shuangpin(syls, sch)
                if sp_keys:
                    if try_add("shuangpin_all3", sch, app, "", sp_keys, seg):
                        counts["shuangpin_all3"] += 1
                        continue

            # Contextual full sentence (4-14 chars with >= 16 char real left context)
            if need("contextual_sentence") and 4 <= n_chars <= 14 and len(left) >= 16:
                if try_add("contextual_sentence", "full", app, left, keys, seg):
                    counts["contextual_sentence"] += 1
                    continue

            # Standalone full sentence (5-14 chars, no context)
            if need("standalone_sentence") and 5 <= n_chars <= 14:
                if try_add("standalone_sentence", "full", app, "", keys, seg):
                    counts["standalone_sentence"] += 1
                    continue

    # Second pass if any quota fell slightly short: top up from remaining sentences
    if len(rows) < target_total:
        for domain, passage in docs:
            if len(rows) >= target_total:
                break
            for left, seg in extract_cjk_clauses(passage):
                if len(rows) >= target_total:
                    break
                if not (5 <= len(seg) <= 14):
                    continue
                syls = to_syllables(seg)
                if not syls:
                    continue
                keys = "".join(syls)
                cat = "contextual_sentence" if len(left) >= 16 else "standalone_sentence"
                if try_add(cat, "full", "com.apple.Notes", left if len(left) >= 16 else "", keys, seg):
                    counts[cat] += 1

    print(f"[{split_name}] Total={len(rows)} | Breakdown: {dict(sorted(counts.items()))}", flush=True)
    return rows, used_targets


def write_tsv(path: pathlib.Path, rows: list[tuple[str, str, str, str, str, str]]) -> None:
    lines = ["category\tscheme\tapp\tleft_ctx\tkeys\texpected"]
    for cat, scheme, app, ctx, keys, exp in rows:
        lines.append(f"{cat}\t{scheme}\t{app}\t{ctx}\t{keys}\t{exp}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dev_seed = load_existing_tsv(BENCH_DIR / "dev_suite.tsv") + load_existing_tsv(
        BENCH_DIR / "contextual_open_suite.tsv"
    )
    holdout_seed = load_existing_tsv(BENCH_DIR / "test_holdout.tsv")

    holdout_seed_targets = {r[5] for r in holdout_seed if len(r[5]) >= 2}
    dev_seed_targets = {r[5] for r in dev_seed if len(r[5]) >= 2}

    buckets = stream_disjoint_passages()

    dev_rows, dev_targets = build_split(
        "dev", dev_seed, buckets["dev"], forbidden_targets=holdout_seed_targets, target_total=10_000
    )
    holdout_rows, holdout_targets = build_split(
        "holdout",
        holdout_seed,
        buckets["holdout"],
        forbidden_targets=dev_targets | dev_seed_targets,
        target_total=10_000,
    )

    # Hard leak assertion: 0 overlapping multi-char targets between dev_10k and holdout_10k
    overlap = {r[5] for r in dev_rows if len(r[5]) >= 2} & {r[5] for r in holdout_rows if len(r[5]) >= 2}
    print(f"Leakage check (shared multi-char expected targets): {len(overlap)}", flush=True)
    assert len(overlap) == 0, f"Unexpected overlap: {list(overlap)[:5]}"

    write_tsv(BENCH_DIR / "dev_suite_10k.tsv", dev_rows)
    write_tsv(BENCH_DIR / "test_holdout_10k.tsv", holdout_rows)
    print(f"Wrote {BENCH_DIR / 'dev_suite_10k.tsv'} ({len(dev_rows)} rows)")
    print(f"Wrote {BENCH_DIR / 'test_holdout_10k.tsv'} ({len(holdout_rows)} rows)")


if __name__ == "__main__":
    main()
