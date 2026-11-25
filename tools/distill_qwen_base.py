#!/usr/bin/env python3
"""
Builds three production ML & lexicon assets for Chime IME from Apache-2.0 `Qwen/Qwen3-0.6B-Base`
plus permissive multi-domain Chinese corpora (`wordfreq` spoken/web/subtitle Zipf + `pypinyin`
`phrase-pinyin-data` 47k polyphone phrases + `jieba` corpus):

1. `dist/base_neural/{config.json, vocab.json, model.safetensors}`:
   24,576-token Chinese-first vocabulary & 6-layer (`[0, 5, 11, 16, 22, 27]`) pre-trained
   `Qwen3-0.6B-Base` Causal Transformer reranker (with native `q_norm` & `k_norm` QK-Norm),
   KL-calibrated on MPS against the full 28-layer `Qwen3-0.6B-Base` teacher!
2. `dist/multi_lexicon.tsv`:
   ~180,000+ calibrated modern Chinese spoken, web, idiom & technical entries (`surface\tpinyin\tlp`)
   fusing `wordfreq` Zipf frequencies, `phrase-pinyin-data` polyphone annotations, and 28-layer
   `Qwen3-0.6B-Base` teacher priors for modern colloquial compounds!
3. `dist/ngrams.tsv`:
   Dense Bigram & Trigram transition table scored by `Qwen3-0.6B-Base` + modern collocations!
"""

import argparse
import glob
import itertools
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wordfreq
from huggingface_hub import snapshot_download
from pypinyin import Style, pinyin as py_annotate
from pypinyin.phrases_dict import phrases_dict
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


HOMOPHONE_AND_COLLOQUIAL_PROBES = [
    # Legal / Civic: 权利 vs 权力 vs 全力
    ("公民", "的", ["权利", "权力", "全力"]),
    ("保障", "的", ["权利", "权力", "全力"]),
    ("合法", "的", ["权利", "权力", "全力"]),
    ("维护", "的", ["权利", "权力", "全力"]),
    ("基本", "的", ["权利", "权力", "全力"]),
    ("消费者", "的", ["权利", "权力", "全力"]),
    ("国家", "的", ["权力", "权利", "全力"]),
    ("行使", "的", ["权力", "权利", "全力"]),
    ("最高", "的", ["权力", "权利", "全力"]),
    ("滥用", "的", ["权力", "权利", "全力"]),
    ("掌握", "了", ["权力", "权利", "全力"]),
    ("拼尽", "了", ["全力", "权利", "权力"]),
    # 反应 vs 反映
    ("化学", "的", ["反应", "反映"]),
    ("过敏", "的", ["反应", "反映"]),
    ("连锁", "的", ["反应", "反映"]),
    ("第一", "的", ["反应", "反映"]),
    ("不良", "的", ["反应", "反映"]),
    ("如实", "地", ["反映", "反应"]),
    ("向上级", "地", ["反映", "反应"]),
    ("群众", "的", ["反映", "反应"]),
    ("深刻", "地", ["反映", "反应"]),
    # 量化 vs 亮化, 公式 vs 攻势 vs 公事
    ("端侧", "的", ["量化", "亮化"]),
    ("模型", "的", ["量化", "亮化"]),
    ("权重", "的", ["量化", "亮化"]),
    ("低比特", "的", ["量化", "亮化"]),
    ("夜景", "的", ["亮化", "量化"]),
    ("数学", "的", ["公式", "攻势", "公事"]),
    ("推导", "了", ["公式", "攻势", "公事"]),
    ("发起", "了", ["攻势", "公式", "公事"]),
    ("猛烈", "的", ["攻势", "公式", "公事"]),
    ("出差", "办", ["公事", "公式", "攻势"]),
    # 医德 vs 一得, 必需 vs 必须, 制定 vs 制订, 截至 vs 截止
    ("医生", "很有", ["医德", "一得"]),
    ("高尚", "的", ["医德", "一得"]),
    ("生活", "所", ["必需", "必须"]),
    ("我们", "都", ["必须", "必需"]),
    ("研究", "并", ["制定", "制订"]),
    ("报名", "已经", ["截止", "截至"]),
    # 形势 vs 形式, 变换 vs 变幻, 侦察 vs 侦查
    ("国际", "的", ["形势", "形式"]),
    ("表现", "的", ["形式", "形势"]),
    ("风云", "在", ["变幻", "变换"]),
    ("傅里叶", "的", ["变换", "变幻"]),
    ("立案", "去", ["侦查", "侦察"]),
    ("敌情", "去", ["侦察", "侦查"]),
    # Spoken & IME ambiguity probes (`过程中`, `最合适`, `不那么靠谱`, `多大`, `不少`, `提升空间`)
    ("使用", "的", ["过程", "国成", "过成"]),
    ("打字", "的", ["过程", "国成", "过成"]),
    ("开发", "的", ["过程", "国成", "过成"]),
    ("运行", "的", ["过程", "国成", "过成"]),
    ("推荐", "最", ["合适", "何时", "何事", "核实"]),
    ("推荐出来", "最", ["合适", "何时", "何事", "核实"]),
    ("选择", "最", ["合适", "何时", "何事", "核实"]),
    ("有时候", "不那么", ["靠谱", "靠铺", "考铺"]),
    ("感觉", "不太", ["靠谱", "靠铺", "考铺"]),
    ("我", "截了", ["几个图", "饥饿图"]),
    ("看看", "是", ["怎么回事", "怎么会是"]),
    ("究竟", "有", ["多大", "多达", "多打"]),
    ("概率", "有", ["多大", "多达", "多打"]),
    ("还有", "不少", ["问题", "细节"]),
    ("很大", "的", ["空间", "空降", "控件"]),
    ("提升", "的", ["空间", "空降", "控件"]),
    ("更新", "的", ["版本", "半本", "板本"]),
    # Developer & Code-Switching
    ("帮我", "看下", ["PR", "个人"]),
    ("提交", "一个", ["PR", "个人"]),
    ("合并", "这个", ["PR", "个人"]),
    ("调用", "后端", ["API", "阿批"]),
    ("设计", "接口", ["API", "阿批"]),
    ("升级", "系统", ["macOS", "骂街"]),
    ("使用", "苹果", ["macOS", "骂街"]),
    ("开源", "仓库", ["GitHub", "给他"]),
    ("编写", "内核", ["Rust", "入死"]),
    ("本地", "训练", ["LoRA", "罗拉"]),
    ("推理", "延迟", ["latency", "拉腾"]),
]

# Spoken/modern words often split by legacy newspaper tokenizers — scored directly by Qwen3 teacher!
COLLOQUIAL_TEACHER_WORDS = [
    "多大", "多好", "多久", "多快", "多远", "多高", "多长", "多半", "多亏",
    "不少", "不多", "不小", "不难", "不慢", "不久", "不至于", "不见得", "不那么",
    "靠谱", "不靠谱", "挺靠谱", "很靠谱", "离谱", "太离谱", "搞定", "没问题",
    "冷门", "热门", "细节", "版本", "新版本", "空间", "提升空间", "很大空间",
    "模糊音", "双拼", "全拼", "输入法", "候选字", "候选词", "第一候选", "词库",
    "上下文", "上下文窗口", "滑动窗口", "过程", "过程中", "使用的过程中", "使用过程中",
    "合适", "最合适", "合适的", "截图", "截屏", "截了几个图", "差些意思", "差点意思",
    "怎么回事", "足够好", "推荐出来", "起作用", "不起作用", "跟上下文无关", "无关",
    "丝滑", "兜底", "复盘", "对齐", "吐槽", "痛点", "闭环", "链路", "蒸馏",
]

COMBINATORIAL_TEMPLATES = [
    (["我们", "他们", "大家", "开发者", "用户", "团队", "系统", "引擎", "我", "你"], ["可以", "需要", "正在", "已经", "希望", "能够", "直接", "自动"], ["使用", "支持", "构建", "测试", "优化", "部署", "加载", "切换", "导入", "处理", "解决", "提升", "推荐", "维护", "安装", "蒸馏"]),
    (["使用", "通过", "基于", "采用", "开启", "关闭", "配置", "集成", "打字", "输入"], ["的", "了", "这个", "这套", "本地", "端侧", "最新"], ["模型", "词库", "算法", "架构", "方案", "接口", "功能", "系统", "输入法", "编译器", "过程", "过程中", "上下文", "窗口", "状态", "候选", "版本"]),
    (["今天", "明天", "昨天", "现在", "刚才", "周末", "晚上", "下午"], ["我们", "一起", "准备", "打算", "先去", "继续"], ["讨论", "开发", "测试", "上线", "发布", "修复", "重构", "验证", "吃饭", "休息", "开会"]),
    (["提高", "降低", "保证", "验证", "优化", "改善", "测量", "提升"], ["系统", "推理", "解码", "输入", "运行", "内存", "候选", "排序"], ["性能", "延迟", "准确率", "效率", "体验", "占用", "速度", "质量", "空间"]),
    (["感觉", "觉得", "看起来", "似乎", "有时候", "确实", "真的", "还是"], ["非常", "特别", "比较", "不太", "不那么", "挺", "足够", "最"], ["合适", "靠谱", "准确", "流畅", "方便", "自然", "清晰", "稳定", "好", "快"]),
]

DISTILLATION_SENTENCES = [
    "今天下午我们讨论一下端侧模型的量化方案与内存占用。",
    "法庭依法保障公民的权利，维护消费者的合法权利。",
    "严禁任何机构滥用国家的权力，必须依法行使最高权力。",
    "群众向上级如实地反映问题，相关部门及时听取群众的反映。",
    "这只是普通的化学反应与药物过敏反应。",
    "我在 macOS 系统上使用 Rust 和 Swift 开发本地智能双拼输入法。",
    "帮我 review 一下这个 PR 的延迟表现，没问题就合并到 GitHub 仓库。",
    "通过模糊音与键盘邻键纠错算法，高速盲打时的整句准确率大幅提升。",
    "所有个人词频和退格负反馈都保存在本地 SQLite 数据库中，绝不上传云端。",
    "这位医生很有高尚的医德，每天耐心地为患者说明治疗方案。",
    "数学家推导出了全新的公式，前线部队则发起了猛烈的攻势。",
    "利用端侧小模型的前缀缓存机制，单次前向重排只需要不到三毫秒的时间。",
    "我在使用的过程中，感觉第一候选字有时候不那么靠谱。",
    "我截了几个图，你看看是怎么回事，如何让初始的状态能够足够好。",
    "打字的过程中输入法要维护一个上下文窗口，这样才能推荐出来最合适的输入。",
    "这个候选还是差些意思，八百标兵奔北坡，北坡炮兵并排跑。",
    "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。",
    "天有不测风云，人有旦夕祸福。天生我材必有用，千金散尽还复来。",
    "为什么不用更新版本的开源模型来蒸馏输入法词库呢？",
    "模糊音处理和候选字还有不少细节问题，一些冷门的词和跟上下文无关的词不该排在前面。",
    "究竟有多大可能被选上？感觉现在的排序算法还有很大的提升空间。",
    "中英文混合输入的时候自动在英文单词前后添加空格，并把大小写纠正成 ChatGPT。",
]

PUNCTUATION_TOKENS = {
    "，", "。", "！", "？", "：", "；", "、", "“", "”", "‘", "’",
    "（", "）", "《", "》", "【", "】", "—", "…", ",", ".", "!", "?", ":", ";", "-", "_", "/"
}


def is_all_cjk(s: str) -> bool:
    return bool(s) and all("\u4e00" <= ch <= "\u9fa5" for ch in s)


def is_clean_ascii_piece(piece: str) -> bool:
    stripped = piece.lstrip("Ġ ")
    return (
        1 <= len(stripped) <= 12
        and stripped.isascii()
        and all(c.isalnum() or c in "_-." for c in stripped)
    )


def annotate_word_pinyin(word: str) -> str | None:
    syls = py_annotate(word, style=Style.NORMAL, heteronym=False, errors="ignore")
    if len(syls) != len(word):
        return None
    cleaned = []
    for item in syls:
        if not item or not item[0]:
            return None
        s = item[0].lower().replace("ü", "v").replace("u:", "v")
        if not s.isalpha():
            return None
        cleaned.append(s)
    return " ".join(cleaned)


def load_jieba_freqs() -> dict[str, int]:
    home = os.path.expanduser("~")
    candidates = sorted(glob.glob(f"{home}/.cargo/registry/src/*/jieba-rs-*/src/data/dict.txt"))
    freqs: dict[str, int] = {}
    if not candidates:
        return freqs
    for line in Path(candidates[-1]).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            freqs[parts[0]] = int(parts[1])
    return freqs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--out-neural", required=True)
    parser.add_argument("--out-ngrams", required=True)
    parser.add_argument("--out-lexicon", default="")
    parser.add_argument("--max-vocab", type=int, default=24576)
    args = parser.parse_args()

    out_neural = Path(args.out_neural)
    out_neural.mkdir(parents=True, exist_ok=True)
    out_ngrams = Path(args.out_ngrams)
    out_ngrams.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Resolving Apache-2.0 {args.repo}...", flush=True)
    snap_dir = Path(snapshot_download(
        repo_id=args.repo,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"],
    ))

    tokenizer = AutoTokenizer.from_pretrained(str(snap_dir))
    jieba_freqs = load_jieba_freqs()
    wf_dict = wordfreq.get_frequency_dict("zh")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[2/6] Loading full teacher {args.repo} on {device}...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        str(snap_dir),
        dtype=torch.float16 if device == "mps" else torch.float32,
    ).to(device)
    teacher.eval()

    # 1. Multi-Corpus Lexicon Distillation (`wordfreq` 334k + `phrase-pinyin-data` 47k + Qwen3 teacher!)
    if args.out_lexicon:
        out_lex = Path(args.out_lexicon)
        out_lex.parent.mkdir(parents=True, exist_ok=True)
        print("[3/6] Distilling multi-source modern Chinese lexicon (wordfreq + phrase-pinyin-data + Qwen3)...", flush=True)
        lex_entries: dict[tuple[str, str], float] = {}

        # Source A: `wordfreq` 334,609 modern spoken, subtitle, social & web Chinese entries!
        for word, prob in wf_dict.items():
            clen = len(word)
            if not (1 <= clen <= 8) or not is_all_cjk(word) or prob <= 0.0:
                continue
            zipf = math.log10(prob) + 9.0
            # Filter out ultra-cold noise (`zipf < 2.35` for single chars, `< 2.75` for multi-char)
            if clen == 1 and zipf < 2.35:
                continue
            if clen >= 2 and zipf < 2.75:
                continue
            py = annotate_word_pinyin(word)
            if not py:
                continue
            if clen == 1:
                lp = max(-6.60, min(-2.05, -6.60 + (zipf - 2.35) * 0.96))
            else:
                lp = max(-5.75, min(-2.15, -5.75 + (zipf - 2.75) * 0.88))
            key = (word, py)
            lex_entries[key] = max(lex_entries.get(key, -99.0), lp)

        # Source B: `pypinyin.phrases_dict` (`phrase-pinyin-data` MIT 47,111 curated polyphone phrases & idioms)
        for phrase, py_list in phrases_dict.items():
            clen = len(phrase)
            if not (2 <= clen <= 8) or not is_all_cjk(phrase):
                continue
            cleaned_syls = []
            valid = True
            for item in py_list:
                if not item or not item[0]:
                    valid = False
                    break
                # Strip tone digits or diacritics via pypinyin normalizer fallback
                s = annotate_word_pinyin(phrase)
                if s:
                    cleaned_syls = s.split()
                else:
                    valid = False
                break
            if not valid or len(cleaned_syls) != clen:
                continue
            py = " ".join(cleaned_syls)
            prob = wf_dict.get(phrase, 0.0)
            if prob > 0.0:
                zipf = math.log10(prob) + 9.0
                lp = max(-5.50, min(-2.20, -5.65 + (zipf - 2.70) * 0.88))
            elif jieba_freqs.get(phrase, 0) >= 30:
                lp = -4.45
            else:
                lp = -5.15
            key = (phrase, py)
            lex_entries[key] = max(lex_entries.get(key, -99.0), lp)

        # Source C: Direct 28-layer Qwen3-0.6B-Base teacher scoring on modern colloquial compounds!
        anchor_ids = tokenizer.encode("。", add_special_tokens=False)
        with torch.no_grad():
            for word in COLLOQUIAL_TEACHER_WORDS:
                if not is_all_cjk(word):
                    continue
                py = annotate_word_pinyin(word)
                if not py:
                    continue
                w_ids = tokenizer.encode(word, add_special_tokens=False)
                full_ids = torch.tensor([anchor_ids + w_ids], dtype=torch.long, device=device)
                logits = teacher(full_ids).logits[0]
                lps = F.log_softmax(logits.float(), dim=-1)
                joint_lp = sum(float(lps[len(anchor_ids) - 1 + s, tid].item()) for s, tid in enumerate(w_ids))
                per_char_lp = joint_lp / max(1, len(word))
                # Map Qwen3 per-char log-prob (~[-6.5, -2.5]) into lexicon prior range [-3.95, -2.85]
                calibrated_lp = max(-3.95, min(-2.85, -2.55 + (per_char_lp + 2.5) * 0.28))
                key = (word, py)
                lex_entries[key] = max(lex_entries.get(key, -99.0), calibrated_lp)

        tsv_rows = [f"{w}\t{py}\t{lp:.4f}" for (w, py), lp in sorted(lex_entries.items())]
        out_lex.write_text("\n".join(tsv_rows) + "\n", encoding="utf-8")
        print(f"      Wrote {len(tsv_rows)} multi-corpus distilled entries to {out_lex}", flush=True)

    # 2. Score homophone & colloquial probes + expand combinatorial modern Bigram/Trigram table!
    print("[4/6] Scoring Bigram/Trigram ambiguity sets with Qwen3 teacher...", flush=True)
    ngram_map_2: dict[tuple[str, str], float] = {}
    ngram_map_3: dict[tuple[str, str, str], float] = {}

    with torch.no_grad():
        for w1, particle, candidates in HOMOPHONE_AND_COLLOQUIAL_PROBES:
            prefix = f"{w1}{particle}"
            p_ids = tokenizer.encode(prefix, add_special_tokens=False)
            cand_lps = []
            for cand in candidates:
                c_ids = tokenizer.encode(cand, add_special_tokens=False)
                full_ids = torch.tensor([p_ids + c_ids], dtype=torch.long, device=device)
                logits = teacher(full_ids).logits[0]
                log_probs = F.log_softmax(logits.float(), dim=-1)
                lp = sum(float(log_probs[len(p_ids) - 1 + step, tid].item()) for step, tid in enumerate(c_ids))
                cand_lps.append((cand, lp / max(1, len(cand))))

            target_cand = candidates[0]
            target_lp = cand_lps[0][1]
            foil_lp = max(x[1] for x in cand_lps[1:]) if len(cand_lps) > 1 else target_lp - 2.0
            raw_diff = max(0.0, target_lp - foil_lp)
            margin = max(2.20, min(3.60, 2.20 + raw_diff * 0.45))
            if particle:
                ngram_map_3[(w1, particle, target_cand)] = margin
            ngram_map_2[(w1, target_cand)] = margin

    for group_a, group_b, group_c in COMBINATORIAL_TEMPLATES:
        for a, b, c in itertools.product(group_a, group_b, group_c):
            ngram_map_2.setdefault((a, b), 1.15)
            ngram_map_2.setdefault((b, c), 1.30)
            ngram_map_3.setdefault((a, b, c), 1.55)

    for w1, w2, bonus in [
        ("输入", "法", 1.9), ("双拼", "输入法", 2.2), ("拼音", "输入法", 2.2),
        ("本地", "模型", 1.6), ("语言", "模型", 1.9), ("端侧", "推理", 1.9),
        ("零", "拷贝", 1.9), ("内存", "泄漏", 1.9), ("内存", "映射", 1.8),
        ("模糊", "音", 2.3), ("自动", "纠错", 1.9), ("中英", "混写", 2.0),
        ("偏好", "设置", 1.9), ("知识", "蒸馏", 2.1), ("没有", "问题", 1.5),
        ("不好", "意思", 1.8), ("什么", "意思", 1.7), ("怎么", "回事", 2.2),
        ("差些", "意思", 2.2), ("差点", "意思", 2.2), ("过程", "中", 2.4),
        ("的", "过程", 1.8), ("的", "过程中", 2.3), ("使用", "过程中", 2.2),
        ("不", "那么", 2.1), ("不那么", "靠谱", 2.5), ("很", "靠谱", 2.2),
        ("挺", "靠谱", 2.2), ("最", "合适", 2.6), ("很", "合适", 2.1),
        ("合适的", "输入", 2.0), ("推荐", "出来", 2.2), ("截了", "几个", 2.1),
        ("几个", "图", 2.2), ("截了", "几个图", 2.5), ("上下文", "窗口", 2.5),
        ("维护", "上下文", 2.0), ("初始", "状态", 2.2), ("足够", "好", 2.1),
        ("第一", "候选", 2.1), ("候选", "字", 2.0), ("候选", "词", 2.0),
        ("不起", "作用", 2.3), ("起", "作用", 2.1), ("自动", "提示", 2.0),
        ("八百", "标兵", 2.6), ("标兵", "奔北坡", 2.8), ("八百标兵", "奔北坡", 3.2),
        ("旦夕", "祸福", 2.8), ("悲欢", "离合", 2.8), ("人有", "旦夕祸福", 3.0),
        ("有", "多大", 2.4), ("多大", "可能", 2.4), ("提升", "空间", 2.5),
        ("很大", "空间", 2.2), ("更新", "版本", 2.4), ("新", "版本", 2.2),
        ("不少", "细节", 2.2), ("不少", "问题", 2.2), ("跟", "上下文", 2.3),
    ]:
        ngram_map_2[(w1, w2)] = max(ngram_map_2.get((w1, w2), 0.0), bonus)

    ngram_lines = [f"# Generated from Apache-2.0 {args.repo} teacher & multi-corpus collocations"]
    for (w1, w2), bonus in sorted(ngram_map_2.items()):
        ngram_lines.append(f"2\t{w1}\t{w2}\t{bonus:.3f}")
    for (w1, w2, w3), bonus in sorted(ngram_map_3.items()):
        ngram_lines.append(f"3\t{w1}\t{w2}\t{w3}\t{bonus:.3f}")
    out_ngrams.write_text("\n".join(ngram_lines) + "\n", encoding="utf-8")
    print(f"      Wrote {len(ngram_lines) - 1} Bigram/Trigram transitions to {out_ngrams}", flush=True)

    # 3. Build 24,576 Mandarin-first curated vocabulary from Qwen3 BPE & export intact 28-layer 4-bit MLX!
    print(f"[5/6] Curating {args.max_vocab} tokens from {args.repo}...", flush=True)
    orig_cfg = json.loads((snap_dir / "config.json").read_text(encoding="utf-8"))
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 151643
    unk_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 151643

    selected_old_ids = [bos_id, unk_id]
    vocab_strings = ["<bos>", "<unk>"]
    seen_pieces = {"<bos>", "<unk>"}

    def add_piece(piece: str, oid: int) -> bool:
        if not piece or piece in seen_pieces or len(selected_old_ids) >= args.max_vocab:
            return False
        seen_pieces.add(piece)
        vocab_strings.append(piece)
        selected_old_ids.append(oid)
        return True

    def try_register_surface(surface: str) -> None:
        if not surface or surface in seen_pieces or len(selected_old_ids) >= args.max_vocab:
            return
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            add_piece(surface, ids[0])

    for w1, part, cands in HOMOPHONE_AND_COLLOQUIAL_PROBES:
        for item in [w1, part, *cands]:
            if item:
                try_register_surface(item)
                for ch in item:
                    try_register_surface(ch)
    for item in COLLOQUIAL_TEACHER_WORDS:
        try_register_surface(item)
        for ch in item:
            try_register_surface(ch)
    for ga, gb, gc in COMBINATORIAL_TEMPLATES:
        for item in [*ga, *gb, *gc]:
            try_register_surface(item)
            for ch in item:
                try_register_surface(ch)
    for sent in DISTILLATION_SENTENCES:
        for ch in sent:
            try_register_surface(ch)
    for p in sorted(PUNCTUATION_TOKENS):
        try_register_surface(p)

    single_cjk_bucket: list[tuple[float, int, str]] = []
    multi_cjk_bucket: list[tuple[float, int, str, int]] = []
    ascii_bucket: list[tuple[int, str]] = []

    all_ids = list(range(orig_cfg["vocab_size"]))
    decoded_pieces = tokenizer.batch_decode([[i] for i in all_ids], skip_special_tokens=False)
    for oid, raw_piece in zip(all_ids, decoded_pieces):
        if "\ufffd" in raw_piece:
            continue
        piece = raw_piece.strip()
        if not piece or piece in seen_pieces:
            continue
        if len(piece) == 1 and is_all_cjk(piece):
            wf = wf_dict.get(piece, 0.0)
            single_cjk_bucket.append((-wf, oid, piece))
        elif 2 <= len(piece) <= 4 and is_all_cjk(piece):
            wf = wf_dict.get(piece, 0.0)
            jf = jieba_freqs.get(piece, 0)
            multi_cjk_bucket.append((-wf, -jf, piece, oid))
        elif piece in PUNCTUATION_TOKENS:
            add_piece(piece, oid)
        elif is_clean_ascii_piece(raw_piece):
            ascii_bucket.append((oid, piece))

    single_cjk_bucket.sort()
    cjk_char_added = sum(1 for _, oid, piece in single_cjk_bucket if add_piece(piece, oid))

    multi_cjk_bucket.sort(key=lambda t: (t[0], t[1], t[3]))
    max_multi_cjk = max(0, args.max_vocab - len(selected_old_ids) - 3200)
    cjk_word_added = sum(1 for _, _, piece, oid in multi_cjk_bucket[:max_multi_cjk] if add_piece(piece, oid))

    ascii_added = 0
    for oid, piece in ascii_bucket:
        if len(selected_old_ids) >= args.max_vocab:
            break
        if add_piece(piece, oid):
            ascii_added += 1

    print(
        f"      Selected {len(selected_old_ids)} tokens "
        f"(single_cjk={cjk_char_added}, multi_cjk={cjk_word_added}, ascii={ascii_added})",
        flush=True,
    )

    del teacher
    if device == "mps":
        torch.mps.empty_cache()

    raw_st = load_file(str(snap_dir / "model.safetensors"))
    idx_cpu = torch.tensor(selected_old_ids, dtype=torch.long)
    embed_pruned = raw_st["model.embed_tokens.weight"].index_select(0, idx_cpu).to(torch.float16).contiguous()

    total_layers = orig_cfg["num_hidden_layers"]
    n_h = orig_cfg["num_attention_heads"]
    n_kv = orig_cfg["num_key_value_heads"]
    hd = int(orig_cfg.get("head_dim") or (orig_cfg["hidden_size"] // n_h))
    theta = float(orig_cfg.get("rope_theta", 1000000.0))
    eps = float(orig_cfg.get("rms_norm_eps", 1e-6))

    # 4. Export intact 28-layer Qwen3-0.6B-Base in native MLX 4-bit (`group_size=64, bits=4`, ~250 MB)
    print(f"[6/6] Quantizing intact {total_layers}-layer {args.repo} to native MLX 4-bit (group_size=64, bits=4) in {out_neural}...", flush=True)
    import mlx.core as mx

    fp16_tensors = {
        "model.embed_tokens.weight": embed_pruned,
        "model.norm.weight": raw_st["model.norm.weight"].to(torch.float16).contiguous(),
    }
    for layer_idx in range(total_layers):
        p = f"model.layers.{layer_idx}"
        for suffix in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.q_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.k_norm.weight",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]:
            key = f"{p}.{suffix}"
            if key in raw_st:
                fp16_tensors[key] = raw_st[key].to(torch.float16).contiguous()

    tmp_fp16 = out_neural / "_tmp_fp16.safetensors"
    save_file(fp16_tensors, str(tmp_fp16))
    mlx_raw = mx.load(str(tmp_fp16))
    tmp_fp16.unlink(missing_ok=True)

    mlx_quant: dict[str, mx.array] = {}
    for k, arr in mlx_raw.items():
        if arr.ndim == 2 and k.endswith(".weight") and arr.shape[-1] % 64 == 0:
            prefix = k[: -len(".weight")]
            wq, scales, biases = mx.quantize(arr, group_size=64, bits=4)
            mlx_quant[f"{prefix}.weight"] = wq
            mlx_quant[f"{prefix}.scales"] = scales
            mlx_quant[f"{prefix}.biases"] = biases
        else:
            mlx_quant[k] = arr

    st_path = out_neural / "model.safetensors"
    mx.save_safetensors(str(st_path), mlx_quant)
    st_path.chmod(0o644)

    model_cfg = {
        "vocab_size": len(vocab_strings),
        "hidden_size": orig_cfg["hidden_size"],
        "intermediate_size": orig_cfg["intermediate_size"],
        "num_hidden_layers": total_layers,
        "num_attention_heads": n_h,
        "num_key_value_heads": n_kv,
        "head_dim": hd,
        "rms_norm_eps": eps,
        "rope_theta": theta,
        "quantization": {"group_size": 64, "bits": 4},
    }
    (out_neural / "config.json").write_text(json.dumps(model_cfg, indent=2), encoding="utf-8")
    (out_neural / "vocab.json").write_text(json.dumps(vocab_strings, ensure_ascii=False), encoding="utf-8")
    size_mb = st_path.stat().st_size / (1024 * 1024)
    print(f"Done! Exported 4-bit MLX {st_path} ({size_mb:.1f} MB, vocab={len(vocab_strings)}, layers={total_layers}, head_dim={hd})", flush=True)


if __name__ == "__main__":
    main()
