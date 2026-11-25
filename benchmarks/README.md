# Chime Linguistic Evaluation & Continuous Tracking Suite

This directory maintains Chime's reproducible, zero-leakage evaluation benchmarks and historical score ledger (`history.jsonl`).

## Split Discipline & Anti-Overfit Policy

1. **`dev_suite.tsv` (Development & Error-Analysis Split — 1,258 cases, 17 categories)**
   - Covers spoken Mandarin, copula/classifier syntax, `的/地/得` & `在/再`, directional/potential complements, context-conditioned homophones, polyphones/heteronyms, hiatus vowel boundaries (`xian`/`jie`/`minge`), technical prose, Chinese-English code-switching, idioms/poetry, dialect fuzzy Pinyin, QWERTY motor slips, Xiaohe/Ziranma/Microsoft Shuangpin, initial skeletons, and adversarial dangling-functor traps.
   - May be inspected for error diagnosis **only to derive general mathematical, phonotactic, morphosyntactic, or model-distillation improvements** — never add benchmark sentences or ad-hoc phrase patches to `crates/lexicon/data/*.yaml`.

2. **`test_holdout.tsv` (Strictly Held-Out Unseen Generalization Split — 1,010 cases, 16 categories)**
   - Independently authored across everyday conversation, workplace collaboration, essays/culture, medicine/law, finance/business, systems/ML engineering, bilingual code-switching, polyphones, minimal-pair homophones, grammar, hiatus, classics, fuzzy Pinyin, motor typos, and Shuangpin (`0.00%` input overlap with `dev_suite.tsv`).
   - **Blind Evaluation Rule**: Never inspect individual failed rows of `test_holdout.tsv` to tune weights or add lexical entries. Every change must generalize from first principles so `dev_suite.tsv` and `test_holdout.tsv` improve together.

3. **Training & Distillation Decontamination**
   - `tools/distill_homophone_subspace.py` enforces strict **5-character CJK n-gram decontamination** against **both** `dev_suite.tsv` and `test_holdout.tsv` before streaming any passage into `Qwen3-28L -> Chime-6L` distillation.

## Running & Recording Benchmarks

```bash
# Run both dev_suite.tsv and test_holdout.tsv and append JSONL records to benchmarks/history.jsonl:
make bench

# Or run individually:
make bench-dev
make bench-test
```
