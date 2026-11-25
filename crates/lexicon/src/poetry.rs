//! Public-domain classical Chinese poetry couplets, prose quotes, proverbs, and four-character
//! idioms loaded from compile-time YAML (`crates/lexicon/data/classical_couplets.yaml`).

use crate::{
    Lexicon,
    generated::{COUPLET_CHAINS, STANDALONE_IDIOMS},
};

/// Populates `lex` with classical poetry couplets, ancient prose quotes, proverbs, idioms,
/// and registers next-clause continuation chains for `Tab` auto-completion.
pub fn populate_poetry_and_idioms(lex: &mut Lexicon) {
    for &(clause_a, py_a, joiner, clause_b, py_b) in COUPLET_CHAINS {
        let clean_py_a = py_a.replace('，', "");
        lex.insert_pinyin_word(clause_a, &clean_py_a, -3.45, false);
        let clean_py_b = py_b.replace('，', "");
        lex.insert_pinyin_word(clause_b, &clean_py_b, -3.55, false);
        lex.insert_continuation(clause_a, format!("{joiner}{clause_b}"));
        lex.insert_bigram_boost(clause_a, clause_b, 2.8);
    }
    for &(idiom, py) in STANDALONE_IDIOMS {
        lex.insert_pinyin_word(idiom, py, -3.65, false);
    }
}
