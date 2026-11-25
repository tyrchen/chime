//! Declarative closed-class Mandarin grammar, canonical English casing, and unit-test seed loader
//! backed by compile-time validated YAML tables (`crates/lexicon/data/*.yaml`).

use crate::{
    Lexicon,
    generated::{
        CLOSED_GRAMMAR_ENTRIES, CODE_SWITCH_BIGRAMS, ENGLISH_CASING_TERMS, UNIT_TEST_SEED_WORDS,
    },
};

/// Populates closed-class Mandarin grammatical items, canonical English proper-casing, and
/// classical couplets from declarative YAML tables (used when building `core.imedict` or mounting
/// runtime overlay without open-domain test-word contamination).
pub fn populate_core_grammar_and_casing(lex: &mut Lexicon) {
    for &(surface, py, log_prob) in CLOSED_GRAMMAR_ENTRIES {
        lex.insert_pinyin_word(surface, py, log_prob, false);
    }
    for &(surface, log_prob, is_tech) in ENGLISH_CASING_TERMS {
        lex.insert_english_term(surface, log_prob, is_tech);
    }
    for &(prev, next, bonus) in CODE_SWITCH_BIGRAMS {
        lex.insert_bigram_boost(prev, next, bonus);
    }
    crate::poetry::populate_poetry_and_idioms(lex);
}

/// Populates `lex` with [`populate_core_grammar_and_casing`] plus the minimal unit-test seed table
/// (`data/unit_test_seed.yaml`) for in-memory tests when `dist/core.imedict` is not mounted.
pub fn populate_seed_lexicon(lex: &mut Lexicon) {
    populate_core_grammar_and_casing(lex);
    for &(surface, py, log_prob, is_tech) in UNIT_TEST_SEED_WORDS {
        lex.insert_pinyin_word(surface, py, log_prob, is_tech);
    }
}
