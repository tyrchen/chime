//! Build-time compiler turning declarative YAML tables (`data/*.yaml`) into static zero-overhead
//! Rust slices (`$OUT_DIR/generated_lexicon_data.rs`).

use std::{collections::BTreeMap, env, error::Error, fmt::Write as _, fs, path::Path};

type Record = BTreeMap<String, String>;
type Sections = BTreeMap<String, Vec<Record>>;

fn unquote(val: &str) -> String {
    let trimmed = val.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        trimmed[1..trimmed.len() - 1]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
    } else {
        trimmed.to_string()
    }
}

/// Minimal strict parser for flat sectioned YAML lists of key-value maps (`section:` -> `- key:
/// val`).
fn parse_sectioned_yaml_records(path: &Path) -> Result<Sections, Box<dyn Error>> {
    let raw = fs::read_to_string(path)?;
    let mut sections: Sections = BTreeMap::new();
    let mut cur_section = String::new();
    let mut cur_record: Option<Record> = None;

    for (line_no, raw_line) in raw.lines().enumerate() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent = raw_line.len() - raw_line.trim_start().len();
        if indent == 0 && trimmed.ends_with(':') {
            if let Some(rec) = cur_record.take()
                && !cur_section.is_empty()
            {
                sections.entry(cur_section.clone()).or_default().push(rec);
            }
            cur_section = trimmed.trim_end_matches(':').trim().to_string();
            sections.entry(cur_section.clone()).or_default();
            continue;
        }
        if let Some(item_rest) = trimmed.strip_prefix("- ") {
            if let Some(rec) = cur_record.take() {
                sections.entry(cur_section.clone()).or_default().push(rec);
            }
            let (k, v) = item_rest.split_once(':').ok_or_else(|| {
                format!(
                    "expected `key: value` at {}:{}, got `{trimmed}`",
                    path.display(),
                    line_no + 1
                )
            })?;
            let mut map = BTreeMap::new();
            map.insert(k.trim().to_string(), unquote(v));
            cur_record = Some(map);
        } else if let Some((k, v)) = trimmed.split_once(':') {
            if let Some(map) = cur_record.as_mut() {
                map.insert(k.trim().to_string(), unquote(v));
            }
        } else {
            return Err(format!(
                "invalid YAML syntax at {}:{}: `{trimmed}`",
                path.display(),
                line_no + 1
            )
            .into());
        }
    }
    if let Some(rec) = cur_record.take()
        && !cur_section.is_empty()
    {
        sections.entry(cur_section).or_default().push(rec);
    }
    Ok(sections)
}

fn req<'a>(rec: &'a Record, key: &str, file: &str) -> Result<&'a str, Box<dyn Error>> {
    rec.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing `{key}` in {file}").into())
}

fn parse_f32(rec: &Record, key: &str, file: &str) -> Result<String, Box<dyn Error>> {
    let raw = req(rec, key, file)?;
    let v: f32 = raw
        .parse()
        .map_err(|e| format!("invalid float `{raw}` for `{key}` in {file}: {e}"))?;
    Ok(format!("{v:.4}_f32"))
}

fn parse_bool(rec: &Record, key: &str) -> &'static str {
    match rec.get(key).map(String::as_str) {
        Some("true") => "true",
        _ => "false",
    }
}

fn emit_polyphones_and_grammar(data_dir: &Path, out: &mut String) -> Result<(), Box<dyn Error>> {
    let poly = parse_sectioned_yaml_records(&data_dir.join("polyphones.yaml"))?;
    out.push_str("pub const MODERN_DAILY_HETERONYM_CHARS: &[(&str, &str, f32)] = &[\n");
    for rec in poly.get("character_readings").into_iter().flatten() {
        writeln!(
            out,
            "    ({:?}, {:?}, {}),",
            req(rec, "surface", "polyphones.yaml")?,
            req(rec, "pinyin", "polyphones.yaml")?,
            parse_f32(rec, "log_prob", "polyphones.yaml")?,
        )?;
    }
    out.push_str("];\n\n");

    out.push_str("pub const POLYPHONE_WORD_OVERRIDES: &[(&str, &str)] = &[\n");
    for rec in poly.get("word_overrides").into_iter().flatten() {
        writeln!(
            out,
            "    ({:?}, {:?}),",
            req(rec, "surface", "polyphones.yaml")?,
            req(rec, "pinyin", "polyphones.yaml")?,
        )?;
    }
    out.push_str("];\n\n");

    let gram = parse_sectioned_yaml_records(&data_dir.join("closed_grammar.yaml"))?;
    out.push_str("pub const CLOSED_GRAMMAR_ENTRIES: &[(&str, &str, f32)] = &[\n");
    for rec in gram.get("grammar_entries").into_iter().flatten() {
        writeln!(
            out,
            "    ({:?}, {:?}, {}),",
            req(rec, "surface", "closed_grammar.yaml")?,
            req(rec, "pinyin", "closed_grammar.yaml")?,
            parse_f32(rec, "log_prob", "closed_grammar.yaml")?,
        )?;
    }
    out.push_str("];\n\n");
    Ok(())
}

fn emit_english_and_classics(data_dir: &Path, out: &mut String) -> Result<(), Box<dyn Error>> {
    let eng = parse_sectioned_yaml_records(&data_dir.join("english_casing.yaml"))?;
    out.push_str("pub const ENGLISH_CASING_TERMS: &[(&str, f32, bool)] = &[\n");
    for rec in eng.get("english_terms").into_iter().flatten() {
        writeln!(
            out,
            "    ({:?}, {}, {}),",
            req(rec, "surface", "english_casing.yaml")?,
            parse_f32(rec, "log_prob", "english_casing.yaml")?,
            parse_bool(rec, "tech"),
        )?;
    }
    out.push_str("];\n\n");

    out.push_str("pub const CODE_SWITCH_BIGRAMS: &[(&str, &str, f32)] = &[\n");
    for rec in eng.get("code_switch_bigrams").into_iter().flatten() {
        writeln!(
            out,
            "    ({:?}, {:?}, {}),",
            req(rec, "prev", "english_casing.yaml")?,
            req(rec, "next", "english_casing.yaml")?,
            parse_f32(rec, "bonus", "english_casing.yaml")?,
        )?;
    }
    out.push_str("];\n\n");

    let cls = parse_sectioned_yaml_records(&data_dir.join("classical_couplets.yaml"))?;
    out.push_str("pub const COUPLET_CHAINS: &[(&str, &str, &str, &str, &str)] = &[\n");
    for rec in cls.get("couplets").into_iter().flatten() {
        writeln!(
            out,
            "    ({:?}, {:?}, {:?}, {:?}, {:?}),",
            req(rec, "clause_a", "classical_couplets.yaml")?,
            req(rec, "pinyin_a", "classical_couplets.yaml")?,
            req(rec, "joiner", "classical_couplets.yaml")?,
            req(rec, "clause_b", "classical_couplets.yaml")?,
            req(rec, "pinyin_b", "classical_couplets.yaml")?,
        )?;
    }
    out.push_str("];\n\n");

    out.push_str("pub const STANDALONE_IDIOMS: &[(&str, &str)] = &[\n");
    for rec in cls.get("idioms").into_iter().flatten() {
        writeln!(
            out,
            "    ({:?}, {:?}),",
            req(rec, "surface", "classical_couplets.yaml")?,
            req(rec, "pinyin", "classical_couplets.yaml")?,
        )?;
    }
    out.push_str("];\n\n");

    let seed = parse_sectioned_yaml_records(&data_dir.join("unit_test_seed.yaml"))?;
    out.push_str("pub const UNIT_TEST_SEED_WORDS: &[(&str, &str, f32, bool)] = &[\n");
    for rec in seed.get("seed_words").into_iter().flatten() {
        writeln!(
            out,
            "    ({:?}, {:?}, {}, {}),",
            req(rec, "surface", "unit_test_seed.yaml")?,
            req(rec, "pinyin", "unit_test_seed.yaml")?,
            parse_f32(rec, "log_prob", "unit_test_seed.yaml")?,
            parse_bool(rec, "tech"),
        )?;
    }
    out.push_str("];\n");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let data_dir = Path::new(&manifest_dir).join("data");
    for name in [
        "polyphones.yaml",
        "closed_grammar.yaml",
        "english_casing.yaml",
        "classical_couplets.yaml",
        "unit_test_seed.yaml",
    ] {
        println!("cargo:rerun-if-changed={}", data_dir.join(name).display());
    }

    let mut out = String::from("// @generated by crates/lexicon/build.rs from data/*.yaml\n\n");
    emit_polyphones_and_grammar(&data_dir, &mut out)?;
    emit_english_and_classics(&data_dir, &mut out)?;

    let out_dir = env::var("OUT_DIR")?;
    fs::write(Path::new(&out_dir).join("generated_lexicon_data.rs"), out)?;
    Ok(())
}
