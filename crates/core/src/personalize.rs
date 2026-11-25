//! Local reverse-G2P noisy training pair synthesizer (for Google Colab `LoRA` fine-tuning)
//! and `.imepack` personal brain bundle installer/packager.

use std::{collections::HashMap, path::Path};

use chime_lexicon::{
    CompiledDictBuilder, Lexicon, TypingSample, annotate_chinese_word, parse_tsv_dict,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::sanitize::{SanitizeReport, sanitize_corpus_text};

/// Errors raised during `.imepack` installation or corpus export.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PersonalizeError {
    /// Filesystem I/O failure.
    #[error("personalize I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON manifest or JSONL serialization error.
    #[error("personalize JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Dictionary compilation failure.
    #[error("dictionary error: {0}")]
    Lexicon(#[from] chime_lexicon::LexiconError),
}

/// Single JSONL training record consumed by `colab/train_personal_brain.ipynb`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingRecord {
    /// Left context preceding the target phrase.
    pub left_context: String,
    /// Simulated raw user keystroke sequence (with realistic abbreviation/fuzzy/typo noise).
    pub raw_keys: String,
    /// Ground-truth mixed Chinese-English target phrase.
    pub target: String,
}

/// Metadata stored inside `<name>.imepack/manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImePackManifest {
    /// Format version (`"1.0"`).
    pub format_version: String,
    /// Profile display name.
    pub name: String,
    /// Base architecture identifier (e.g. `"chime-qwen2.5-80m-24k"`).
    pub base_model: String,
    /// True if `neural/model.safetensors` is bundled.
    pub has_safetensors: bool,
    /// True if `personal.imedict` is bundled.
    pub has_dictionary: bool,
}

/// Scrubs raw personal documents, synthesizes noisy reverse-G2P `(left_context, raw_keys, target)`
/// records, merges opt-in `TypingSample`s, and returns newline-delimited JSONL text.
#[must_use]
pub fn build_colab_training_jsonl(
    raw_corpus_text: &str,
    recorded_samples: &[TypingSample],
) -> (String, SanitizeReport, usize) {
    let (clean_lines, report) = sanitize_corpus_text(raw_corpus_text);
    let empty_overrides = HashMap::new();
    let mut records: Vec<TrainingRecord> = Vec::new();

    for sample in recorded_samples {
        let (_, sub_rep) = sanitize_corpus_text(&sample.committed_surface);
        if sub_rep.redacted_items == 0 && !sample.raw_keys.is_empty() {
            records.push(TrainingRecord {
                left_context: sample.left_context.clone(),
                raw_keys: sample.raw_keys.clone(),
                target: sample.committed_surface.clone(),
            });
        }
    }

    for (line_idx, line) in clean_lines.iter().enumerate() {
        let clauses: Vec<&str> = line
            .split(['，', '。', '！', '？', '；', ',', ';'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let mut rolling_ctx = String::new();

        for (c_idx, clause) in clauses.into_iter().enumerate() {
            if clause.contains("<REDACTED>") {
                continue;
            }
            if let Some(clean_keys) = reverse_g2p_mixed_clause(clause, &empty_overrides) {
                records.push(TrainingRecord {
                    left_context: rolling_ctx.clone(),
                    raw_keys: clean_keys.clone(),
                    target: clause.to_string(),
                });

                // Deterministic realistic noise augmentation (swap typo / fuzzy / abbreviation)
                let seed = line_idx.wrapping_mul(31).wrapping_add(c_idx);
                if let Some(noisy) = inject_realistic_typing_noise(&clean_keys, seed) {
                    records.push(TrainingRecord {
                        left_context: rolling_ctx.clone(),
                        raw_keys: noisy,
                        target: clause.to_string(),
                    });
                }
            }
            rolling_ctx.push_str(clause);
            if rolling_ctx.chars().count() > 48 {
                rolling_ctx = rolling_ctx
                    .chars()
                    .rev()
                    .take(48)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
            }
        }
    }

    let mut jsonl_lines = Vec::with_capacity(records.len());
    for rec in &records {
        if let Ok(line) = serde_json::to_string(rec) {
            jsonl_lines.push(line);
        }
    }
    (jsonl_lines.join("\n"), report, jsonl_lines.len())
}

fn reverse_g2p_mixed_clause(clause: &str, overrides: &HashMap<&str, &str>) -> Option<String> {
    let mut keys = String::new();
    let mut cjk_run = String::new();

    for ch in clause.chars() {
        if ('\u{4E00}'..='\u{9FA5}').contains(&ch) {
            cjk_run.push(ch);
        } else {
            if !cjk_run.is_empty() {
                let py = annotate_chinese_word(&cjk_run, overrides)?;
                for syl in py.split_whitespace() {
                    keys.push_str(syl);
                }
                cjk_run.clear();
            }
            if ch.is_ascii_alphanumeric() {
                keys.push(ch.to_ascii_lowercase());
            }
        }
    }
    if !cjk_run.is_empty() {
        let py = annotate_chinese_word(&cjk_run, overrides)?;
        for syl in py.split_whitespace() {
            keys.push_str(syl);
        }
    }

    if keys.len() >= 2 { Some(keys) } else { None }
}

fn inject_realistic_typing_noise(clean_keys: &str, seed: usize) -> Option<String> {
    if clean_keys.len() < 5 || !clean_keys.is_ascii() {
        return None;
    }
    match seed % 4 {
        0 if clean_keys.contains("sh") => Some(clean_keys.replacen("sh", "s", 1)),
        1 if clean_keys.contains("ang") => Some(clean_keys.replacen("ang", "an", 1)),
        2 => {
            // Adjacent keystroke swap in the middle of the string
            let mut bytes = clean_keys.as_bytes().to_vec();
            let pos = (bytes.len() / 2).min(bytes.len() - 2);
            if bytes[pos] == bytes[pos + 1] {
                None
            } else {
                bytes.swap(pos, pos + 1);
                String::from_utf8(bytes).ok()
            }
        }
        _ => None,
    }
}

/// Packages a directory into a validated `.imepack` directory (compiling `personal.tsv` if
/// present).
///
/// # Errors
///
/// Returns [`PersonalizeError`] on I/O, JSON, or dictionary compilation failure.
pub fn build_imepack_dir(
    name: &str,
    personal_tsv: Option<&Path>,
    neural_dir: Option<&Path>,
    out_pack_dir: &Path,
) -> Result<ImePackManifest, PersonalizeError> {
    std::fs::create_dir_all(out_pack_dir)?;

    let mut has_dictionary = false;
    if let Some(tsv_path) = personal_tsv
        && tsv_path.exists()
    {
        let raw = std::fs::read_to_string(tsv_path)?;
        let mut lex = Lexicon::new();
        let stats = parse_tsv_dict(&raw, &mut lex);
        if stats.imported_entries > 0 {
            let builder = CompiledDictBuilder::from_lexicon(&lex);
            builder.write_to_file(&out_pack_dir.join("personal.imedict"))?;
            has_dictionary = true;
        }
    }

    let mut has_safetensors = false;
    if let Some(n_dir) = neural_dir
        && n_dir.join("model.safetensors").exists()
    {
        let dst_neural = out_pack_dir.join("neural");
        std::fs::create_dir_all(&dst_neural)?;
        for file_name in ["config.json", "vocab.json", "model.safetensors"] {
            let src = n_dir.join(file_name);
            if src.exists() {
                std::fs::copy(&src, dst_neural.join(file_name))?;
            }
        }
        has_safetensors = true;
    }

    let manifest = ImePackManifest {
        format_version: "1.0".to_string(),
        name: name.to_string(),
        base_model: "chime-qwen2.5-80m-24k".to_string(),
        has_safetensors,
        has_dictionary,
    };
    std::fs::write(
        out_pack_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

/// Installs a `.imepack` directory into `app_support_dir` (`dicts/personal.imedict` + `neural/`).
///
/// # Errors
///
/// Returns [`PersonalizeError`] if `manifest.json` is missing or copying assets fails.
pub fn install_imepack_dir(
    pack_dir: &Path,
    app_support_dir: &Path,
) -> Result<ImePackManifest, PersonalizeError> {
    let manifest_bytes = std::fs::read(pack_dir.join("manifest.json"))?;
    let manifest: ImePackManifest = serde_json::from_slice(&manifest_bytes)?;

    if manifest.has_dictionary {
        let src_dict = pack_dir.join("personal.imedict");
        if src_dict.exists() {
            let dicts_dir = app_support_dir.join("dicts");
            std::fs::create_dir_all(&dicts_dir)?;
            std::fs::copy(&src_dict, dicts_dir.join("personal.imedict"))?;
        }
    }

    if manifest.has_safetensors {
        let src_neural = pack_dir.join("neural");
        let dst_neural = app_support_dir.join("neural");
        std::fs::create_dir_all(&dst_neural)?;
        for file_name in ["config.json", "vocab.json", "model.safetensors"] {
            let src = src_neural.join(file_name);
            if src.exists() {
                std::fs::copy(&src, dst_neural.join(file_name))?;
            }
        }
    }

    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_synthesize_sanitized_noisy_g2p_jsonl_for_colab() {
        let raw = "\
今天在 macOS 上测试双拼输入法，端侧模型量化很快！
OPENAI_API_KEY=sk-secret123456
";
        let (jsonl, report, count) = build_colab_training_jsonl(raw, &[]);
        assert_eq!(report.redacted_items, 1);
        assert!(count >= 2);
        assert!(!jsonl.contains("sk-secret"));
        assert!(jsonl.contains("macos"));
    }
}
