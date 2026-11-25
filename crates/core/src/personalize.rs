//! Local reverse-G2P noisy training pair synthesizer (for Google Colab `LoRA` fine-tuning)
//! and `.imepack` v2 `Multi-LoRA` + zero-copy `.imedict` cartridge packager/installer.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Command,
};

use chime_lexicon::{
    CompiledDictBuilder, Lexicon, TypingSample, annotate_chinese_word, parse_tsv_dict,
};
use chime_neural::LoraRouting;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::sanitize::{SanitizeReport, sanitize_corpus_text};

/// Errors raised during `.imepack` installation, archive extraction, or corpus export.
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
    /// Archive extraction or invalid `.imepack` structure.
    #[error("invalid .imepack archive: {0}")]
    InvalidArchive(String),
}

/// Single JSONL training record consumed by Colab `LoRA` scripts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingRecord {
    /// Left context preceding the target phrase.
    pub left_context: String,
    /// Simulated raw user keystroke sequence (with realistic abbreviation/fuzzy/typo noise).
    pub raw_keys: String,
    /// Ground-truth mixed Chinese-English target phrase.
    pub target: String,
}

fn default_pack_version() -> String {
    "0.6.0".to_string()
}

const fn default_lora_scale() -> f32 {
    1.0
}

/// Metadata stored inside `<id>.imepack/manifest.json` (`v2.0` `Multi-LoRA` + `.imedict` cartridge,
/// backward-compatible with `v1.0` manifests).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct ImePackManifest {
    /// Format version (`"2.0"`).
    pub format_version: String,
    /// Stable kebab-case identifier (`"classics-poetry"`, `"tech-developer"`, …).
    #[serde(default)]
    pub id: String,
    /// Profile display name.
    pub name: String,
    /// Short summary of what this cartridge adds.
    #[serde(default)]
    pub description: String,
    /// Semantic version of the cartridge.
    #[serde(default = "default_pack_version")]
    pub version: String,
    /// Base architecture identifier (e.g. `"chime-qwen3-6l-24k"`).
    pub base_model: String,
    /// Dynamic context/app routing mode for `lora.safetensors`.
    #[serde(default)]
    pub routing: LoraRouting,
    /// Base `LoRA` multiplier (`alpha / rank`).
    #[serde(default = "default_lora_scale")]
    pub lora_scale: f32,
    /// True if `lora.safetensors` is bundled in this cartridge.
    #[serde(default)]
    pub has_lora: bool,
    /// True if legacy full `neural/model.safetensors` is bundled.
    #[serde(default)]
    pub has_safetensors: bool,
    /// True if `pack.imedict` (or legacy `personal.imedict`) is bundled.
    #[serde(default)]
    pub has_dictionary: bool,
    /// True if this cartridge contains private personal corpus weights (never publish/upload).
    #[serde(default)]
    pub private_pack: bool,
}

impl ImePackManifest {
    /// Reads and parses a `manifest.json` file from disk.
    ///
    /// # Errors
    ///
    /// Returns [`PersonalizeError`] if reading or JSON parsing fails.
    pub fn load_file(path: &Path) -> Result<Self, PersonalizeError> {
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Returns the canonical directory slug for storing under `packs/<slug>.imepack`.
    #[must_use]
    pub fn effective_id(&self) -> String {
        let raw = if self.id.trim().is_empty() {
            &self.name
        } else {
            &self.id
        };
        let mut slug = String::with_capacity(raw.len());
        for ch in raw.chars() {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                slug.push(ch.to_ascii_lowercase());
            } else if ch.is_whitespace() && !slug.ends_with('-') {
                slug.push('-');
            }
        }
        let trimmed = slug.trim_matches('-');
        if trimmed.is_empty() {
            "custom-pack".to_string()
        } else {
            trimmed.to_string()
        }
    }
}

/// Options for building a `v2.0` `.imepack` cartridge directory.
#[derive(Debug, Clone)]
pub struct PackBuildOptions<'a> {
    /// Stable kebab-case identifier.
    pub id: &'a str,
    /// Human-readable display name.
    pub name: &'a str,
    /// Summary description.
    pub description: &'a str,
    /// Pack semantic version.
    pub version: &'a str,
    /// Routing policy (`Always`, `Developer`, `Classical`).
    pub routing: LoraRouting,
    /// Base `LoRA` scale (`alpha / rank`).
    pub lora_scale: f32,
    /// Optional lexical TSV (`surface\tpinyin\tlog_prob[\tis_tech]`).
    pub lexicon_tsv: Option<&'a Path>,
    /// Optional `Tab` continuations TSV (`trigger\tnext_clause`).
    pub continuations_tsv: Option<&'a Path>,
    /// Optional N-gram boosts TSV (`2\tw1\tw2\tbonus` / `3\tw1\tw2\tw3\tbonus`).
    pub ngrams_tsv: Option<&'a Path>,
    /// Optional `lora.safetensors` low-rank adapter file.
    pub lora_safetensors: Option<&'a Path>,
    /// Optional legacy full `neural/` directory.
    pub neural_dir: Option<&'a Path>,
    /// Marks the cartridge as strictly private to the local machine.
    pub private_pack: bool,
}

/// Installed `.imepack` cartridge discovered under `app_support_dir/packs/`.
#[derive(Debug, Clone)]
pub struct InstalledImePack {
    /// Cartridge directory path (`.../packs/<id>.imepack`).
    pub dir: PathBuf,
    /// Parsed `manifest.json`.
    pub manifest: ImePackManifest,
    /// Resolved path to `pack.imedict` (or `personal.imedict`) if present.
    pub dict_path: Option<PathBuf>,
    /// Resolved path to `lora.safetensors` if present.
    pub lora_path: Option<PathBuf>,
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

/// Builds a `v2.0` `.imepack` cartridge directory containing `manifest.json`, optional
/// `pack.imedict` (compiled with words, English terms, N-grams, and `Tab` continuations!), and
/// optional `lora.safetensors`.
///
/// # Errors
///
/// Returns [`PersonalizeError`] on I/O, JSON, or dictionary compilation failure.
#[allow(clippy::too_many_lines)]
pub fn build_imepack_v2(
    opts: &PackBuildOptions<'_>,
    out_pack_dir: &Path,
) -> Result<ImePackManifest, PersonalizeError> {
    std::fs::create_dir_all(out_pack_dir)?;

    let mut lex = Lexicon::new();
    let mut dict_items = 0usize;

    if let Some(tsv_path) = opts.lexicon_tsv
        && tsv_path.exists()
    {
        let raw = std::fs::read_to_string(tsv_path)?;
        for raw_line in raw.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            match cols.as_slice() {
                ["EN", surface, lp_str, tech_str] => {
                    if let Ok(lp) = lp_str.parse::<f32>() {
                        lex.insert_english_term(*surface, lp, *tech_str == "1");
                        dict_items += 1;
                    }
                }
                [surface, py, lp_str, tech_str] => {
                    if let Ok(lp) = lp_str.parse::<f32>()
                        && lex.insert_pinyin_word(*surface, py, lp, *tech_str == "1")
                    {
                        dict_items += 1;
                    }
                }
                [surface, py, lp_str] => {
                    if let Ok(lp) = lp_str.parse::<f32>() {
                        if *py == "EN" {
                            lex.insert_english_term(*surface, lp, true);
                            dict_items += 1;
                        } else if lex.insert_pinyin_word(*surface, py, lp, false) {
                            dict_items += 1;
                        }
                    }
                }
                _ => {
                    let st = parse_tsv_dict(line, &mut lex);
                    dict_items += st.imported_entries;
                }
            }
        }
    }

    if let Some(cont_path) = opts.continuations_tsv
        && cont_path.exists()
    {
        let raw_cont = std::fs::read_to_string(cont_path)?;
        dict_items += lex.load_continuations_tsv(&raw_cont);
    }

    if let Some(ng_path) = opts.ngrams_tsv
        && ng_path.exists()
    {
        let raw_ng = std::fs::read_to_string(ng_path)?;
        dict_items += lex.load_ngram_tsv(&raw_ng);
    }

    let has_dictionary = if dict_items > 0 {
        let builder = CompiledDictBuilder::from_lexicon(&lex);
        builder.write_to_file(&out_pack_dir.join("pack.imedict"))?;
        true
    } else {
        false
    };

    let mut has_lora = false;
    if let Some(lora_src) = opts.lora_safetensors
        && lora_src.exists()
    {
        let dst = out_pack_dir.join("lora.safetensors");
        if lora_src != dst {
            std::fs::copy(lora_src, &dst)?;
        }
        has_lora = true;
    }

    let mut has_safetensors = false;
    if let Some(n_dir) = opts.neural_dir
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
        format_version: "2.0".to_string(),
        id: opts.id.to_string(),
        name: opts.name.to_string(),
        description: opts.description.to_string(),
        version: opts.version.to_string(),
        base_model: "chime-qwen3-6l-24k".to_string(),
        routing: opts.routing,
        lora_scale: opts.lora_scale,
        has_lora,
        has_safetensors,
        has_dictionary,
        private_pack: opts.private_pack,
    };
    std::fs::write(
        out_pack_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

/// Backward-compatible wrapper packaging a basic personal `.imepack` directory.
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
    build_imepack_v2(
        &PackBuildOptions {
            id: name,
            name,
            description: "Personal vocabulary and neural adaptation pack",
            version: "0.6.0",
            routing: LoraRouting::Always,
            lora_scale: 1.0,
            lexicon_tsv: personal_tsv,
            continuations_tsv: None,
            ngrams_tsv: None,
            lora_safetensors: None,
            neural_dir,
            private_pack: true,
        },
        out_pack_dir,
    )
}

/// Installs either a `<name>.imepack` directory OR a `<name>.imepack.zip` archive into
/// `app_support_dir/packs/<id>.imepack/`!
///
/// # Errors
///
/// Returns [`PersonalizeError`] if extraction fails, `manifest.json` is missing, or copying fails.
pub fn install_imepack_dir(
    pack_path: &Path,
    app_support_dir: &Path,
) -> Result<ImePackManifest, PersonalizeError> {
    if pack_path.is_file()
        && pack_path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("zip"))
    {
        let tmp_extract =
            std::env::temp_dir().join(format!("chime_pack_extract_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp_extract);
        std::fs::create_dir_all(&tmp_extract)?;
        extract_zip_archive(pack_path, &tmp_extract)?;
        let resolved = find_manifest_dir(&tmp_extract).ok_or_else(|| {
            let _ = std::fs::remove_dir_all(&tmp_extract);
            PersonalizeError::InvalidArchive("zip archive contains no manifest.json".to_string())
        })?;
        let res = install_unpacked_imepack_dir(&resolved, app_support_dir);
        let _ = std::fs::remove_dir_all(&tmp_extract);
        return res;
    }

    install_unpacked_imepack_dir(pack_path, app_support_dir)
}

fn install_unpacked_imepack_dir(
    pack_dir: &Path,
    app_support_dir: &Path,
) -> Result<ImePackManifest, PersonalizeError> {
    let manifest_bytes = std::fs::read(pack_dir.join("manifest.json"))?;
    let manifest: ImePackManifest = serde_json::from_slice(&manifest_bytes)?;
    let slug = manifest.effective_id();

    let dst_pack_dir = app_support_dir
        .join("packs")
        .join(format!("{slug}.imepack"));
    std::fs::create_dir_all(&dst_pack_dir)?;
    std::fs::write(
        dst_pack_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    if manifest.has_dictionary {
        let src_dict = if pack_dir.join("pack.imedict").exists() {
            pack_dir.join("pack.imedict")
        } else {
            pack_dir.join("personal.imedict")
        };
        if src_dict.exists() {
            let dst_dict = dst_pack_dir.join("pack.imedict");
            if src_dict != dst_dict {
                std::fs::copy(&src_dict, &dst_dict)?;
            }
        }
    }

    if (manifest.has_lora || pack_dir.join("lora.safetensors").exists())
        && pack_dir.join("lora.safetensors").exists()
    {
        let src_lora = pack_dir.join("lora.safetensors");
        let dst_lora = dst_pack_dir.join("lora.safetensors");
        if src_lora != dst_lora {
            std::fs::copy(&src_lora, &dst_lora)?;
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

/// Discovers and validates all installed `.imepack` cartridges under `app_support_dir/packs/`,
/// ordered so general public packs (`classics-poetry`, `tech-developer`) mount first and personal
/// `Always` packs (`personal-*`) mount at highest lexical priority!
#[must_use]
pub fn discover_installed_packs(app_support_dir: &Path) -> Vec<InstalledImePack> {
    let packs_root = app_support_dir.join("packs");
    let Ok(entries) = std::fs::read_dir(packs_root) else {
        return Vec::new();
    };
    let mut packs = Vec::new();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_dir() || !path.join("manifest.json").exists() {
            continue;
        }
        let Ok(bytes) = std::fs::read(path.join("manifest.json")) else {
            continue;
        };
        let Ok(manifest) = serde_json::from_slice::<ImePackManifest>(&bytes) else {
            continue;
        };
        let dict_path = ["pack.imedict", "personal.imedict"]
            .into_iter()
            .map(|f| path.join(f))
            .find(|p| p.exists());
        let lora_cand = path.join("lora.safetensors");
        let lora_path = lora_cand.exists().then_some(lora_cand);
        packs.push(InstalledImePack {
            dir: path,
            manifest,
            dict_path,
            lora_path,
        });
    }
    // `mount_compiled_mmap` inserts at index 0 (later mounts win ties), so sort general packs
    // before personal `private_pack` / `Always` packs!
    packs.sort_by(|a, b| {
        a.manifest
            .private_pack
            .cmp(&b.manifest.private_pack)
            .then_with(|| a.manifest.effective_id().cmp(&b.manifest.effective_id()))
    });
    packs
}

fn find_manifest_dir(root: &Path) -> Option<PathBuf> {
    if root.join("manifest.json").exists() {
        return Some(root.to_path_buf());
    }
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.filter_map(Result::ok) {
        let p = entry.path();
        if p.is_dir() {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name == "__MACOSX" {
                continue;
            }
            if p.join("manifest.json").exists() {
                return Some(p);
            }
        }
    }
    None
}

fn extract_zip_archive(zip_path: &Path, dst_dir: &Path) -> Result<(), PersonalizeError> {
    let ditto = Path::new("/usr/bin/ditto");
    let status = if ditto.exists() {
        Command::new(ditto)
            .arg("-x")
            .arg("-k")
            .arg(zip_path)
            .arg(dst_dir)
            .status()?
    } else {
        Command::new("unzip")
            .arg("-q")
            .arg("-o")
            .arg(zip_path)
            .arg("-d")
            .arg(dst_dir)
            .status()?
    };
    if status.success() {
        Ok(())
    } else {
        Err(PersonalizeError::InvalidArchive(format!(
            "failed to extract {}",
            zip_path.display()
        )))
    }
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
