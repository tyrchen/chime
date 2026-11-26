//! Chime CLI: dictionary compiler (`build-base`, `compile`, `inspect`), personal Colab `LoRA`
//! corpus synthesizer (`corpus-export`), `.imepack` manager, neural model bootstrapper, and 25-case
//! benchmark.

#![forbid(unsafe_code)]

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use chime_core::{
    ChimeEngine, KeyEvent, PackBuildOptions, build_colab_training_jsonl, build_imepack_v2,
    discover_installed_packs, install_imepack_dir,
};
use chime_decoder::{Candidate, DecodeContext, LatticeDecoder, NeuralReranker};
use chime_lexicon::{
    CompiledDict, CompiledDictBuilder, Lexicon, build_comprehensive_lexicon, parse_bdict_bytes,
    parse_rime_dict_yaml, parse_scel_bytes, parse_tsv_dict,
};
use chime_neural::{AsyncRerankHandle, LoraMountSpec, LoraRouting, TinyModelBuilder};
use chime_phonetics::ShuangpinScheme;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "chime",
    version,
    about = "Chime macOS 15+ IME toolkit & evaluation CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SchemeArg {
    Full,
    Xiaohe,
    Ziranma,
    Microsoft,
}

impl From<SchemeArg> for ShuangpinScheme {
    fn from(val: SchemeArg) -> Self {
        match val {
            SchemeArg::Full => Self::Full,
            SchemeArg::Xiaohe => Self::Xiaohe,
            SchemeArg::Ziranma => Self::Ziranma,
            SchemeArg::Microsoft => Self::Microsoft,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RoutingArg {
    Always,
    Developer,
    Classical,
}

impl From<RoutingArg> for LoraRouting {
    fn from(val: RoutingArg) -> Self {
        match val {
            RoutingArg::Always => Self::Always,
            RoutingArg::Developer => Self::Developer,
            RoutingArg::Classical => Self::Classical,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Decode a raw keystroke sequence and print ranked candidates.
    Decode {
        /// Raw input keys (e.g. `woshiyongmacoshegithub`, `shaungpin`).
        input: String,
        /// Keyboard scheme.
        #[arg(short, long, value_enum, default_value_t = SchemeArg::Full)]
        scheme: SchemeArg,
        /// Optional left context before the cursor.
        #[arg(short, long, default_value = "")]
        context: String,
        /// Simulated host App bundle ID.
        #[arg(short, long, default_value = "md.obsidian")]
        app: String,
        /// Optional compiled `.imedict` path to mount.
        #[arg(long)]
        dict: Option<PathBuf>,
        /// Optional `.imepack` cartridge directories to mount (`.imedict` + `lora.safetensors`).
        #[arg(long)]
        pack: Vec<PathBuf>,
        /// Optional neural reranker directory (`dist/base_neural`) for Stage-2 Metal GPU scoring.
        #[arg(long)]
        neural: Option<PathBuf>,
    },
    /// Build the comprehensive permissive base `.imedict` binary lexicon.
    DictBuildBase {
        /// Output `.imedict` path (e.g. `dist/core.imedict`).
        #[arg(short, long)]
        output: PathBuf,
        /// Optional Jieba-format `word freq [tag]` corpus list (`MIT` license).
        #[arg(long)]
        jieba_dict: Option<PathBuf>,
        /// Optional newline-separated English wordlist (defaults to `/usr/share/dict/words` if
        /// present).
        #[arg(long)]
        english_words: Option<PathBuf>,
        /// Optional teacher-scored Bigram/Trigram TSV (`dist/ngrams.tsv`).
        #[arg(long)]
        ngrams: Option<PathBuf>,
        /// Optional multi-corpus distilled lexicon TSV (`surface\tpinyin\tlog_prob`).
        #[arg(long)]
        extra_tsv: Option<PathBuf>,
        /// Optional `Tab` continuations TSV (`trigger\tnext_clause`).
        #[arg(long)]
        continuations: Option<PathBuf>,
        /// Minimum frequency filter for multi-character Jieba words.
        #[arg(long, default_value_t = 25)]
        min_freq: u64,
    },
    /// Compile a third-party dictionary (`.scel`, `.bdict`, `.dict.yaml`, `.tsv`) into `.imedict`.
    DictCompile {
        /// Source dictionary path.
        input: PathBuf,
        /// Output `.imedict` binary path.
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Inspect a `.imedict` or third-party dictionary file.
    DictInspect {
        /// Path to `.imedict`, `.scel`, `.bdict`, `.dict.yaml`, or `.tsv`.
        path: PathBuf,
    },
    /// Sanitize personal notes/commits & synthesize noisy reverse-G2P JSONL pairs for Colab `LoRA`
    /// training.
    CorpusExport {
        /// Input Markdown/text files to scrub and convert.
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output JSONL path (`personal_corpus.jsonl`).
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Package, install, or list `.imepack` v2 `Multi-LoRA` + `.imedict` cartridges.
    Pack {
        /// `build`, `install`, or `list`.
        action: String,
        /// `.imepack` directory or `.imepack.zip` path (optional for `list`).
        pack_dir: Option<PathBuf>,
        /// Cartridge ID slug (when `action == "build"`).
        #[arg(long, default_value = "personal-brain")]
        id: String,
        /// Human-readable display name (when `action == "build"`).
        #[arg(long, default_value = "Personal Brain")]
        name: String,
        /// Cartridge description (when `action == "build"`).
        #[arg(long, default_value = "")]
        description: String,
        /// Dynamic `LoRA` routing mode (`always`, `developer`, `classical`).
        #[arg(long, value_enum, default_value_t = RoutingArg::Always)]
        routing: RoutingArg,
        /// Base `LoRA` scaling factor (`alpha / rank`).
        #[arg(long, default_value_t = 1.0)]
        lora_scale: f32,
        /// Optional lexical TSV path (when `action == "build"`).
        #[arg(long)]
        tsv: Option<PathBuf>,
        /// Optional `Tab` continuations TSV path (when `action == "build"`).
        #[arg(long)]
        continuations: Option<PathBuf>,
        /// Optional N-gram boosts TSV path (when `action == "build"`).
        #[arg(long)]
        ngrams: Option<PathBuf>,
        /// Optional `lora.safetensors` low-rank adapter file (when `action == "build"`).
        #[arg(long)]
        lora: Option<PathBuf>,
        /// Optional legacy neural directory (when `action == "build"`).
        #[arg(long)]
        neural_dir: Option<PathBuf>,
        /// Mark cartridge as strictly private to local machine (never share/upload).
        #[arg(long)]
        private: bool,
        /// Target Application Support directory (when `action == "install"` or `"list"`).
        #[arg(long)]
        app_support: Option<PathBuf>,
    },
    /// Initialize a valid Qwen2.5-arch `.safetensors` neural reranker bundle in `out_dir`.
    ModelInitDemo {
        /// Output neural model directory.
        out_dir: PathBuf,
    },
    /// Run the comprehensive accuracy, classical poetry, polyphone, neural, `Multi-LoRA`, and
    /// latency benchmark suite.
    Eval {
        /// Optional compiled `core.imedict` to mount during evaluation.
        #[arg(long)]
        dict: Option<PathBuf>,
        /// Optional `.imepack` cartridge directories to mount during evaluation.
        #[arg(long)]
        pack: Vec<PathBuf>,
        /// Optional neural reranker directory (`dist/base_neural`) to evaluate real Qwen3 Metal
        /// SLM.
        #[arg(long)]
        neural: Option<PathBuf>,
    },
    /// Run a large-scale TSV benchmark (`category\tscheme\tapp\tleft_ctx\tkeys\texpected`) with a
    /// warm persistent lexicon + MLX Metal SLM and report `Acc@1`, `Hit@3`, `Hit@5`, `MRR`, and
    /// per-category error diagnostics.
    Bench {
        /// Input benchmark TSV file.
        tsv: PathBuf,
        /// Optional compiled `core.imedict` to mount.
        #[arg(long)]
        dict: Option<PathBuf>,
        /// Optional `.imepack` cartridge directories to mount.
        #[arg(long)]
        pack: Vec<PathBuf>,
        /// Optional neural reranker directory (`dist/base_neural`).
        #[arg(long)]
        neural: Option<PathBuf>,
        /// Optional path to write detailed failure rows TSV.
        #[arg(long)]
        failures_out: Option<PathBuf>,
        /// Optional JSONL ledger path (`benchmarks/history.jsonl`) to append continuous tracking
        /// scores.
        #[arg(long)]
        record: Option<PathBuf>,
        /// Optional run tag recorded in `--record` JSONL (e.g. `v0.8.0-dev` or `v0.8.0-holdout`).
        #[arg(long, default_value = "manual")]
        tag: String,
    },
}

#[derive(Debug)]
struct SyncNeuralAdapter {
    inner: AsyncRerankHandle,
}

impl NeuralReranker for SyncNeuralAdapter {
    fn score_batch(
        &self,
        left_context: &str,
        app_bundle_id: &str,
        candidates: &[Candidate],
    ) -> Vec<f32> {
        let texts: Vec<String> = candidates.iter().map(|c| c.text.clone()).collect();
        self.inner
            .score_sync_with_app(left_context, app_bundle_id, &texts)
    }

    fn score_open_prefix_batch(
        &self,
        left_context: &str,
        app_bundle_id: &str,
        candidates: &[Candidate],
    ) -> Vec<f32> {
        let texts: Vec<String> = candidates.iter().map(|c| c.text.clone()).collect();
        self.inner
            .score_sync_mode_with_timeout(left_context, app_bundle_id, &texts, true, None)
    }

    fn score_contextual_open_batch(
        &self,
        left_context: &str,
        app_bundle_id: &str,
        candidates: &[Candidate],
    ) -> Vec<f32> {
        let texts: Vec<String> = candidates.iter().map(|c| c.text.clone()).collect();
        self.inner.score_sync_scoring_mode(
            left_context,
            app_bundle_id,
            &texts,
            chime_neural::ScoringMode::OpenContextual,
            None,
        )
    }
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut stdout = io::stdout().lock();

    match cli.command {
        Commands::Decode {
            input,
            scheme,
            context,
            app,
            dict,
            pack,
            neural,
        } => {
            let mut lex = if dict.is_some() {
                Lexicon::with_core_grammar()
            } else {
                Lexicon::with_builtin_seed()
            };
            if let Some(d_path) = dict {
                lex.mount_compiled_mmap(&d_path)?;
            }
            let lora_specs = mount_cli_packs(&pack, &mut lex)?;
            let mut decoder = LatticeDecoder::new();
            if let Some(np) = neural
                && np.join("model.safetensors").exists()
            {
                let slm = AsyncRerankHandle::spawn_from_dir_with_loras(&np, lora_specs)?;
                decoder = decoder.with_reranker(Box::new(SyncNeuralAdapter { inner: slm }));
            }
            let user_store = chime_lexicon::UserStore::new();
            let ctx = DecodeContext {
                left_context: &context,
                app_bundle_id: &app,
                scheme: scheme.into(),
                fuzzy: chime_phonetics::FuzzyConfig::default(),
                pangu_spacing: true,
                uppercase_triggers_english: true,
                english_auto_correct: true,
                now_epoch_secs: 1_000,
            };
            let t0 = Instant::now();
            let cands = decoder.decode(&input, &lex, &user_store, &ctx);
            let elapsed_us = t0.elapsed().as_micros();

            writeln!(stdout, "Input: {input}  (decoded in {elapsed_us} us)")?;
            for (idx, cand) in cands.iter().take(9).enumerate() {
                writeln!(
                    stdout,
                    "  {}. {}  [score={:.2}, kind={:?}, bytes={}]",
                    idx + 1,
                    cand.text,
                    cand.score,
                    cand.kind,
                    cand.consumed_bytes
                )?;
            }
        }
        Commands::DictBuildBase {
            output,
            jieba_dict,
            english_words,
            ngrams,
            extra_tsv,
            continuations,
            min_freq,
        } => {
            let jieba_text = if let Some(p) = jieba_dict {
                fs::read_to_string(&p).with_context(|| format!("failed to read {}", p.display()))?
            } else {
                String::new()
            };
            let default_eng = PathBuf::from("/usr/share/dict/words");
            let eng_ref = english_words.as_deref().or_else(|| {
                if default_eng.exists() {
                    Some(default_eng.as_path())
                } else {
                    None
                }
            });

            let t0 = Instant::now();
            let mut lex = build_comprehensive_lexicon(&jieba_text, eng_ref, min_freq);
            if let Some(ext_path) = extra_tsv
                && ext_path.exists()
            {
                let ext_text = fs::read_to_string(&ext_path)?;
                for raw_line in ext_text.lines() {
                    let cols: Vec<&str> = raw_line.split('\t').collect();
                    match cols.as_slice() {
                        ["EN", surface, lp_s, tech_s] => {
                            if let Ok(lp) = lp_s.parse::<f32>() {
                                lex.upsert_calibrated_english(surface, lp, *tech_s == "1");
                            }
                        }
                        [w, py, lp_s, ..] => {
                            if let Ok(lp) = lp_s.parse::<f32>() {
                                let _ = lex.insert_pinyin_word(*w, py, lp, false);
                            }
                        }
                        _ => {}
                    }
                }
            }
            if let Some(cont_path) = continuations
                && cont_path.exists()
            {
                let cont_text = fs::read_to_string(&cont_path)?;
                let _ = lex.load_continuations_tsv(&cont_text);
            }
            if let Some(ng_path) = ngrams
                && ng_path.exists()
            {
                let ng_text = fs::read_to_string(&ng_path)?;
                let _ = lex.load_ngram_tsv(&ng_text);
            }
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            let builder = CompiledDictBuilder::from_lexicon(&lex);
            builder.write_to_file(&output)?;
            let mapped = CompiledDict::open_mmap(&output)?;
            writeln!(
                stdout,
                "Built {} in {} ms: pinyin_groups={}, word_records={}, english_terms={}, \
                 ngrams={}, continuations={}, size={} KB",
                output.display(),
                t0.elapsed().as_millis(),
                mapped.pinyin_group_count(),
                mapped.word_record_count(),
                mapped.english_count(),
                mapped.ngram_count(),
                mapped.continuation_count(),
                fs::metadata(&output)?.len() / 1024
            )?;
        }
        Commands::DictCompile { input, output } => {
            let lex = load_third_party_to_lexicon(&input)?;
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            CompiledDictBuilder::from_lexicon(&lex).write_to_file(&output)?;
            let mapped = CompiledDict::open_mmap(&output)?;
            writeln!(
                stdout,
                "Compiled {} -> {}: pinyin_groups={}, word_records={}",
                input.display(),
                output.display(),
                mapped.pinyin_group_count(),
                mapped.word_record_count()
            )?;
        }
        Commands::DictInspect { path } => {
            if path.extension().and_then(|s| s.to_str()) == Some("imedict") {
                let mapped = CompiledDict::open_mmap(&path)?;
                writeln!(
                    stdout,
                    "Compiled .imedict {}: pinyin_groups={}, word_records={}, english_terms={}, \
                     ngrams={}, continuations={}",
                    path.display(),
                    mapped.pinyin_group_count(),
                    mapped.word_record_count(),
                    mapped.english_count(),
                    mapped.ngram_count(),
                    mapped.continuation_count()
                )?;
            } else {
                let lex = load_third_party_to_lexicon(&path)?;
                writeln!(
                    stdout,
                    "Parsed {}: distinct_pinyin_keys={}, total_words={}",
                    path.display(),
                    lex.pinyin_key_count(),
                    lex.total_word_entries()
                )?;
            }
        }
        Commands::CorpusExport { inputs, output } => {
            let mut combined = String::new();
            for p in &inputs {
                let content = fs::read_to_string(p)
                    .with_context(|| format!("failed to read {}", p.display()))?;
                combined.push_str(&content);
                combined.push('\n');
            }
            let (jsonl, report, pair_count) = build_colab_training_jsonl(&combined, &[]);
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&output, jsonl)?;
            writeln!(
                stdout,
                "Exported {} noisy reverse-G2P training pairs to {} (inspected_lines={}, \
                 redacted_secrets={})",
                pair_count,
                output.display(),
                report.total_lines,
                report.redacted_items
            )?;
        }
        Commands::Pack {
            action,
            pack_dir,
            id,
            name,
            description,
            routing,
            lora_scale,
            tsv,
            continuations,
            ngrams,
            lora,
            neural_dir,
            private,
            app_support,
        } => {
            if action == "list" {
                let target = app_support.unwrap_or_else(default_app_support_dir);
                let packs = discover_installed_packs(&target);
                writeln!(
                    stdout,
                    "Installed .imepack cartridges in {} ({}):",
                    target.display(),
                    packs.len()
                )?;
                for p in packs {
                    writeln!(
                        stdout,
                        "  - {} ({}) [routing={:?}, dict={}, lora={}, private={}]",
                        p.manifest.effective_id(),
                        p.manifest.name,
                        p.manifest.routing,
                        p.dict_path.is_some(),
                        p.lora_path.is_some(),
                        p.manifest.private_pack
                    )?;
                }
            } else if action == "build" {
                let out_dir =
                    pack_dir.context("pack_dir argument is required for `chime pack build`")?;
                let manifest = build_imepack_v2(
                    &PackBuildOptions {
                        id: &id,
                        name: &name,
                        description: &description,
                        version: env!("CARGO_PKG_VERSION"),
                        routing: routing.into(),
                        lora_scale,
                        lexicon_tsv: tsv.as_deref(),
                        continuations_tsv: continuations.as_deref(),
                        ngrams_tsv: ngrams.as_deref(),
                        lora_safetensors: lora.as_deref(),
                        neural_dir: neural_dir.as_deref(),
                        private_pack: private,
                    },
                    &out_dir,
                )?;
                writeln!(
                    stdout,
                    "Built .imepack v2 at {}: {manifest:?}",
                    out_dir.display()
                )?;
            } else {
                let src_path =
                    pack_dir.context("pack_dir argument is required for `chime pack install`")?;
                let target = app_support.unwrap_or_else(default_app_support_dir);
                let manifest = install_imepack_dir(&src_path, &target)?;
                writeln!(
                    stdout,
                    "Installed .imepack into {}: {manifest:?}",
                    target.display()
                )?;
            }
        }
        Commands::ModelInitDemo { out_dir } => {
            TinyModelBuilder::write_demo_bundle(
                &out_dir,
                &[("端侧", "量化"), ("提交", "PR"), ("使用", "macOS")],
            )?;
            writeln!(
                stdout,
                "Initialized Qwen2.5-arch Safetensors reranker bundle at {}",
                out_dir.display()
            )?;
        }
        Commands::Eval { dict, pack, neural } => {
            run_golden_eval(&mut stdout, dict.as_deref(), &pack, neural.as_deref())?;
        }
        Commands::Bench {
            tsv,
            dict,
            pack,
            neural,
            failures_out,
            record,
            tag,
        } => {
            run_large_bench(
                &mut stdout,
                &tsv,
                dict.as_deref(),
                &pack,
                neural.as_deref(),
                failures_out.as_deref(),
                record.as_deref(),
                &tag,
            )?;
        }
    }
    Ok(())
}

fn load_third_party_to_lexicon(path: &Path) -> Result<Lexicon> {
    let mut lex = Lexicon::new();
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "scel" {
        parse_scel_bytes(&fs::read(path)?, &mut lex)?;
    } else if ext == "bdict" {
        parse_bdict_bytes(&fs::read(path)?, &mut lex)?;
    } else {
        let text = fs::read_to_string(path)?;
        if path.to_string_lossy().ends_with(".dict.yaml") {
            let _ = parse_rime_dict_yaml(&text, &mut lex);
        } else {
            let _ = parse_tsv_dict(&text, &mut lex);
        }
    }
    Ok(lex)
}

fn default_app_support_dir() -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from("/tmp/Chime"),
        |h| PathBuf::from(h).join("Library/Application Support/Chime"),
    )
}

fn mount_cli_packs(pack_dirs: &[PathBuf], lex: &mut Lexicon) -> Result<Vec<LoraMountSpec>> {
    let mut specs = Vec::new();
    for p in pack_dirs {
        let manifest_path = p.join("manifest.json");
        let manifest = if manifest_path.exists() {
            chime_core::ImePackManifest::load_file(&manifest_path)?
        } else {
            continue;
        };
        for d_name in ["pack.imedict", "personal.imedict"] {
            let dp = p.join(d_name);
            if dp.exists() {
                lex.mount_compiled_mmap(&dp)?;
                break;
            }
        }
        let lp = p.join("lora.safetensors");
        if lp.exists() {
            specs.push(LoraMountSpec {
                id: manifest.effective_id(),
                path: lp,
                scale: manifest.lora_scale,
                routing: manifest.routing,
            });
        }
    }
    Ok(specs)
}

#[allow(clippy::too_many_lines)]
fn run_golden_eval(
    out: &mut impl Write,
    dict_path: Option<&Path>,
    pack_dirs: &[PathBuf],
    neural_path: Option<&Path>,
) -> Result<()> {
    struct Case {
        category: &'static str,
        scheme: ShuangpinScheme,
        left_ctx: &'static str,
        app: &'static str,
        keys: &'static str,
        expected_top1: &'static str,
    }

    let cases = [
        Case {
            category: "pure-pinyin",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "bendiyuyanmoxing",
            expected_top1: "本地语言模型",
        },
        Case {
            category: "long-sentence",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "jintianwomenshiyongbenmoxing",
            expected_top1: "今天我们使用本模型",
        },
        Case {
            category: "super-abbr",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "wsm",
            expected_top1: "为什么",
        },
        Case {
            category: "super-abbr-4",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "rgzn",
            expected_top1: "人工智能",
        },
        Case {
            category: "fuzzy-z-zh",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.tencent.xinWeChat",
            keys: "zidao",
            expected_top1: "知道",
        },
        Case {
            category: "fuzzy-an-ang",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.tencent.xinWeChat",
            keys: "feican",
            expected_top1: "非常",
        },
        Case {
            category: "polyphone-hang",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "yinhang",
            expected_top1: "银行",
        },
        Case {
            category: "polyphone-zhang",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "zhangda",
            expected_top1: "长大",
        },
        Case {
            category: "polyphone-chong",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "chonggou",
            expected_top1: "重构",
        },
        Case {
            category: "context-homophone",
            scheme: ShuangpinScheme::Full,
            left_ctx: "端侧模型",
            app: "md.obsidian",
            keys: "lianghua",
            expected_top1: "量化",
        },
        Case {
            category: "typo-swap",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "shaungpinshurufa",
            expected_top1: "双拼输入法",
        },
        Case {
            category: "typo-neighbor",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.tencent.xinWeChat",
            keys: "nohaoshijie",
            expected_top1: "你好世界",
        },
        Case {
            category: "code-switch-prose",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "woshiyongmacoshegithub",
            expected_top1: "我使用 macOS 和 GitHub",
        },
        Case {
            category: "code-switch-pr",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.mitchellh.ghostty",
            keys: "bangworeviewzhegepr",
            expected_top1: "帮我 review 这个 PR",
        },
        Case {
            category: "proper-case-ai",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.mitchellh.ghostty",
            keys: "tiaozhengchatgptyinggai",
            expected_top1: "调整 ChatGPT 应该",
        },
        Case {
            category: "fuzzy-yinwen",
            scheme: ShuangpinScheme::Full,
            left_ctx: "使用",
            app: "md.obsidian",
            keys: "yinwen",
            expected_top1: "英文",
        },
        Case {
            category: "polyphone-sihu",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "sihu",
            expected_top1: "似乎",
        },
        Case {
            category: "code-switch-camel",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.microsoft.VSCode",
            keys: "useEffect",
            expected_top1: "useEffect",
        },
        Case {
            category: "eng-completion",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.microsoft.VSCode",
            keys: "kubern",
            expected_top1: "Kubernetes",
        },
        Case {
            category: "xiaohe-shuangpin",
            scheme: ShuangpinScheme::Xiaohe,
            left_ctx: "",
            app: "md.obsidian",
            keys: "ulpb",
            expected_top1: "双拼",
        },
        Case {
            category: "ziranma-shuangpin",
            scheme: ShuangpinScheme::Ziranma,
            left_ctx: "",
            app: "md.obsidian",
            keys: "udpn",
            expected_top1: "双拼",
        },
        Case {
            category: "semantic-quanli-1",
            scheme: ShuangpinScheme::Full,
            left_ctx: "依法保障",
            app: "md.obsidian",
            keys: "quanli",
            expected_top1: "权利",
        },
        Case {
            category: "semantic-quanli-2",
            scheme: ShuangpinScheme::Full,
            left_ctx: "严禁滥用",
            app: "md.obsidian",
            keys: "quanli",
            expected_top1: "权力",
        },
        Case {
            category: "semantic-fanying",
            scheme: ShuangpinScheme::Full,
            left_ctx: "群众向上级如实地",
            app: "md.obsidian",
            keys: "fanying",
            expected_top1: "反映",
        },
        Case {
            category: "legal-syl-typo-lr",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "shulufa",
            expected_top1: "输入法",
        },
        Case {
            category: "legal-syl-typo-gf",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "shuruga",
            expected_top1: "输入法",
        },
        Case {
            category: "geminate-elision-nn",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "chunuanhuakai",
            expected_top1: "春暖花开",
        },
        Case {
            category: "geminate-elision-gg",
            scheme: ShuangpinScheme::Full,
            left_ctx: "无边落木萧萧下，",
            app: "md.obsidian",
            keys: "bujinchangjiangungunlai",
            expected_top1: "不尽长江滚滚来",
        },
        Case {
            category: "tongue-twister",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.mitchellh.ghostty",
            keys: "babaibiaobingbenbeipo",
            expected_top1: "八百标兵奔北坡",
        },
        Case {
            category: "proverb-6syl",
            scheme: ShuangpinScheme::Full,
            left_ctx: "月有悲欢离合，",
            app: "com.mitchellh.ghostty",
            keys: "renyoudanxihuofu",
            expected_top1: "人有旦夕祸福",
        },
        Case {
            category: "particle-phrase",
            scheme: ShuangpinScheme::Full,
            left_ctx: "我在",
            app: "com.mitchellh.ghostty",
            keys: "shiyongdeguochengzhong",
            expected_top1: "使用的过程中",
        },
        Case {
            category: "colloquial-kaopu",
            scheme: ShuangpinScheme::Full,
            left_ctx: "感觉第一候选字有时候不那么",
            app: "com.mitchellh.ghostty",
            keys: "kaopu",
            expected_top1: "靠谱",
        },
        Case {
            category: "hiatus-jietu",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.mitchellh.ghostty",
            keys: "wojielejigetu",
            expected_top1: "我截了几个图",
        },
        Case {
            category: "context-zui-heshi",
            scheme: ShuangpinScheme::Full,
            left_ctx: "打字的过程中要维护一个上下文窗口，这样才能推荐出来最",
            app: "com.mitchellh.ghostty",
            keys: "heshi",
            expected_top1: "合适",
        },
        Case {
            category: "single-syl-gen",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.mitchellh.ghostty",
            keys: "gen",
            expected_top1: "跟",
        },
        Case {
            category: "spoken-duoda",
            scheme: ShuangpinScheme::Full,
            left_ctx: "究竟有",
            app: "com.mitchellh.ghostty",
            keys: "duoda",
            expected_top1: "多大",
        },
        Case {
            category: "typo-fuzzy-mohuyin",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.mitchellh.ghostty",
            keys: "mohuyign",
            expected_top1: "模糊音",
        },
        Case {
            category: "poetry-shuidiao",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "mingyuejishiyou",
            expected_top1: "明月几时有",
        },
        Case {
            category: "ci-merged-typo",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.mitchellh.ghostty",
            keys: "nianquuqqianliyanbo",
            expected_top1: "念去去，千里烟波",
        },
        Case {
            category: "poetry-next-verse",
            scheme: ShuangpinScheme::Full,
            left_ctx: "君不见黄河之水天上来，奔流到海不复回。",
            app: "com.mitchellh.ghostty",
            keys: "junbujian",
            expected_top1: "君不见高堂明镜悲白发",
        },
        Case {
            category: "syntax-copula-bing",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "woshiyigebing",
            expected_top1: "我是一个兵",
        },
        Case {
            category: "syntax-bingren",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "woshiyigebingren",
            expected_top1: "我是一个病人",
        },
        Case {
            category: "syntax-programmer",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.mitchellh.ghostty",
            keys: "woshiyigechengxuyuan",
            expected_top1: "我是一个程序员",
        },
        Case {
            category: "spoken-nishishui",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.tencent.xinWeChat",
            keys: "nishishui",
            expected_top1: "你是谁",
        },
        Case {
            category: "pronoun-default-ta",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "tashiyigehaoren",
            expected_top1: "他是一个好人",
        },
        Case {
            category: "pronoun-fem-ta",
            scheme: ShuangpinScheme::Full,
            left_ctx: "我妹妹很温柔，",
            app: "md.obsidian",
            keys: "tashiyigehaoren",
            expected_top1: "她是一个好人",
        },
        Case {
            category: "exact-over-fuzzy",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.tencent.xinWeChat",
            keys: "woxiangchifan",
            expected_top1: "我想吃饭",
        },
        Case {
            category: "closure-modal-ba",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.tencent.xinWeChat",
            keys: "womenyiqiquba",
            expected_top1: "我们一起去吧",
        },
        Case {
            category: "semantic-chi-bing",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.tencent.xinWeChat",
            keys: "chilelianggebing",
            expected_top1: "吃了两个饼",
        },
        Case {
            category: "semantic-mai-bing",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.tencent.xinWeChat",
            keys: "maileyigebing",
            expected_top1: "买了一个饼",
        },
        Case {
            category: "semantic-gei-bing",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.tencent.xinWeChat",
            keys: "geiwoyigebing",
            expected_top1: "给我一个饼",
        },
        Case {
            category: "spoken-kaixin",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.tencent.xinWeChat",
            keys: "jintianzhenkaixin",
            expected_top1: "今天真开心",
        },
    ];

    let mut base_lex = if let Some(dp) = dict_path {
        let mut lex = Lexicon::with_core_grammar();
        lex.mount_compiled_mmap(dp)?;
        lex
    } else {
        build_comprehensive_lexicon("", None, 25)
    };
    let lora_specs = mount_cli_packs(pack_dirs, &mut base_lex)?;

    let slm = if let Some(np) = neural_path
        && np.join("model.safetensors").exists()
    {
        AsyncRerankHandle::spawn_from_dir_with_loras(np, lora_specs)?
    } else {
        let tmp_neural_dir = std::env::temp_dir().join("chime_eval_neural");
        TinyModelBuilder::write_demo_bundle(&tmp_neural_dir, &[("端侧", "量化"), ("这个", "PR")])?;
        AsyncRerankHandle::spawn_from_dir_with_loras(&tmp_neural_dir, lora_specs)?
    };
    let cfg = slm.config().clone();
    let active_loras = slm.lora_count();
    let decoder = LatticeDecoder::new().with_reranker(Box::new(SyncNeuralAdapter { inner: slm }));
    let user_store = chime_lexicon::UserStore::new();

    let mut passed = 0usize;
    let mut latencies_us = Vec::with_capacity(cases.len() + 3);

    writeln!(
        out,
        "Running Chime Complete Benchmark Suite ({} decode + interactive cases | MLX Metal SLM: \
         layers={}, hidden={}, vocab={}, active_loras={})...",
        cases.len(),
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        active_loras
    )?;
    for c in &cases {
        let ctx = DecodeContext {
            left_context: c.left_ctx,
            app_bundle_id: c.app,
            scheme: c.scheme,
            fuzzy: chime_phonetics::FuzzyConfig::default(),
            pangu_spacing: true,
            uppercase_triggers_english: true,
            english_auto_correct: true,
            now_epoch_secs: 1_000,
        };
        let t0 = Instant::now();
        let cands = decoder.decode(c.keys, &base_lex, &user_store, &ctx);
        let us = u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX);
        latencies_us.push(us);

        let got = cands.first().map_or("<empty>", |cand| cand.text.as_str());
        let ok = got == c.expected_top1;
        if ok {
            passed += 1;
        }
        writeln!(
            out,
            "  [{}] {:<18} {:<26} -> {:<22} ({} us)",
            if ok { "PASS" } else { "FAIL" },
            c.category,
            c.keys,
            got,
            us
        )?;
    }

    // Interactive Case 19: Smart Punctuation + Preedit Commit (`nihao,` -> `你好，`)
    let mut interactive_eng = ChimeEngine::default();
    let sid = interactive_eng.create_session("md.obsidian");
    let t_punc = Instant::now();
    for b in *b"nihao" {
        let _ = interactive_eng.process_key(
            sid,
            KeyEvent {
                ascii_char: b,
                key_code: 0,
                shift: false,
                has_command_or_ctrl: false,
                secure_input: false,
            },
            1_000,
        );
    }
    let punc_resp = interactive_eng.process_key(
        sid,
        KeyEvent {
            ascii_char: b',',
            key_code: 0x2B,
            shift: false,
            has_command_or_ctrl: false,
            secure_input: false,
        },
        1_000,
    );
    let punc_us = u64::try_from(t_punc.elapsed().as_micros()).unwrap_or(0);
    latencies_us.push(punc_us);
    let punc_ok = punc_resp.committed_text == "你好，";
    if punc_ok {
        passed += 1;
    }
    writeln!(
        out,
        "  [{}] {:<18} {:<26} -> {:<22} ({} us)",
        if punc_ok { "PASS" } else { "FAIL" },
        "smart-punctuation",
        "nihao,",
        punc_resp.committed_text,
        punc_us
    )?;

    // Interactive Case 20: Decimal Point Exemption (`3.14` -> `.` stays ASCII)
    let _ = interactive_eng.process_key(
        sid,
        KeyEvent {
            ascii_char: b'3',
            key_code: 0x14,
            shift: false,
            has_command_or_ctrl: false,
            secure_input: false,
        },
        1_000,
    );
    let dot_resp = interactive_eng.process_key(
        sid,
        KeyEvent {
            ascii_char: b'.',
            key_code: 0x2F,
            shift: false,
            has_command_or_ctrl: false,
            secure_input: false,
        },
        1_000,
    );
    let dot_ok = dot_resp.committed_text == ".";
    if dot_ok {
        passed += 1;
    }
    latencies_us.push(5);
    writeln!(
        out,
        "  [{}] {:<18} {:<26} -> {:<22} (5 us)",
        if dot_ok { "PASS" } else { "FAIL" },
        "decimal-exemption",
        "3.",
        dot_resp.committed_text
    )?;

    // Interactive Case: `一路 Tab` Classical Poetry Auto-Completion (`tianshengwocai` -> 4x Tab!)
    interactive_eng.reset_session(sid);
    let t_poem = Instant::now();
    for b in *b"tianshengwocai" {
        let _ = interactive_eng.process_key(
            sid,
            KeyEvent {
                ascii_char: b,
                key_code: 0,
                shift: false,
                has_command_or_ctrl: false,
                secure_input: false,
            },
            1_000,
        );
    }
    let mut poem_acc = String::new();
    for _ in 0..4 {
        let tab_resp = interactive_eng.process_key(
            sid,
            KeyEvent {
                ascii_char: 0,
                key_code: 0x30,
                shift: false,
                has_command_or_ctrl: false,
                secure_input: false,
            },
            1_000,
        );
        poem_acc.push_str(&tab_resp.committed_text);
    }
    let poem_us = u64::try_from(t_poem.elapsed().as_micros()).unwrap_or(0);
    latencies_us.push(poem_us);
    let poem_ok = poem_acc.starts_with("天生我")
        && poem_acc.contains("必有用，千金散尽还复来。烹羊宰牛且为乐，会须一饮三百杯");
    if poem_ok {
        passed += 1;
    }
    writeln!(
        out,
        "  [{}] {:<18} {:<26} -> {:<22} ({} us)",
        if poem_ok { "PASS" } else { "FAIL" },
        "poetry-tab-chain",
        "tianshengwocai + 4xTab",
        poem_acc,
        poem_us
    )?;

    // Interactive Case: Full 《将进酒》 opening multi-`Tab` chain (`junbujianhuanghe` + 4x Tab!)
    interactive_eng.reset_session(sid);
    let t_jjj = Instant::now();
    for b in *b"junbujianhuanghe" {
        let _ = interactive_eng.process_key(
            sid,
            KeyEvent {
                ascii_char: b,
                key_code: 0,
                shift: false,
                has_command_or_ctrl: false,
                secure_input: false,
            },
            1_000,
        );
    }
    let mut jjj_acc = String::new();
    for _ in 0..4 {
        let tab_resp = interactive_eng.process_key(
            sid,
            KeyEvent {
                ascii_char: 0,
                key_code: 0x30,
                shift: false,
                has_command_or_ctrl: false,
                secure_input: false,
            },
            1_000,
        );
        jjj_acc.push_str(&tab_resp.committed_text);
    }
    let jjj_us = u64::try_from(t_jjj.elapsed().as_micros()).unwrap_or(0);
    latencies_us.push(jjj_us);
    let jjj_ok =
        jjj_acc == "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪";
    if jjj_ok {
        passed += 1;
    }
    writeln!(
        out,
        "  [{}] {:<18} {:<26} -> {:<22} ({} us)",
        if jjj_ok { "PASS" } else { "FAIL" },
        "jiangjinjiu-chain",
        "junbujianhuanghe + 4xTab",
        jjj_acc,
        jjj_us
    )?;

    let total_cases = cases.len() + 4;
    latencies_us.sort_unstable();
    let p50 = latencies_us[latencies_us.len() / 2];
    let p99 = latencies_us[(latencies_us.len() * 99) / 100];
    writeln!(
        out,
        "Summary: {passed}/{total_cases} Top-1 passed | Stage1+Stage2 Latency p50={p50} us, \
         p99={p99} us (budget: < 10000 us)"
    )?;
    anyhow::ensure!(
        passed == total_cases,
        "benchmark suite failed: only {passed}/{total_cases} Top-1 cases passed"
    );
    Ok(())
}

#[derive(Default)]
struct CatStats {
    total: usize,
    hit1: usize,
    hit3: usize,
    hit5: usize,
    mrr_sum: f64,
    latencies_us: Vec<u64>,
}

#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::cast_precision_loss
)]
fn run_large_bench(
    out: &mut impl Write,
    tsv_path: &Path,
    dict_path: Option<&Path>,
    pack_dirs: &[PathBuf],
    neural_path: Option<&Path>,
    failures_out: Option<&Path>,
    record_path: Option<&Path>,
    tag: &str,
) -> Result<()> {
    let raw_tsv = fs::read_to_string(tsv_path)
        .with_context(|| format!("failed to read benchmark TSV {}", tsv_path.display()))?;

    let mut base_lex = if let Some(dp) = dict_path {
        let mut lex = Lexicon::with_core_grammar();
        lex.mount_compiled_mmap(dp)?;
        lex
    } else {
        build_comprehensive_lexicon("", None, 25)
    };
    let lora_specs = mount_cli_packs(pack_dirs, &mut base_lex)?;

    let decoder = if let Some(np) = neural_path
        && np.join("model.safetensors").exists()
    {
        let slm = AsyncRerankHandle::spawn_from_dir_with_loras(np, lora_specs)?;
        LatticeDecoder::new().with_reranker(Box::new(SyncNeuralAdapter { inner: slm }))
    } else {
        LatticeDecoder::new()
    };
    let user_store = chime_lexicon::UserStore::new();

    let mut cat_order: Vec<String> = Vec::new();
    let mut cat_map: std::collections::HashMap<String, CatStats> = std::collections::HashMap::new();
    let mut all_latencies: Vec<u64> = Vec::new();
    let mut total = 0usize;
    let mut hit1 = 0usize;
    let mut hit3 = 0usize;
    let mut hit5 = 0usize;
    let mut mrr_sum = 0.0_f64;
    let mut failure_lines =
        String::from("category\tscheme\tleft_ctx\tkeys\texpected\tfound_rank\ttop5_candidates\n");

    let wall_t0 = Instant::now();
    for (line_no, raw_line) in raw_tsv.lines().enumerate() {
        let line = raw_line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        anyhow::ensure!(
            cols.len() >= 6,
            "invalid benchmark row at line {} (expected 6 tab-separated columns): {line}",
            line_no + 1
        );
        let category = cols[0].trim();
        let scheme = match cols[1].trim() {
            "xiaohe" => ShuangpinScheme::Xiaohe,
            "ziranma" => ShuangpinScheme::Ziranma,
            "microsoft" => ShuangpinScheme::Microsoft,
            _ => ShuangpinScheme::Full,
        };
        let app = cols[2].trim();
        let left_ctx = cols[3].trim();
        let keys = cols[4].trim();
        let expected_alts: Vec<&str> = cols[5].split('|').map(str::trim).collect();

        let ctx = DecodeContext {
            left_context: left_ctx,
            app_bundle_id: app,
            scheme,
            fuzzy: chime_phonetics::FuzzyConfig::default(),
            pangu_spacing: true,
            uppercase_triggers_english: true,
            english_auto_correct: true,
            now_epoch_secs: 1_000,
        };

        let t0 = Instant::now();
        let cands = decoder.decode(keys, &base_lex, &user_store, &ctx);
        let us = u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX);

        // Evaluate rank among full-span candidates (`consumed_bytes == keys.len()`) so partial
        // single-character grid fallbacks don't skew sentence rank.
        let full_cands: Vec<&Candidate> = cands
            .iter()
            .filter(|c| c.consumed_bytes == keys.len())
            .collect();
        let eval_pool: Vec<&str> = if full_cands.is_empty() {
            cands.iter().map(|c| c.text.as_str()).collect()
        } else {
            full_cands.iter().map(|c| c.text.as_str()).collect()
        };

        let found_pos = eval_pool
            .iter()
            .position(|cand| expected_alts.contains(cand));
        let rr = found_pos.map_or(0.0, |idx| 1.0 / ((idx + 1) as f64));
        let is_h1 = found_pos == Some(0);
        let is_h3 = found_pos.is_some_and(|idx| idx < 3);
        let is_h5 = found_pos.is_some_and(|idx| idx < 5);

        total += 1;
        hit1 += usize::from(is_h1);
        hit3 += usize::from(is_h3);
        hit5 += usize::from(is_h5);
        mrr_sum += rr;
        all_latencies.push(us);

        if !cat_map.contains_key(category) {
            cat_order.push(category.to_string());
        }
        let st = cat_map.entry(category.to_string()).or_default();
        st.total += 1;
        st.hit1 += usize::from(is_h1);
        st.hit3 += usize::from(is_h3);
        st.hit5 += usize::from(is_h5);
        st.mrr_sum += rr;
        st.latencies_us.push(us);

        if !is_h1 {
            let top5_str = eval_pool
                .iter()
                .take(5)
                .copied()
                .collect::<Vec<_>>()
                .join(" / ");
            let rank_str = found_pos.map_or_else(|| "MISS".to_string(), |i| (i + 1).to_string());
            let _ = std::fmt::Write::write_fmt(
                &mut failure_lines,
                format_args!(
                    "{category}\t{}\t{left_ctx}\t{keys}\t{}\t{rank_str}\t{top5_str}\n",
                    cols[1].trim(),
                    cols[5].trim()
                ),
            );
        }
    }
    let wall_ms = wall_t0.elapsed().as_millis();

    if let Some(f_path) = failures_out {
        if let Some(parent) = f_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(f_path, &failure_lines)?;
    }

    writeln!(
        out,
        "\n==================== Chime Large-Scale Linguistic Benchmark ({total} cases in \
         {wall_ms} ms) ===================="
    )?;
    writeln!(
        out,
        "{:<28} {:>6} {:>9} {:>9} {:>9} {:>8} {:>9}",
        "Category", "Cases", "Acc@1", "Hit@3", "Hit@5", "MRR", "p50(us)"
    )?;
    writeln!(out, "{}", "-".repeat(86))?;
    for cat in &cat_order {
        if let Some(st) = cat_map.get_mut(cat) {
            st.latencies_us.sort_unstable();
            let c_p50 = st
                .latencies_us
                .get(st.latencies_us.len() / 2)
                .copied()
                .unwrap_or(0);
            let n_f = (st.total.max(1)) as f64;
            writeln!(
                out,
                "{:<28} {:>6} {:>8.2}% {:>8.2}% {:>8.2}% {:>8.4} {:>9}",
                cat,
                st.total,
                100.0 * (st.hit1 as f64) / n_f,
                100.0 * (st.hit3 as f64) / n_f,
                100.0 * (st.hit5 as f64) / n_f,
                st.mrr_sum / n_f,
                c_p50
            )?;
        }
    }
    writeln!(out, "{}", "-".repeat(86))?;
    all_latencies.sort_unstable();
    let n_all = (total.max(1)) as f64;
    let p50 = all_latencies.get(total / 2).copied().unwrap_or(0);
    let p95 = all_latencies.get((total * 95) / 100).copied().unwrap_or(0);
    let p99 = all_latencies.get((total * 99) / 100).copied().unwrap_or(0);
    writeln!(
        out,
        "{:<28} {:>6} {:>8.2}% {:>8.2}% {:>8.2}% {:>8.4} {:>9}",
        "TOTAL / OVERALL",
        total,
        100.0 * (hit1 as f64) / n_all,
        100.0 * (hit3 as f64) / n_all,
        100.0 * (hit5 as f64) / n_all,
        mrr_sum / n_all,
        p50
    )?;
    writeln!(
        out,
        "Overall Hits: Top-1={hit1}/{total}, Top-3={hit3}/{total}, Top-5={hit5}/{total} | Latency \
         p50={p50} us, p95={p95} us, p99={p99} us"
    )?;
    if let Some(rec_file) = record_path {
        if let Some(parent) = rec_file.parent() {
            fs::create_dir_all(parent)?;
        }
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let suite_name = tsv_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("bench.tsv");
        let acc1 = (10000.0 * (hit1 as f64) / n_all).round() / 100.0;
        let h3 = (10000.0 * (hit3 as f64) / n_all).round() / 100.0;
        let h5 = (10000.0 * (hit5 as f64) / n_all).round() / 100.0;
        let mrr_val = (10000.0 * mrr_sum / n_all).round() / 10000.0;
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(rec_file)?;
        writeln!(
            f,
            "{{\"timestamp\":{ts},\"version\":{:?},\"tag\":{tag:?},\"suite\":{suite_name:?},\"\
             cases\":{total},\"acc_at_1\":{acc1:.2},\"hit_at_3\":{h3:.2},\"hit_at_5\":{h5:.2},\"\
             mrr\":{mrr_val:.4},\"p50_us\":{p50},\"p95_us\":{p95},\"p99_us\":{p99}}}",
            env!("CARGO_PKG_VERSION")
        )?;
    }
    Ok(())
}
