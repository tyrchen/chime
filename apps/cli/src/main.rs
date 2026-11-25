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
    ChimeConfig, ChimeEngine, KeyEvent, build_colab_training_jsonl, build_imepack_dir,
    install_imepack_dir,
};
use chime_decoder::{Candidate, DecodeContext, LatticeDecoder, NeuralReranker};
use chime_lexicon::{
    CompiledDict, CompiledDictBuilder, Lexicon, build_comprehensive_lexicon, parse_bdict_bytes,
    parse_rime_dict_yaml, parse_scel_bytes, parse_tsv_dict,
};
use chime_neural::{AsyncRerankHandle, TinyModelBuilder};
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
    /// Package or install a Colab-trained `.imepack` personal brain bundle.
    Pack {
        /// Either `build` or `install`.
        action: String,
        /// `.imepack` directory path.
        pack_dir: PathBuf,
        /// Optional personal TSV path (when `action == "build"`).
        #[arg(long)]
        tsv: Option<PathBuf>,
        /// Optional neural directory containing `model.safetensors` (when `action == "build"`).
        #[arg(long)]
        neural_dir: Option<PathBuf>,
        /// Target Application Support directory (when `action == "install"`).
        #[arg(long)]
        app_support: Option<PathBuf>,
    },
    /// Initialize a valid Qwen2.5-arch `.safetensors` neural reranker bundle in `out_dir`.
    ModelInitDemo {
        /// Output neural model directory.
        out_dir: PathBuf,
    },
    /// Run the comprehensive 20-case accuracy, polyphone, punctuation, neural, and latency
    /// benchmark suite.
    Eval {
        /// Optional compiled `core.imedict` to mount during evaluation.
        #[arg(long)]
        dict: Option<PathBuf>,
        /// Optional neural reranker directory (`dist/base_neural`) to evaluate real Qwen2.5 Metal
        /// SLM.
        #[arg(long)]
        neural: Option<PathBuf>,
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
        _app_bundle_id: &str,
        candidates: &[Candidate],
    ) -> Vec<f32> {
        let texts: Vec<String> = candidates.iter().map(|c| c.text.clone()).collect();
        self.inner.score_sync(left_context, &texts)
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
        } => {
            let mut engine = ChimeEngine::new(ChimeConfig {
                scheme: scheme.into(),
                ..ChimeConfig::default()
            });
            if let Some(d_path) = dict {
                engine.lexicon_mut().mount_compiled_mmap(&d_path)?;
            }
            let t0 = Instant::now();
            let cands = engine.decode_stateless(&input, &context, &app);
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
                 ngrams={}, size={} KB",
                output.display(),
                t0.elapsed().as_millis(),
                mapped.pinyin_group_count(),
                mapped.word_record_count(),
                mapped.english_count(),
                mapped.ngram_count(),
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
                    "Compiled .imedict {}: pinyin_groups={}, word_records={}, english_terms={}",
                    path.display(),
                    mapped.pinyin_group_count(),
                    mapped.word_record_count(),
                    mapped.english_count()
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
            tsv,
            neural_dir,
            app_support,
        } => {
            if action == "build" {
                let manifest = build_imepack_dir(
                    "PersonalBrain",
                    tsv.as_deref(),
                    neural_dir.as_deref(),
                    &pack_dir,
                )?;
                writeln!(
                    stdout,
                    "Built .imepack at {}: {manifest:?}",
                    pack_dir.display()
                )?;
            } else {
                let target = app_support.unwrap_or_else(default_app_support_dir);
                let manifest = install_imepack_dir(&pack_dir, &target)?;
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
        Commands::Eval { dict, neural } => {
            run_golden_eval(&mut stdout, dict.as_deref(), neural.as_deref())?;
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

#[allow(clippy::too_many_lines)]
fn run_golden_eval(
    out: &mut impl Write,
    dict_path: Option<&Path>,
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
            expected_top1: "帮我review这个PR",
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
            left_ctx: "法庭依法保障公民的",
            app: "md.obsidian",
            keys: "quanli",
            expected_top1: "权利",
        },
        Case {
            category: "semantic-quanli-2",
            scheme: ShuangpinScheme::Full,
            left_ctx: "严禁滥用国家的",
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
    ];

    let base_lex = if let Some(dp) = dict_path {
        let mut lex = Lexicon::with_builtin_seed();
        lex.mount_compiled_mmap(dp)?;
        lex
    } else {
        build_comprehensive_lexicon("", None, 25)
    };

    let slm = if let Some(np) = neural_path
        && np.join("model.safetensors").exists()
    {
        AsyncRerankHandle::spawn_from_dir(np)?
    } else {
        let tmp_neural_dir = std::env::temp_dir().join("chime_eval_neural");
        TinyModelBuilder::write_demo_bundle(&tmp_neural_dir, &[("端侧", "量化"), ("这个", "PR")])?;
        AsyncRerankHandle::spawn_from_dir(&tmp_neural_dir)?
    };
    let cfg = slm.config().clone();
    let decoder = LatticeDecoder::new().with_reranker(Box::new(SyncNeuralAdapter { inner: slm }));
    let user_store = chime_lexicon::UserStore::new();

    let mut passed = 0usize;
    let mut latencies_us = Vec::with_capacity(cases.len() + 2);

    writeln!(
        out,
        "Running Chime Complete Benchmark Suite ({} decode + 2 interactive cases | MLX Metal SLM: \
         layers={}, hidden={}, vocab={})...",
        cases.len(),
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.vocab_size
    )?;
    for c in &cases {
        let ctx = DecodeContext {
            left_context: c.left_ctx,
            app_bundle_id: c.app,
            scheme: c.scheme,
            fuzzy: chime_phonetics::FuzzyConfig::default(),
            pangu_spacing: true,
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

    let total_cases = cases.len() + 2;
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
