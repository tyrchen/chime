//! Developer CLI for Chime IME: test decoding, inspect dictionaries, scrub corpora, and run
//! benchmarks.

#![forbid(unsafe_code)]

use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

use anyhow::{Context, Result};
use chime_core::{ChimeConfig, ChimeEngine, sanitize_corpus_text};
use chime_lexicon::{Lexicon, parse_rime_dict_yaml, parse_scel_bytes, parse_tsv_dict};
use chime_phonetics::ShuangpinScheme;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "chime",
    version,
    about = "Chime macOS IME developer & evaluation CLI"
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
        /// Simulated host App bundle ID (e.g. `com.microsoft.VSCode` or `com.tencent.xinWeChat`).
        #[arg(short, long, default_value = "md.obsidian")]
        app: String,
    },
    /// Inspect and validate a third-party dictionary (`.scel`, `.dict.yaml`, or `.tsv`).
    DictInspect {
        /// Path to the dictionary file.
        path: PathBuf,
    },
    /// Sanitize a personal text file locally before uploading to Google Colab for `LoRA` training.
    Sanitize {
        /// Input raw text file.
        input: PathBuf,
        /// Output sanitized text file.
        output: PathBuf,
    },
    /// Run the built-in golden accuracy & latency benchmark suite.
    Eval,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut stdout = io::stdout().lock();

    match cli.command {
        Commands::Decode {
            input,
            scheme,
            context,
            app,
        } => {
            let cfg = ChimeConfig {
                scheme: scheme.into(),
                ..ChimeConfig::default()
            };
            let engine = ChimeEngine::new(cfg);
            let t0 = Instant::now();
            let cands = engine.decode_stateless(&input, &context, &app);
            let elapsed_us = t0.elapsed().as_micros();

            writeln!(stdout, "Input: {input}  (decoded in {elapsed_us} us)")?;
            for (idx, cand) in cands.iter().take(9).enumerate() {
                writeln!(
                    stdout,
                    "  {}. {}  [score={:.2}, kind={:?}]",
                    idx + 1,
                    cand.text,
                    cand.score,
                    cand.kind
                )?;
            }
        }
        Commands::DictInspect { path } => {
            let mut lex = Lexicon::new();
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let stats = if ext == "scel" {
                let bytes = fs::read(&path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                parse_scel_bytes(&bytes, &mut lex)?
            } else {
                let text = fs::read_to_string(&path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                if path.to_string_lossy().ends_with(".dict.yaml") {
                    parse_rime_dict_yaml(&text, &mut lex)
                } else {
                    parse_tsv_dict(&text, &mut lex)
                }
            };
            writeln!(
                stdout,
                "Imported {}: total_rows={}, imported={}, skipped={}, distinct_pinyin_keys={}",
                path.display(),
                stats.total_rows,
                stats.imported_entries,
                stats.skipped_rows,
                lex.pinyin_key_count()
            )?;
        }
        Commands::Sanitize { input, output } => {
            let raw = fs::read_to_string(&input)
                .with_context(|| format!("failed to read {}", input.display()))?;
            let (clean_lines, report) = sanitize_corpus_text(&raw);
            fs::write(&output, clean_lines.join("\n"))
                .with_context(|| format!("failed to write {}", output.display()))?;
            writeln!(
                stdout,
                "Sanitized {} -> {}: total_lines={}, kept_lines={}, redacted_items={}",
                input.display(),
                output.display(),
                report.total_lines,
                report.kept_lines,
                report.redacted_items
            )?;
        }
        Commands::Eval => run_golden_eval(&mut stdout)?,
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn run_golden_eval(out: &mut impl Write) -> Result<()> {
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
            category: "super-abbr",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "md.obsidian",
            keys: "wsm",
            expected_top1: "为什么",
        },
        Case {
            category: "fuzzy-pinyin",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.tencent.xinWeChat",
            keys: "zidao",
            expected_top1: "知道",
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
            category: "code-switch-dev",
            scheme: ShuangpinScheme::Full,
            left_ctx: "",
            app: "com.microsoft.VSCode",
            keys: "useEffect",
            expected_top1: "useEffect",
        },
        Case {
            category: "xiaohe-shuangpin",
            scheme: ShuangpinScheme::Xiaohe,
            left_ctx: "",
            app: "md.obsidian",
            keys: "ulpb",
            expected_top1: "双拼",
        },
    ];

    let full_engine = ChimeEngine::new(ChimeConfig::default());
    let xiaohe_engine = ChimeEngine::new(ChimeConfig {
        scheme: ShuangpinScheme::Xiaohe,
        ..ChimeConfig::default()
    });

    let mut passed = 0usize;
    let mut latencies_us = Vec::with_capacity(cases.len());

    writeln!(
        out,
        "Running Chime Golden Benchmark Suite ({} cases)...",
        cases.len()
    )?;
    for c in &cases {
        let eng = if c.scheme == ShuangpinScheme::Xiaohe {
            &xiaohe_engine
        } else {
            &full_engine
        };
        let t0 = Instant::now();
        let cands = eng.decode_stateless(c.keys, c.left_ctx, c.app);
        let us = u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX);
        latencies_us.push(us);

        let got = cands.first().map_or("<empty>", |cand| cand.text.as_str());
        let ok = got == c.expected_top1;
        if ok {
            passed += 1;
        }
        writeln!(
            out,
            "  [{}] {:<18} {:<24} -> {:<22} ({} us)",
            if ok { "PASS" } else { "FAIL" },
            c.category,
            c.keys,
            got,
            us
        )?;
    }

    latencies_us.sort_unstable();
    let p50 = latencies_us[latencies_us.len() / 2];
    let p99 = latencies_us[(latencies_us.len() * 99) / 100];
    writeln!(
        out,
        "Summary: {passed}/{} Top-1 passed | Latency p50={} us, p99={} us (budget: < 2000 us)",
        cases.len(),
        p50,
        p99
    )?;
    Ok(())
}
