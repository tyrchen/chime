//! Optional on-device `SenseVoice` final for bounded utterances. Never used on HTTP sessions.
use std::{
    path::{Path, PathBuf},
    process::Child,
    sync::{Mutex, MutexGuard, PoisonError},
};

use serde::Deserialize;

const RATE: usize = 16_000;
#[derive(Debug)]
pub(crate) struct SecondPass {
    contents: PathBuf,
    model: PathBuf,
    samples: Vec<f32>,
    punctuation: bool,
    layout: bool,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl SecondPass {
    /// `None` unless the directory holds `SenseVoice` weights. `punctuation` keeps the model's
    /// own marks; `layout` adds paragraphs and digit spacing.
    pub(crate) fn new(
        contents: &Path,
        model: &Path,
        punctuation: bool,
        layout: bool,
    ) -> Option<Self> {
        (model.join("model.int8.onnx").is_file() && model.join("tokens.txt").is_file()).then(|| {
            Self {
                contents: contents.into(),
                model: model.into(),
                samples: vec![],
                punctuation,
                layout,
            }
        })
    }
    pub(crate) fn append(&mut self, frame: &[f32]) -> bool {
        if self.samples.len() + frame.len() > RATE * 45 {
            return false;
        }
        self.samples.extend_from_slice(frame);
        true
    }
    /// Re-recognize the whole utterance in a separate helper. `slot` holds the process the
    /// session's `Drop` kills on cancel: the finished streaming helper is reaped first so both
    /// models are never resident together, then the second-pass process takes its place.
    #[cfg(unix)]
    pub(crate) fn finish(
        self,
        original: &str,
        slot: &Mutex<Child>,
        cancelled: impl Fn() -> bool,
    ) -> Option<String> {
        use std::{
            io::{Read, Write},
            net::Shutdown,
            os::{fd::OwnedFd, unix::net::UnixStream},
            process::{Command, Stdio},
            time::Duration,
        };
        if self.samples.len() < RATE * 4 || cancelled() {
            return None;
        }
        let (mut parent, peer) = UnixStream::pair().ok()?;
        parent.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
        parent
            .set_write_timeout(Some(Duration::from_secs(2)))
            .ok()?;
        {
            let mut proc = lock(slot);
            let _ = proc.kill();
            let _ = proc.wait();
            *proc = Command::new(self.contents.join("Helpers/ChimeVoice"))
                .arg("--sense-final")
                .current_dir(&self.model)
                .stdin(OwnedFd::from(peer.try_clone().ok()?))
                .stdout(OwnedFd::from(peer))
                .stderr(Stdio::null())
                .spawn()
                .ok()?;
        }
        // `Drop` may have run between the check above and the swap.
        if cancelled() {
            let _ = lock(slot).kill();
            return None;
        }
        let reply = (|| -> Option<Decoded> {
            parent
                .write_all(&u32::try_from(self.samples.len()).ok()?.to_le_bytes())
                .ok()?;
            let pcm: Vec<u8> = self.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
            parent.write_all(&pcm).ok()?;
            parent.shutdown(Shutdown::Write).ok()?;
            let mut bytes = vec![];
            parent.take(262_145).read_to_end(&mut bytes).ok()?;
            if bytes.len() > 262_144 {
                return None;
            }
            serde_json::from_slice(&bytes).ok()
        })();
        let mut proc = lock(slot);
        if reply.is_none() {
            let _ = proc.kill();
        }
        if !proc.wait().ok()?.success() {
            return None;
        }
        drop(proc);
        reply?.render(original, self.punctuation, self.layout)
    }
    #[cfg(not(unix))]
    pub(crate) fn finish(self, _: &str, _: &Mutex<Child>, _: impl Fn() -> bool) -> Option<String> {
        None
    }
}
#[derive(Deserialize)]
struct Decoded {
    text: String,
    #[serde(default)]
    tokens: Vec<String>,
    #[serde(default)]
    timestamps: Vec<f32>,
}
fn han(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}
fn end_mark(t: &str) -> bool {
    matches!(t, "。" | "！" | "？")
}
// Recognize a repeated label immediately around two quantities (e.g. 库存80库存120).
// Only insert a comma if the decoder also observed a pause; never guess a label or number.
fn paused_parallel_quantity(before: &str, after: &[String]) -> bool {
    let stem = before.trim_end_matches(|c: char| c.is_ascii_digit());
    if stem.len() == before.len() {
        return false;
    }
    let next: String = after
        .iter()
        .take(6)
        .flat_map(|s| s.chars())
        .take(5)
        .collect();
    let Some(at) = next.find(|c: char| c.is_ascii_digit()) else {
        return false;
    };
    let label = &next[..at];
    (2..=4).contains(&label.chars().count()) && label.chars().all(han) && stem.ends_with(label)
}
impl Decoded {
    fn render(&self, original: &str, punctuation: bool, layout: bool) -> Option<String> {
        let mut out = self.render_marked(original, layout)?;
        if !punctuation {
            out.retain(|c| !"，。！？、；：".contains(c));
        }
        Some(out)
    }

    fn render_marked(&self, original: &str, layout: bool) -> Option<String> {
        let length = |s: &str| s.chars().filter(|c| c.is_alphanumeric()).count();
        let src = length(original);
        let n = length(&self.text);
        // Reject empty or grossly truncated/expanded offline hypotheses. This is a sanity
        // check, not a semantic correctness guarantee.
        if n == 0 || src == 0 || n * 2 < src || src * 2 < n {
            return None;
        }
        let mut out = String::new();
        if self.tokens.len() == self.timestamps.len() && self.tokens.concat() == self.text {
            let mut paragraph_start = 0;
            let mut sentence_start = 0;
            for (i, t) in self.tokens.iter().enumerate() {
                if i > 0 && self.tokens[i - 1] == "。" && t == "。" {
                    continue;
                }
                if i >= 2 && end_mark(&self.tokens[i - 1]) && !end_mark(t) {
                    // Only break at an actual predicted sentence end AND a spoken pause. Keep
                    // adjacent parallel sentences (same opening two tokens) in one paragraph.
                    let parallel = self.tokens.get(i..i + 2).is_some_and(|head| {
                        self.tokens.get(sentence_start..sentence_start + 2) == Some(head)
                    });
                    if layout
                        && !parallel
                        && out[paragraph_start..].chars().count() >= 24
                        && self.timestamps[i] - self.timestamps[i - 2] >= 0.50
                    {
                        out.push_str("\n\n");
                        paragraph_start = out.len();
                    }
                    sentence_start = i;
                }
                if layout
                    && i > 0
                    && self.timestamps[i] - self.timestamps[i - 1] >= 0.50
                    && paused_parallel_quantity(&out, &self.tokens[i..])
                {
                    out.push('，');
                }
                out.push_str(t);
            }
        } else {
            out.push_str(self.text.trim());
        }
        if !layout {
            return Some(out);
        }
        let mut spaced = String::new();
        let mut prev = ' ';
        for ch in out.chars() {
            if (han(prev) && ch.is_ascii_digit()) || (prev.is_ascii_digit() && han(ch)) {
                spaced.push(' ');
            }
            spaced.push(ch);
            prev = ch;
        }
        Some(spaced)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_guessing_from_missing_or_misaligned_timing() {
        let mut d = Decoded {
            text: "买12支笔。".into(),
            tokens: vec!["错".into()],
            timestamps: vec![1.],
        };
        assert_eq!(
            d.render("买十二支笔", true, true).as_deref(),
            Some("买 12 支笔。")
        );
        d.text = "买".into();
        assert!(d.render("买十二支笔", true, true).is_none());
    }
    #[test]
    fn repeated_quantity_needs_acoustic_pause_and_same_label() {
        let raw = "库存80库存120件。";
        let tokens: Vec<_> = raw.chars().map(|c| c.to_string()).collect();
        let mut d = Decoded {
            text: raw.into(),
            timestamps: vec![0.; tokens.len()],
            tokens,
        };
        assert_eq!(
            d.render(raw, true, true).as_deref(),
            Some("库存 80 库存 120 件。")
        );
        d.timestamps[4..].fill(0.7);
        assert_eq!(
            d.render(raw, true, true).as_deref(),
            Some("库存 80，库存 120 件。")
        );
        assert_eq!(d.render(raw, true, false).as_deref(), Some(raw));
        assert!(!paused_parallel_quantity(
            "库存80",
            &["需求".into(), "120".into()]
        ));
    }
    #[test]
    fn break_only_after_long_sentence_and_pause() {
        let first = "这是一段用来验证长语音分段效果的测试内容不涉及个人信息";
        let text = format!("{first}。接下来再说另一件事。。");
        let tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let mut timestamps = vec![0.; tokens.len()];
        timestamps[first.chars().count() + 1..].fill(1.);
        let d = Decoded {
            text: text.clone(),
            tokens,
            timestamps,
        };
        assert_eq!(
            d.render(&text, true, true),
            Some(format!("{first}。\n\n接下来再说另一件事。"))
        );
    }
}
