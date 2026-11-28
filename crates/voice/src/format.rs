//! Final-only formatting. The model may propose punctuation, never replacement words.
use std::path::Path;

fn mark(c: char) -> bool {
    "，。？！、；：,.?!;:".contains(c)
}

/// Project insertions onto the original (including its spaces, casing and technical terms).
/// Reject the entire proposal if any non-space input character is lost or replaced.
fn overlay(raw: &str, suggested: &str) -> Option<String> {
    let mut original = raw.chars().peekable();
    let mut out = String::new();
    for c in suggested.chars().filter(|c| !c.is_whitespace()) {
        if mark(c) && original.peek() != Some(&c) {
            // No extra punctuation inside an ASCII token (e.g. a URL, time or version).
            if out.ends_with(|p: char| p.is_ascii_alphanumeric())
                && original.peek().is_some_and(char::is_ascii_alphanumeric)
            {
                return None;
            }
            out.push(c);
        } else {
            while original.peek().is_some_and(|c| c.is_whitespace()) {
                out.push(original.next()?);
            }
            let source = original.next()?;
            if !source.eq_ignore_ascii_case(&c) {
                return None;
            }
            out.push(source);
        }
    }
    for c in original {
        if !c.is_whitespace() {
            return None;
        }
        out.push(c);
    }
    Some(out)
}

fn boundary(c: char) -> bool {
    "，。！？；,!?;\n".contains(c)
}

fn tidy(raw: &str) -> String {
    let mut clauses: Vec<(&str, &str)> = Vec::new();
    for part in raw.split_inclusive(boundary) {
        let body = part.trim_end_matches(boundary);
        let suffix = &part[body.len()..];
        let mut body = body;
        // Only unambiguous clause-initial restarts. Preserve lexical reduplication (好好),
        // emphasis (非常非常), numbers, negations, and spoken self-corrections.
        for intro in ["我觉得", "我的意思是", "然后", "就是说", "我们先"] {
            while body
                .strip_prefix(intro)
                .is_some_and(|rest| rest.starts_with(intro))
            {
                body = &body[intro.len()..];
            }
        }
        // Also handle a whole repeated clause when punctuation restoration is off.
        if let Some((left, right)) = body.split_at_checked(body.len() / 2)
            && left == right
            && left.chars().count() >= 4
            && left.chars().all(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
        {
            body = left;
        }
        if matches!(body.trim(), "呃" | "嗯嗯" | "嗯") && matches!(suffix, "，" | ",") {
            continue;
        }
        if let Some((prev, sep)) = clauses.last_mut()
            && matches!(*sep, "，" | "," | "。")
            && body.chars().count() >= 4
            && body.trim() == prev.trim()
        {
            *sep = suffix;
        } else {
            clauses.push((body, suffix));
        }
    }
    let mut out = String::with_capacity(raw.len());
    for (body, sep) in clauses {
        out.push_str(body);
        out.push_str(sep);
    }
    if out.trim().is_empty() {
        raw.to_owned()
    } else {
        out
    }
}

#[cfg(unix)]
fn predict(contents: &Path, raw: &str) -> std::io::Result<String> {
    use std::{
        io::{Read, Write},
        net::Shutdown,
        os::{fd::OwnedFd, unix::net::UnixStream},
        process::{Command, Stdio},
        time::Duration,
    };
    // Use a local socketpair: bounded waits, no temporary files, no transcripts in argv,
    // and a dead helper cannot send SIGPIPE to the Swift host.
    let (mut parent, peer) = UnixStream::pair()?;
    parent.set_read_timeout(Some(Duration::from_secs(4)))?;
    parent.set_write_timeout(Some(Duration::from_secs(1)))?;
    let mut child = Command::new(contents.join("Helpers/ChimeVoice"))
        .arg("--punctuate")
        .current_dir(contents.join("Resources/voice"))
        .stdin(OwnedFd::from(peer.try_clone()?))
        .stdout(OwnedFd::from(peer))
        .stderr(Stdio::null())
        .spawn()?;
    let result = (|| {
        parent.write_all(raw.as_bytes())?;
        parent.shutdown(Shutdown::Write)?;
        let mut out = String::new();
        parent.take(131_073).read_to_string(&mut out)?;
        if out.len() > 131_072 {
            return Err(std::io::Error::other("oversized punctuation result"));
        }
        Ok(out)
    })();
    if result.is_err() {
        let _ = child.kill();
    }
    if !child.wait()?.success() {
        return Err(std::io::Error::other("punctuation helper failed"));
    }
    result
}

/// Format only this utterance, off the UI thread. Preserve supplied punctuation from HTTP
/// backends; on model failure, timeout or an unsafe suggestion keep the transcription.
#[must_use]
pub fn format_final(contents: &Path, raw: &str, punctuation: bool, organize: bool) -> String {
    #[cfg(unix)]
    let punctuated = (punctuation
        && !raw.trim().is_empty()
        && raw.len() <= 32768
        && !raw.contains(['，', '。', '？', '！', '\n'])
        && !raw.ends_with(['.', '!', '?']))
    .then(|| predict(contents, raw).ok().and_then(|s| overlay(raw, &s)))
    .flatten();
    #[cfg(not(unix))]
    let punctuated: Option<String> = {
        let _ = (contents, punctuation);
        None
    };
    let text = punctuated.as_deref().unwrap_or(raw);
    if organize {
        tidy(text)
    } else {
        text.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn insertions_only() {
        assert_eq!(
            overlay("我没有买 Rust 3.14", "我没有买rust 3.14。"),
            Some("我没有买 Rust 3.14。".into())
        );
        assert!(overlay("我没有买", "我们要买。").is_none());
        assert!(overlay("API", "A，PI。").is_none());
    }
    #[test]
    fn light_edit_not_paraphrase() {
        assert_eq!(
            tidy("嗯，我觉得我觉得可以。明天再说，明天再说。"),
            "我觉得可以。明天再说。"
        );
        assert_eq!(
            tidy("我没有好好珍惜，非常非常后悔。不是今天，是明天。"),
            "我没有好好珍惜，非常非常后悔。不是今天，是明天。"
        );
        assert_eq!(tidy("嗯，"), "嗯，");
        assert_eq!(tidy("明天早点到明天早点到"), "明天早点到");
        assert_eq!(tidy("12341234"), "12341234");
    }
    #[test]
    fn independent_switches_and_missing_model() {
        let missing = Path::new("/nonexistent-chime-app");
        let raw = "我觉得我觉得没有问题";
        assert_eq!(format_final(missing, raw, false, false), raw);
        assert_eq!(format_final(missing, raw, true, false), raw);
        assert_eq!(format_final(missing, raw, false, true), "我觉得没有问题");
        assert_eq!(format_final(missing, "", true, true), "");
    }
}
