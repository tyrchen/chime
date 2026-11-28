//! Transcript clean-up and joining helpers.

/// Returns `true` for CJK ideographs and full-width punctuation where no space is wanted.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3000..=0x303F   // CJK symbols & punctuation
        | 0x3400..=0x4DBF // Extension A
        | 0x4E00..=0x9FFF // Unified ideographs
        | 0xF900..=0xFAFF // Compatibility ideographs
        | 0xFF00..=0xFFEF // Full-width forms
        | 0x20000..=0x2FFFF)
}

/// Joins consecutive utterance transcripts, inserting a space only between two Latin/number runs.
#[must_use]
pub fn join_segments<'a, I>(parts: I) -> String
where
    I: IntoIterator<Item = &'a str>,
{
    let mut out = String::new();
    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let (Some(prev), Some(next)) = (out.chars().last(), part.chars().next())
            && !is_cjk(prev)
            && !is_cjk(next)
            && !prev.is_whitespace()
        {
            out.push(' ');
        }
        out.push_str(part);
    }
    out
}

/// Cleans one raw server transcript.
///
/// * removes `SenseVoice`-style control tags such as `<|zh|><|NEUTRAL|><|Speech|>`;
/// * drops a transcript that merely echoes the decoding `prompt` or is a known silence
///   hallucination of Whisper-family models;
/// * when `chinese_punctuation` is set, converts ASCII `,.?!:;` that directly follow a Chinese
///   character into full-width punctuation (keeping `3.14`, `12:30`, URLs and English intact).
#[must_use]
pub fn normalize_transcript(raw: &str, prompt: &str, chinese_punctuation: bool) -> String {
    let mut stripped = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(start) = rest.find("<|") {
        stripped.push_str(&rest[..start]);
        match rest[start..].find("|>") {
            Some(end) => rest = &rest[start + end + 2..],
            None => {
                rest = "";
            }
        }
    }
    stripped.push_str(rest);
    let text = stripped.trim();

    if text.is_empty() || is_hallucination(text, prompt) {
        return String::new();
    }
    if !chinese_punctuation {
        return text.to_string();
    }

    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let prev_cjk = i > 0 && is_cjk(chars[i - 1]);
        let full = match c {
            ',' => Some('，'),
            '.' => Some('。'),
            '?' => Some('？'),
            '!' => Some('！'),
            ':' => Some('：'),
            ';' => Some('；'),
            _ => None,
        };
        if let (true, Some(f)) = (prev_cjk, full) {
            out.push(f);
            // Drop the space Whisper often emits after ASCII punctuation.
            while i + 1 < chars.len() && chars[i + 1] == ' ' {
                i += 1;
            }
        } else {
            out.push(c);
        }
        i += 1;
    }
    out
}

/// Subtitle credits Whisper-family models emit on silence or noise.
const HALLUCINATIONS: [&str; 4] = ["Amara.org", "明镜与点点", "请不吝点赞", "字幕由"];

fn is_hallucination(text: &str, prompt: &str) -> bool {
    let core: String = text
        .chars()
        .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation() && !is_cjk_punct(*c))
        .collect();
    if core.is_empty() {
        return true;
    }
    let prompt_core: String = prompt
        .chars()
        .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation() && !is_cjk_punct(*c))
        .collect();
    if !prompt_core.is_empty() && core == prompt_core {
        return true;
    }
    HALLUCINATIONS.iter().any(|k| text.contains(k))
}

fn is_cjk_punct(c: char) -> bool {
    matches!(c as u32, 0x3000..=0x303F | 0xFF01..=0xFF0F | 0xFF1A..=0xFF20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_without_spaces_between_chinese() {
        assert_eq!(
            join_segments(["今天天气不错，", "我们去公园吧。"]),
            "今天天气不错，我们去公园吧。"
        );
        assert_eq!(join_segments(["打开", "GitHub"]), "打开GitHub");
        assert_eq!(join_segments(["hello", "world", ""]), "hello world");
        assert_eq!(join_segments([" 用 Rust ", " 写代码 "]), "用 Rust写代码");
    }

    #[test]
    fn strips_sensevoice_tags() {
        assert_eq!(
            normalize_transcript("<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好世界", "", true),
            "你好世界"
        );
        assert_eq!(normalize_transcript("<|zh|>", "", true), "");
        assert_eq!(normalize_transcript("abc<|unterminated", "", true), "abc");
    }

    #[test]
    fn converts_punctuation_only_after_chinese() {
        assert_eq!(
            normalize_transcript("你好, 世界. 圆周率是3.14, 时间12:30!", "", true),
            "你好，世界。圆周率是3.14, 时间12:30!"
        );
        assert_eq!(
            normalize_transcript("Hello, world.", "", true),
            "Hello, world."
        );
        assert_eq!(
            normalize_transcript("你好, 世界.", "", false),
            "你好, 世界."
        );
    }

    #[test]
    fn drops_prompt_echo_and_known_hallucinations() {
        let prompt = "以下是普通话的句子，使用简体中文。";
        assert_eq!(
            normalize_transcript("以下是普通话的句子,使用简体中文", prompt, true),
            ""
        );
        assert_eq!(
            normalize_transcript("字幕由Amara.org社区提供", "", true),
            ""
        );
        assert_eq!(normalize_transcript("。。。", "", true), "");
        assert_eq!(
            normalize_transcript("正常的一句话。", prompt, true),
            "正常的一句话。"
        );
    }
}
