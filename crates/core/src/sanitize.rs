//! High-recall local privacy sanitizer for preparing personal Colab `LoRA` training corpora.

use serde::{Deserialize, Serialize};

/// Summary of redacted sensitive items before exporting a personal training corpus.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SanitizeReport {
    /// Total input lines inspected.
    pub total_lines: usize,
    /// Clean lines retained for training.
    pub kept_lines: usize,
    /// Lines dropped or redacted due to secret tokens, credentials, or PII.
    pub redacted_items: usize,
}

/// Scrubs API keys, tokens, emails, phone numbers, and high-entropy secrets from `raw_text`.
#[must_use]
pub fn sanitize_corpus_text(raw_text: &str) -> (Vec<String>, SanitizeReport) {
    let mut kept = Vec::new();
    let mut report = SanitizeReport::default();

    for line in raw_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        report.total_lines += 1;

        if contains_secret_marker(trimmed) {
            report.redacted_items += 1;
            continue;
        }

        let mut cleaned_tokens = Vec::new();
        for token in trimmed.split_whitespace() {
            if looks_like_pii_or_high_entropy_secret(token) {
                report.redacted_items += 1;
                cleaned_tokens.push("<REDACTED>");
            } else {
                cleaned_tokens.push(token);
            }
        }
        let rebuilt = cleaned_tokens.join(" ");
        if !rebuilt.is_empty() && rebuilt != "<REDACTED>" {
            kept.push(rebuilt);
            report.kept_lines += 1;
        }
    }

    (kept, report)
}

fn contains_secret_marker(line: &str) -> bool {
    let markers = [
        "-----BEGIN ",
        "PRIVATE KEY",
        "sk-",
        "ghp_",
        "gho_",
        "github_pat_",
        "AKIA",
        "ASIA",
        "xoxb-",
        "xoxp-",
        "eyJhbGci",
        "password=",
        "passwd=",
        "secret=",
        "api_key=",
    ];
    markers.iter().any(|m| line.contains(m))
}

fn looks_like_pii_or_high_entropy_secret(token: &str) -> bool {
    let clean = token.trim_matches(|c: char| matches!(c, ',' | ';' | '"' | '\'' | '(' | ')'));
    // Email address check
    if clean.contains('@') && clean.contains('.') && clean.len() >= 5 {
        return true;
    }
    // Phone / national ID / credit card digit runs (>= 11 digits)
    let digit_count = clean.chars().filter(char::is_ascii_digit).count();
    if digit_count >= 11 {
        return true;
    }
    // Long mixed-case/digit ASCII secret token (>= 28 chars with digits + upper + lower)
    if clean.len() >= 28
        && clean.is_ascii()
        && clean.bytes().any(|b| b.is_ascii_digit())
        && clean.bytes().any(|b| b.is_ascii_uppercase())
        && clean.bytes().any(|b| b.is_ascii_lowercase())
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_strip_secrets_and_pii_before_colab_export() {
        let sample = "\
帮我 review 一下这个 PR 的 latency
export OPENAI_API_KEY=sk-proj-1234567890abcdef
我的邮箱是 user@example.com 手机号 13800138000
今天使用 macOS 测试双拼输入法
";
        let (lines, report) = sanitize_corpus_text(sample);
        assert_eq!(report.total_lines, 4);
        assert!(report.redacted_items >= 3);
        assert_eq!(lines.len(), 3);
        assert!(!lines.iter().any(|l| l.contains("sk-proj")));
        assert!(!lines.iter().any(|l| l.contains("user@example.com")));
        assert!(!lines.iter().any(|l| l.contains("13800138000")));
    }
}
