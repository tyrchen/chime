//! Stateful Chinese full-width smart punctuation and paired quotation mapper.

/// Per-session quotation parity state for alternating `“”` and `‘’`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PunctuationState {
    double_quote_open: bool,
    single_quote_open: bool,
}

impl PunctuationState {
    /// Maps an ASCII punctuation byte into Chinese full-width punctuation when appropriate.
    /// Exempts `.` immediately following an ASCII digit (`3.14` stays `3.14`!).
    pub fn map_punctuation(&mut self, ascii: u8, prev_char: Option<char>) -> Option<&'static str> {
        match ascii {
            b',' => Some("，"),
            b'.' => {
                if prev_char.is_some_and(|c| c.is_ascii_digit()) {
                    Some(".")
                } else {
                    Some("。")
                }
            }
            b'?' => Some("？"),
            b'!' => Some("！"),
            b':' => {
                if prev_char.is_some_and(|c| c.is_ascii_digit()) {
                    Some(":")
                } else {
                    Some("：")
                }
            }
            b';' => Some("；"),
            b'\\' => Some("、"),
            b'(' => Some("（"),
            b')' => Some("）"),
            b'[' => Some("【"),
            b']' => Some("】"),
            b'<' => Some("《"),
            b'>' => Some("》"),
            b'^' => Some("……"),
            b'_' => Some("——"),
            b'$' => Some("￥"),
            b'"' => {
                self.double_quote_open = !self.double_quote_open;
                if self.double_quote_open {
                    Some("“")
                } else {
                    Some("”")
                }
            }
            b'\'' => {
                self.single_quote_open = !self.single_quote_open;
                if self.single_quote_open {
                    Some("‘")
                } else {
                    Some("’")
                }
            }
            _ => None,
        }
    }

    /// Returns whether `ascii` triggers auto-commit of the active preedit candidate before emitting
    /// punctuation.
    #[must_use]
    pub const fn triggers_preedit_commit(ascii: u8) -> bool {
        matches!(
            ascii,
            b',' | b'?'
                | b'!'
                | b':'
                | b';'
                | b'\\'
                | b'('
                | b')'
                | b'['
                | b']'
                | b'<'
                | b'>'
                | b'"'
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_alternate_smart_quotes_and_exempt_decimal_point() {
        let mut st = PunctuationState::default();
        assert_eq!(st.map_punctuation(b'"', None), Some("“"));
        assert_eq!(st.map_punctuation(b'"', None), Some("”"));
        assert_eq!(st.map_punctuation(b'.', Some('3')), Some("."));
        assert_eq!(st.map_punctuation(b'.', Some('好')), Some("。"));
    }
}
