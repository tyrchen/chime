//! `MacBook` QWERTY keyboard spatial proximity and transposition typo model.

use smallvec::SmallVec;

use crate::syllables::{SyllableId, lookup_syllable};

/// Kind of keystroke typo edit applied to recover a valid syllable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TypoEdit {
    /// Two adjacent keystrokes arrived out of order (`shaung` -> `shuang`).
    AdjacentSwap {
        /// Penalty cost for transposition.
        cost: f32,
    },
    /// One key was substituted by a physically adjacent QWERTY neighbor (`nohao` -> `nihao`).
    NeighborKey {
        /// Spatial penalty cost based on Euclidean key distance.
        cost: f32,
    },
}

impl TypoEdit {
    /// Returns the numeric cost penalty associated with this typo edit.
    #[must_use]
    pub const fn cost(self) -> f32 {
        match self {
            Self::AdjacentSwap { cost } | Self::NeighborKey { cost } => cost,
        }
    }
}

/// Physical key-cap center coordinates `(row_x, row_y)` on an ANSI `MacBook` QWERTY layout.
#[must_use]
const fn key_coords(ch: u8) -> Option<(f32, f32)> {
    match ch {
        b'q' => Some((0.0, 0.0)),
        b'w' => Some((1.0, 0.0)),
        b'e' => Some((2.0, 0.0)),
        b'r' => Some((3.0, 0.0)),
        b't' => Some((4.0, 0.0)),
        b'y' => Some((5.0, 0.0)),
        b'u' => Some((6.0, 0.0)),
        b'i' => Some((7.0, 0.0)),
        b'o' => Some((8.0, 0.0)),
        b'p' => Some((9.0, 0.0)),
        b'a' => Some((0.25, 1.0)),
        b's' => Some((1.25, 1.0)),
        b'd' => Some((2.25, 1.0)),
        b'f' => Some((3.25, 1.0)),
        b'g' => Some((4.25, 1.0)),
        b'h' => Some((5.25, 1.0)),
        b'j' => Some((6.25, 1.0)),
        b'k' => Some((7.25, 1.0)),
        b'l' => Some((8.25, 1.0)),
        b'z' => Some((0.75, 2.0)),
        b'x' => Some((1.75, 2.0)),
        b'c' => Some((2.75, 2.0)),
        b'v' => Some((3.75, 2.0)),
        b'b' => Some((4.75, 2.0)),
        b'n' => Some((5.75, 2.0)),
        b'm' => Some((6.75, 2.0)),
        _ => None,
    }
}

/// Computes the substitution cost between two lowercase QWERTY keys if physically adjacent.
#[must_use]
pub fn keyboard_substitution_cost(typed: u8, intended: u8) -> Option<f32> {
    if typed == intended {
        return Some(0.0);
    }
    let (x1, y1) = key_coords(typed)?;
    let (x2, y2) = key_coords(intended)?;
    let dist_sq = (x1 - x2).mul_add(x1 - x2, (y1 - y2) * (y1 - y2));
    // Restrict to immediate physical neighbors (distance <= ~1.35 key pitches).
    if dist_sq <= 1.85 {
        Some(0.85 + 0.45 * dist_sq)
    } else {
        None
    }
}

/// Generates plausible 1-edit typo corrections (transposition or neighbor key) for a token slice.
#[must_use]
pub fn generate_typo_variants(token: &str) -> SmallVec<[(SyllableId, TypoEdit); 6]> {
    let mut out = SmallVec::new();
    let bytes = token.as_bytes();
    if bytes.len() < 2 || bytes.len() > 6 || !bytes.iter().all(u8::is_ascii_lowercase) {
        return out;
    }

    // 1. Adjacent character transposition (`shaung` -> `shuang`, `nhiao` -> `nihao`).
    let mut buf = bytes.to_vec();
    for i in 0..(bytes.len() - 1) {
        if buf[i] == buf[i + 1] {
            continue;
        }
        buf.swap(i, i + 1);
        if let Ok(candidate) = std::str::from_utf8(&buf)
            && let Some(id) = lookup_syllable(candidate)
            && !out.iter().any(|(existing, _)| *existing == id)
        {
            out.push((id, TypoEdit::AdjacentSwap { cost: 1.05 }));
        }
        buf.swap(i, i + 1);
    }

    // 2. Single-key physical QWERTY neighbor substitution (`no` -> `ni`, `hso` -> `hao`).
    for i in 0..bytes.len() {
        let orig = buf[i];
        for cand_key in b'a'..=b'z' {
            if cand_key == orig {
                continue;
            }
            if let Some(cost) = keyboard_substitution_cost(orig, cand_key) {
                buf[i] = cand_key;
                if let Ok(candidate) = std::str::from_utf8(&buf)
                    && let Some(id) = lookup_syllable(candidate)
                    && !out.iter().any(|(existing, _)| *existing == id)
                {
                    out.push((id, TypoEdit::NeighborKey { cost }));
                }
                buf[i] = orig;
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syllables::canonical_syllable;

    #[test]
    fn test_should_recover_transposition_and_neighbor_typos() {
        let swap_hits = generate_typo_variants("shaung");
        assert!(
            swap_hits
                .iter()
                .any(|(id, _)| canonical_syllable(*id) == "shuang")
        );

        let neighbor_hits = generate_typo_variants("no");
        assert!(
            neighbor_hits
                .iter()
                .any(|(id, _)| canonical_syllable(*id) == "ni")
        );
    }
}
