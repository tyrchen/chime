//! Energy-based voice activity detection that cuts a live PCM stream into utterances.
//!
//! The detector tracks an adaptive noise floor and marks a 30 ms frame as voiced when its RMS is
//! clearly above both the floor and an absolute threshold. An utterance closes after a pause of
//! `end_silence_ms`, or unconditionally at `max_segment_ms` (Whisper-style models decode 30 s
//! windows). Leading silence is trimmed so silent audio is never sent to the server, which also
//! avoids the well-known Whisper hallucinations on silence.

/// Tunables for [`Segmenter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadParams {
    /// Analysis frame length.
    pub frame_ms: u32,
    /// Minimum voiced audio before a segment counts as speech.
    pub min_speech_ms: u32,
    /// Pause length that closes an utterance.
    pub end_silence_ms: u32,
    /// Utterances shorter than this are not cut at pauses (keeps context for the recognizer).
    pub min_segment_ms: u32,
    /// Hard cut length.
    pub max_segment_ms: u32,
    /// Silence kept in front of speech onset.
    pub lead_in_ms: u32,
    /// Silence kept after the last voiced frame of a closed utterance.
    pub tail_ms: u32,
    /// Absolute RMS below which a frame is never voiced.
    pub abs_threshold: f32,
    /// A frame is voiced when its RMS exceeds `noise_floor * noise_ratio`.
    pub noise_ratio: f32,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            frame_ms: 30,
            min_speech_ms: 240,
            end_silence_ms: 650,
            min_segment_ms: 1_200,
            max_segment_ms: 24_000,
            lead_in_ms: 300,
            tail_ms: 250,
            abs_threshold: 0.006,
            noise_ratio: 2.5,
        }
    }
}

/// Streaming utterance segmenter.
#[derive(Debug, Clone)]
pub struct Segmenter {
    params: VadParams,
    frame_len: usize,
    min_speech: usize,
    end_silence: usize,
    min_segment: usize,
    max_segment: usize,
    lead_in: usize,
    tail: usize,
    pending_frame: Vec<f32>,
    open: Vec<f32>,
    voiced: usize,
    trailing_silence: usize,
    noise_floor: f32,
    last_rms: f32,
}

fn ms_to_samples(ms: u32, sample_rate: u32) -> usize {
    usize::try_from(u64::from(ms) * u64::from(sample_rate) / 1000).unwrap_or(usize::MAX)
}

impl Segmenter {
    /// Creates a segmenter for `sample_rate` Hz mono audio.
    #[must_use]
    pub fn new(sample_rate: u32, params: VadParams) -> Self {
        let sr = sample_rate.max(1);
        Self {
            params,
            frame_len: ms_to_samples(params.frame_ms, sr).max(1),
            min_speech: ms_to_samples(params.min_speech_ms, sr),
            end_silence: ms_to_samples(params.end_silence_ms, sr),
            min_segment: ms_to_samples(params.min_segment_ms, sr),
            max_segment: ms_to_samples(params.max_segment_ms, sr).max(1),
            lead_in: ms_to_samples(params.lead_in_ms, sr),
            tail: ms_to_samples(params.tail_ms, sr),
            pending_frame: Vec::new(),
            open: Vec::new(),
            voiced: 0,
            trailing_silence: 0,
            noise_floor: 0.002,
            last_rms: 0.0,
        }
    }

    /// Feeds samples and returns every utterance closed by them, in order.
    pub fn push(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut closed = Vec::new();
        let mut rest = samples;
        while !rest.is_empty() {
            let need = self.frame_len - self.pending_frame.len();
            let take = need.min(rest.len());
            self.pending_frame.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.pending_frame.len() == self.frame_len {
                let frame = std::mem::take(&mut self.pending_frame);
                if let Some(seg) = self.process_frame(&frame) {
                    closed.push(seg);
                }
                self.pending_frame = frame;
                self.pending_frame.clear();
            }
        }
        closed
    }

    /// Audio of the utterance currently being spoken (may be only lead-in silence).
    #[must_use]
    pub fn open_audio(&self) -> &[f32] {
        &self.open
    }

    /// Whether the open utterance already contains enough voiced audio to be worth transcribing.
    #[must_use]
    pub const fn open_has_speech(&self) -> bool {
        self.voiced >= self.min_speech && self.voiced > 0
    }

    /// Flushes the stream and returns the final open utterance when it contains speech.
    pub fn finish(&mut self) -> Option<Vec<f32>> {
        let pending = std::mem::take(&mut self.pending_frame);
        self.open.extend_from_slice(&pending);
        let has_speech = self.open_has_speech();
        let mut seg = std::mem::take(&mut self.open);
        self.voiced = 0;
        if has_speech {
            let keep = seg.len()
                - self
                    .trailing_silence
                    .saturating_sub(self.tail)
                    .min(seg.len());
            seg.truncate(keep);
            self.trailing_silence = 0;
            Some(seg)
        } else {
            self.trailing_silence = 0;
            None
        }
    }

    /// Loudness of the latest frame mapped to `0.0..=1.0` (for the level meter).
    #[must_use]
    pub fn level(&self) -> f32 {
        if self.last_rms <= 1e-6 {
            return 0.0;
        }
        let db = 20.0 * self.last_rms.log10();
        ((db + 55.0) / 45.0).clamp(0.0, 1.0)
    }

    fn process_frame(&mut self, frame: &[f32]) -> Option<Vec<f32>> {
        #[allow(clippy::cast_precision_loss)]
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        self.last_rms = rms;

        let threshold = self
            .params
            .abs_threshold
            .max(self.noise_floor * self.params.noise_ratio);
        let voiced = rms > threshold;
        // Fast-falling, slow-rising floor follows background noise without absorbing speech.
        if rms < self.noise_floor {
            self.noise_floor = self.noise_floor * 0.9 + rms * 0.1;
        } else if !voiced {
            self.noise_floor += (rms - self.noise_floor) * 0.05;
        }
        self.noise_floor = self.noise_floor.clamp(1e-4, 0.05);

        self.open.extend_from_slice(frame);
        if voiced {
            self.voiced += frame.len();
            self.trailing_silence = 0;
        } else {
            self.trailing_silence += frame.len();
        }

        if !self.open_has_speech() {
            if self.voiced > 0 && self.trailing_silence >= self.end_silence {
                // A short click or bump: forget it.
                self.voiced = 0;
            }
            if self.voiced == 0 && self.open.len() > self.lead_in {
                let excess = self.open.len() - self.lead_in;
                self.open.drain(..excess);
            }
            // Unvoiced-but-noisy audio must not grow without bound either.
            if self.open.len() >= self.max_segment {
                let excess = self.open.len() - self.lead_in;
                self.open.drain(..excess);
                self.voiced = 0;
            }
            return None;
        }

        if self.trailing_silence >= self.end_silence && self.open.len() >= self.min_segment {
            let keep =
                self.open.len() - self.trailing_silence + self.tail.min(self.trailing_silence);
            let carry_from = self.open.len().saturating_sub(self.lead_in).max(keep);
            let carry = self.open[carry_from..].to_vec();
            let mut seg = std::mem::replace(&mut self.open, carry);
            seg.truncate(keep);
            self.voiced = 0;
            self.trailing_silence = 0;
            return Some(seg);
        }

        if self.open.len() >= self.max_segment {
            let seg = std::mem::take(&mut self.open);
            self.voiced = 0;
            self.trailing_silence = 0;
            return Some(seg);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;

    fn tone(ms: u32) -> Vec<f32> {
        let n = ms_to_samples(ms, SR);
        #[allow(clippy::cast_precision_loss)]
        (0..n).map(|i| 0.2 * (i as f32 * 0.07).sin()).collect()
    }

    fn hush(ms: u32) -> Vec<f32> {
        let n = ms_to_samples(ms, SR);
        #[allow(clippy::cast_precision_loss)]
        (0..n).map(|i| 0.0008 * (i as f32 * 1.3).sin()).collect()
    }

    #[test]
    fn pure_silence_never_produces_segments_or_grows() {
        let mut seg = Segmenter::new(SR, VadParams::default());
        for _ in 0..200 {
            assert_eq!(seg.push(&hush(100)).len(), 0);
        }
        assert!(!seg.open_has_speech());
        assert!(seg.open_audio().len() <= ms_to_samples(330, SR));
        assert!(seg.finish().is_none());
    }

    #[test]
    fn cuts_at_pause_and_keeps_order() {
        let mut seg = Segmenter::new(SR, VadParams::default());
        let mut out = Vec::new();
        out.extend(seg.push(&hush(500)));
        out.extend(seg.push(&tone(1500)));
        assert_eq!(out.len(), 0);
        assert!(seg.open_has_speech());
        out.extend(seg.push(&hush(900)));
        assert_eq!(out.len(), 1, "a 900 ms pause closes the first utterance");
        // lead-in (<=300 ms) + speech (1500 ms) + tail (<=250 ms)
        let len_ms = out[0].len() * 1000 / SR as usize;
        assert!(
            (1500..=2100).contains(&len_ms),
            "segment length {len_ms} ms"
        );

        out.extend(seg.push(&tone(800)));
        assert!(seg.open_has_speech());
        let last = seg.finish();
        assert!(last.is_some());
        assert!(!seg.open_has_speech());
    }

    #[test]
    fn short_utterances_are_not_cut_at_pauses() {
        let mut seg = Segmenter::new(SR, VadParams::default());
        let mut out = seg.push(&tone(400));
        out.extend(seg.push(&hush(700)));
        assert!(
            out.is_empty(),
            "segment shorter than min_segment_ms must stay open"
        );
        out.extend(seg.push(&tone(1200)));
        out.extend(seg.push(&hush(700)));
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn clicks_are_ignored() {
        let mut seg = Segmenter::new(SR, VadParams::default());
        seg.push(&hush(300));
        seg.push(&tone(60));
        seg.push(&hush(1000));
        assert!(!seg.open_has_speech());
        assert!(seg.finish().is_none());
    }

    #[test]
    fn long_monologue_is_force_cut() {
        let params = VadParams {
            max_segment_ms: 3_000,
            ..VadParams::default()
        };
        let mut seg = Segmenter::new(SR, params);
        let mut out = Vec::new();
        for _ in 0..7 {
            out.extend(seg.push(&tone(1000)));
        }
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|s| s.len() == ms_to_samples(3_000, SR)));
    }

    #[test]
    fn level_tracks_loudness() {
        let mut seg = Segmenter::new(SR, VadParams::default());
        seg.push(&hush(60));
        let quiet = seg.level();
        seg.push(&tone(60));
        let loud = seg.level();
        assert!(loud > quiet + 0.3, "quiet={quiet} loud={loud}");
    }
}
