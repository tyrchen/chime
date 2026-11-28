//! One hold-to-talk dictation session: audio intake, VAD segmentation, and a background
//! recognition worker that produces live partial and final transcripts.

use std::{
    sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError},
    thread,
    time::{Duration, Instant},
};

use serde::Serialize;

use crate::{
    Segmenter, Transcriber, VadParams, VoiceSettings,
    text::{join_segments, normalize_transcript},
};

/// Lifecycle of a [`VoiceSession`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceState {
    /// Microphone audio is being accepted.
    Recording,
    /// Recording stopped; remaining utterances are being recognized.
    Finishing,
    /// All audio recognized; `text` is final.
    Done,
    /// The session was cancelled; `text` must be discarded.
    Cancelled,
}

/// UI-facing state of a session, serialized as JSON across the FFI boundary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VoiceSnapshot {
    /// Current lifecycle state.
    pub state: VoiceState,
    /// Best transcript so far: final text of closed utterances, then provisional text.
    pub text: String,
    /// Latest input loudness in `0.0..=1.0`.
    pub level: f32,
    /// Recorded audio duration.
    pub elapsed_ms: u64,
    /// Whether a recognition request is currently running.
    pub busy: bool,
    /// Most recent recognition error, if any.
    pub error: Option<String>,
    /// Recording stopped because `max_record_secs` was reached.
    pub limit_reached: bool,
}

#[derive(Debug, Default)]
struct Slot {
    audio: Option<Vec<f32>>,
    provisional: String,
    text: Option<String>,
}

#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)]
struct Inner {
    segmenter: Segmenter,
    slots: Vec<Slot>,
    next_final: usize,
    open_partial: String,
    open_samples_at_partial: usize,
    last_partial_at: Option<Instant>,
    total_samples: usize,
    max_samples: usize,
    stop_requested: bool,
    flushed: bool,
    cancelled: bool,
    done: bool,
    busy: bool,
    partial_error: Option<String>,
    final_error: Option<String>,
    limit_reached: bool,
}

#[derive(Debug)]
struct Shared {
    inner: Mutex<Inner>,
    cv: Condvar,
    settings: VoiceSettings,
    sample_rate: u32,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

enum Job {
    Final(usize, Vec<f32>),
    Partial(usize, Vec<f32>),
}

/// A running dictation session. Dropping it cancels any outstanding work.
#[derive(Debug)]
pub struct VoiceSession {
    shared: Arc<Shared>,
}

impl VoiceSession {
    /// Starts a session for `sample_rate` Hz mono audio and spawns its recognition worker.
    #[must_use]
    pub fn start(
        settings: VoiceSettings,
        transcriber: Arc<dyn Transcriber>,
        sample_rate: u32,
    ) -> Self {
        Self::start_with_vad(settings, transcriber, sample_rate, VadParams::default())
    }

    /// Like [`Self::start`] with explicit VAD tunables.
    #[must_use]
    pub fn start_with_vad(
        settings: VoiceSettings,
        transcriber: Arc<dyn Transcriber>,
        sample_rate: u32,
        vad: VadParams,
    ) -> Self {
        let sample_rate = sample_rate.max(1);
        let max_samples = usize::try_from(
            settings
                .max_record_secs
                .max(1)
                .saturating_mul(u64::from(sample_rate)),
        )
        .unwrap_or(usize::MAX);
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                segmenter: Segmenter::new(sample_rate, vad),
                slots: Vec::new(),
                next_final: 0,
                open_partial: String::new(),
                open_samples_at_partial: 0,
                last_partial_at: None,
                total_samples: 0,
                max_samples,
                stop_requested: false,
                flushed: false,
                cancelled: false,
                done: false,
                busy: false,
                partial_error: None,
                final_error: None,
                limit_reached: false,
            }),
            cv: Condvar::new(),
            settings,
            sample_rate,
        });
        let worker_shared = Arc::clone(&shared);
        let spawned = thread::Builder::new()
            .name("chime-voice-asr".into())
            .spawn(move || worker(&worker_shared, transcriber.as_ref()));
        if let Err(e) = spawned {
            let mut g = shared.lock();
            g.final_error = Some(format!("cannot start recognition thread: {e}"));
            g.done = true;
        }
        Self { shared }
    }

    /// Appends microphone samples. Ignored once recording has stopped.
    pub fn push_audio(&self, samples: &[f32]) {
        let mut g = self.shared.lock();
        if g.stop_requested || g.cancelled || samples.is_empty() {
            return;
        }
        let room = g.max_samples.saturating_sub(g.total_samples);
        let take = room.min(samples.len());
        g.total_samples += take;
        let closed = g.segmenter.push(&samples[..take]);
        for audio in closed {
            let provisional = std::mem::take(&mut g.open_partial);
            g.slots.push(Slot {
                audio: Some(audio),
                provisional,
                text: None,
            });
            g.open_samples_at_partial = 0;
        }
        if take < samples.len() || g.total_samples >= g.max_samples {
            g.limit_reached = true;
            g.stop_requested = true;
        }
        drop(g);
        self.shared.cv.notify_all();
    }

    /// Stops recording; the worker recognizes the remaining audio and then reports
    /// [`VoiceState::Done`].
    pub fn stop(&self) {
        self.shared.lock().stop_requested = true;
        self.shared.cv.notify_all();
    }

    /// Cancels the session and discards all text.
    pub fn cancel(&self) {
        let mut g = self.shared.lock();
        if !g.done {
            g.cancelled = true;
        }
        drop(g);
        self.shared.cv.notify_all();
    }

    /// Returns the current UI state.
    #[must_use]
    pub fn snapshot(&self) -> VoiceSnapshot {
        let g = self.shared.lock();
        let state = if g.cancelled {
            VoiceState::Cancelled
        } else if g.done {
            VoiceState::Done
        } else if g.stop_requested {
            VoiceState::Finishing
        } else {
            VoiceState::Recording
        };
        let text = if g.cancelled {
            String::new()
        } else {
            join_segments(
                g.slots
                    .iter()
                    .map(|s| s.text.as_deref().unwrap_or(&s.provisional))
                    .chain(std::iter::once(g.open_partial.as_str())),
            )
        };
        let elapsed_ms =
            (g.total_samples as u64).saturating_mul(1000) / u64::from(self.shared.sample_rate);
        VoiceSnapshot {
            state,
            text,
            level: if state == VoiceState::Recording {
                g.segmenter.level()
            } else {
                0.0
            },
            elapsed_ms,
            busy: g.busy,
            error: g.final_error.clone().or_else(|| g.partial_error.clone()),
            limit_reached: g.limit_reached,
        }
    }

    /// Blocks until the session is done or cancelled, or `timeout` elapses.
    #[must_use]
    pub fn wait_finished(&self, timeout: Duration) -> VoiceSnapshot {
        let deadline = Instant::now() + timeout;
        let mut g = self.shared.lock();
        while !g.done && !g.cancelled {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            g = self
                .shared
                .cv
                .wait_timeout(g, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        drop(g);
        self.snapshot()
    }
}

impl Drop for VoiceSession {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn next_job(shared: &Shared) -> Option<Job> {
    let interval = Duration::from_millis(shared.settings.partial_interval_ms);
    let min_new_audio = (shared.sample_rate / 5) as usize; // 200 ms
    let mut g = shared.lock();
    loop {
        if g.cancelled {
            return None;
        }
        let idx = g.next_final;
        if idx < g.slots.len() {
            g.next_final += 1;
            if let Some(audio) = g.slots[idx].audio.take() {
                g.busy = true;
                return Some(Job::Final(idx, audio));
            }
            continue;
        }
        if g.stop_requested {
            if !g.flushed {
                g.flushed = true;
                let provisional = std::mem::take(&mut g.open_partial);
                if let Some(audio) = g.segmenter.finish() {
                    g.slots.push(Slot {
                        audio: Some(audio),
                        provisional,
                        text: None,
                    });
                }
                continue;
            }
            g.done = true;
            drop(g);
            shared.cv.notify_all();
            return None;
        }
        if shared.settings.live_preview && g.segmenter.open_has_speech() {
            let open_len = g.segmenter.open_audio().len();
            let due = g.last_partial_at.is_none_or(|t| t.elapsed() >= interval);
            if due && open_len >= g.open_samples_at_partial + min_new_audio {
                g.open_samples_at_partial = open_len;
                g.last_partial_at = Some(Instant::now());
                g.busy = true;
                let idx = g.slots.len();
                return Some(Job::Partial(idx, g.segmenter.open_audio().to_vec()));
            }
        }
        g = shared
            .cv
            .wait_timeout(g, Duration::from_millis(40))
            .unwrap_or_else(PoisonError::into_inner)
            .0;
    }
}

fn worker(shared: &Shared, transcriber: &dyn Transcriber) {
    let settings = &shared.settings;
    while let Some(job) = next_job(shared) {
        let (idx, audio, is_final) = match job {
            Job::Final(i, a) => (i, a, true),
            Job::Partial(i, a) => (i, a, false),
        };
        let result = transcriber
            .transcribe(&audio, shared.sample_rate)
            .map(|raw| normalize_transcript(&raw, &settings.prompt, settings.chinese_punctuation));
        let mut g = shared.lock();
        g.busy = false;
        if g.cancelled {
            return;
        }
        match (is_final, result) {
            (true, Ok(text)) => {
                g.partial_error = None;
                if let Some(slot) = g.slots.get_mut(idx) {
                    slot.text = Some(text);
                }
            }
            (true, Err(e)) => {
                // Keep whatever the live preview recognized for this utterance.
                if let Some(slot) = g.slots.get_mut(idx) {
                    slot.text = Some(std::mem::take(&mut slot.provisional));
                }
                g.final_error = Some(e.to_string());
            }
            (false, Ok(text)) => {
                g.partial_error = None;
                if idx == g.slots.len() {
                    g.open_partial = text;
                } else if let Some(slot) = g.slots.get_mut(idx)
                    && slot.text.is_none()
                {
                    slot.provisional = text;
                }
            }
            (false, Err(e)) => g.partial_error = Some(e.to_string()),
        }
        drop(g);
        shared.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::VoiceError;

    const SR: u32 = 16_000;

    /// Returns `"<n>"` where `n` is the clip length in tenths of a second, so tests can tell
    /// partial (shorter) and final (complete) results apart.
    #[derive(Default)]
    struct FakeAsr {
        calls: AtomicUsize,
        fail_finals: bool,
        delay_ms: u64,
    }

    impl Transcriber for FakeAsr {
        fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String, VoiceError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.delay_ms > 0 {
                thread::sleep(Duration::from_millis(self.delay_ms));
            }
            if self.fail_finals && n > 0 {
                return Err(VoiceError::Transport("connection refused".into()));
            }
            Ok(format!("[{}]", samples.len() * 10 / sample_rate as usize))
        }
    }

    fn tone(ms: usize) -> Vec<f32> {
        #[allow(clippy::cast_precision_loss)]
        (0..ms * 16)
            .map(|i| 0.2 * (i as f32 * 0.07).sin())
            .collect()
    }

    fn hush(ms: usize) -> Vec<f32> {
        vec![0.0; ms * 16]
    }

    fn settings(live_preview: bool) -> VoiceSettings {
        VoiceSettings {
            live_preview,
            partial_interval_ms: 0,
            ..VoiceSettings::default()
        }
    }

    fn feed(s: &VoiceSession, audio: &[f32]) {
        for chunk in audio.chunks(1600) {
            s.push_audio(chunk);
        }
    }

    #[test]
    fn two_utterances_are_recognized_in_order() {
        let asr = Arc::new(FakeAsr::default());
        let s = VoiceSession::start(settings(false), asr.clone(), SR);
        feed(&s, &tone(1500));
        feed(&s, &hush(900));
        feed(&s, &tone(2000));
        assert_eq!(s.snapshot().state, VoiceState::Recording);
        s.stop();
        let snap = s.wait_finished(Duration::from_secs(5));
        assert_eq!(snap.state, VoiceState::Done);
        assert_eq!(asr.calls.load(Ordering::SeqCst), 2);
        // first: 1.5 s speech + tail, second: lead-in + 2.0 s speech
        assert!(
            snap.text.starts_with("[17]") || snap.text.starts_with("[18]"),
            "{}",
            snap.text
        );
        assert!(
            snap.text.ends_with("[22]") || snap.text.ends_with("[23]"),
            "{}",
            snap.text
        );
        assert!(snap.error.is_none());
    }

    #[test]
    fn silence_sends_nothing() {
        let asr = Arc::new(FakeAsr::default());
        let s = VoiceSession::start(settings(true), asr.clone(), SR);
        feed(&s, &hush(3000));
        s.stop();
        let snap = s.wait_finished(Duration::from_secs(5));
        assert_eq!(snap.state, VoiceState::Done);
        assert_eq!(snap.text, "");
        assert_eq!(asr.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn live_preview_shows_text_before_release() {
        let asr = Arc::new(FakeAsr::default());
        let s = VoiceSession::start(settings(true), asr.clone(), SR);
        feed(&s, &tone(1000));
        let deadline = Instant::now() + Duration::from_secs(3);
        while s.snapshot().text.is_empty() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        let live = s.snapshot();
        assert_eq!(live.state, VoiceState::Recording);
        assert!(
            !live.text.is_empty(),
            "partial transcript expected while recording"
        );
        s.stop();
        let done = s.wait_finished(Duration::from_secs(5));
        assert_eq!(done.state, VoiceState::Done);
        assert_eq!(done.text, "[10]");
    }

    #[test]
    fn cancel_discards_everything() {
        let asr = Arc::new(FakeAsr {
            delay_ms: 50,
            ..FakeAsr::default()
        });
        let s = VoiceSession::start(settings(true), asr, SR);
        feed(&s, &tone(1500));
        s.cancel();
        let snap = s.wait_finished(Duration::from_secs(2));
        assert_eq!(snap.state, VoiceState::Cancelled);
        assert_eq!(snap.text, "");
    }

    #[test]
    fn failed_final_falls_back_to_preview_and_reports_error() {
        let asr = Arc::new(FakeAsr {
            fail_finals: true,
            ..FakeAsr::default()
        });
        let s = VoiceSession::start(settings(true), asr.clone(), SR);
        feed(&s, &tone(1000));
        let deadline = Instant::now() + Duration::from_secs(3);
        while asr.calls.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        // Let the first (successful) partial land before stopping.
        while s.snapshot().text.is_empty() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        s.stop();
        let snap = s.wait_finished(Duration::from_secs(5));
        assert_eq!(snap.state, VoiceState::Done);
        assert_ne!(snap.text, "");
        assert!(
            snap.error
                .as_deref()
                .is_some_and(|e| e.contains("connection refused"))
        );
    }

    #[test]
    fn recording_limit_stops_intake() {
        let asr = Arc::new(FakeAsr::default());
        let cfg = VoiceSettings {
            max_record_secs: 1,
            ..settings(false)
        };
        let s = VoiceSession::start(cfg, asr, SR);
        feed(&s, &tone(2500));
        let snap = s.wait_finished(Duration::from_secs(5));
        assert!(snap.limit_reached);
        assert_eq!(snap.state, VoiceState::Done);
        assert_eq!(snap.elapsed_ms, 1000);
        assert_eq!(snap.text, "[10]");
    }
}
