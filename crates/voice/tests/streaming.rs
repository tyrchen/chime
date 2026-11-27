//! Streaming session lifecycle and an opt-in smoke test against the actual packaged model.
use std::{
    fs, io,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use chime_voice::{StreamingSession, VoiceSnapshot, VoiceState};

fn done(s: &StreamingSession) -> VoiceSnapshot {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let result = s.snapshot();
        if result.state == VoiceState::Done {
            return result;
        }
        assert!(
            Instant::now() < deadline,
            "decoder never completed: {result:?}"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn missing_model_fails_without_starting_a_session() -> io::Result<()> {
    let dir = tempfile::tempdir()?;
    assert!(
        StreamingSession::start(dir.path(), true, 10)
            .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn helper_crash_finishes_with_error() -> io::Result<()> {
    let dir = tempfile::tempdir()?;
    let contents = dir.path();
    fs::create_dir(contents.join("Helpers"))?;
    fs::create_dir_all(contents.join("Resources/voice"))?;
    for name in ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"] {
        fs::write(contents.join("Resources/voice").join(name), [])?;
    }
    std::os::unix::fs::symlink("/usr/bin/false", contents.join("Helpers/ChimeVoice"))?;
    let mut s = StreamingSession::start(contents, true, 10)?;
    s.push_audio(&[0.1; 800]);
    s.stop();
    let result = done(&s);
    assert!(result.error.is_some() && result.text.is_empty());
    Ok(())
}

#[test]
#[ignore = "requires a packaged arm64 Chime.app and a 16 kHz mono speech fixture"]
fn bundled_dictation() -> Result<(), Box<dyn std::error::Error>> {
    let contents = PathBuf::from(std::env::var("CHIME_TEST_APP")?)
        .canonicalize()?
        .join("Contents");
    let bytes = fs::read(std::env::var("CHIME_TEST_PCM")?)?;
    let expected = std::env::var("CHIME_TEST_TEXT")?;
    let samples: Vec<_> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
        .collect();
    // Cancellation while loading must not poison the next hold.
    let mut cancelled = StreamingSession::start(&contents, true, 180)?;
    cancelled.push_audio(&[0.1; 800]);
    drop(cancelled);
    for preview in [true, false] {
        let mut s = StreamingSession::start(&contents, preview, 180)?;
        let mut live = String::new();
        for chunk in samples.chunks(800) {
            s.push_audio(chunk);
            thread::sleep(Duration::from_millis(50));
            let snap = s.snapshot();
            assert_eq!(snap.state, VoiceState::Recording, "{snap:?}");
            if snap.text != live {
                eprintln!("{} ms: {}", snap.elapsed_ms, snap.text);
                live = snap.text;
            }
        }
        assert_eq!(!live.is_empty(), preview);
        s.stop();
        let result = done(&s);
        assert_eq!(result.error, None);
        assert_eq!(result.text, expected);
    }
    let mut s = StreamingSession::start(&contents, true, 1)?;
    s.push_audio(&vec![0.0; 20_000]);
    let silence = done(&s);
    assert!(silence.limit_reached && silence.text.is_empty() && silence.error.is_none());
    assert_eq!(silence.elapsed_ms, 1000);
    Ok(())
}
