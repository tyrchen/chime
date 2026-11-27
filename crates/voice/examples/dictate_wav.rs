//! Streams a WAV file through a [`VoiceSession`] in real time against a speech server, printing the
//! live transcript. Handy for checking a self-hosted ASR deployment without the macOS UI.
//!
//! ```bash
//! ffmpeg -i speech.m4a -ar 16000 -ac 1 -c:a pcm_s16le /tmp/speech.wav
//! cargo run -p chime-voice --example dictate_wav -- /tmp/speech.wav \
//!     http://127.0.0.1:8178/inference [model] [api-key]
//! ```

use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use chime_voice::{HttpTranscriber, VoiceSession, VoiceSettings};

fn read_wav_pcm16_mono(path: &str) -> Result<(Vec<f32>, u32), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let (mut pos, mut rate, mut fmt_ok) = (12usize, 16_000u32, false);
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        if id == b"fmt " && body.len() >= 16 {
            let channels = u16::from_le_bytes([body[2], body[3]]);
            let bits = u16::from_le_bytes([body[14], body[15]]);
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            fmt_ok = channels == 1 && bits == 16;
        } else if id == b"data" {
            if !fmt_ok {
                return Err(
                    "expected 16-bit mono PCM (convert with ffmpeg -ac 1 -c:a pcm_s16le)".into(),
                );
            }
            let samples = body
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0)
                .collect();
            return Ok((samples, rate));
        }
        pos += 8 + len + (len & 1);
    }
    Err("no data chunk".into())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(path), Some(endpoint)) = (args.get(1), args.get(2)) else {
        return Err("usage: dictate_wav <file.wav> <endpoint> [model] [api-key]".into());
    };
    let (samples, rate) = read_wav_pcm16_mono(path)?;
    let settings = VoiceSettings {
        endpoint: endpoint.clone(),
        model: args.get(3).cloned().unwrap_or_else(|| "whisper-1".into()),
        prompt: "以下是普通话的句子，使用简体中文，并带有标点符号。".into(),
        ..VoiceSettings::default()
    };
    let client =
        HttpTranscriber::new(&settings, args.get(4).cloned()).map_err(|e| e.to_string())?;
    let session = VoiceSession::start(settings, Arc::new(client), rate);

    let started = Instant::now();
    let chunk = (rate / 10) as usize; // 100 ms, like the microphone tap
    let mut last = String::new();
    for (i, part) in samples.chunks(chunk).enumerate() {
        session.push_audio(part);
        thread::sleep(Duration::from_millis(100));
        let snap = session.snapshot();
        if snap.text != last {
            println!("[{:>5} ms audio] {}", (i + 1) * 100, snap.text);
            last = snap.text;
        }
    }
    let released = Instant::now();
    session.stop();
    let done = session.wait_finished(Duration::from_secs(60));
    println!(
        "final ({:?}, {} ms after release, {} ms total): {}",
        done.state,
        released.elapsed().as_millis(),
        started.elapsed().as_millis(),
        done.text
    );
    if let Some(err) = done.error {
        println!("error: {err}");
    }
    Ok(())
}
