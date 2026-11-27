//! One native Paraformer helper and one persistent decoder stream per key hold.
use std::{
    io::{self, Read, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex, MutexGuard, PoisonError, mpsc},
    thread,
};

use crate::{VoiceSnapshot, VoiceState};

const RATE: usize = 16_000;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Session owned by the IME. Only the worker accesses pipes; the audio callback never waits on IO.
#[derive(Debug)]
pub struct StreamingSession {
    tx: Option<mpsc::SyncSender<Vec<f32>>>,
    state: Arc<Mutex<VoiceSnapshot>>,
    child: Arc<Mutex<Child>>,
    samples: usize,
    limit: usize,
}

impl StreamingSession {
    /// Starts the signed helper relative to the app's Contents directory. Model loading is
    /// asynchronous inside that process, so the first microphone buffers are queued, not lost.
    ///
    /// # Errors
    /// Returns an error if the resources, helper or worker cannot be opened.
    pub fn start(contents: &Path, dir: &Path, preview: bool, seconds: u64) -> io::Result<Self> {
        for name in ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"] {
            std::fs::File::open(dir.join(name)).map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("本地语音模型 {name} 不可用，请在「语音输入」中重新下载：{e}"),
                )
            })?;
        }
        let mut child = Command::new(contents.join("Helpers/ChimeVoice"))
            .current_dir(dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let mut input = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("missing helper stdin"))?;
        let mut output = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("missing helper stdout"))?;
        let child = Arc::new(Mutex::new(child));
        let state = Arc::new(Mutex::new(VoiceSnapshot {
            state: VoiceState::Recording,
            text: String::new(),
            level: 0.0,
            elapsed_ms: 0,
            busy: false,
            error: None,
            limit_reached: false,
        }));
        // Bounded cold-start allowance; fail visibly rather than silently losing audio.
        let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(256);
        let session = Self {
            tx: Some(tx),
            state: state.clone(),
            child: child.clone(),
            samples: 0,
            limit: usize::try_from(seconds.clamp(1, 600)).unwrap_or(600) * RATE,
        };
        let worker = thread::Builder::new()
            .name("chime-streaming-asr".into())
            .spawn(move || {
                let result = (|| -> io::Result<()> {
                    // The host is Swift, not a Rust main that ignores SIGPIPE. An early helper
                    // exit must produce BrokenPipe, not terminate the entire input method.
                    #[cfg(unix)]
                    {
                        use nix::sys::signal::{SigSet, Signal};
                        let mut mask = SigSet::empty();
                        mask.add(Signal::SIGPIPE);
                        mask.thread_block()?;
                    }
                    loop {
                        // Closing the sender on stop drains all queued PCM before the single flush.
                        let frame = rx.recv().unwrap_or_default();
                        if matches!(lock(&state).state, VoiceState::Cancelled | VoiceState::Done) {
                            break;
                        }
                        let text = exchange(&mut input, &mut output, &frame)?;
                        let mut s = lock(&state);
                        if matches!(s.state, VoiceState::Cancelled | VoiceState::Done) {
                            break;
                        }
                        if preview || frame.is_empty() {
                            text.trim().clone_into(&mut s.text);
                        }
                        if frame.is_empty() {
                            break;
                        }
                    }
                    Ok(())
                })();
                drop(input);
                let mut s = lock(&state);
                if matches!(s.state, VoiceState::Recording | VoiceState::Finishing) {
                    s.state = VoiceState::Done;
                    s.busy = false;
                    s.level = 0.0;
                    if let Err(e) = result {
                        s.error = Some(format!("内置识别器中断：{e}"));
                    }
                }
                drop(s);
                // Also releases ORT allocations. Never wait for teardown on the IMK or tap threads.
                let mut proc = lock(&child);
                let _ = proc.kill();
                let _ = proc.wait();
            });
        if let Err(e) = worker {
            let mut proc = lock(&session.child);
            let _ = proc.kill();
            let _ = proc.wait();
            return Err(e);
        }
        Ok(session)
    }

    /// Enqueue samples without blocking on decoder progress.
    pub fn push_audio(&mut self, pcm: &[f32]) {
        let mut s = lock(&self.state);
        if s.state != VoiceState::Recording {
            return;
        }
        let take = pcm.len().min(self.limit - self.samples);
        for chunk in pcm[..take].chunks(RATE) {
            if self
                .tx
                .as_ref()
                .is_none_or(|tx| tx.try_send(chunk.to_vec()).is_err())
            {
                s.state = VoiceState::Done;
                s.error = Some("内置识别器未能及时接收音频，请重试".into());
                self.tx.take();
                return;
            }
        }
        s.level = (pcm[..take].iter().fold(0.0_f32, |m, x| m.max(x.abs())) * 4.0).min(1.0);
        self.samples += take;
        s.elapsed_ms = (self.samples / 16) as u64;
        if self.samples == self.limit {
            s.limit_reached = true;
            drop(s);
            self.stop();
        }
    }

    /// Stop capture and flush buffered PCM, including model right context.
    pub fn stop(&mut self) {
        let mut s = lock(&self.state);
        if s.state == VoiceState::Recording {
            s.state = VoiceState::Finishing;
            s.busy = true;
            s.level = 0.0;
        }
        self.tx.take();
    }

    /// Current UI state, isolated from previous/cancelled sessions.
    #[must_use]
    pub fn snapshot(&self) -> VoiceSnapshot {
        lock(&self.state).clone()
    }
}

impl Drop for StreamingSession {
    fn drop(&mut self) {
        lock(&self.state).state = VoiceState::Cancelled;
        // Interrupt even a blocked model load/read. Worker owns reaping the process.
        if let Ok(mut proc) = self.child.try_lock() {
            let _ = proc.kill();
        }
    }
}

fn exchange(input: &mut impl Write, output: &mut impl Read, pcm: &[f32]) -> io::Result<String> {
    let mut frame = Vec::with_capacity(4 + pcm.len() * 4);
    frame.extend_from_slice(
        &u32::try_from(pcm.len())
            .map_err(io::Error::other)?
            .to_le_bytes(),
    );
    for x in pcm {
        frame.extend_from_slice(&x.to_le_bytes());
    }
    input.write_all(&frame)?;
    let mut size = [0; 4];
    output.read_exact(&mut size)?;
    let size = u32::from_le_bytes(size) as usize;
    if size > 1024 * 1024 {
        return Err(io::Error::other("invalid transcript length"));
    }
    let mut bytes = vec![0; size];
    output.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(io::Error::other)
}
