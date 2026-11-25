//! Dedicated actor thread owning the non-`Send` `mlx_rs::Array` Metal GPU reranker and
//! communicating strictly via `flume` channels.

use std::{path::Path, time::Instant};

use flume::{Receiver, Sender, TrySendError};

use crate::{ModelConfig, NeuralError, SafetensorsReranker};

/// Asynchronous rerank job dispatched off the synchronous keystroke hot path.
#[derive(Debug, Clone)]
pub struct RerankJob {
    /// Target session ID.
    pub session_id: u64,
    /// Monotonic keystroke generation ID.
    pub generation_id: u64,
    /// Left context string before the cursor.
    pub left_context: String,
    /// Candidate surface strings from Stage 1.
    pub candidate_texts: Vec<String>,
    /// Timestamp when the keystroke arrived.
    pub dispatched_at: Instant,
}

/// Completed Stage-2 rerank scores delivered back to the session coordinator.
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Target session ID.
    pub session_id: u64,
    /// Monotonic keystroke generation ID.
    pub generation_id: u64,
    /// Additive neural score deltas aligned with `candidate_texts`.
    pub bonus_scores: Vec<f32>,
    /// End-to-end elapsed time in milliseconds since `dispatched_at`.
    pub elapsed_ms: u64,
}

enum ActorCmd {
    Async(RerankJob),
    Sync {
        left_context: String,
        candidate_texts: Vec<String>,
        reply_tx: Sender<Vec<f32>>,
    },
}

/// Thread-safe (`Send + Sync`) handle to the dedicated MLX Metal GPU actor thread.
#[derive(Debug, Clone)]
pub struct AsyncRerankHandle {
    cmd_tx: Sender<ActorCmd>,
    result_rx: Receiver<RerankResult>,
    config: ModelConfig,
}

impl AsyncRerankHandle {
    /// Spawns a dedicated actor thread that loads `model_dir` onto Metal GPU and owns the
    /// `SafetensorsReranker` exclusively without locks.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError`] if loading or validating the model inside the worker thread fails.
    pub fn spawn_from_dir(model_dir: &Path) -> Result<Self, NeuralError> {
        let dir_owned = model_dir.to_path_buf();
        let (init_tx, init_rx) = flume::bounded::<Result<ModelConfig, NeuralError>>(1);
        let (cmd_tx, cmd_rx) = flume::bounded::<ActorCmd>(16);
        let (result_tx, result_rx) = flume::bounded::<RerankResult>(16);

        std::thread::spawn(move || {
            let mut reranker = match SafetensorsReranker::load_dir(&dir_owned) {
                Ok(r) => {
                    let _ = init_tx.send(Ok(r.config().clone()));
                    r
                }
                Err(e) => {
                    let _ = init_tx.send(Err(e));
                    return;
                }
            };

            while let Ok(cmd) = cmd_rx.recv() {
                match cmd {
                    ActorCmd::Async(mut job) => {
                        // Coalesce burst async keystrokes to the newest queued job!
                        while let Ok(ActorCmd::Async(newer)) = cmd_rx.try_recv() {
                            job = newer;
                        }
                        let bonuses =
                            reranker.score_candidates(&job.left_context, &job.candidate_texts);
                        let elapsed_ms = u64::try_from(job.dispatched_at.elapsed().as_millis())
                            .unwrap_or(u64::MAX);
                        let _ = result_tx.try_send(RerankResult {
                            session_id: job.session_id,
                            generation_id: job.generation_id,
                            bonus_scores: bonuses,
                            elapsed_ms,
                        });
                    }
                    ActorCmd::Sync {
                        left_context,
                        candidate_texts,
                        reply_tx,
                    } => {
                        let bonuses = reranker.score_candidates(&left_context, &candidate_texts);
                        let _ = reply_tx.send(bonuses);
                    }
                }
            }
        });

        let config = init_rx
            .recv()
            .map_err(|e| NeuralError::Safetensors(format!("worker init channel closed: {e}")))??;

        Ok(Self {
            cmd_tx,
            result_rx,
            config,
        })
    }

    /// Returns the loaded model hyperparameters.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Enqueues a Stage-2 rerank job without ever blocking the UI thread.
    #[must_use]
    pub fn submit(&self, job: RerankJob) -> bool {
        !matches!(
            self.cmd_tx.try_send(ActorCmd::Async(job)),
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_))
        )
    }

    /// Synchronously scores a candidate batch via the dedicated MLX Metal worker (used by CLI
    /// `eval`).
    #[must_use]
    pub fn score_sync(&self, left_context: &str, candidate_texts: &[String]) -> Vec<f32> {
        let (reply_tx, reply_rx) = flume::bounded(1);
        if self
            .cmd_tx
            .send(ActorCmd::Sync {
                left_context: left_context.to_string(),
                candidate_texts: candidate_texts.to_vec(),
                reply_tx,
            })
            .is_err()
        {
            return vec![0.0; candidate_texts.len()];
        }
        reply_rx
            .recv()
            .unwrap_or_else(|_| vec![0.0; candidate_texts.len()])
    }

    /// Polls any completed Stage-2 rerank results without blocking.
    #[must_use]
    pub fn drain_results(&self) -> Vec<RerankResult> {
        self.result_rx.try_iter().collect()
    }
}
