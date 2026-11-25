//! Non-blocking background worker thread for Stage-2 SLM candidate rescoring.

use std::{sync::Arc, time::Instant};

use flume::{Receiver, Sender, TrySendError};

use crate::SafetensorsReranker;

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

/// Handle to the bounded background Stage-2 neural rerank worker.
#[derive(Debug, Clone)]
pub struct AsyncRerankHandle {
    job_tx: Sender<RerankJob>,
    result_rx: Receiver<RerankResult>,
}

impl AsyncRerankHandle {
    /// Spawns a dedicated background worker thread owning `reranker`.
    #[must_use]
    pub fn spawn(reranker: Arc<SafetensorsReranker>) -> Self {
        let (job_tx, job_rx) = flume::bounded::<RerankJob>(8);
        let (result_tx, result_rx) = flume::bounded::<RerankResult>(16);

        std::thread::spawn(move || {
            while let Ok(mut job) = job_rx.recv() {
                // Coalesce burst keystrokes: drain to the freshest queued job!
                while let Ok(newer) = job_rx.try_recv() {
                    job = newer;
                }
                let bonuses = reranker.score_candidates(&job.left_context, &job.candidate_texts);
                let elapsed_ms =
                    u64::try_from(job.dispatched_at.elapsed().as_millis()).unwrap_or(u64::MAX);
                let _ = result_tx.try_send(RerankResult {
                    session_id: job.session_id,
                    generation_id: job.generation_id,
                    bonus_scores: bonuses,
                    elapsed_ms,
                });
            }
        });

        Self { job_tx, result_rx }
    }

    /// Enqueues a Stage-2 rerank job without ever blocking the UI thread.
    #[must_use]
    pub fn submit(&self, job: RerankJob) -> bool {
        !matches!(
            self.job_tx.try_send(job),
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_))
        )
    }

    /// Polls any completed Stage-2 rerank results without blocking.
    #[must_use]
    pub fn drain_results(&self) -> Vec<RerankResult> {
        self.result_rx.try_iter().collect()
    }
}
