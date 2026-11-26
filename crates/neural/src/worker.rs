//! Dedicated actor thread owning the non-`Send` `mlx_rs::Array` Metal GPU reranker + `Multi-LoRA`
//! sidecar adapters, communicating strictly via `flume` channels.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use chime_phonetics::{FuzzyConfig, ShuangpinScheme};
use flume::{Receiver, Sender, TrySendError};

use crate::{
    LoraMountSpec, ModelConfig, NeuralBeamCandidate, NeuralError, SafetensorsReranker, ScoringMode,
};

/// Request for Route A (Pinyin-constrained autoregressive neural beam search + exact-lexicon
/// Bayesian fusion) dispatched to the dedicated Metal GPU actor thread.
#[derive(Debug, Clone)]
pub struct RouteADecodeRequest {
    /// Active `macOS` host application bundle identifier (for `Multi-LoRA` dynamic routing).
    pub app_bundle_id: String,
    /// Left context string before the cursor.
    pub left_context: String,
    /// Raw preedit keystroke buffer.
    pub raw_input: String,
    /// Active keyboard scheme (`Full` or Shuangpin).
    pub scheme: ShuangpinScheme,
    /// Active fuzzy Pinyin configuration.
    pub fuzzy: FuzzyConfig,
    /// Whether to apply Pangu spacing around English tokens.
    pub pangu_spacing: bool,
    /// Scoring mode (`ClosedSentence` vs `OpenContextual`).
    pub scoring_mode: ScoringMode,
    /// Autoregressive beam width (typically `8`).
    pub beam_width: usize,
    /// Optional exact-pinyin full-span lexicon / `.imepack` / user-custom phrase proposals scored
    /// strictly on the exact same Bayesian `log P_LM(Y | C)` scale (zero static unigram bias).
    pub extra_exact_candidates: Vec<String>,
}

/// Asynchronous rerank job dispatched off the synchronous keystroke hot path.
#[derive(Debug, Clone)]
pub struct RerankJob {
    /// Target session ID.
    pub session_id: u64,
    /// Monotonic keystroke generation ID.
    pub generation_id: u64,
    /// Active `macOS` host application bundle identifier (for `Multi-LoRA` dynamic routing).
    pub app_bundle_id: String,
    /// Left context string before the cursor.
    pub left_context: String,
    /// Candidate surface strings from Stage 1.
    pub candidate_texts: Vec<String>,
    /// Scoring mode (closed sentence vs mid-sentence open contextual continuation).
    pub scoring_mode: ScoringMode,
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

/// Completed asynchronous Route A neural beam decode delivered back to the session coordinator
/// whenever a keystroke's GPU beam search exceeds the frame-0 synchronous budget.
#[derive(Debug, Clone)]
pub struct RouteAResult {
    /// Target session ID.
    pub session_id: u64,
    /// Monotonic keystroke generation ID.
    pub generation_id: u64,
    /// Ranked Route A neural beam candidates.
    pub candidates: Vec<NeuralBeamCandidate>,
    /// End-to-end elapsed time in milliseconds since `dispatched_at`.
    pub elapsed_ms: u64,
}

enum ActorCmd {
    Async(RerankJob),
    TimedOrAsync {
        job: RerankJob,
        sync_reply_tx: Sender<Vec<f32>>,
    },
    Sync {
        left_context: String,
        app_bundle_id: String,
        candidate_texts: Vec<String>,
        scoring_mode: ScoringMode,
        reply_tx: Sender<Vec<f32>>,
    },
    RouteADecode {
        req: RouteADecodeRequest,
        reply_tx: Sender<Vec<NeuralBeamCandidate>>,
    },
    RouteATimedOrAsync {
        session_id: u64,
        generation_id: u64,
        dispatched_at: Instant,
        req: RouteADecodeRequest,
        sync_reply_tx: Sender<Vec<NeuralBeamCandidate>>,
    },
}

/// Thread-safe (`Send + Sync`) handle to the dedicated MLX Metal GPU actor thread.
#[derive(Debug, Clone)]
pub struct AsyncRerankHandle {
    cmd_tx: Sender<ActorCmd>,
    result_rx: Receiver<RerankResult>,
    route_a_result_rx: Receiver<RouteAResult>,
    config: ModelConfig,
    lora_count: usize,
    has_constrained_decoder: bool,
}

impl AsyncRerankHandle {
    /// Spawns a dedicated actor thread that loads `model_dir` onto Metal GPU and owns the
    /// `SafetensorsReranker` exclusively without locks.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError`] if loading or validating the model inside the worker thread fails.
    pub fn spawn_from_dir(model_dir: &Path) -> Result<Self, NeuralError> {
        Self::spawn_from_dir_with_loras(model_dir, Vec::new())
    }

    /// Spawns a dedicated actor thread loading `model_dir` plus zero or more `Multi-LoRA` sidecar
    /// adapters onto Metal GPU.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError`] if loading the base model fails.
    #[allow(clippy::too_many_lines)]
    pub fn spawn_from_dir_with_loras(
        model_dir: &Path,
        loras: Vec<LoraMountSpec>,
    ) -> Result<Self, NeuralError> {
        let dir_owned = model_dir.to_path_buf();
        let (init_tx, init_rx) =
            flume::bounded::<Result<(ModelConfig, usize, bool), NeuralError>>(1);
        let (cmd_tx, cmd_rx) = flume::bounded::<ActorCmd>(16);
        let (result_tx, result_rx) = flume::bounded::<RerankResult>(16);
        let (route_a_result_tx, route_a_result_rx) = flume::bounded::<RouteAResult>(16);

        std::thread::spawn(move || {
            let mut reranker = match SafetensorsReranker::load_dir_with_loras(&dir_owned, &loras) {
                Ok(r) => {
                    let _ = init_tx.send(Ok((
                        r.config().clone(),
                        r.lora_count(),
                        r.has_constrained_decoder(),
                    )));
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
                        let bonuses = reranker.score_candidates_with_scoring_mode(
                            &job.left_context,
                            &job.app_bundle_id,
                            &job.candidate_texts,
                            job.scoring_mode,
                        );
                        let elapsed_ms = u64::try_from(job.dispatched_at.elapsed().as_millis())
                            .unwrap_or(u64::MAX);
                        let _ = result_tx.try_send(RerankResult {
                            session_id: job.session_id,
                            generation_id: job.generation_id,
                            bonus_scores: bonuses,
                            elapsed_ms,
                        });
                    }
                    ActorCmd::TimedOrAsync {
                        mut job,
                        mut sync_reply_tx,
                    } => {
                        // Coalesce burst keystrokes in `TimedOrAsync` mode: if newer keystrokes
                        // are already queued, skip superseded jobs so the Metal GPU thread never
                        // builds a FIFO backlog that causes the final keystroke to miss its frame
                        // budget!
                        while let Ok(next_cmd) = cmd_rx.try_recv() {
                            match next_cmd {
                                ActorCmd::TimedOrAsync {
                                    job: newer_job,
                                    sync_reply_tx: newer_tx,
                                } => {
                                    job = newer_job;
                                    sync_reply_tx = newer_tx;
                                }
                                ActorCmd::Async(newer_job) => {
                                    job = newer_job;
                                    let (dead_tx, _) = flume::bounded(0);
                                    sync_reply_tx = dead_tx;
                                }
                                ActorCmd::Sync {
                                    left_context,
                                    app_bundle_id,
                                    candidate_texts,
                                    scoring_mode,
                                    reply_tx,
                                } => {
                                    let bonuses = reranker.score_candidates_with_scoring_mode(
                                        &left_context,
                                        &app_bundle_id,
                                        &candidate_texts,
                                        scoring_mode,
                                    );
                                    let _ = reply_tx.send(bonuses);
                                }
                                ActorCmd::RouteADecode { req, reply_tx } => {
                                    let out = run_route_a_request(&mut reranker, &req);
                                    let _ = reply_tx.send(out);
                                }
                                ActorCmd::RouteATimedOrAsync {
                                    session_id,
                                    generation_id,
                                    dispatched_at,
                                    req,
                                    sync_reply_tx: ra_tx,
                                } => {
                                    let out = run_route_a_request(&mut reranker, &req);
                                    if ra_tx.send(out.clone()).is_err() {
                                        let elapsed_ms =
                                            u64::try_from(dispatched_at.elapsed().as_millis())
                                                .unwrap_or(u64::MAX);
                                        let _ = route_a_result_tx.try_send(RouteAResult {
                                            session_id,
                                            generation_id,
                                            candidates: out,
                                            elapsed_ms,
                                        });
                                    }
                                }
                            }
                        }
                        let bonuses = reranker.score_candidates_with_scoring_mode(
                            &job.left_context,
                            &job.app_bundle_id,
                            &job.candidate_texts,
                            job.scoring_mode,
                        );
                        // Deliver synchronously if caller is still within its frame budget;
                        // otherwise seamlessly fall back to async `result_tx`!
                        if sync_reply_tx.send(bonuses.clone()).is_err() {
                            let elapsed_ms = u64::try_from(job.dispatched_at.elapsed().as_millis())
                                .unwrap_or(u64::MAX);
                            let _ = result_tx.try_send(RerankResult {
                                session_id: job.session_id,
                                generation_id: job.generation_id,
                                bonus_scores: bonuses,
                                elapsed_ms,
                            });
                        }
                    }
                    ActorCmd::Sync {
                        left_context,
                        app_bundle_id,
                        candidate_texts,
                        scoring_mode,
                        reply_tx,
                    } => {
                        let bonuses = reranker.score_candidates_with_scoring_mode(
                            &left_context,
                            &app_bundle_id,
                            &candidate_texts,
                            scoring_mode,
                        );
                        let _ = reply_tx.send(bonuses);
                    }
                    ActorCmd::RouteADecode {
                        mut req,
                        mut reply_tx,
                    } => {
                        // Coalesce burst keystrokes in Route A mode so rapid typing always
                        // evaluates the latest keystroke immediately with zero FIFO backlog!
                        while let Ok(next_cmd) = cmd_rx.try_recv() {
                            match next_cmd {
                                ActorCmd::RouteADecode {
                                    req: newer_req,
                                    reply_tx: newer_tx,
                                } => {
                                    req = newer_req;
                                    reply_tx = newer_tx;
                                }
                                ActorCmd::Sync {
                                    left_context,
                                    app_bundle_id,
                                    candidate_texts,
                                    scoring_mode,
                                    reply_tx: s_tx,
                                } => {
                                    let bonuses = reranker.score_candidates_with_scoring_mode(
                                        &left_context,
                                        &app_bundle_id,
                                        &candidate_texts,
                                        scoring_mode,
                                    );
                                    let _ = s_tx.send(bonuses);
                                }
                                ActorCmd::Async(_)
                                | ActorCmd::TimedOrAsync { .. }
                                | ActorCmd::RouteATimedOrAsync { .. } => {}
                            }
                        }
                        let out = run_route_a_request(&mut reranker, &req);
                        let _ = reply_tx.send(out);
                    }
                    ActorCmd::RouteATimedOrAsync {
                        mut session_id,
                        mut generation_id,
                        mut dispatched_at,
                        mut req,
                        mut sync_reply_tx,
                    } => {
                        // Coalesce rapid keystrokes so burst typing NEVER queues stale Route A
                        // GPU passes behind earlier keystrokes!
                        while let Ok(next_cmd) = cmd_rx.try_recv() {
                            match next_cmd {
                                ActorCmd::RouteATimedOrAsync {
                                    session_id: next_session,
                                    generation_id: next_gen,
                                    dispatched_at: next_at,
                                    req: next_req,
                                    sync_reply_tx: next_tx,
                                } => {
                                    session_id = next_session;
                                    generation_id = next_gen;
                                    dispatched_at = next_at;
                                    req = next_req;
                                    sync_reply_tx = next_tx;
                                }
                                ActorCmd::RouteADecode {
                                    req: r_req,
                                    reply_tx: r_tx,
                                } => {
                                    let out = run_route_a_request(&mut reranker, &r_req);
                                    let _ = r_tx.send(out);
                                }
                                ActorCmd::Sync {
                                    left_context,
                                    app_bundle_id,
                                    candidate_texts,
                                    scoring_mode,
                                    reply_tx: s_tx,
                                } => {
                                    let bonuses = reranker.score_candidates_with_scoring_mode(
                                        &left_context,
                                        &app_bundle_id,
                                        &candidate_texts,
                                        scoring_mode,
                                    );
                                    let _ = s_tx.send(bonuses);
                                }
                                ActorCmd::Async(_) | ActorCmd::TimedOrAsync { .. } => {}
                            }
                        }
                        let out = run_route_a_request(&mut reranker, &req);
                        if sync_reply_tx.send(out.clone()).is_err() {
                            let elapsed_ms = u64::try_from(dispatched_at.elapsed().as_millis())
                                .unwrap_or(u64::MAX);
                            let _ = route_a_result_tx.try_send(RouteAResult {
                                session_id,
                                generation_id,
                                candidates: out,
                                elapsed_ms,
                            });
                        }
                    }
                }
            }
        });

        let (config, lora_count, has_constrained_decoder) = init_rx
            .recv()
            .map_err(|e| NeuralError::Safetensors(format!("worker init channel closed: {e}")))??;

        Ok(Self {
            cmd_tx,
            result_rx,
            route_a_result_rx,
            config,
            lora_count,
            has_constrained_decoder,
        })
    }

    /// Returns true if the loaded model bundle includes `token_pinyin.json` and supports Route A
    /// constrained autoregressive neural beam decoding.
    #[must_use]
    pub const fn has_constrained_decoder(&self) -> bool {
        self.has_constrained_decoder
    }

    /// Returns true if the loaded model bundle uses full-sentence Pinyin lookahead conditioning
    /// (`pinyin_char_v1`).
    #[must_use]
    pub fn is_pinyin_conditioned(&self) -> bool {
        self.config.is_pinyin_conditioned()
    }

    /// Runs Route A constrained autoregressive neural beam search (+ pure-Bayesian exact-lexicon
    /// hypothesis fusion) on the Metal GPU worker within `budget`.
    #[must_use]
    pub fn decode_route_a_with_budget(
        &self,
        req: RouteADecodeRequest,
        budget: Duration,
    ) -> Option<Vec<NeuralBeamCandidate>> {
        if !self.has_constrained_decoder {
            return None;
        }
        let (reply_tx, reply_rx) = flume::bounded(1);
        if matches!(
            self.cmd_tx
                .try_send(ActorCmd::RouteADecode { req, reply_tx }),
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_))
        ) {
            return None;
        }
        reply_rx.recv_timeout(budget).ok()
    }

    /// Runs Route A on the Metal GPU worker with a short frame-0 `budget` (e.g. `12 ms`).
    /// If it finishes within `budget`, returns `Some(candidates)` synchronously; otherwise returns
    /// `None` immediately (keeping keystrokes ultra-snappy!) and delivers the finished candidates
    /// asynchronously via [`Self::drain_route_a_results`].
    #[must_use]
    pub fn decode_route_a_timed_or_async(
        &self,
        session_id: u64,
        generation_id: u64,
        req: RouteADecodeRequest,
        budget: Duration,
    ) -> Option<Vec<NeuralBeamCandidate>> {
        if !self.has_constrained_decoder {
            return None;
        }
        let (sync_reply_tx, sync_reply_rx) = flume::bounded(1);
        if matches!(
            self.cmd_tx.try_send(ActorCmd::RouteATimedOrAsync {
                session_id,
                generation_id,
                dispatched_at: Instant::now(),
                req,
                sync_reply_tx,
            }),
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_))
        ) {
            return None;
        }
        if budget.is_zero() {
            return None;
        }
        sync_reply_rx.recv_timeout(budget).ok()
    }

    /// Polls any completed asynchronous Route A decode results without blocking.
    #[must_use]
    pub fn drain_route_a_results(&self) -> Vec<RouteAResult> {
        self.route_a_result_rx.try_iter().collect()
    }

    /// Returns the loaded model hyperparameters.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns the number of mounted `Multi-LoRA` sidecar adapters.
    #[must_use]
    pub const fn lora_count(&self) -> usize {
        self.lora_count
    }

    /// Enqueues a Stage-2 rerank job without ever blocking the UI thread.
    #[must_use]
    pub fn submit(&self, job: RerankJob) -> bool {
        !matches!(
            self.cmd_tx.try_send(ActorCmd::Async(job)),
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_))
        )
    }

    /// Submits a rerank job and waits up to `budget` (e.g., `10ms`) for an immediate Metal GPU
    /// result so 99% of keystrokes render Stage-2 contextually reranked candidates on frame 0!
    /// If the GPU exceeds `budget`, returns `None` and delivers the result asynchronously via
    /// [`Self::drain_results`].
    #[must_use]
    pub fn submit_with_budget(&self, job: RerankJob, budget: Duration) -> Option<Vec<f32>> {
        let (sync_reply_tx, sync_reply_rx) = flume::bounded(1);
        if matches!(
            self.cmd_tx
                .try_send(ActorCmd::TimedOrAsync { job, sync_reply_tx }),
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_))
        ) {
            return None;
        }
        sync_reply_rx.recv_timeout(budget).ok()
    }

    /// Synchronously scores a candidate batch via the dedicated MLX Metal worker (used by CLI
    /// `eval`).
    #[must_use]
    pub fn score_sync(&self, left_context: &str, candidate_texts: &[String]) -> Vec<f32> {
        self.score_sync_with_app(left_context, "", candidate_texts)
    }

    /// Synchronously scores a candidate batch with `app_bundle_id` `Multi-LoRA` routing.
    #[must_use]
    pub fn score_sync_with_app(
        &self,
        left_context: &str,
        app_bundle_id: &str,
        candidate_texts: &[String],
    ) -> Vec<f32> {
        self.score_sync_mode_with_timeout(left_context, app_bundle_id, candidate_texts, false, None)
    }

    /// Scores an open mid-sentence prefix batch (without right-boundary punctuation closure),
    /// optionally bounded by `timeout`.
    #[must_use]
    pub fn score_sync_mode_with_timeout(
        &self,
        left_context: &str,
        app_bundle_id: &str,
        candidate_texts: &[String],
        open_prefix: bool,
        timeout: Option<Duration>,
    ) -> Vec<f32> {
        let mode = if open_prefix {
            ScoringMode::BeamPrefixGuidance
        } else {
            ScoringMode::ClosedSentence
        };
        self.score_sync_scoring_mode(left_context, app_bundle_id, candidate_texts, mode, timeout)
    }

    /// Synchronously scores a candidate batch under an explicit [`ScoringMode`], optionally
    /// bounded by `timeout`.
    #[must_use]
    pub fn score_sync_scoring_mode(
        &self,
        left_context: &str,
        app_bundle_id: &str,
        candidate_texts: &[String],
        scoring_mode: ScoringMode,
        timeout: Option<Duration>,
    ) -> Vec<f32> {
        let (reply_tx, reply_rx) = flume::bounded(1);
        if self
            .cmd_tx
            .send(ActorCmd::Sync {
                left_context: left_context.to_string(),
                app_bundle_id: app_bundle_id.to_string(),
                candidate_texts: candidate_texts.to_vec(),
                scoring_mode,
                reply_tx,
            })
            .is_err()
        {
            return vec![0.0; candidate_texts.len()];
        }
        if let Some(dur) = timeout {
            reply_rx
                .recv_timeout(dur)
                .unwrap_or_else(|_| vec![0.0; candidate_texts.len()])
        } else {
            reply_rx
                .recv()
                .unwrap_or_else(|_| vec![0.0; candidate_texts.len()])
        }
    }

    /// Polls any completed Stage-2 rerank results without blocking.
    #[must_use]
    pub fn drain_results(&self) -> Vec<RerankResult> {
        self.result_rx.try_iter().collect()
    }
}

fn run_route_a_request(
    reranker: &mut SafetensorsReranker,
    req: &RouteADecodeRequest,
) -> Vec<NeuralBeamCandidate> {
    let mut ncands = reranker.decode_constrained(
        &req.left_context,
        &req.app_bundle_id,
        &req.raw_input,
        req.scheme,
        req.fuzzy,
        req.pangu_spacing,
        req.scoring_mode,
        req.beam_width,
    );
    if !req.extra_exact_candidates.is_empty() {
        let mut to_score = Vec::new();
        for txt in &req.extra_exact_candidates {
            if !ncands.iter().any(|c| c.text == *txt) && !to_score.contains(txt) {
                to_score.push(txt.clone());
            }
        }
        if !to_score.is_empty() {
            let lower_raw = req.raw_input.to_ascii_lowercase();
            let is_single_syl = req.scheme == ShuangpinScheme::Full
                && !req.raw_input.contains('\'')
                && chime_phonetics::is_valid_syllable(&lower_raw);
            let eff_mode = if is_single_syl
                || (!req.left_context.trim().is_empty() && req.raw_input.len() <= 12)
            {
                ScoringMode::OpenContextual
            } else {
                req.scoring_mode
            };
            let lps = reranker.score_candidates_exact_bayesian_with_pinyin(
                &req.left_context,
                &req.app_bundle_id,
                &req.raw_input,
                req.scheme,
                &to_score,
                eff_mode,
            );
            for (txt, lp) in to_score.into_iter().zip(lps) {
                if lp.is_finite() {
                    ncands.push(NeuralBeamCandidate {
                        text: txt,
                        score: lp,
                        lm_log_prob: lp,
                        channel_cost: 0.0,
                        is_exact: true,
                        consumed_bytes: req.raw_input.len(),
                    });
                }
            }
            ncands.sort_by(|a, b| b.score.total_cmp(&a.score));
        }
    }
    ncands
}
