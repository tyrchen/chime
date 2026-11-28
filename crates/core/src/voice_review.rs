//! One-shot review of an ASR final. Shares the keyboard model, not its composing sessions.
use std::{
    collections::HashMap,
    task::Poll,
    time::{Duration, Instant},
};

use crate::ChimeEngine;

/// Private reply ticket. Dropping it cancels interest in the result.
#[derive(Debug)]
pub struct VoiceReview {
    texts: Vec<String>,
    scores: flume::Receiver<Vec<f32>>,
    expires: Instant,
}

impl ChimeEngine {
    /// Dictionary-only suggestion. Long-word homophones go to the keyboard model only when it
    /// can actually score them; without a loaded model the plain dictionary match still applies.
    #[must_use]
    pub fn voice_lexicon_suggestion(&self, text: &str) -> Option<String> {
        let voice = &self.config.voice;
        if !voice.lexicon_suggestions {
            return None;
        }
        if voice.pinyin_rerank && self.async_reranker.is_some() {
            self.lexicon.voice_verse_suggestion(text)
        } else {
            self.lexicon.voice_suggestion(text)
        }
    }

    /// Submit local dictionary proposals for final-only scoring. Returns immediately if disabled,
    /// missing a model, no strong phonetic proposals, or no queue capacity.
    #[must_use]
    pub fn start_voice_review(&self, text: &str) -> Option<VoiceReview> {
        if !self.config.voice.lexicon_suggestions || !self.config.voice.pinyin_rerank {
            return None;
        }
        let worker = self.async_reranker.as_ref()?;
        let texts = self.lexicon.voice_pinyin_candidates(text);
        if texts.len() < 2 {
            return None;
        }
        let pinyin = if worker.is_pinyin_conditioned() {
            let han: String = text
                .chars()
                .filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c))
                .collect();
            chime_lexicon::base_builder::annotate_chinese_word(&han, &HashMap::new())?
                .replace(' ', "")
        } else {
            String::new()
        };
        let expires = Instant::now() + Duration::from_millis(350);
        let scores = worker.try_score_voice(&texts, pinyin, expires)?;
        Some(VoiceReview {
            texts,
            scores,
            expires,
        })
    }
}

impl VoiceReview {
    /// Nonblocking; never leaks a late result into another utterance.
    pub fn poll(&self) -> Poll<Option<&str>> {
        if Instant::now() >= self.expires {
            return Poll::Ready(None);
        }
        match self.scores.try_recv() {
            Ok(scores) if scores.len() == self.texts.len() => {
                Poll::Ready(winner(&scores).map(|i| self.texts[i].as_str()))
            }
            Err(flume::TryRecvError::Empty) => Poll::Pending,
            _ => Poll::Ready(None),
        }
    }
}

// Unnormalized full-utterance log likelihoods; all alternatives have identical character
// length. This is a conservative experiment threshold, not calibrated ASR confidence.
fn winner(scores: &[f32]) -> Option<usize> {
    if scores.len() < 2 || scores.iter().any(|s| !s.is_finite()) {
        return None;
    }
    let (best, &score) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    (best != 0
        && score - scores[0] >= 3.0
        && scores
            .iter()
            .enumerate()
            .all(|(i, s)| i == best || score - s >= 1.5))
    .then_some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_unique_clear_improvements() {
        assert_eq!(winner(&[-14., -10., -13.]), Some(1));
        for scores in [
            &[-10., -10., -13.],
            &[-14., -10., -11.],
            &[-10., -13., -14.],
            &[f32::NEG_INFINITY, -10., -15.],
        ] {
            assert_eq!(winner(scores), None);
        }
    }

    #[test]
    fn pending_timeout_and_disconnection_keep_original() {
        let (tx, scores) = flume::bounded(1);
        let mut ticket = VoiceReview {
            texts: vec!["原稿".into(), "候选".into()],
            scores,
            expires: Instant::now() + Duration::from_secs(1),
        };
        assert_eq!(ticket.poll(), Poll::Pending);
        drop(tx);
        assert_eq!(ticket.poll(), Poll::Ready(None));
        let (tx, rx) = flume::bounded(1);
        assert!(tx.send(vec![-20., -3.]).is_ok());
        ticket.scores = rx;
        ticket.expires = Instant::now();
        assert_eq!(ticket.poll(), Poll::Ready(None));
    }

    #[test]
    #[ignore = "set CHIME_VOICE_RESOURCES to the installed app's Contents/Resources to replay the \
                local MLX model"]
    fn replay_real_model() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let resources = std::env::var("CHIME_VOICE_RESOURCES")?;
        let mut engine =
            ChimeEngine::with_data_dir(dir.path(), Some(std::path::Path::new(&resources)))?;
        engine.config.voice.pinyin_rerank = true;
        assert!(engine.async_reranker.is_some());
        for (raw, expected) in [
            ("请打开浏览汽。", Some("请打开浏览器。")),
            ("请打开浏览器。", None),
            ("这个数剧库需要备份。", Some("这个数据库需要备份。")),
            ("这个数据库需要备份。", None),
            ("我们用计算鸡处理数据。", Some("我们用计算机处理数据。")),
            ("这里无山也无水。", None),
            ("我没有买 API 3.14。", None),
        ] {
            let before = Instant::now();
            let result = engine.start_voice_review(raw).and_then(|review| {
                loop {
                    match review.poll() {
                        Poll::Ready(v) => {
                            break v.map(str::to_owned);
                        }
                        Poll::Pending => std::thread::sleep(Duration::from_millis(5)),
                    }
                }
            });
            println!(
                "{raw} => {} ({} ms)",
                result.as_deref().unwrap_or("[保留原稿]"),
                before.elapsed().as_millis()
            );
            assert_eq!(result.as_deref(), expected, "{raw}");
        }
        Ok(())
    }
}
