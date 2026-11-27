//! Client for self-hosted speech-recognition servers.

use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::{VoiceError, VoiceSettings, wav::encode_wav_pcm16};

/// Something that turns a mono PCM clip into text. Implemented by [`HttpTranscriber`]; tests use
/// in-memory fakes.
pub trait Transcriber: Send + Sync {
    /// Transcribes one utterance.
    ///
    /// # Errors
    ///
    /// Returns a [`VoiceError`] when the backend fails.
    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String, VoiceError>;
}

/// Sends WAV clips to an OpenAI-compatible `multipart/form-data` transcription endpoint.
///
/// Compatible servers include whisper.cpp `server` (`/inference`), faster-whisper / speaches,
/// vLLM, `LocalAI`, and `FunASR` / `SenseVoice` behind an OpenAI-compatible wrapper. The request
/// carries `file`, `model`, `response_format=json`, `temperature=0`, and the optional `language`
/// and `prompt` fields; the response may be `{"text": …}`, a `{"result": [{"text": …}]}` list, or
/// plain text.
pub struct HttpTranscriber {
    agent: ureq::Agent,
    endpoint: String,
    model: String,
    language: String,
    prompt: String,
    api_key: Option<String>,
}

impl fmt::Debug for HttpTranscriber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpTranscriber")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .field("language", &self.language)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish_non_exhaustive()
    }
}

impl HttpTranscriber {
    /// Builds a client from `settings`. `api_key` (if any) is sent as `Authorization: Bearer`.
    ///
    /// # Errors
    ///
    /// Returns [`VoiceError::InvalidSettings`] when the endpoint is not an HTTP(S) URL.
    pub fn new(settings: &VoiceSettings, api_key: Option<String>) -> Result<Self, VoiceError> {
        settings.validate()?;
        let tls = ureq::tls::TlsConfig::builder()
            .provider(ureq::tls::TlsProvider::NativeTls)
            .root_certs(ureq::tls::RootCerts::PlatformVerifier)
            .build();
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_millis(
                settings.request_timeout_ms.clamp(1_000, 300_000),
            )))
            .http_status_as_error(false)
            // The endpoint is usually loopback or LAN; never route audio through an env proxy.
            .proxy(None)
            .tls_config(tls)
            .user_agent("chime-voice")
            .build()
            .into();
        Ok(Self {
            agent,
            endpoint: settings.endpoint.trim().to_string(),
            model: settings.model.trim().to_string(),
            language: settings.language.trim().to_string(),
            prompt: settings.prompt.trim().to_string(),
            api_key: api_key.filter(|k| !k.trim().is_empty()),
        })
    }
}

impl Transcriber for HttpTranscriber {
    fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<String, VoiceError> {
        let wav = encode_wav_pcm16(samples, sample_rate);
        let boundary = new_boundary();
        let mut fields: Vec<(&str, &str)> = vec![("response_format", "json"), ("temperature", "0")];
        if !self.model.is_empty() {
            fields.push(("model", &self.model));
        }
        if !self.language.is_empty() {
            fields.push(("language", &self.language));
        }
        if !self.prompt.is_empty() {
            fields.push(("prompt", &self.prompt));
        }
        let body = build_multipart_body(&boundary, &fields, "audio.wav", "audio/wav", &wav);

        let mut req = self.agent.post(&self.endpoint).header(
            "Content-Type",
            format!("multipart/form-data; boundary={boundary}"),
        );
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key.trim()));
        }
        let mut resp = req
            .send(&body[..])
            .map_err(|e| VoiceError::Transport(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| VoiceError::Transport(e.to_string()))?;
        if !(200..300).contains(&status) {
            let snippet: String = text.chars().take(240).collect();
            return Err(VoiceError::Status {
                status,
                body: snippet,
            });
        }
        parse_transcript(&text)
    }
}

fn new_boundary() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("----chime-voice-{nanos:x}-{n:x}")
}

/// Builds a `multipart/form-data` body with text `fields` followed by one file part named `file`.
#[must_use]
pub fn build_multipart_body(
    boundary: &str,
    fields: &[(&str, &str)],
    file_name: &str,
    file_mime: &str,
    file_bytes: &[u8],
) -> Vec<u8> {
    let mut body = Vec::with_capacity(file_bytes.len() + 512);
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; \
             filename=\"{file_name}\"\r\nContent-Type: {file_mime}\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(file_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

/// Extracts the transcript from a server response body.
///
/// # Errors
///
/// Returns [`VoiceError::Response`] for JSON bodies without a recognizable text field.
pub fn parse_transcript(body: &str) -> Result<String, VoiceError> {
    let trimmed = body.trim();
    let Ok(json) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        // `response_format=text` servers reply with the bare transcript.
        return Ok(trimmed.to_string());
    };
    if let Some(s) = json.as_str() {
        return Ok(s.trim().to_string());
    }
    if let Some(s) = json.get("text").and_then(serde_json::Value::as_str) {
        return Ok(s.trim().to_string());
    }
    for key in ["result", "results", "segments"] {
        if let Some(items) = json.get(key).and_then(serde_json::Value::as_array) {
            let parts: Vec<&str> = items
                .iter()
                .filter_map(|it| it.get("text").and_then(serde_json::Value::as_str))
                .collect();
            if !parts.is_empty() {
                return Ok(crate::join_segments(parts.iter().copied()));
            }
        }
    }
    if let Some(msg) = json.get("error").map(|e| {
        e.get("message")
            .and_then(serde_json::Value::as_str)
            .map_or_else(|| e.to_string(), str::to_string)
    }) {
        return Err(VoiceError::Response(msg));
    }
    let snippet: String = trimmed.chars().take(160).collect();
    Err(VoiceError::Response(snippet))
}

#[cfg(test)]
mod tests {
    use std::{
        io::{BufRead, BufReader, Read, Write},
        net::TcpListener,
        sync::mpsc,
        thread,
    };

    use super::*;

    #[test]
    fn parses_common_response_shapes() {
        assert_eq!(
            parse_transcript(r#"{"text":" 你好 "}"#).ok().as_deref(),
            Some("你好")
        );
        assert_eq!(
            parse_transcript(r#"{"result":[{"key":"a","text":"第一句"},{"text":"第二句"}]}"#)
                .ok()
                .as_deref(),
            Some("第一句第二句")
        );
        assert_eq!(parse_transcript("纯文本\n").ok().as_deref(), Some("纯文本"));
        assert!(parse_transcript(r#"{"error":{"message":"model not found"}}"#).is_err());
        assert!(parse_transcript(r#"{"foo":1}"#).is_err());
    }

    #[test]
    fn multipart_body_layout() {
        let body = build_multipart_body("B", &[("model", "m")], "a.wav", "audio/wav", b"XYZ");
        let s = String::from_utf8_lossy(&body);
        assert!(
            s.starts_with("--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n")
        );
        assert!(s.contains(
            "name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nXYZ\r\n--B--\r\n"
        ));
    }

    /// Serves exactly one HTTP request, returning the raw request bytes through the channel.
    fn one_shot_server(
        status: &'static str,
        reply: &'static str,
    ) -> (String, mpsc::Receiver<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| std::process::abort());
        let addr = listener
            .local_addr()
            .unwrap_or_else(|_| std::process::abort());
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let Ok((stream, _)) = listener.accept() else {
                return;
            };
            let mut reader = BufReader::new(stream);
            let mut raw = Vec::new();
            let mut content_len = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                raw.extend_from_slice(line.as_bytes());
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_len = v.trim().parse().unwrap_or(0);
                }
                if line == "\r\n" {
                    break;
                }
            }
            let mut body = vec![0u8; content_len];
            let _ = reader.read_exact(&mut body);
            raw.extend_from_slice(&body);
            let mut stream = reader.into_inner();
            let resp = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: \
                 {}\r\nConnection: close\r\n\r\n{reply}",
                reply.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = tx.send(raw);
        });
        (format!("http://{addr}/v1/audio/transcriptions"), rx)
    }

    #[test]
    fn http_round_trip_against_local_server() {
        let (endpoint, rx) = one_shot_server("200 OK", r#"{"text":"测试成功"}"#);
        let settings = VoiceSettings {
            endpoint,
            model: "whisper-large-v3".into(),
            language: "zh".into(),
            prompt: "简体中文".into(),
            ..VoiceSettings::default()
        };
        let client = HttpTranscriber::new(&settings, Some("sk-local".into()))
            .unwrap_or_else(|_| std::process::abort());
        let text = client.transcribe(&[0.0; 1600], 16_000);
        assert_eq!(text.as_deref(), Ok("测试成功"));

        let raw = rx.recv().unwrap_or_default();
        let req = String::from_utf8_lossy(&raw);
        assert!(req.starts_with("POST /v1/audio/transcriptions HTTP/1.1"));
        assert!(
            req.contains("authorization: Bearer sk-local")
                || req.contains("Authorization: Bearer sk-local")
        );
        assert!(req.contains("name=\"model\"\r\n\r\nwhisper-large-v3"));
        assert!(req.contains("name=\"language\"\r\n\r\nzh"));
        assert!(req.contains("name=\"prompt\"\r\n\r\n简体中文"));
        assert!(req.contains("RIFF"));
    }

    #[test]
    fn http_error_status_is_reported() {
        let (endpoint, _rx) = one_shot_server("503 Service Unavailable", r#"{"error":"loading"}"#);
        let settings = VoiceSettings {
            endpoint,
            ..VoiceSettings::default()
        };
        let client =
            HttpTranscriber::new(&settings, None).unwrap_or_else(|_| std::process::abort());
        let res = client.transcribe(&[0.0; 160], 16_000);
        assert!(
            matches!(&res, Err(VoiceError::Status { status: 503, body }) if body.contains("loading")),
            "unexpected {res:?}"
        );
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let client = HttpTranscriber::new(&VoiceSettings::default(), Some("secret-key".into()))
            .unwrap_or_else(|_| std::process::abort());
        assert!(!format!("{client:?}").contains("secret-key"));
    }
}
