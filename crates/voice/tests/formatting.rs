//! Exercise the shipped punctuation helper through the production Rust formatter.
use std::path::Path;

#[test]
#[ignore = "requires CHIME_TEST_APP pointing to a packaged arm64 Chime.app"]
fn local_punctuation_and_cleanup() -> Result<(), Box<dyn std::error::Error>> {
    let app = std::env::var("CHIME_TEST_APP")?;
    let contents = Path::new(&app).canonicalize()?.join("Contents");
    let text = "我没有同意取消明天的会议你别发通知";
    assert_eq!(
        chime_voice::format_final(&contents, text, true, false),
        "我没有同意取消明天的会议，你别发通知。"
    );
    let text = "嗯我觉得我觉得明天可以开会你有时间吗";
    let edited = chime_voice::format_final(&contents, text, true, true);
    assert_eq!(edited, "我觉得明天可以开会，你有时间吗？");
    Ok(())
}
