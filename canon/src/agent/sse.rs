//! SSE delta extractor — parses raw ChatGPT SSE frames into plain text.
//!
//! ChatGPT sends two shapes over /backend-api/f/conversation:
//!   append: {"p": "/message/content/parts/0", "o": "append", "v": "text"}
//!   patch:  {"p": "", "o": "patch", "v": [{...}, ...]}
//!
//! Group chat (calpico) WS frames are passed through as-is if they
//! do not match the SSE shape — Rust will handle them separately later.

use serde_json::Value;

/// Extract accumulative text from one raw SSE line or WS frame.
/// Returns None for [DONE], empty lines, non-content frames.
pub fn extract_sse_delta(raw: &str) -> Option<String> {
    let data = raw.strip_prefix("data: ").unwrap_or(raw).trim();

    if data == "[DONE]" || data.is_empty() {
        return None;
    }

    let v: Value = serde_json::from_str(data).ok()?;
    let op = v.get("o").and_then(|o| o.as_str())?;

    match op {
        "append" => v.get("v").and_then(|v| v.as_str()).map(|s| s.to_string()),
        "patch" => {
            let arr = v.get("v").and_then(|v| v.as_array())?;
            let mut out = String::new();
            for item in arr {
                if item.get("o").and_then(|o| o.as_str()) != Some("append") {
                    continue;
                }
                let p = item.get("p").and_then(|p| p.as_str()).unwrap_or("");
                if p.contains("parts") {
                    if let Some(s) = item.get("v").and_then(|v| v.as_str()) {
                        out.push_str(s);
                    }
                }
            }
            if out.is_empty() { None } else { Some(out) }
        }
        _ => None,
    }
}

/// Returns true if this raw line signals end of stream.
pub fn is_done(raw: &str) -> bool {
    let data = raw.strip_prefix("data: ").unwrap_or(raw).trim();
    data == "[DONE]"
}
