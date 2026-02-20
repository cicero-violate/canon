use crate::ChromeConnection;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Deserialize)]
pub struct LocalMessage {
    pub conversation_id: Option<String>,
    pub content: String,
}

const BASE_DIR: &str = "/home/cicero-arch-omen/ai_sandbox/chatgpt-website/chatgpt_messages";
const RESPONSE_JS: &str = include_str!("response_capture_sse.js");

#[derive(Debug, Serialize, Deserialize)]
pub struct CapturedMessage {
    #[serde(rename = "conversationId")]
    pub conversation_id: String,
    #[serde(rename = "messageId")]
    pub message_id: String,
    pub timestamp: u64,
    #[serde(rename = "responseText")]
    pub response_text: Option<String>,
    #[serde(rename = "sseEvents")]
    pub sse_events: Vec<Value>,
}

struct SseMessage {
    conversation_id: String,
    message_id: String,
    event: Value,
}

pub fn setup_response_listener(conn: &mut ChromeConnection) -> Result<(), Box<dyn std::error::Error>> {
    println!("Injecting response capture script...");

    // Register a CDP binding so JS can push messages directly to Rust
    conn.send_command(
        "Runtime.addBinding",
        json!({
            "name": "__chromiumMessenger"
        }),
    )?;

    conn.send_command(
        "Runtime.evaluate",
        json!({
            "expression": RESPONSE_JS,
            "awaitPromise": false,
        }),
    )?;
    std::thread::sleep(Duration::from_millis(200));

    let listener_script = r#"
        window.addEventListener("message", (event) => {
            if (event.data && event.data.type === "CHATGPT_MESSAGE_CAPTURE") {
                __chromiumMessenger(JSON.stringify(event.data));
            }
        });
    "#;

    conn.send_command(
        "Runtime.evaluate",
        json!({
            "expression": listener_script,
            "awaitPromise": false,
        }),
    )?;
    std::thread::sleep(Duration::from_millis(100));
    Ok(())
}

pub fn wait_for_response(conn: &mut ChromeConnection, send_timestamp: u64, persist: bool, tx: &std::sync::mpsc::Sender<serde_json::Value>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Waiting for response...");
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(60);
    let mut last_message_id: Option<String> = None;

    while start.elapsed() < timeout {
        if let Some(event) = conn.read_event()? {
            if event.get("method").and_then(|m| m.as_str()) == Some("Runtime.bindingCalled") {
                let payload = event["params"]["payload"].as_str().ok_or("missing binding payload")?;

                let value: Value = serde_json::from_str(payload)?;

                let _ = tx.send(value.clone());

                let msg_id = value["messageId"].as_str().unwrap_or("");
                let timestamp = value["timestamp"].as_u64().unwrap_or(0);

                if !msg_id.is_empty() && timestamp >= send_timestamp && last_message_id.as_deref() != Some(msg_id) {
                    let captured = CapturedMessage {
                        conversation_id: value["conversationId"].as_str().unwrap_or("").to_string(),
                        message_id: msg_id.to_string(),
                        timestamp,
                        response_text: value["responseText"].as_str().map(|s| s.to_string()),
                        sse_events: value["sseEvents"].as_array().cloned().unwrap_or_default(),
                    };

                    if persist {
                        save_message(&captured)?;
                    }
                    println!("Response captured successfully!");
                    return Ok(());
                }
            }
        }
    }

    Err("Timeout waiting for response".into())
}

fn save_message(msg: &CapturedMessage) -> Result<(), Box<dyn std::error::Error>> {
    save_sse_messages(msg)?;

    let path = PathBuf::from(BASE_DIR).join(&msg.conversation_id).join(&msg.message_id);

    println!("[chromium_messenger] Saving message: conv={}, msg={}", msg.conversation_id, msg.message_id);

    fs::create_dir_all(&path)?;

    let file_path = path.join(format!("{}.json", msg.message_id));
    println!("[chromium_messenger] Writing to: {}", file_path.display());

    write_json_atomic(&file_path, &serde_json::to_value(msg)?)?;

    println!("[chromium_messenger] Message saved successfully");

    notify_watch_service(&msg.conversation_id, &msg.message_id);

    Ok(())
}

fn save_sse_messages(msg: &CapturedMessage) -> Result<(), Box<dyn std::error::Error>> {
    let sse_messages = extract_sse_messages(&msg.sse_events, &msg.conversation_id);
    if sse_messages.is_empty() {
        return Ok(());
    }

    let mut seen = HashSet::new();
    for entry in sse_messages {
        if entry.message_id == msg.message_id {
            continue;
        }
        if !seen.insert(entry.message_id.clone()) {
            continue;
        }

        let path = PathBuf::from(BASE_DIR).join(&entry.conversation_id).join(&entry.message_id);
        let file_path = path.join(format!("{}.json", entry.message_id));
        if file_path.is_file() {
            continue;
        }

        fs::create_dir_all(&path)?;
        let json = json!({
            "conversationId": entry.conversation_id,
            "messageId": entry.message_id,
            "timestamp": msg.timestamp,
            "responseText": null,
            "sseEvents": [entry.event],
        });
        write_json_atomic(&file_path, &json)?;
    }

    Ok(())
}

fn extract_sse_messages(events: &[Value], fallback_conversation_id: &str) -> Vec<SseMessage> {
    let mut messages = Vec::new();
    for event in events {
        let message = match extract_event_message(event) {
            Some(message) => message,
            None => continue,
        };
        let message_id = match message.get("id").and_then(Value::as_str) {
            Some(id) => id.to_string(),
            None => continue,
        };

        // Skip context messages that are not actual conversation messages
        if let Some(content) = message.get("content") {
            if let Some(content_type) = content.get("content_type").and_then(Value::as_str) {
                // Skip user/model editable context messages
                if content_type == "user_editable_context" || content_type == "model_editable_context" {
                    continue;
                }
            }

            // Skip system messages with empty text content
            if let Some(author) = message.get("author") {
                if let Some(role) = author.get("role").and_then(Value::as_str) {
                    if role == "system" && content.get("content_type").and_then(Value::as_str) == Some("text") {
                        if let Some(parts) = content.get("parts").and_then(Value::as_array) {
                            let is_empty = parts.iter().all(|p| p.as_str().map(|s| s.trim().is_empty()).unwrap_or(true));
                            if is_empty {
                                continue;
                            }
                        }
                    }
                }
            }
        }

        let conversation_id = message.get("conversation_id").and_then(Value::as_str).unwrap_or(fallback_conversation_id).to_string();

        messages.push(SseMessage { conversation_id, message_id, event: event.clone() });
    }

    messages
}

fn extract_event_message(event: &Value) -> Option<Value> {
    if let Value::Object(map) = event {
        if let Some(message) = map.get("message") {
            return Some(message.clone());
        }
        if let Some(message) = map.get("input_message") {
            return Some(message.clone());
        }
        if let Some(value) = map.get("v") {
            if let Some(message) = value.get("message") {
                return Some(message.clone());
            }
            if let Some(message) = value.get("input_message") {
                return Some(message.clone());
            }
        }
    }
    None
}

fn write_json_atomic(path: &PathBuf, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let filename = path.file_name().and_then(|name| name.to_str()).ok_or("invalid json file name")?;
    let tmp_name = format!("{}.tmp.{}", filename, std::process::id());
    let tmp_path = path.with_file_name(tmp_name);
    let contents = serde_json::to_string_pretty(value)?;
    fs::write(&tmp_path, contents)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn notify_watch_service(conversation_id: &str, message_id: &str) {
    let socket_path = format!("/tmp/watch_{}.sock", conversation_id);
    match UnixStream::connect(&socket_path) {
        Ok(mut stream) => {
            if let Err(e) = writeln!(stream, "{}", message_id) {
                eprintln!("[chromium_messenger] Failed to write to socket: {}", e);
            } else {
                println!("[chromium_messenger] Notified watch_service: {}", message_id);
            }
        }
        Err(e) => {
            eprintln!("[chromium_messenger] Socket connection failed ({}): {}", socket_path, e);
            eprintln!("[chromium_messenger] watch_service may not be running");
        }
    }
}

pub fn save_message_from_value(value: Value) -> Result<(), Box<dyn std::error::Error>> {
    let captured = CapturedMessage {
        conversation_id: value["conversationId"].as_str().unwrap_or("").to_string(),
        message_id: value["messageId"].as_str().unwrap_or("").to_string(),
        timestamp: value["timestamp"].as_u64().unwrap_or(0),
        response_text: value["responseText"].as_str().map(|s| s.to_string()),
        sse_events: value["sseEvents"].as_array().cloned().unwrap_or_default(),
    };
    save_message(&captured)
}
