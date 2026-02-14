//! LLM provider bridge — connects AgentCallInput to ChatGPT via the
//! chromium_messenger daemon.
//!
//! Flow:
//!   AgentCallInput
//!     → format_prompt()          — serialize ir_slice + stage + predecessors
//!     → write to Unix socket     — /tmp/chromium_messenger.sock as LocalMessage
//!     → poll BASE_DIR            — wait for new CapturedMessage on disk
//!     → parse_response()         — extract ```json block from responseText
//!     → AgentCallOutput
//!
//! The daemon must be running before any call is made.
//! Uses temporary-chat so each session is fully stateless.
//! No LLM calls inside this file — it is pure I/O plumbing.

use std::fs;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use super::call::{AgentCallId, AgentCallInput, AgentCallOutput};

/// Path to the chromium_messenger daemon Unix socket.
const DAEMON_SOCKET: &str = "/tmp/chromium_messenger.sock";

/// Base directory where the daemon writes captured messages.
const BASE_DIR: &str =
    "/home/cicero-arch-omen/ai_sandbox/chatgpt-website/chatgpt_messages";

/// How long to wait for a response before timing out.
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(120);

/// How often to poll the filesystem for a new response file.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Error type for LLM provider operations.
#[derive(Debug)]
pub enum LlmProviderError {
    /// Could not connect to or write to the daemon socket.
    SocketError(String),
    /// Timed out waiting for a response from the daemon.
    Timeout,
    /// The response file existed but contained no `responseText`.
    EmptyResponse,
    /// The responseText did not contain a valid ```json block.
    NoJsonBlock,
    /// The ```json block could not be parsed as a JSON object.
    MalformedJson(String),
    /// Filesystem I/O error while polling for response.
    Io(String),
}

impl std::fmt::Display for LlmProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmProviderError::SocketError(e) => write!(f, "daemon socket error: {e}"),
            LlmProviderError::Timeout => write!(f, "timeout waiting for LLM response"),
            LlmProviderError::EmptyResponse => write!(f, "response file had no responseText"),
            LlmProviderError::NoJsonBlock => {
                write!(f, "responseText contained no ```json block")
            }
            LlmProviderError::MalformedJson(e) => write!(f, "json block parse error: {e}"),
            LlmProviderError::Io(e) => write!(f, "filesystem poll error: {e}"),
        }
    }
}

impl std::error::Error for LlmProviderError {}

/// Send an AgentCallInput to ChatGPT via the daemon and return the response
/// as an AgentCallOutput.
///
/// Blocking — caller should run this on a dedicated thread if needed.
pub fn call_llm(input: &AgentCallInput) -> Result<AgentCallOutput, LlmProviderError> {
    let send_time = system_time_ms();

    // 1. Format the prompt.
    let prompt = format_prompt(input);

    // 2. Send to daemon via Unix socket.
    send_to_daemon(&prompt)?;

    // 3. Poll filesystem for a new response file.
    let (conversation_id, message_id, response_text) = poll_for_response(send_time)?;

    // 4. Parse the JSON payload block from responseText.
    let payload = parse_json_block(&response_text)?;

    // 5. Extract optional proof_id and delta_ids from payload.
    let proof_id = payload
        .get("proof_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let emitted_delta_ids = payload
        .get("emitted_delta_ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Ok(AgentCallOutput {
        call_id: format!("llm-{}-{}", conversation_id, message_id),
        node_id: input.node_id.clone(),
        payload,
        proof_id,
        emitted_delta_ids,
        stage: input.stage.clone(),
    })
}

// ---------------------------------------------------------------------------
// Prompt formatter
// ---------------------------------------------------------------------------

/// Formats an AgentCallInput into a structured prompt string.
///
/// The prompt instructs the LLM to reply with a single ```json block
/// containing the required fields for the current pipeline stage.
fn format_prompt(input: &AgentCallInput) -> String {
    let stage = format!("{:?}", input.stage);
    let ir_slice = serde_json::to_string_pretty(&input.ir_slice)
        .unwrap_or_else(|_| "{}".to_string());

    let predecessors = if input.predecessor_outputs.is_empty() {
        "none".to_string()
    } else {
        input
            .predecessor_outputs
            .iter()
            .map(|o| {
                format!(
                    "node={} stage={:?} payload={}",
                    o.node_id,
                    o.stage,
                    serde_json::to_string(&o.payload).unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let required_fields = required_fields_for_stage(&stage);

    format!(
        "You are a capability node in a Tier 7 self-modifying AI architecture.\n\
         Your role: {stage}\n\
         Node ID: {node_id}\n\
         Call ID: {call_id}\n\
         \n\
         ## IR Slice (your read-only view of the system)\n\
         ```json\n{ir_slice}\n```\n\
         \n\
         ## Predecessor outputs\n\
         {predecessors}\n\
         \n\
         ## Your task\n\
         Analyse the IR slice and produce a response for the {stage} stage.\n\
         \n\
         ## Required response format\n\
         Reply with ONLY a single ```json block containing these fields:\n\
         {required_fields}\n\
         \n\
         Do not include any text outside the ```json block.",
        stage = stage,
        node_id = input.node_id,
        call_id = input.call_id,
        ir_slice = ir_slice,
        predecessors = predecessors,
        required_fields = required_fields,
    )
}

/// Returns the required JSON fields description for each pipeline stage.
fn required_fields_for_stage(stage: &str) -> &'static str {
    match stage {
        "Observe" => r#"{
  "observation": "<string: summary of IR artifacts observed>",
  "hottest_modules": ["<module_id>", ...],
  "issues_found": ["<string>", ...]
}"#,
        "Learn" => r#"{
  "rationale": "<string: reasoning about what should change>",
  "proposed_kind": "<split_module|merge_modules|move_artifact|rename_artifact|add_edge|remove_edge>",
  "target_artifact_id": "<string>",
  "destination_id": "<string or null>"
}"#,
        "Decide" => r#"{
  "proof_id": "<string: proof identifier>",
  "verification_notes": "<string>"
}"#,
        "Plan" => r#"{
  "decision": "<accept|reject>",
  "admission_id": "<string: admission identifier>",
  "rationale": "<string>"
}"#,
        _ => r#"{
  "observation": "<string>",
  "rationale": "<string>"
}"#,
    }
}

// ---------------------------------------------------------------------------
// Daemon socket write
// ---------------------------------------------------------------------------

fn send_to_daemon(prompt: &str) -> Result<(), LlmProviderError> {
    let msg = json!({
        "conversation_id": null,
        "content": prompt,
    });
    let payload = serde_json::to_string(&msg)
        .map_err(|e| LlmProviderError::SocketError(e.to_string()))?;

    let mut stream = UnixStream::connect(DAEMON_SOCKET)
        .map_err(|e| LlmProviderError::SocketError(e.to_string()))?;

    stream
        .write_all(payload.as_bytes())
        .map_err(|e| LlmProviderError::SocketError(e.to_string()))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Filesystem poller
// ---------------------------------------------------------------------------

/// Polls BASE_DIR for a new `<message_id>.json` file whose timestamp
/// is >= send_time. Returns (conversation_id, message_id, response_text).
fn poll_for_response(
    send_time_ms: u64,
) -> Result<(String, String, String), LlmProviderError> {
    let base = Path::new(BASE_DIR);
    let deadline = std::time::Instant::now() + RESPONSE_TIMEOUT;

    while std::time::Instant::now() < deadline {
        if let Ok(conv_dirs) = fs::read_dir(base) {
            for conv_entry in conv_dirs.flatten() {
                let conv_path = conv_entry.path();
                if !conv_path.is_dir() {
                    continue;
                }
                let conversation_id = conv_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();

                if let Ok(msg_dirs) = fs::read_dir(&conv_path) {
                    for msg_entry in msg_dirs.flatten() {
                        let msg_path = msg_entry.path();
                        if !msg_path.is_dir() {
                            continue;
                        }
                        let message_id = msg_path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("")
                            .to_string();

                        let json_path = msg_path.join(format!("{}.json", message_id));
                        if !json_path.is_file() {
                            continue;
                        }

                        // Check file modification time >= send_time.
                        let mtime_ms = file_mtime_ms(&json_path)
                            .unwrap_or(0);
                        if mtime_ms < send_time_ms {
                            continue;
                        }

                        // Read and check responseText is populated.
                        let data = fs::read(&json_path)
                            .map_err(|e| LlmProviderError::Io(e.to_string()))?;
                        let value: Value = serde_json::from_slice(&data)
                            .map_err(|e| LlmProviderError::Io(e.to_string()))?;

                        if let Some(text) = value
                            .get("responseText")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.trim().is_empty())
                        {
                            return Ok((conversation_id, message_id, text.to_string()));
                        }
                    }
                }
            }
        }
        std::thread::sleep(POLL_INTERVAL);
    }

    Err(LlmProviderError::Timeout)
}

// ---------------------------------------------------------------------------
// Response parser
// ---------------------------------------------------------------------------

/// Extracts the first ```json ... ``` block from a response string
/// and parses it as a JSON Value.
fn parse_json_block(text: &str) -> Result<Value, LlmProviderError> {
    // Find opening fence.
    let open = text.find("```json")
        .or_else(|| text.find("```JSON"))
        .ok_or(LlmProviderError::NoJsonBlock)?;

    let after_fence = &text[open + 7..]; // skip "```json"

    // Find closing fence.
    let close = after_fence
        .find("```")
        .ok_or(LlmProviderError::NoJsonBlock)?;

    let json_str = after_fence[..close].trim();

    serde_json::from_str(json_str)
        .map_err(|e| LlmProviderError::MalformedJson(e.to_string()))
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn system_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn file_mtime_ms(path: &PathBuf) -> Option<u64> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}
