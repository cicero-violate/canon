use crate::{ChromeConnection, TargetDescriptor};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;
use tungstenite::{connect, Message as WsMessage};
use url::form_urlencoded::Serializer;
use url::Url;

const REQUEST_JS: &str = include_str!("request_message_dom.js");
const REQUEST_HOOK_JS: &str = include_str!("request_hook.js");

pub fn inject_request_hook(conn: &mut ChromeConnection) -> Result<(), Box<dyn std::error::Error>> {
    println!("Injecting request hook...");
    conn.send_command(
        "Runtime.evaluate",
        json!({
            "expression": REQUEST_HOOK_JS,
            "awaitPromise": false,
        }),
    )?;
    std::thread::sleep(Duration::from_millis(100));
    Ok(())
}

pub fn send_message(conn: &mut ChromeConnection, message_text: &str) -> Result<u64, Box<dyn std::error::Error>> {
    let send_timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis() as u64;

    println!("Sending message: '{}'", message_text);
    let request_call = format!("({})({{{}: {}}})", REQUEST_JS, "text", serde_json::to_string(message_text)?);

    conn.send_command(
        "Runtime.evaluate",
        json!({
            "expression": request_call,
            "awaitPromise": true,
        }),
    )?;

    Ok(send_timestamp)
}

pub fn open_conversation_tab(chrome_port: u16, conversation_id: &str) -> Result<TargetDescriptor, Box<dyn std::error::Error>> {
    let base_url = format!("https://chatgpt.com/c/{}", conversation_id);
    open_url_tab(chrome_port, &base_url)
}

pub fn open_url_tab(chrome_port: u16, base_url: &str) -> Result<TargetDescriptor, Box<dyn std::error::Error>> {
    let query = Serializer::new(String::new()).append_pair("url", &base_url).finish();
    let new_url = format!("http://localhost:{}/json/new?{}", chrome_port, query);
    let response = ureq::get(&new_url).timeout(Duration::from_secs(2)).call();
    let tab: TargetDescriptor = match response {
        Ok(resp) => resp.into_json()?,
        Err(ureq::Error::Status(405, _)) => match ureq::post(&new_url).timeout(Duration::from_secs(2)).call() {
            Ok(resp) => resp.into_json()?,
            Err(_) => open_conversation_tab_via_browser(chrome_port, &base_url)?,
        },
        Err(_) => open_conversation_tab_via_browser(chrome_port, &base_url)?,
    };
    Ok(tab)
}

fn open_conversation_tab_via_browser(chrome_port: u16, url: &str) -> Result<TargetDescriptor, Box<dyn std::error::Error>> {
    let version_url = format!("http://localhost:{}/json/version", chrome_port);
    let version: Value = ureq::get(&version_url).timeout(Duration::from_secs(2)).call()?.into_json()?;
    let ws_url = version.get("webSocketDebuggerUrl").and_then(Value::as_str).ok_or("Missing browser WebSocket debugger URL")?;
    let (mut socket, _) = connect(Url::parse(ws_url)?)?;
    let payload = json!({
        "id": 1,
        "method": "Target.createTarget",
        "params": { "url": url }
    });
    socket.send(WsMessage::Text(payload.to_string()))?;
    let mut target_id: Option<String> = None;
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(2) {
        match socket.read() {
            Ok(WsMessage::Text(text)) => {
                if let Ok(value) = serde_json::from_str::<Value>(&text) {
                    if value.get("id").and_then(Value::as_u64) == Some(1) {
                        target_id = value.get("result").and_then(|r| r.get("targetId")).and_then(Value::as_str).map(|s| s.to_string());
                        break;
                    }
                }
            }
            Ok(_) => {}
            Err(tungstenite::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(err) => return Err(Box::new(err)),
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let target_id = target_id.ok_or("Failed to create target")?;
    let list_url = format!("http://localhost:{}/json", chrome_port);
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(2) {
        let targets: Vec<TargetDescriptor> = ureq::get(&list_url).timeout(Duration::from_secs(2)).call()?.into_json()?;
        if let Some(tab) = targets.into_iter().find(|target| target.id.as_deref() == Some(&target_id)) {
            return Ok(tab);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err("Created target not found in target list".into())
}
