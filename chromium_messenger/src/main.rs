use serde::Deserialize;
use serde_json::{json, Value};
use std::io::Read;
use std::net::TcpStream;
use std::time::Duration;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{connect, Message, WebSocket};
use url::Url;

use std::os::unix::net::UnixListener;
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::thread;

mod incoming;
mod outgoing;

const LOCAL_SOCKET_PATH: &str = "/tmp/chromium_messenger.sock";

#[derive(Debug, Deserialize, Clone)]
pub struct TargetDescriptor {
    id: Option<String>,
    #[serde(rename = "type")]
    kind: String,
    url: String,
    title: Option<String>,
    #[serde(rename = "webSocketDebuggerUrl")]
    web_socket_debugger_url: Option<String>,
}

pub struct ChromeConnection {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    next_id: u64,
}

enum DaemonEvent {
    ReceiveChromiumMessage(serde_json::Value),
    ReceiveLocalMessage(incoming::LocalMessage),
}

impl ChromeConnection {
    fn connect(
        chrome_port: u16,
        conversation_id: Option<&str>,
        new_tab: bool,
    ) -> Result<(Self, Option<String>), Box<dyn std::error::Error>> {
        let json_url = format!("http://localhost:{}/json", chrome_port);
        let response: Vec<TargetDescriptor> = ureq::get(&json_url)
            .timeout(Duration::from_secs(2))
            .call()?
            .into_json()?;

        let chatgpt_tab = if let Some(conv_id) = conversation_id {
            let conversation_path = format!("/c/{}", conv_id);
            if let Some(tab) = response
                .iter()
                .find(|target| target.kind == "page" && target.url.contains(&conversation_path))
                .cloned()
            {
                tab
            } else {
                outgoing::open_conversation_tab(chrome_port, conv_id)?
            }
        } else if new_tab {
            outgoing::open_url_tab(chrome_port, "https://chatgpt.com/")?
        } else {
            response
                .iter()
                .find(|target| {
                    target.kind == "page"
                        && (target.url.contains("chatgpt.com")
                            || target.url.contains("chat.openai.com"))
                })
                .cloned()
                .ok_or(
                    "No matching ChatGPT tab found. Please open https://chatgpt.com in Chrome.",
                )?
        };

        let ws_url = chatgpt_tab
            .web_socket_debugger_url
            .ok_or("ChatGPT tab has no WebSocket debugger URL")?;

        println!(
            "Found ChatGPT tab: {:?}",
            chatgpt_tab.title.as_deref().unwrap_or("(untitled)")
        );
        println!("URL: {}", chatgpt_tab.url);

        let url = Url::parse(&ws_url)?;
        let (mut socket, _) = connect(url)?;

        // Set non-blocking so the daemon event loop can interleave
        // local socket messages with Chromium events.
        match socket.get_ref() {
            tungstenite::stream::MaybeTlsStream::Plain(tcp) => {
                tcp.set_nonblocking(true)?;
            }
            _ => {}
        }

        Ok((Self { socket, next_id: 1 }, chatgpt_tab.id.clone()))
    }

    pub fn send_command(
        &mut self,
        method: &str,
        params: Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let id = self.next_id;
        self.next_id += 1;
        let payload = json!({
            "id": id,
            "method": method,
            "params": params,
        });
        self.socket.send(Message::Text(payload.to_string()))?;
        Ok(())
    }

    fn wait_for_page_load(&mut self, timeout_ms: u64) -> Result<(), Box<dyn std::error::Error>> {
        println!("Waiting for page load...");
        let start = std::time::Instant::now();
        let timeout = Duration::from_millis(timeout_ms);

        loop {
            if start.elapsed() > timeout {
                return Err("Timeout waiting for page load".into());
            }

            if let Some(event) = self.read_event()? {
                if let Some(method) = event.get("method").and_then(|v| v.as_str()) {
                    if method == "Page.loadEventFired" {
                        println!("Page load complete");
                        return Ok(());
                    }
                }
            }

            std::thread::sleep(Duration::from_millis(50));
        }
    }

    pub fn read_event(&mut self) -> Result<Option<Value>, Box<dyn std::error::Error>> {
        match self.socket.read() {
            Ok(Message::Text(text)) => Ok(Some(serde_json::from_str(&text)?)),
            Err(tungstenite::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::WouldBlock => {
                Ok(None)
            }
            Err(e) => Err(Box::new(e)),
            _ => Ok(None),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(|s| s.as_str()) == Some("--daemon") {
        let chrome_port = args
            .get(2)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(9222);

        if let Err(e) = run_daemon(chrome_port) {
            eprintln!("Daemon error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    if args.len() < 2 {
        eprintln!(
            "Usage: {} <message_text> [chrome_port] [conversation_id]",
            args[0]
        );
        eprintln!("       {} --stdin [chrome_port] [conversation_id]", args[0]);
        eprintln!(
            "       {} --no-wait --stdin [chrome_port] [conversation_id]",
            args[0]
        );
        eprintln!("       {} --new-tab <message_text> [chrome_port]", args[0]);
        eprintln!("       {} --daemon [chrome_port]", args[0]);
        eprintln!("  chrome_port defaults to 9222");
        eprintln!("  conversation_id targets a specific tab via https://chatgpt.com/c/<id>");
        std::process::exit(1);
    }

    let mut no_wait = false;
    let mut new_tab = false;
    let mut use_stdin = false;
    let mut positionals: Vec<String> = Vec::new();
    let mut iter = args.into_iter().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--no-wait" => no_wait = true,
            "--new-tab" => new_tab = true,
            "--stdin" | "-" => use_stdin = true,
            _ => positionals.push(arg),
        }
    }

    let message_text = if use_stdin {
        let mut buffer = String::new();
        if let Err(err) = std::io::stdin().read_to_string(&mut buffer) {
            eprintln!("Failed to read stdin: {err}");
            std::process::exit(1);
        }
        buffer
    } else if let Some(first) = positionals.first() {
        first.clone()
    } else {
        eprintln!("Missing message text or --stdin");
        std::process::exit(1);
    };

    let mut positional_idx = if use_stdin { 0 } else { 1 };
    let chrome_port = positionals
        .get(positional_idx)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(9222);
    if positionals
        .get(positional_idx)
        .and_then(|s| s.parse::<u16>().ok())
        .is_some()
    {
        positional_idx += 1;
    }
    let conversation_id = positionals.get(positional_idx).map(|s| s.as_str());

    if let Err(e) = run_messenger(
        &message_text,
        chrome_port,
        conversation_id,
        no_wait,
        new_tab,
    ) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_daemon(chrome_port: u16) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Starting chromium_messenger daemon on port {}...",
        chrome_port
    );

    let (mut conn, _) = ChromeConnection::connect(chrome_port, None, false)?;

    conn.send_command("Runtime.enable", json!({}))?;
    std::thread::sleep(Duration::from_millis(100));

    incoming::setup_response_listener(&mut conn)?;

    use std::sync::mpsc::channel;
    let (tx, rx) = channel::<DaemonEvent>();

    // Start local socket listener
    spawn_local_listener(tx.clone())?;

    loop {
        // 1. Handle Chromium → daemon (already works)
        if let Some(event) = conn.read_event()? {
            if event.get("method").and_then(|m| m.as_str()) == Some("Runtime.bindingCalled") {
                let payload = event["params"]["payload"].as_str().unwrap_or("");
                if let Ok(value) = serde_json::from_str::<Value>(payload) {
                    // persist all Chromium messages
                    if let Some(msg_id) = value.get("messageId").and_then(|v| v.as_str()) {
                        println!("[daemon] Chromium message {}", msg_id);
                    }
                    incoming::save_message_from_value(value)?;
                }
            }
        }

        // 2. Handle Local → daemon → Chromium
        while let Ok(event) = rx.try_recv() {
            if let DaemonEvent::ReceiveLocalMessage(msg) = event {
                println!("[daemon] Local message → Chromium");
                outgoing::send_message(&mut conn, &msg.content)?;
            }
        }

        std::thread::sleep(Duration::from_millis(10));
    }
}

fn run_messenger(
    message_text: &str,
    chrome_port: u16,
    conversation_id: Option<&str>,
    no_wait: bool,
    new_tab: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Connecting to Chrome on port {}...", chrome_port);
    let (mut conn, target_id) = ChromeConnection::connect(chrome_port, conversation_id, new_tab)?;

    conn.send_command("Runtime.enable", json!({}))?;
    std::thread::sleep(Duration::from_millis(100));

    if new_tab {
        println!("New tab detected, enabling Page domain...");
        conn.send_command("Page.enable", json!({}))?;
        std::thread::sleep(Duration::from_millis(100));
        conn.wait_for_page_load(10_000)?;
        std::thread::sleep(Duration::from_millis(500));
    }

    if let Some(target_id) = target_id {
        conn.send_command("Target.activateTarget", json!({ "targetId": target_id }))?;
        conn.send_command("Page.bringToFront", json!({}))?;
        std::thread::sleep(Duration::from_millis(100));
    }

    outgoing::inject_request_hook(&mut conn)?;

    if !no_wait {
        incoming::setup_response_listener(&mut conn)?;
    }

    let send_timestamp = outgoing::send_message(&mut conn, message_text)?;

    if no_wait {
        return Ok(());
    }

    // run_messenger is send-only; daemon owns persistence
    let (_tx, rx) = mpsc::channel::<Value>();
    incoming::wait_for_response(&mut conn, send_timestamp, false, &_tx)
}
fn spawn_local_listener(tx: Sender<DaemonEvent>) -> std::io::Result<()> {
    let _ = std::fs::remove_file(LOCAL_SOCKET_PATH);
    let listener = UnixListener::bind(LOCAL_SOCKET_PATH)?;

    thread::spawn(move || {
        for stream in listener.incoming() {
            if let Ok(mut stream) = stream {
                use std::io::Read;
                let mut len_buf = [0u8; 4];
                if stream.read_exact(&mut len_buf).is_err() {
                    eprintln!("[listener] failed to read length prefix");
                    continue;
                }
                let msg_len = u32::from_le_bytes(len_buf) as usize;
                let mut body = vec![0u8; msg_len];
                if stream.read_exact(&mut body).is_err() {
                    eprintln!("[listener] failed to read body ({} bytes)", msg_len);
                    continue;
                }
                match serde_json::from_slice::<incoming::LocalMessage>(&body) {
                    Ok(msg) => {
                        let _ = tx.send(DaemonEvent::ReceiveLocalMessage(msg));
                    }
                    Err(e) => eprintln!("[listener] parse failed: {e}"),
                }
            }
        }
    });

    Ok(())
}
