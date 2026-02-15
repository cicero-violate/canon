// src/background.ts — service worker
// Plain JSON over WebSocket to Rust at :8787.
// Rust → EXT: OPEN_TAB | CLOSE_TAB | TURN
// EXT → Rust: TAB_OPENED | TAB_CLOSED | CHUNK

const RUST_WS = "ws://127.0.0.1:8787";
const RECONNECT_DELAY_MS = 2000;
const CHATGPT_URL = "https://chatgpt.com";

// tabId → true (tracked open ChatGPT tabs)
const liveTabs = new Set<number>();

let ws: WebSocket | null = null;
let wsReady = false;
const sendQueue: string[] = [];

// ─────────────────────────────────────────────
// WebSocket to Rust
// ─────────────────────────────────────────────
function sendToRust(msg: object) {
  const frame = JSON.stringify(msg);
  if (wsReady && ws) {
    ws.send(frame);
  } else {
    sendQueue.push(frame);
  }
}

function connectRust() {
  if (
    ws &&
    (ws.readyState === WebSocket.OPEN ||
      ws.readyState === WebSocket.CONNECTING)
  ) {
    return;
  }

  ws = new WebSocket(RUST_WS);

  ws.onopen = () => {
    console.log("[BG] WS open");
    wsReady = true;
    while (sendQueue.length) {
      ws!.send(sendQueue.shift()!);
    }
  };

  ws.onmessage = (ev: MessageEvent) => {
    let msg: any;
    try {
      msg = JSON.parse(ev.data as string);
    } catch (e) {
      console.warn("[BG] WS non-JSON frame:", ev.data);
      return;
    }

    const type: string = msg?.type ?? "";
    console.log("[BG] ← Rust:", type, msg);

    if (type === "OPEN_TAB") {
      openChatGPTTab();
      return;
    }

    if (type === "CLOSE_TAB") {
      const tabId: number = msg.tabId;
      if (tabId) {
        chrome.tabs.remove(tabId, () => {
          void chrome.runtime.lastError;
        });
      }
      return;
    }

    if (type === "TURN") {
      const tabId: number = msg.tabId;
      const text: string = msg.text ?? "";
      const target = tabId ?? firstLiveTab();
      if (target == null) {
        console.error("[BG] TURN: no live tab");
        return;
      }
      chrome.tabs.sendMessage(
        target,
        { type: "TURN", text },
        () => void chrome.runtime.lastError,
      );
      return;
    }

    console.warn("[BG] unknown Rust message type:", type);
  };

  ws.onclose = () => {
    console.log("[BG] WS closed — reconnecting in", RECONNECT_DELAY_MS, "ms");
    wsReady = false;
    ws = null;
    setTimeout(connectRust, RECONNECT_DELAY_MS);
  };

  ws.onerror = () => {
    console.warn("[BG] WS error");
    ws?.close();
  };
}

// ─────────────────────────────────────────────
// Tab helpers
// ─────────────────────────────────────────────
function firstLiveTab(): number | undefined {
  return liveTabs.values().next().value;
}

function openChatGPTTab() {
  chrome.tabs.create({ url: CHATGPT_URL }, (tab) => {
    if (!tab?.id) return;
    const newTabId = tab.id;

    // Wait for the tab to finish loading before reporting ready
    const listener = (tabId: number, changeInfo: any) => {
      if (tabId !== newTabId || changeInfo.status !== "complete") return;
      chrome.tabs.onUpdated.removeListener(listener);
      sendToRust({ type: "TAB_OPENED", tabId: newTabId });
    };

    chrome.tabs.onUpdated.addListener(listener);
  });
}

// ─────────────────────────────────────────────
// Track ChatGPT tabs
// ─────────────────────────────────────────────
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete" || !tab?.url) return;
  const isChatGPT =
    tab.url.includes("chatgpt.com") || tab.url.includes("chat.openai.com");
  if (isChatGPT) {
    liveTabs.add(tabId);
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  if (liveTabs.has(tabId)) {
    liveTabs.delete(tabId);
    sendToRust({ type: "TAB_CLOSED", tabId });
  }
});

// On startup, register any already-open ChatGPT tabs
chrome.tabs.query({}, (tabs) => {
  for (const tab of tabs) {
    if (
      tab?.id &&
      tab?.url &&
      (tab.url.includes("chatgpt.com") ||
        tab.url.includes("chat.openai.com"))
    ) {
      liveTabs.add(tab.id);
    }
  }
});

// ─────────────────────────────────────────────
// content → background → Rust (raw chunk passthrough)
// ─────────────────────────────────────────────
chrome.runtime.onMessage.addListener(
  (
    message: any,
    sender: chrome.runtime.MessageSender,
    sendResponse: (r: any) => void,
  ) => {
    if (message?.type === "CHUNK") {
      const tabId = sender?.tab?.id ?? firstLiveTab();
      sendToRust({ type: "CHUNK", tabId, raw: message.raw });
      sendResponse({ ok: true });
      return true;
    }

    sendResponse({ ok: false, error: "unknown type" });
    return true;
  },
);

// ─────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────
connectRust();
