// src/content.ts — pure relay, no logic, no imports

function inject(path: string) {
  const s = document.createElement("script");
  s.src = chrome.runtime.getURL(path);
  (document.head || document.documentElement).appendChild(s);
  s.onload = () => s.remove();
}

inject("page.js");
inject("request_hook.js");
inject("request_hook_group.js");

// ── page → content → background (raw chunk passthrough) ──
window.addEventListener("message", (e) => {
  if (e.source !== window) return;
  if (e.data?.type !== "CHUNK") return;

  if (!chrome?.runtime?.sendMessage) return;

  chrome.runtime.sendMessage(
    { type: "CHUNK", raw: e.data.raw },
    () => void chrome.runtime.lastError,
  );
});

// ── background → content → page ──
chrome.runtime.onMessage.addListener((msg: any, _sender, sendResponse) => {
  if (msg?.type === "TURN") {
    window.postMessage({ type: "TURN", text: msg.text }, "*");
    sendResponse({ ok: true });
    return false;
  }
});
