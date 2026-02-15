// src/page.ts — MAIN world. No imports.
// 1. Fetch hook  → fire raw SSE lines to content as CHUNK
// 2. WS hook     → fire raw WS frames to content as CHUNK
// 3. TURN        → inject text into editor with retry, press Enter

(function () {
  if ((window as any).__relay_installed) return;
  (window as any).__relay_installed = true;

  // ─────────────────────────────────────────────
  // Helpers
  // ─────────────────────────────────────────────
  function fireChunk(raw: string) {
    window.postMessage({ type: "CHUNK", raw }, "*");
  }

  // ─────────────────────────────────────────────
  // WebSocket hook — captures ChatGPT group chat WS frames
  // ─────────────────────────────────────────────
  const OrigWebSocket = window.WebSocket;

  (window as any).WebSocket = function (
    url: string,
    protocols?: string | string[],
  ) {
    const ws = protocols
      ? new OrigWebSocket(url, protocols)
      : new OrigWebSocket(url);

    ws.addEventListener("message", (ev: MessageEvent) => {
      let raw = "";
      if (typeof ev.data === "string") {
        raw = ev.data;
      } else if (ev.data instanceof ArrayBuffer) {
        raw = new TextDecoder().decode(ev.data);
      }
      if (raw) fireChunk(raw);
    });

    return ws;
  };

  (window as any).WebSocket.prototype = OrigWebSocket.prototype;

  // ─────────────────────────────────────────────
  // Fetch hook — tees ChatGPT SSE responses, fires raw lines
  // ─────────────────────────────────────────────
  const TARGETS = [
    { origin: "https://chatgpt.com", path: "/backend-api/f/conversation" },
    { origin: "https://chat.openai.com", path: "/backend-api/f/conversation" },
    { origin: "https://chatgpt.com", path: "/backend-api/calpico" },
    { origin: "https://chat.openai.com", path: "/backend-api/calpico" },
  ];

  function matchesTarget(input: string): boolean {
    try {
      const url = new URL(input, location.href);
      return TARGETS.some(
        (t) => url.origin === t.origin && url.pathname.startsWith(t.path),
      );
    } catch {
      return false;
    }
  }

  const origFetch = window.fetch;

  window.fetch = async function (input: any, init?: any) {
    const url = typeof input === "string" ? input : input?.url ?? "";
    const isTarget = matchesTarget(url);

    const response = await origFetch(input, init);
    if (!isTarget || !response.body) return response;

    const [toPage, toCapture] = response.body.tee();

    (async () => {
      const reader = toCapture.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            if (line) fireChunk(line);
          }
        }
        if (buffer) fireChunk(buffer);
      } catch {}
    })();

    return new Response(toPage, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };

  // ─────────────────────────────────────────────
  // TURN handler — inject text with retry, press Enter
  // ─────────────────────────────────────────────
  function handleTurn(text: string) {
    (window as any).__pendingPromptInjection = text;
    tryInject(text, 20);
  }

  function tryInject(text: string, attemptsLeft: number) {
    const editor = document.querySelector(
      'div[contenteditable="true"]',
    ) as HTMLElement | null;

    if (!editor) {
      if (attemptsLeft <= 0) {
        console.warn("[page] TURN: editor never appeared, giving up");
        return;
      }
      console.warn("[page] TURN: editor not ready, retrying...");
      setTimeout(() => tryInject(text, attemptsLeft - 1), 500);
      return;
    }

    editor.textContent = "<PROMPT>";
    editor.dispatchEvent(new Event("input", { bubbles: true }));
    editor.focus();

    setTimeout(() => {
      const dispatch = (evType: string) =>
        editor.dispatchEvent(
          new KeyboardEvent(evType, {
            key: "Enter",
            code: "Enter",
            which: 13,
            keyCode: 13,
            bubbles: true,
            cancelable: true,
            composed: true,
          }),
        );
      dispatch("keydown");
      dispatch("keypress");
    }, 80);
  }

  window.addEventListener("message", (e) => {
    if (e.source !== window) return;
    if (e.data?.type === "TURN") {
      handleTurn(e.data.text ?? "");
    }
  });
})();
