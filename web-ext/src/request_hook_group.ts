// src/request_hook_group.ts

interface Window {
  __RequestHookGroupInstalled?: boolean;
  __pendingPromptInjection?: string | null;
  __promptInjectionMode?: "auto" | "buffer";
  __promptInjectionQueue?: string[];
}

(function () {
  if (window.__RequestHookGroupInstalled) return;
  window.__RequestHookGroupInstalled = true;

  const TARGETS = [
    {
      origin: "https://chatgpt.com",
      path: "/backend-api/calpico/chatgpt/rooms",
    },
    {
      origin: "https://chat.openai.com",
      path: "/backend-api/calpico/chatgpt/rooms",
    },
  ];

  console.log("[RequestHookGroup] Installing request modifier");

  // Globals
  window.__pendingPromptInjection = window.__pendingPromptInjection || null;
  window.__promptInjectionMode = window.__promptInjectionMode || "auto";
  window.__promptInjectionQueue = window.__promptInjectionQueue || [];

  function matchesTarget(input: RequestInfo): boolean {
    try {
      // Safely extract URL string
      const urlString =
        typeof input === "string"
          ? input
          : input instanceof Request
            ? input.url
            : ((input as any)?.url ?? "");

      if (!urlString) return false;

      const abs = new URL(urlString, location.href);
      return TARGETS.some(
        (t) => abs.origin === t.origin && abs.pathname.startsWith(t.path),
      );
    } catch {
      return false;
    }
  }

  const originalFetch = window.fetch;

  window.fetch = async function (
    input: RequestInfo,
    init?: RequestInit,
  ): Promise<Response> {
    // CASE 1: fetch(Request)
    if (input instanceof Request) {
      const url = input.url;
      if (!matchesTarget(url)) {
        return originalFetch(input);
      }
      if (input.method !== "POST") {
        return originalFetch(input);
      }
      try {
        const text = await input.clone().text();
        if (!text) return originalFetch(input);
        const payload = JSON.parse(text);

        // Calpico group chat only
        if (payload?.content?.text === "<PROMPT>") {
          payload.content.text =
            window.__pendingPromptInjection ||
            (window.__promptInjectionQueue ?? []).join("\n\n") ||
            "";

          const newReq = new Request(input, {
            body: JSON.stringify(payload),
          });

          if (window.__promptInjectionMode === "auto") {
            window.__pendingPromptInjection = null;
            window.__promptInjectionQueue = [];
          }

          console.log("✅ INJECTED (Request)");
          return originalFetch(newReq);
        }
      } catch (e) {
        console.warn("Hook parse failed:", e);
      }
      return originalFetch(input);
    }

    // CASE 2: fetch(url, init)
    const url = typeof input === "string" ? input : (input as Request)?.url;
    if (matchesTarget(url) && init && typeof init.body === "string") {
      try {
        const payload = JSON.parse(init.body);
        if (payload?.content?.text === "<PROMPT>") {
          payload.content.text =
            window.__pendingPromptInjection ||
            (window.__promptInjectionQueue ?? []).join("\n\n") ||
            "";

          init.body = JSON.stringify(payload);

          if (window.__promptInjectionMode === "auto") {
            window.__pendingPromptInjection = null;
            window.__promptInjectionQueue = [];
          }

          console.log("✅ INJECTED (init.body)");
        }
      } catch (e) {
        console.warn("Hook parse failed:", e);
      }
    }

    return originalFetch(input, init);
  };
})();
