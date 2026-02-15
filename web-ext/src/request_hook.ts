// src/request_hook.ts

// Tell TypeScript these properties are allowed on window
interface Window {
  __RequestHookPrivateInstalled?: boolean;
  __pendingPromptInjection?: string | null;
  __promptInjectionMode?: "auto" | "buffer";
  __promptInjectionQueue?: string[];
}

(function () {
  if (window.__RequestHookPrivateInstalled) return;
  window.__RequestHookPrivateInstalled = true;

  const TARGETS = [
    { origin: "https://chatgpt.com", path: "/backend-api/f/conversation" },
    { origin: "https://chat.openai.com", path: "/backend-api/f/conversation" },
  ];

  console.log("[RequestHookPrivate] Installing request modifier");

  // Safe initialization of globals
  window.__pendingPromptInjection = window.__pendingPromptInjection ?? null;
  window.__promptInjectionMode = window.__promptInjectionMode ?? "auto";
  window.__promptInjectionQueue = window.__promptInjectionQueue ?? [];

  function matchesTarget(input: RequestInfo): boolean {
    try {
      const url = typeof input === "string" ? input : (input as Request).url;
      const abs = new URL(url, location.href);
      return TARGETS.some(
        (target) =>
          abs.origin === target.origin && abs.pathname.startsWith(target.path),
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
    const isTarget = matchesTarget(input);

    if (isTarget && init && typeof init.body === "string") {
      console.log(
        "[RequestHookPrivate] Intercepted:",
        typeof input === "string" ? input : (input as Request).url,
      );

      try {
        const payload = JSON.parse(init.body);

        if (Array.isArray(payload?.messages) && payload.messages.length > 0) {
          const lastMessage = payload.messages[payload.messages.length - 1];

          if (lastMessage?.content?.parts) {
            const hasPlaceholder = lastMessage.content.parts.some(
              (part) => typeof part === "string" && part.includes("<PROMPT>"),
            );

            if (hasPlaceholder) {
              // Combine queue + pending
              const parts = [...window.__promptInjectionQueue!];
              if (window.__pendingPromptInjection) {
                parts.push(window.__pendingPromptInjection);
              }
              const combined = parts.join("\n\n");

              // Replace entire parts array
              lastMessage.content.parts = [combined];

              console.log(
                "[RequestHookPrivate] Replaced <PROMPT> with:",
                combined.substring(0, 120) +
                  (combined.length > 120 ? "..." : ""),
              );

              window.__promptInjectionQueue = [];

              init.body = JSON.stringify(payload);

              if (window.__promptInjectionMode === "auto") {
                window.__pendingPromptInjection = null;
              }
            }
          }
        }
      } catch (err) {
        console.warn("[RequestHookPrivate] Modification failed:", err);
      }
    }

    return originalFetch(input, init);
  };
})();
