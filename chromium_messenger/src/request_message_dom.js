(async function modifyAndSendMessage({
  text = "Hello from CDP ✅",
  pollInterval = 150,
  timeout = 60000
} = {}) {
  const sleep = (ms) => new Promise(res => setTimeout(res, ms));
  console.log("[ChatGPT Submit] SCRIPT STARTED, text length:", text.length);

  // ========== SET PENDING INJECTION ==========
  window.__pendingPromptInjection = text;
  console.log('[ChatGPT Submit] Set pending injection:', text.length, 'chars', 'First 100 chars:', text.substring(0, 100));
  // ===========================================

  // Wait for editable div
  const findEditable = async () => {
    const t0 = performance.now();
    while (performance.now() - t0 < timeout) {
      const e = document.querySelector('div[contenteditable="true"]');
      // Don't check visibility - works in background tabs
      if (e) return e;
      await sleep(pollInterval);
    }
    return null;
  };

  const E = await findEditable();
  if (!E) { console.log('[ChatGPT Submit] NO_INPUT'); return { ok: false, reason: "NO_INPUT" }; }

  // Fill with placeholder (request hook swaps this at send time)
  const placeholder = "<PROMPT>";
  console.log('[ChatGPT Submit] Filling editable with placeholder');
  // Don't call focus() - not needed and may fail in background tabs
  // E.focus();
  E.textContent = placeholder;
  E.dispatchEvent(new InputEvent("input", { bubbles: true, data: placeholder }));
  console.log('[ChatGPT Submit] Placeholder applied, waiting for send button');
  await sleep(500);

  // Wait for send button and click
  const start = performance.now();
  while (performance.now() - start < timeout) {
    const B = document.querySelector('button[data-testid="send-button"]');
    if (B && !B.disabled) {
      console.log('[ChatGPT Submit] Clicking send button');
      B.click();
      console.log("✅ Message sent via CDP");
      return { ok: true, sent: true };
    }
    await sleep(pollInterval);
  }

  console.log('[ChatGPT Submit] TIMEOUT_WAITING_FOR_BUTTON');
  return { ok: false, reason: "TIMEOUT_WAITING_FOR_BUTTON" };
})
