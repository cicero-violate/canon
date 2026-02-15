// Injected script that hooks into ChatGPT's fetch API
(function() {
  if (window.__ChatGPTCaptureInstalled) return;
  window.__ChatGPTCaptureInstalled = true;

  const DEBUG = false;
  const log = (...args) => {
    if (DEBUG) console.log('[ChatGPT Capture]', ...args);
  };

  const TARGETS = [
    {origin: "https://chatgpt.com", path: "/backend-api/f/conversation"},
    {origin: "https://chat.openai.com", path: "/backend-api/f/conversation"}
  ];

  function matchesTarget(input) {
    try {
      const url = new URL(input, location.href);
      for (const target of TARGETS) {
        if (url.origin === target.origin && url.pathname.startsWith(target.path)) {
          return url.href;
        }
      }
    } catch (err) {
      console.warn('[ChatGPT Capture] URL parse failed', err);
    }
    return null;
  }

  function updateFromEvent(data, state) {
    // New ChatGPT SSE format wraps the payload under a "v" key.
    const unwrapped = (data && typeof data === 'object' && 'v' in data) ? data.v : data;

    if (!state.conversationId) {
      state.conversationId = unwrapped.conversation_id || unwrapped.conversationId || unwrapped.message?.conversation_id;
    }
    if (!state.messageId) {
      state.messageId = unwrapped.message_id || unwrapped.messageId || unwrapped.message?.id;
    }

    state.sseEvents.push(data);

    const message = unwrapped.message;
    const role = message?.author?.role;
    const parts = message?.content?.parts;
    if (role === 'assistant' && Array.isArray(parts) && parts.length > 0) {
      const text = parts.filter(p => typeof p === 'string').join('\n');
      if (text.trim()) state.responseText = text;
    }
  }

  const originalFetch = window.fetch;
  window.fetch = async function(input, init) {
    const url = matchesTarget(typeof input === 'string' ? input : input?.url);

    const response = await originalFetch(input, init);
    if (!url) return response;

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('text/event-stream') || !response.body) {
      return response;
    }

    const [toPage, toCapture] = response.body.tee();

    (async () => {
      const reader = toCapture.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const state = {
        conversationId: null,
        messageId: null,
        responseText: null,
        sseEvents: []
      };

      try {
        while (true) {
          const {value, done} = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, {stream: true});
          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            if (line === 'data: [DONE]') continue;

            try {
              const data = JSON.parse(line.slice(6));
              updateFromEvent(data, state);
            } catch (err) {
              // Ignore parse errors for individual lines
            }
          }
        }

        if (state.conversationId && state.messageId && state.sseEvents.length > 0) {
          window.postMessage({
            type: 'CHATGPT_MESSAGE_CAPTURE',
            conversationId: state.conversationId,
            messageId: state.messageId,
            responseText: state.responseText,
            sseEvents: state.sseEvents,
            timestamp: Date.now()
          }, '*');
        } else {
          log('Not sending - missing conversation ID, message ID, or response text');
        }
      } catch (err) {
        console.error('[ChatGPT Capture] Stream error', err);
      }
    })();

    return new Response(toPage, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers
    });
  };

  log('Hook installed');
})();
