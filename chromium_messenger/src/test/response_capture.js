// Response capture script for ChatGPT
(function() {
    console.log("[MMSB] Response capture initialized");
    
    // Intercept fetch responses
    const originalFetch = window.fetch;
    window.fetch = async function(...args) {
        const response = await originalFetch.apply(this, args);
        
        // Clone response to read body
        const cloned = response.clone();
        
        try {
            const text = await cloned.text();
            
            // Check if this is a ChatGPT response
            if (args[0].includes('conversation')) {
                // Extract conversation and message IDs from URL or response
                const data = {
                    type: "CHATGPT_MESSAGE_CAPTURE",
                    conversationId: extractConversationId(args[0]),
                    messageId: extractMessageId(text),
                    timestamp: Date.now(),
                    responseText: text
                };
                
                // Send to Rust via CDP binding
                if (typeof __chromiumMessenger !== 'undefined') {
                    __chromiumMessenger(JSON.stringify(data));
                }
            }
        } catch (e) {
            console.error("[MMSB] Failed to capture response:", e);
        }
        
        return response;
    };
    
    function extractConversationId(url) {
        const match = url.match(/conversation\/([a-f0-9-]+)/);
        return match ? match[1] : 'unknown';
    }
    
    function extractMessageId(text) {
        try {
            const json = JSON.parse(text);
            return json.message?.id || 'unknown';
        } catch {
            return 'unknown';
        }
    }
})();
