import json
import time
from streamlit_js_eval import streamlit_js_eval

def load_cache():
    """
    Loads chat sessions from the browser's Local Storage or a cookie.
    This function injects JavaScript to retrieve the data.
    """
    load_js = """
    (function() {
        // Try to load from Local Storage first
        let data = localStorage.getItem('chat_sessions');
        
        // If not in Local Storage, try to load from cookie
        if (!data) {
            const match = document.cookie.match(/chat_sessions=([^;]+)/);
            if (match) {
                try {
                    data = decodeURIComponent(match[1]);
                } catch (e) {
                    console.error("Error decoding cookie data:", e);
                    return "CACHE_EMPTY";
                }
            }
        }
        
        console.log("📥 Loaded chat sessions from browser:", data);
        return data || "CACHE_EMPTY";
    })();
    """
    return streamlit_js_eval(js_expressions=load_js, key="load_cache")

def save_cache(sessions):
    """
    Saves chat sessions to the browser's Local Storage and as a cookie.
    This function injects JavaScript to save the data.
    """
    # Convert the session data to a JSON string.
    # We escape backslashes and backticks to safely inject this into a JS template literal.
    data_string = json.dumps(sessions).replace('\\', '\\\\').replace('`', '\\`')

    # This JS will read existing Local Storage, parse it, merge with the incoming
    # sessions (incoming keys take precedence), and then write the merged object
    # back to localStorage and cookie. Doing the merge in the browser avoids a
    # race where the server-side state overwrites previously saved sessions.
    save_js_template = '''
    (function() {
        try {
            const incomingJson = `{DATA}`;
            let incoming = {};
            try { incoming = JSON.parse(incomingJson); } catch(e) { console.warn('Incoming sessions JSON parse failed', e); }

            let existingRaw = localStorage.getItem('chat_sessions');
            let existing = {};
            if (existingRaw) {
                try { existing = JSON.parse(existingRaw); } catch(e) { console.warn('Existing sessions JSON parse failed', e); existing = {}; }
            }

            // Merge intelligently per-session to avoid overwriting existing
            // session histories with empty incoming sessions. Rule: prefer
            // incoming when it has messages; otherwise keep existing if it
            // has messages.
            const merged = {};
            for (const k of Object.keys(existing)) {
                merged[k] = existing[k];
            }
            for (const k of Object.keys(incoming)) {
                try {
                    const incVal = incoming[k];
                    const existVal = existing[k];
                    const incLen = Array.isArray(incVal) ? incVal.length : 0;
                    const existLen = Array.isArray(existVal) ? existVal.length : 0;
                    if (incLen === 0 && existLen > 0) {
                        // keep existing (older) non-empty session
                        merged[k] = existVal;
                    } else {
                        // prefer incoming (either non-empty or nothing exists)
                        merged[k] = incVal;
                    }
                } catch (e) {
                    merged[k] = incoming[k];
                }
            }

            const mergedStr = JSON.stringify(merged);
            localStorage.setItem('chat_sessions', mergedStr);
            console.log('✅ Merged and saved sessions to Local Storage.');

            const expiryDate = new Date();
            expiryDate.setFullYear(expiryDate.getFullYear() + 1);
            document.cookie = `chat_sessions=${encodeURIComponent(mergedStr)}; expires=${expiryDate.toUTCString()}; path=/; SameSite=Lax`;
            console.log('✅ Saved merged sessions to cookie backup.');

            return 'CACHE_SAVED';
        } catch (e) {
            console.error('❌ Error merging/saving chat sessions:', e);
            return 'CACHE_ERROR';
        }
    })();
    '''

    save_js = save_js_template.replace('{DATA}', data_string)
    streamlit_js_eval(js_expressions=save_js, key=f"save_cache_{time.time()}")
