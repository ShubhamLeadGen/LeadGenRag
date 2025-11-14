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

    save_js = f'''
    (function() {{
        try {{
            const data = `{data_string}`;

            // 1. Save to Local Storage
            localStorage.setItem('chat_sessions', data);
            console.log("✅ Saved sessions to Local Storage.");

            // 2. Save to Cookie as a backup (expires in 1 year)
            const expiryDate = new Date();
            expiryDate.setFullYear(expiryDate.getFullYear() + 1);
            document.cookie = `chat_sessions=${{encodeURIComponent(data)}}; expires=${{expiryDate.toUTCString()}}; path=/; SameSite=Lax`;
            console.log("✅ Saved sessions to Cookie.");

            return "CACHE_SAVED";
        }} catch (e) {{
            console.error("❌ Error saving chat sessions:", e);
            return "CACHE_ERROR";
        }}
    }})();
    '''
    streamlit_js_eval(js_expressions=save_js, key=f"save_cache_{time.time()}")
