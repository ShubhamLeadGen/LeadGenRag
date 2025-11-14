
import re

def is_gibberish(text: str) -> bool:
    if not text or len(text.strip()) < 3:
        return True
    letters = len(re.findall(r"[a-zA-Z]", text))
    if letters / max(1, len(text)) < 0.4:
        return True
    if len(text.split()) == 1 and letters < 4:
        return True
    return False

def polite_fallback():
    return "I don't have any knowledge about this. Could you please ask something else?"

def beautify_response(text: str) -> str:
    lines = text.strip().split('\n')
    beautified_lines = []
    for line in lines:
        stripped_line = line.strip()
        if re.fullmatch(r'\*+', stripped_line):
            beautified_lines.append('---')
            continue
        if stripped_line.startswith('* '):
            beautified_lines.append('- ' + stripped_line[2:])
        else:
            beautified_lines.append(stripped_line)
    return '\n'.join(beautified_lines)

from src.config import MAX_CHARS_FOR_TEXT_EXTRACTION

def extract_clean_text(docs: list) -> str:
    combined = []
    for doc in docs:
        text = doc.page_content.replace('\ufeff', '').strip()
        text = ' '.join(line.strip() for line in text.splitlines() if line.strip())
        combined.append(text)
    full_text = ' '.join(combined)
    if len(full_text) > MAX_CHARS_FOR_TEXT_EXTRACTION:
        full_text = full_text[:MAX_CHARS_FOR_TEXT_EXTRACTION] + " …"
    return full_text
