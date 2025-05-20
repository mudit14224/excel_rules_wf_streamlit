import re

def extract_code_block(raw_text):
    """Extract code from a Markdown-style Python code block."""
    match = re.search(r"```(?:python)?\s*(.*?)```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

