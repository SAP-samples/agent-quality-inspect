
def ensure_full_stop(text: str) -> str:
    text = text.rstrip()
    return text if text.endswith('.') else text + '.'