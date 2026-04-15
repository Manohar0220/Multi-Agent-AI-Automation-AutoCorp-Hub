import re

def extract_requested_filename(subject: str) -> str:
    """
    Accepts subjects like:
      'Request: resume.pdf'
      'request: offer letter'
    Returns the part after 'Request:' cleaned up, or None if no match.
    """
    print(subject)
    if not subject:
        return None
    m = re.match(r"\s*Request:\s*(.+)$", subject.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    name = re.sub(r"\s+", " ", m.group(1)).strip()
    return name or None
