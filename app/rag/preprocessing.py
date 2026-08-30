"""Document preprocessing for IETF RFCs.

Cleans raw text extracted from IETF RFCs before chunking.
RFC plaintext has a rigidly consistent boilerplate header/footer per page and form-feed page breaks.
"""

import re
from app.utils.logging import logger

def preprocess_text(raw: str, doc_type: str = "rfc") -> str:
    """Clean raw extracted text from an RFC text file.

    Args:
        raw: Raw text extracted from RFC txt.
        doc_type: Type of document (defaults to "rfc").

    Returns:
        Cleaned text ready for chunking.
    """
    original_length = len(raw)

    # Convert form feeds to newlines
    text = raw.replace("\x0c", "\n")
    
    # Remove top page headers (e.g. "RFC 9110 HTTP Semantics June 2022")
    text = re.sub(r"^RFC \d+.*\n", "", text, flags=re.MULTILINE)
    
    # Remove bottom page footers (e.g. "[Page 12]")
    text = re.sub(r"\n\s*\[Page \d+\]\s*\n", "\n", text)

    logger.debug(
        "Preprocessed %s text: %d → %d chars",
        doc_type,
        original_length,
        len(text),
    )

    return text.strip()
