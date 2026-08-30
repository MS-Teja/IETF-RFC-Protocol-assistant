"""Document chunking strategies for IETF RFCs.

Chunks RFCs by their numbered sections (e.g. 1.1, 4.3.2) using regex.
Each chunk is tagged with metadata for label injection during context construction.

Max chunk size is ~400 words to stay under the 512-token limit of
all-MiniLM-L6-v2.
"""

import re
from dataclasses import dataclass, field
from app.utils.logging import logger

MAX_CHUNK_WORDS = 400

@dataclass
class Chunk:
    """A document chunk with metadata for retrieval and citation."""
    content: str
    doc_type: str  # e.g., "rfc9110"
    section_id: str  # e.g., "15.5.1"
    title: str = ""

    @property
    def label(self) -> str:
        """Generate the label for context construction injection."""
        rfc_num = self.doc_type.replace("rfc", "").upper()
        # Format: [RFC 9110, §15.5.1 Bad Request]
        title_part = f" {self.title}" if self.title else ""
        return f"[RFC {rfc_num}, §{self.section_id}{title_part}]"

    @property
    def metadata(self) -> dict:
        """Return metadata dict for vector store storage."""
        return {
            "doc_type": self.doc_type,
            "section_id": self.section_id,
            "title": self.title,
            "label": self.label,
        }

def rfc_chunker(text: str, doc_type: str) -> list[Chunk]:
    """Chunk an RFC text by section boundaries.
    
    Splits on patterns like: "1. Introduction" or "3.1.1. Something"
    """
    chunks = []

    # Regex proposed by user for RFC sections
    section_pattern = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.+)$", re.MULTILINE)

    # Find all matches and their start positions
    matches = list(section_pattern.finditer(text))
    
    if not matches:
        logger.warning("No sections found in %s, chunking globally", doc_type)
        return _split_if_oversized(text, "Global", doc_type, "Entire Document")

    # Add preamble (text before first section)
    preamble = text[:matches[0].start()].strip()
    if len(preamble.split()) > 10:
        chunks.extend(_split_if_oversized(preamble, "Preamble", doc_type, "Abstract/Status"))

    # Iterate over matches to slice text
    for i in range(len(matches)):
        match = matches[i]
        section_num = match.group(1).strip()
        title = match.group(2).strip()
        
        start_idx = match.end()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
        
        section_text = text[start_idx:end_idx].strip()
        
        if section_text:
            chunks.extend(_split_if_oversized(section_text, section_num, doc_type, title))

    logger.info("%s chunked into %d chunks", doc_type, len(chunks))
    return chunks

def _split_if_oversized(
    text: str,
    section_id: str,
    doc_type: str,
    title: str = "",
) -> list[Chunk]:
    """Split a section into multiple chunks if it exceeds MAX_CHUNK_WORDS.

    Splits at paragraph boundaries to keep semantic coherence.
    Each sub-chunk gets a numbered suffix: "15.5.1 (1/3)".
    """
    words = text.split()
    if len(words) <= MAX_CHUNK_WORDS:
        return [Chunk(
            content=text,
            doc_type=doc_type,
            section_id=section_id,
            title=title,
        )]

    # Split on paragraph boundaries (double newline)
    paragraphs = re.split(r"\n\n+", text)

    sub_chunks = []
    current_text = ""
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_word_count + para_words > MAX_CHUNK_WORDS and current_text:
            sub_chunks.append(current_text.strip())
            current_text = para
            current_word_count = para_words
        else:
            current_text += "\n\n" + para if current_text else para
            current_word_count += para_words

    if current_text.strip():
        sub_chunks.append(current_text.strip())

    # Create chunks with numbered suffixes
    total = len(sub_chunks)
    chunks = []
    for i, sub_text in enumerate(sub_chunks):
        sub_section_id = f"{section_id} ({i+1}/{total})" if total > 1 else section_id
        chunks.append(Chunk(
            content=sub_text,
            doc_type=doc_type,
            section_id=sub_section_id,
            title=title,
        ))

    return chunks
