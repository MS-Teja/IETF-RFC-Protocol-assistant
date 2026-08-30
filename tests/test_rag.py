import pytest
from app.rag.preprocessing import preprocess_text
from app.rag.query_processing import contextualize_query
from app.rag.chunking import rfc_chunker, Chunk

def test_preprocess_text():
    # Test line ending normalization
    assert preprocess_text("Hello\x0cWorld") == "Hello\nWorld"
    
    # Test top header removal
    text_with_header = "RFC 9110 HTTP Semantics June 2022\n\n1. Introduction"
    assert "RFC 9110" not in preprocess_text(text_with_header)
    
    # Test bottom footer removal
    text_with_footer = "Some text\n\n[Page 12]\n\nMore text"
    assert "[Page 12]" not in preprocess_text(text_with_footer)

def test_query_contextualization():
    history = [
        {"role": "user", "content": "What is HTTP/3?"},
        {"role": "assistant", "content": "HTTP/3 is the latest version of HTTP."}
    ]
    query = "How does it handle header compression?"
    rewritten = contextualize_query(query, history)
    
    assert "HTTP/3" in rewritten

def test_chunking_metadata():
    text = "1. Introduction\nThis is an RFC."
    chunks = rfc_chunker(text, doc_type="rfc9110")
    
    assert len(chunks) > 0
    chunk = chunks[0]
    
    # Verify metadata fields exist
    assert hasattr(chunk, "doc_type")
    assert hasattr(chunk, "section_id")
    assert "RFC 9110" in chunk.label or "rfc9110" in chunk.label
