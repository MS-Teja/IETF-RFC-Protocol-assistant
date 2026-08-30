import sys
import os

# Add parent directory to path so we can import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.dependencies import get_embedding_service, get_vector_store
from app.rag.ingestion import ingest_documents
from app.utils.logging import logger

def main():
    print("Testing Ingestion...")
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()
    
    # Ingest
    result = ingest_documents(
        data_dir=settings.data_dir,
        embedding_service=embedding_service,
        vector_store=vector_store,
        force_reingest=True
    )
    print(f"Ingestion result: {result}")
    
    # Test retrieval
    query = "What is the punishment for sending offensive messages?"
    print(f"\nQuerying: '{query}'")
    
    query_embedding = embedding_service.embed_query(query)
    chunks = vector_store.query(query_embedding, top_k=2)
    
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get('score', 0)
        label = chunk.get('metadata', {}).get('label', 'Unknown')
        content = chunk.get('content', '')
        print(f"\nResult {i} (Score: {score:.3f}):")
        print(f"Label: {label}")
        print(f"Content: {content[:150]}...")

if __name__ == "__main__":
    main()
