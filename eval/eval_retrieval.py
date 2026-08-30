"""Retrieval evaluation script.

Runs the test_questions.json dataset against the /retrieve endpoint
and calculates standard IR metrics (Hit Rate @ K, MRR @ K).
"""

import json
import httpx
from pathlib import Path

# Adjust port if running on a different port
BASE_URL = "http://localhost:8000"
TOP_K = 5

def load_questions() -> list[dict]:
    path = Path(__file__).parent / "test_questions.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation():
    questions = load_questions()
    total = len(questions)
    hits = 0
    mrr_sum = 0.0

    print(f"Running evaluation on {total} questions (Top-K={TOP_K})...\n")

    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # First check health
        try:
            health = client.get("/health").json()
            if not health.get("chroma_connected"):
                print("Error: ChromaDB is not connected. Run POST /ingest first.")
                return
        except Exception as e:
            print(f"Error: API server is not running at {BASE_URL}")
            return

        for idx, q in enumerate(questions, 1):
            query = q["query"]
            expected_doc = q["expected_document"]
            expected_sec = q["expected_section"]
            
            print(f"Q{idx}: {query}")
            print(f"Expected: {expected_doc} - {expected_sec}")
            
            try:
                response = client.get(f"/retrieve", params={"query": query, "top_k": TOP_K})
                response.raise_for_status()
                results = response.json().get("results", [])
                
                hit = False
                reciprocal_rank = 0.0
                
                for rank, result in enumerate(results, 1):
                    meta = result.get("metadata", {})
                    doc_match = meta.get("doc_type") == expected_doc
                    
                    # Fuzzy match for section (e.g., "Section 66A" in "Section 66A (1/2)")
                    sec_match = expected_sec.lower() in meta.get("section_id", "").lower()
                    
                    if doc_match and sec_match:
                        hit = True
                        reciprocal_rank = 1.0 / rank
                        print(f"  ✓ HIT at rank {rank} (Score: {result.get('score'):.3f})")
                        break
                
                if not hit:
                    print("  ✗ MISS")
                    if results:
                        print(f"  Top prediction was: {results[0].get('metadata', {}).get('label')}")
                
                if hit:
                    hits += 1
                mrr_sum += reciprocal_rank
                print("-" * 40)
                
            except Exception as e:
                print(f"  Error running query: {e}")
                print("-" * 40)

    # Calculate final metrics
    hit_rate = (hits / total) * 100 if total > 0 else 0
    mrr = (mrr_sum / total) if total > 0 else 0
    
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total Questions: {total}")
    print(f"Hit Rate @ {TOP_K}:  {hit_rate:.1f}%")
    print(f"MRR @ {TOP_K}:       {mrr:.3f}")
    print("=" * 40)

if __name__ == "__main__":
    run_evaluation()
