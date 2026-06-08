from app.services.embeddings import query_similar

def retrieve_relevant_chunks(query: str, top_k: int = 5, doc_id: str = None) -> list[dict]:
    """
    - Call query_similar(query, top_k, doc_id) from embeddings.py
    - For each result, return:
        { "content": ..., "doc_id": ..., "filename": ..., 
          "chunk_index": ..., "score": ... }
    - Filter out any results with score < 0.35 (not relevant enough)
    - If no results pass the threshold, return an empty list []
    """
    raw_results = query_similar(query, top_k=top_k, doc_id=doc_id)
    relevant_results = []
    
    for res in raw_results:
        score = res.get("score")
        if score is not None and score >= 0.35:
            relevant_results.append({
                "content": res.get("content"),
                "doc_id": res.get("doc_id"),
                "filename": res.get("filename"),
                "chunk_index": res.get("chunk_index"),
                "score": score
            })
            
    return relevant_results


def is_relevant(results: list[dict]) -> bool:
    """
    - Returns True if at least one result has score >= 0.35
    - Used by the chat endpoint to decide document vs general AI mode
    """
    for res in results:
        score = res.get("score")
        if score is not None and score >= 0.35:
            return True
    return False
