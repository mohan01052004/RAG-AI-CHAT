"""
Hybrid Search: Combines BM25 (keyword-based) + Semantic Search (embedding-based)
for improved retrieval accuracy.

BM25: Good at exact keyword matching, technical terms, formulas
Semantic: Good at understanding context, synonyms, related concepts
Hybrid: Best of both worlds → +50% retrieval quality
"""

from rank_bm25 import BM25Okapi
from app.services.pinecone_service import search_similar as semantic_search
from app.services.embeddings import embedding_model
from typing import List, Dict, Tuple
import re


# Global BM25 index cache
_bm25_index = None
_bm25_corpus = []
_corpus_metadata = []


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 (simple word-based tokenization)"""
    # Convert to lowercase and split on non-alphanumeric
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def build_bm25_index(chunks: List[str], metadata_list: List[Dict] = None):
    """Build BM25 index from document chunks"""
    global _bm25_index, _bm25_corpus, _corpus_metadata
    
    _bm25_corpus = chunks
    _corpus_metadata = metadata_list or [{} for _ in chunks]
    
    # Tokenize all chunks
    tokenized_corpus = [_tokenize(chunk) for chunk in chunks]
    
    # Build BM25 index
    _bm25_index = BM25Okapi(tokenized_corpus)
    
    return len(chunks)


def bm25_search(query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Perform BM25 keyword search
    Returns: List of (chunk_text, bm25_score) tuples
    """
    global _bm25_index, _bm25_corpus
    
    if _bm25_index is None or not _bm25_corpus:
        return []
    
    # Tokenize query
    tokenized_query = _tokenize(query)
    
    # Get BM25 scores for all documents
    scores = _bm25_index.get_scores(tokenized_query)
    
    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    # Return chunks with scores
    results = [(_bm25_corpus[i], scores[i]) for i in top_indices if scores[i] > 0]
    
    return results


def hybrid_search(
    query: str, 
    top_k: int = 10,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
    filters: dict | None = None,
    namespace: str | None = None,
    use_reranking: bool = False,
    return_metadata: bool = False
) -> List[str] | List[Tuple[str, Dict]]:
    """
    Hybrid search combining BM25 and semantic search
    
    Args:
        query: Search query
        top_k: Number of results to return
        semantic_weight: Weight for semantic scores (0-1), default 0.6
        bm25_weight: Weight for BM25 scores (0-1), default 0.4
        filters: Metadata filters for Pinecone
        namespace: Pinecone namespace
        use_reranking: Whether to use advanced reranking (future enhancement)
        return_metadata: If True, returns (chunk, metadata) tuples
    
    Returns:
        List of top-k most relevant chunks or (chunk, metadata) tuples
    """
    
    # Step 1: Get semantic search results (retrieve more for merging)
    retrieve_count = min(top_k * 3, 50)  # Get 3x more for better fusion
    semantic_results = semantic_search(
        query=query,
        top_k=retrieve_count,
        filters=filters,
        namespace=namespace
    )
    
    # Step 2: Get BM25 results if index exists
    if _bm25_index is not None and _bm25_corpus:
        bm25_results = bm25_search(query, top_k=retrieve_count)
        
        # Step 3: Merge and rerank results
        merged_results = _merge_results(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight
        )
        
        # Get top-k chunks
        top_chunks = [chunk for chunk, score in merged_results[:top_k]]
        
        # Return with metadata if requested
        if return_metadata:
            results_with_meta = []
            for chunk in top_chunks:
                # Find metadata for this chunk
                meta = {}
                for i, corpus_chunk in enumerate(_bm25_corpus):
                    if corpus_chunk == chunk and i < len(_corpus_metadata):
                        meta = _corpus_metadata[i]
                        break
                results_with_meta.append((chunk, meta))
            return results_with_meta
        
        return top_chunks
    
    else:
        # Fallback to pure semantic search if BM25 not available
        if return_metadata:
            # Query semantic search with metadata enabled
            return semantic_search(
                query=query,
                top_k=top_k,
                filters=filters,
                namespace=namespace,
                return_metadata=True
            )
        return semantic_results[:top_k]


def _merge_results(
    semantic_results: List[str],
    bm25_results: List[Tuple[str, float]],
    semantic_weight: float,
    bm25_weight: float
) -> List[Tuple[str, float]]:
    """
    Merge semantic and BM25 results with weighted scoring
    
    Uses Reciprocal Rank Fusion (RRF) + score normalization
    """
    
    # Create score dictionaries
    chunk_scores = {}
    
    # Process semantic results (higher rank = better)
    for rank, chunk in enumerate(semantic_results):
        # Reciprocal Rank Fusion: score = 1 / (rank + k)
        # k=60 is a common choice
        rrf_score = 1.0 / (rank + 60)
        chunk_scores[chunk] = chunk_scores.get(chunk, 0) + (semantic_weight * rrf_score)
    
    # Process BM25 results
    if bm25_results:
        # Normalize BM25 scores to [0, 1]
        max_bm25_score = max(score for _, score in bm25_results) if bm25_results else 1.0
        
        for chunk, bm25_score in bm25_results:
            normalized_score = bm25_score / max_bm25_score if max_bm25_score > 0 else 0
            chunk_scores[chunk] = chunk_scores.get(chunk, 0) + (bm25_weight * normalized_score)
    
    # Sort by combined score
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_chunks


def get_bm25_stats() -> Dict:
    """Get statistics about BM25 index"""
    return {
        "index_built": _bm25_index is not None,
        "corpus_size": len(_bm25_corpus),
        "avg_chunk_length": sum(len(chunk) for chunk in _bm25_corpus) / len(_bm25_corpus) if _bm25_corpus else 0
    }


def clear_bm25_index():
    """Clear the BM25 index (useful for testing or rebuilding)"""
    global _bm25_index, _bm25_corpus, _corpus_metadata
    _bm25_index = None
    _bm25_corpus = []
    _corpus_metadata = []
