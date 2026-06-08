"""
Multi-Query Retrieval with Result Fusion - Phase 5

Retrieves documents using multiple query variations and fuses results
using sophisticated ranking algorithms.
"""

from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import hashlib


def _extract_chunk_text(chunk: Any) -> str:
    """Normalize chunk to text for hashing/deduplication."""
    if isinstance(chunk, tuple) and chunk:
        return str(chunk[0])
    return str(chunk)


def _hash_chunk(chunk: Any) -> str:
    """Create hash for deduplication"""
    text = _extract_chunk_text(chunk)
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _reciprocal_rank_fusion(
    results_list: List[List[str]], 
    k: int = 60
) -> List[str]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).
    
    RRF score = sum over all lists: 1 / (rank + k)
    
    Args:
        results_list: List of ranked result lists from different queries
        k: Constant for RRF (default 60, from research)
        
    Returns:
        Fused and re-ranked results
    """
    # Calculate RRF scores
    scores = defaultdict(float)
    chunk_map = {}  # Hash to original chunk
    
    for results in results_list:
        for rank, chunk in enumerate(results):
            chunk_hash = _hash_chunk(chunk)
            chunk_map[chunk_hash] = chunk
            scores[chunk_hash] += 1.0 / (rank + k)
    
    # Sort by scores (descending)
    sorted_hashes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return original chunks in fused order
    return [chunk_map[chunk_hash] for chunk_hash, score in sorted_hashes]


def _weighted_fusion(
    results_list: List[List[str]],
    weights: Optional[List[float]] = None
) -> List[str]:
    """
    Fuse results with weighted voting.
    
    Args:
        results_list: List of ranked result lists
        weights: Weight for each query (if None, equal weights)
        
    Returns:
        Fused results
    """
    if weights is None:
        weights = [1.0] * len(results_list)
    
    scores = defaultdict(float)
    chunk_map = {}
    
    for weight, results in zip(weights, results_list):
        for rank, chunk in enumerate(results):
            chunk_hash = _hash_chunk(chunk)
            chunk_map[chunk_hash] = chunk
            # Higher weight for earlier ranks
            rank_score = 1.0 / (rank + 1)
            scores[chunk_hash] += weight * rank_score
    
    sorted_hashes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[chunk_hash] for chunk_hash, score in sorted_hashes]


def multi_query_retrieve(
    query_variations: List[str],
    search_function,
    top_k_per_query: int = 20,
    final_top_k: int = 10,
    fusion_method: str = "rrf"
) -> List[str]:
    """
    Retrieve documents using multiple query variations and fuse results.
    
    Args:
        query_variations: List of query variations
        search_function: Function that takes (query, top_k) and returns list of chunks
        top_k_per_query: Number of results to retrieve per query
        final_top_k: Number of final results after fusion
        fusion_method: "rrf" (reciprocal rank fusion) or "weighted"
        
    Returns:
        Fused and deduplicated results
    """
    if not query_variations:
        return []
    
    print(f"[INFO] Multi-query retrieval with {len(query_variations)} variations...")
    
    # Retrieve results for each query variation
    all_results = []
    for i, query_var in enumerate(query_variations):
        try:
            results = search_function(query_var, top_k_per_query)
            if results:
                all_results.append(results)
                print(f"  Query {i+1}: {len(results)} results")
        except Exception as e:
            print(f"  [WARNING] Query {i+1} failed: {e}")
            continue
    
    if not all_results:
        return []
    
    # Fuse results
    if fusion_method == "weighted":
        # Give higher weight to original query (first one)
        weights = [2.0] + [1.0] * (len(all_results) - 1)
        fused = _weighted_fusion(all_results, weights)
    else:  # rrf (default)
        fused = _reciprocal_rank_fusion(all_results)
    
    # Return top_k after fusion
    final_results = fused[:final_top_k]
    
    # Calculate deduplication stats
    total_retrieved = sum(len(r) for r in all_results)
    unique_count = len(set(_hash_chunk(c) for results in all_results for c in results))
    
    print(f"[INFO] Fused {total_retrieved} results -> {unique_count} unique -> top {len(final_results)}")
    
    return final_results


def multi_query_retrieve_with_filters(
    query_variations: List[str],
    search_function,
    filters: Optional[Dict] = None,
    top_k_per_query: int = 20,
    final_top_k: int = 10
) -> List[str]:
    """
    Multi-query retrieval with metadata filters.
    
    Args:
        query_variations: List of query variations
        search_function: Function that takes (query, top_k, filters)
        filters: Metadata filters (e.g., {"subject": "Data Structures"})
        top_k_per_query: Results per query
        final_top_k: Final results after fusion
        
    Returns:
        Fused results
    """
    def wrapped_search(query, top_k):
        return search_function(query, top_k, filters)
    
    return multi_query_retrieve(
        query_variations,
        wrapped_search,
        top_k_per_query,
        final_top_k
    )


def smart_deduplication(chunks: List[str], similarity_threshold: float = 0.95) -> List[str]:
    """
    Remove near-duplicate chunks based on text similarity.
    Uses simple character overlap for speed.
    
    Args:
        chunks: List of text chunks
        similarity_threshold: Threshold for considering chunks as duplicates
        
    Returns:
        Deduplicated chunks
    """
    if not chunks:
        return []
    
    def char_similarity(s1: str, s2: str) -> float:
        """Calculate character-level similarity"""
        set1 = set(s1.lower())
        set2 = set(s2.lower())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    deduplicated = []
    seen_hashes = set()
    
    for chunk in chunks:
        chunk_hash = _hash_chunk(chunk)
        
        # Exact duplicate check
        if chunk_hash in seen_hashes:
            continue
        
        # Near-duplicate check
        is_duplicate = False
        for existing in deduplicated:
            if char_similarity(chunk, existing) >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            deduplicated.append(chunk)
            seen_hashes.add(chunk_hash)
    
    if len(deduplicated) < len(chunks):
        print(f"[INFO] Deduplication: {len(chunks)} -> {len(deduplicated)} chunks")
    
    return deduplicated


def get_retrieval_stats(
    query_variations: List[str],
    results_per_query: List[int],
    final_count: int
) -> Dict:
    """Get statistics about multi-query retrieval"""
    return {
        "num_queries": len(query_variations),
        "total_retrieved": sum(results_per_query),
        "avg_per_query": sum(results_per_query) / len(results_per_query) if results_per_query else 0,
        "final_count": final_count,
        "dedup_ratio": final_count / sum(results_per_query) if sum(results_per_query) > 0 else 0
    }
