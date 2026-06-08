"""
RAG Fusion - Combine Multiple Retrieval Strategies

This module fuses results from multiple retrieval approaches:
1. BM25 (keyword-based)
2. Semantic search (embedding-based)
3. HyDE retrieval (hypothetical document embeddings)
4. Self-Query (structured metadata filtering)

Combines their results using Reciprocal Rank Fusion (RRF) to get best of all strategies.
"""

from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import numpy as np


def reciprocal_rank_fusion(rank_lists: List[List[Tuple[str, float]]], 
                           k: int = 60) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) - combine multiple ranking lists.
    
    Formula: score = sum(1 / (k + rank)) for each ranking
    
    Args:
        rank_lists: List of ranked result lists. Each list contains (doc_id, score) tuples.
        k: Parameter for RRF formula (default 60)
        
    Returns:
        Fused ranking with combined scores
    """
    fused_scores = defaultdict(float)
    doc_metadata = {}
    
    # Calculate RRF scores
    for rank_list in rank_lists:
        for rank, (doc_id, score) in enumerate(rank_list, 1):
            rrf_score = 1.0 / (k + rank)
            fused_scores[doc_id] += rrf_score
            
            # Preserve metadata from first occurrence
            if doc_id not in doc_metadata:
                doc_metadata[doc_id] = score
    
    # Sort by fused score
    fused_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return fused_results


def combine_retrieval_scores(retrieval_results: Dict[str, List[Tuple[str, float]]],
                            weights: Dict[str, float] = None) -> List[Tuple[str, float]]:
    """
    Combine results from multiple retrieval methods using weighted averaging.
    
    Args:
        retrieval_results: Dict with retrieval method names as keys:
            - "bm25": List of (doc_id, score) tuples
            - "semantic": List of (doc_id, score) tuples
            - "hyde": List of (doc_id, score) tuples
            - "self_query": List of (doc_id, score) tuples
        weights: Dict with method names and their weights. 
                Defaults to equal weights if not specified.
        
    Returns:
        Combined ranked list
    """
    available_methods = set(retrieval_results.keys())
    
    if weights is None:
        # Equal weights
        weights = {method: 1.0 / len(available_methods) for method in available_methods}
    else:
        # Normalize weights
        total = sum(weights.get(m, 0) for m in available_methods)
        weights = {m: weights.get(m, 0) / total for m in available_methods}
    
    combined_scores = defaultdict(float)
    doc_metadata = {}
    
    # Normalize scores from each method and combine
    for method, results in retrieval_results.items():
        if not results:
            continue
            
        method_weight = weights.get(method, 0)
        if method_weight == 0:
            continue
        
        # Normalize scores in this method to 0-1 range
        scores = [score for _, score in results]
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for doc_id, score in results:
                normalized = (score - min_score) / score_range if score_range > 0 else 0.5
                combined_scores[doc_id] += normalized * method_weight
                
                # Preserve metadata
                if doc_id not in doc_metadata:
                    doc_metadata[doc_id] = {
                        "methods": defaultdict(float),
                        "text": ""
                    }
                doc_metadata[doc_id]["methods"][method] = score
    
    # Sort by combined score
    combined_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return combined_results


def fuse_retrieval_results(bm25_results: List[Tuple[str, float, Dict[str, Any]]],
                          semantic_results: List[Tuple[str, float, Dict[str, Any]]],
                          hyde_results: List[Tuple[str, float, Dict[str, Any]]] = None,
                          self_query_results: List[Tuple[str, float, Dict[str, Any]]] = None,
                          strategy: str = "rrf") -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Fuse results from multiple retrieval strategies.
    
    Args:
        bm25_results: List of (doc_id, score, metadata) from BM25
        semantic_results: List of (doc_id, score, metadata) from semantic search
        hyde_results: List of (doc_id, score, metadata) from HyDE retrieval
        self_query_results: List of (doc_id, score, metadata) from self-query
        strategy: Fusion strategy - "rrf" (reciprocal rank fusion) or "weighted"
        
    Returns:
        Fused results as list of (doc_id, score, metadata) tuples
    """
    
    # Build rank lists without metadata
    rank_lists = []
    metadata_map = {}
    
    # BM25 results
    if bm25_results:
        bm25_list = [(doc_id, score) for doc_id, score, meta in bm25_results]
        rank_lists.append(bm25_list)
        for doc_id, score, meta in bm25_results:
            if doc_id not in metadata_map:
                metadata_map[doc_id] = meta
    
    # Semantic results
    if semantic_results:
        semantic_list = [(doc_id, score) for doc_id, score, meta in semantic_results]
        rank_lists.append(semantic_list)
        for doc_id, score, meta in semantic_results:
            if doc_id not in metadata_map:
                metadata_map[doc_id] = meta
    
    # HyDE results
    if hyde_results:
        hyde_list = [(doc_id, score) for doc_id, score, meta in hyde_results]
        rank_lists.append(hyde_list)
        for doc_id, score, meta in hyde_results:
            if doc_id not in metadata_map:
                metadata_map[doc_id] = meta
    
    # Self-Query results
    if self_query_results:
        sq_list = [(doc_id, score) for doc_id, score, meta in self_query_results]
        rank_lists.append(sq_list)
        for doc_id, score, meta in self_query_results:
            if doc_id not in metadata_map:
                metadata_map[doc_id] = meta
    
    # Fuse using selected strategy
    if strategy == "rrf":
        fused = reciprocal_rank_fusion(rank_lists)
    else:
        # Weighted averaging
        retrieval_dict = {
            "bm25": [(d, s) for d, s, _ in (bm25_results or [])],
            "semantic": [(d, s) for d, s, _ in (semantic_results or [])],
        }
        if hyde_results:
            retrieval_dict["hyde"] = [(d, s) for d, s, _ in hyde_results]
        if self_query_results:
            retrieval_dict["self_query"] = [(d, s) for d, s, _ in self_query_results]
        
        # Adjust weights based on number of methods
        weights = {}
        if bm25_results:
            weights["bm25"] = 0.3
        if semantic_results:
            weights["semantic"] = 0.4
        if hyde_results:
            weights["hyde"] = 0.15
        if self_query_results:
            weights["self_query"] = 0.15
        
        fused = combine_retrieval_scores(retrieval_dict, weights)
    
    # Reconstruct with metadata
    result = []
    for doc_id, score in fused:
        metadata = metadata_map.get(doc_id, {})
        result.append((doc_id, score, metadata))
    
    return result


def deduplicate_fused_results(fused_results: List[Tuple[str, float, Dict[str, Any]]],
                             similarity_threshold: float = 0.90) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Remove near-duplicate results from fused set.
    
    Args:
        fused_results: Results from fuse_retrieval_results()
        similarity_threshold: Threshold for considering documents as duplicates
        
    Returns:
        Deduplicated results
    """
    if not fused_results:
        return fused_results
    
    # For simple deduplication without heavy computation, use content-based approach
    seen_contents = set()
    deduplicated = []
    
    for doc_id, score, metadata in fused_results:
        # Get text content for deduplication
        text_content = metadata.get("text", "")[:200]  # First 200 chars
        
        # Simple hash-based dedup (in production, use semantic similarity)
        content_hash = hash(text_content.lower())
        
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            deduplicated.append((doc_id, score, metadata))
    
    return deduplicated


def rank_by_relevance_diversity(fused_results: List[Tuple[str, float, Dict[str, Any]]],
                               diversity_weight: float = 0.1) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Re-rank fused results to promote diversity while maintaining relevance.
    
    Higher diversity_weight favors diverse topics/concepts.
    
    Args:
        fused_results: Results to re-rank
        diversity_weight: How much to weight diversity vs. relevance (0.0-1.0)
        
    Returns:
        Re-ranked results
    """
    if not fused_results:
        return fused_results
    
    # Track concepts seen so far
    concepts_seen = set()
    reranked = []
    
    for doc_id, score, metadata in fused_results:
        # Get concepts from metadata
        concepts = metadata.get("concepts", [])
        if isinstance(concepts, str):
            concepts = [concepts]
        
        # Calculate diversity score
        new_concepts = len(set(concepts) - concepts_seen)
        total_concepts = len(set(concepts)) or 1
        diversity_score = new_concepts / total_concepts
        
        # Combine relevance and diversity
        adjusted_score = (score * (1 - diversity_weight) + 
                         diversity_score * diversity_weight)
        
        reranked.append((doc_id, adjusted_score, metadata))
        concepts_seen.update(concepts)
    
    # Re-sort by adjusted score
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    return reranked


def apply_rag_fusion(retrieval_methods: Dict[str, List[Tuple[str, float, Dict[str, Any]]]],
                    deduplicate: bool = True,
                    promote_diversity: bool = False) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Apply complete RAG Fusion pipeline.
    
    Args:
        retrieval_methods: Dict of retrieval results from different methods:
            {
              "bm25": [(doc_id, score, metadata), ...],
              "semantic": [...],
              "hyde": [...],
              "self_query": [...]
            }
        deduplicate: Whether to remove duplicates
        promote_diversity: Whether to re-rank for diversity
        
    Returns:
        Final fused and processed results
    """
    
    # Extract individual result lists
    bm25_results = retrieval_methods.get("bm25", [])
    semantic_results = retrieval_methods.get("semantic", [])
    hyde_results = retrieval_methods.get("hyde")
    self_query_results = retrieval_methods.get("self_query")
    
    # Fuse using RRF
    fused = fuse_retrieval_results(
        bm25_results,
        semantic_results,
        hyde_results,
        self_query_results,
        strategy="rrf"
    )
    
    # Deduplicate if requested
    if deduplicate:
        fused = deduplicate_fused_results(fused)
    
    # Promote diversity if requested
    if promote_diversity:
        fused = rank_by_relevance_diversity(fused, diversity_weight=0.2)
    
    return fused
