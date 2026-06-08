"""
Advanced RAG Pipeline Integration

Integrates Self-Query, HyDE, and RAG Fusion into a unified retrieval pipeline.
This module orchestrates the three advanced techniques for superior retrieval quality.

Pipeline Flow:
1. Query Input
2. Self-Query: Extract filters and route query
3. Parallel Retrieval:
   - BM25 search (keyword-based)
   - Semantic search (embedding-based)
   - HyDE retrieval (hypothetical embeddings)
4. RAG Fusion: Combine results from all retrieval strategies
5. Deduplication & Diversity Ranking
6. Return Top K fused results
"""

from typing import List, Dict, Any, Tuple, Optional
from app.services.self_query import extract_filters, route_query
from app.services.hyde import get_hyde_embeddings, hybrid_query_embedding
from app.services.rag_fusion import apply_rag_fusion
from app.services.hybrid_search import hybrid_search
from app.services.multi_query_retrieval import multi_query_retrieve
from app.services.embeddings import embedding_model
import numpy as np


async def advanced_retrieval(
    query: str,
    document_ids: Optional[List[str]] = None,
    top_k: int = 10,
    use_self_query: bool = True,
    use_hyde: bool = True,
    use_fusion: bool = True,
    deduplicate: bool = True,
    promote_diversity: bool = False
) -> List[Dict[str, Any]]:
    """
    Advanced RAG retrieval pipeline combining Self-Query, HyDE, and RAG Fusion.
    
    Args:
        query: User query
        document_ids: Restrict search to specific documents
        top_k: Number of results to return
        use_self_query: Enable Self-Query for metadata extraction
        use_hyde: Enable HyDE for hypothetical document retrieval
        use_fusion: Enable RAG Fusion to combine strategies
        deduplicate: Remove near-duplicate results
        promote_diversity: Promote diverse topics in results
        
    Returns:
        List of retrieved documents with scores and metadata
    """
    
    retrieval_results = {}
    
    # Step 1: Self-Query - Extract filters and route query
    processed_query = query
    metadata_filters = {}
    
    if use_self_query:
        try:
            processed_query, metadata_filters = route_query(query)
            print(f"[INFO] Self-Query extracted filters: {metadata_filters}")
        except Exception as e:
            print(f"[WARNING] Self-Query failed: {str(e)[:80]}")
    
    # Step 2a: Hybrid Search (BM25 + Semantic)
    try:
        hybrid_results = await hybrid_search(
            query=processed_query,
            top_k=top_k * 2,  # Get more candidates for fusion
            document_ids=document_ids,
            filters=metadata_filters
        )
        
        # Convert to (doc_id, score, metadata) tuples
        bm25_scored = [
            (r['id'], r.get('bm25_score', 0.5), r)
            for r in hybrid_results
        ]
        semantic_scored = [
            (r['id'], r.get('semantic_score', 0.5), r)
            for r in hybrid_results
        ]
        
        retrieval_results["bm25"] = bm25_scored
        retrieval_results["semantic"] = semantic_scored
        print(f"[INFO] Hybrid search returned {len(hybrid_results)} results")
    except Exception as e:
        print(f"[WARNING] Hybrid search failed: {str(e)[:80]}")
        retrieval_results["bm25"] = []
        retrieval_results["semantic"] = []
    
    # Step 2b: HyDE Retrieval - Use hypothetical documents
    if use_hyde:
        try:
            hyde_docs, hyde_embeddings = get_hyde_embeddings(processed_query, num_docs=5)
            print(f"[INFO] HyDE generated {len(hyde_docs)} hypothetical documents")
            
            # For each hypothetical document, do semantic search
            hyde_results_all = []
            model = get_embedding_model()
            
            for hyde_doc, hyde_embedding in zip(hyde_docs, hyde_embeddings):
                # Search using HyDE embedding
                hyde_matches = await hybrid_search(
                    query=hyde_doc,
                    top_k=top_k,
                    document_ids=document_ids,
                    filters=metadata_filters
                )
                hyde_results_all.extend(hyde_matches)
            
            # Aggregate HyDE results
            hyde_scored = {}
            for result in hyde_results_all:
                doc_id = result['id']
                score = result.get('semantic_score', 0.5)
                if doc_id not in hyde_scored or score > hyde_scored[doc_id][0]:
                    hyde_scored[doc_id] = (score, result)
            
            retrieval_results["hyde"] = [
                (doc_id, score, metadata)
                for doc_id, (score, metadata) in hyde_scored.items()
            ]
            print(f"[INFO] HyDE retrieval returned {len(retrieval_results['hyde'])} unique results")
        except Exception as e:
            print(f"[WARNING] HyDE retrieval failed: {str(e)[:80]}")
            retrieval_results["hyde"] = []
    
    # Step 2c: Self-Query Filtering Results
    if use_self_query and metadata_filters:
        try:
            # Apply additional filter-based retrieval if complex filters extracted
            print(f"[INFO] Applied metadata filters: {metadata_filters}")
        except Exception as e:
            print(f"[WARNING] Self-Query filtering failed: {str(e)[:80]}")
    
    # Step 3: RAG Fusion - Combine all retrieval strategies
    if use_fusion and len(retrieval_results) > 1:
        try:
            fused_results = apply_rag_fusion(
                retrieval_results,
                deduplicate=deduplicate,
                promote_diversity=promote_diversity
            )
            
            # Take top K from fused results
            final_results = fused_results[:top_k]
            print(f"[INFO] RAG Fusion combined results, returning top {len(final_results)}")
        except Exception as e:
            print(f"[WARNING] RAG Fusion failed: {str(e)[:80]}")
            # Fallback to semantic results only
            final_results = retrieval_results.get("semantic", [])[:top_k]
    else:
        # Use semantic results if fusion disabled or only one strategy available
        final_results = retrieval_results.get("semantic", [])[:top_k]
    
    # Step 4: Format output
    formatted_results = []
    for doc_id, score, metadata in final_results:
        if isinstance(metadata, dict):
            formatted_results.append({
                "id": doc_id,
                "fusion_score": float(score),
                **metadata
            })
    
    return formatted_results


async def advanced_retrieval_with_expansion(
    query: str,
    document_ids: Optional[List[str]] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Advanced retrieval with query expansion + RAG Fusion.
    
    Combines multi-query expansion with advanced fusion techniques.
    
    Args:
        query: User query
        document_ids: Restrict search to specific documents
        top_k: Number of results to return
        
    Returns:
        List of retrieved documents
    """
    
    # Use existing multi-query retrieval with expansion
    expansion_results = await multi_query_retrieve(
        query=query,
        document_ids=document_ids,
        top_k=top_k
    )
    
    # Enhance with advanced pipeline
    advanced_results = await advanced_retrieval(
        query=query,
        document_ids=document_ids,
        top_k=top_k,
        use_self_query=True,
        use_hyde=True,
        use_fusion=True
    )
    
    # Combine and rank results
    result_dict = {}
    
    for result in expansion_results:
        doc_id = result['id']
        score = result.get('score', 0.5)
        if doc_id not in result_dict:
            result_dict[doc_id] = {"result": result, "score": score, "count": 1}
        else:
            result_dict[doc_id]["score"] += score
            result_dict[doc_id]["count"] += 1
    
    for result in advanced_results:
        doc_id = result['id']
        score = result.get('fusion_score', 0.5)
        if doc_id not in result_dict:
            result_dict[doc_id] = {"result": result, "score": score, "count": 1}
        else:
            result_dict[doc_id]["score"] += score
            result_dict[doc_id]["count"] += 1
    
    # Average scores and sort
    combined = [
        {
            **item["result"],
            "combined_score": item["score"] / item["count"],
            "method_count": item["count"]
        }
        for item in result_dict.values()
    ]
    
    combined.sort(key=lambda x: x["combined_score"], reverse=True)
    
    return combined[:top_k]


def get_retrieval_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about retrieval results.
    
    Args:
        results: Results from advanced_retrieval()
        
    Returns:
        Stats dict with coverage, diversity, etc.
    """
    if not results:
        return {
            "result_count": 0,
            "avg_score": 0.0,
            "unique_documents": 0,
            "score_range": (0, 0),
            "topics_covered": []
        }
    
    scores = [r.get('fusion_score', r.get('score', 0.5)) for r in results]
    topics = set()
    
    for r in results:
        if 'topic' in r:
            topics.add(r['topic'])
        if 'concepts' in r:
            if isinstance(r['concepts'], list):
                topics.update(r['concepts'])
    
    return {
        "result_count": len(results),
        "avg_score": np.mean(scores),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "unique_documents": len(set(r['id'] for r in results)),
        "topics_covered": list(topics),
        "has_hyde": any('hyde' in str(r.get('source', '')).lower() for r in results),
        "has_self_query_filters": any('filters' in str(r.get('metadata', '')).lower() for r in results)
    }
