"""
Cross-Encoder Reranking Service for RAG Pipeline
Phase 3: Advanced Reranking

Cross-encoders provide more accurate relevance scoring by processing
query-document pairs jointly, unlike bi-encoders which encode separately.
"""

from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Tuple, Optional

# Global model instance for efficiency
_reranker_model = None
_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def _get_reranker_model():
    """Lazy load cross-encoder model"""
    global _reranker_model
    if _reranker_model is None:
        try:
            print(f"🔄 Loading cross-encoder model: {_model_name}")
            _reranker_model = CrossEncoder(_model_name, max_length=512)
            print(f"✅ Cross-encoder model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load cross-encoder: {e}")
            _reranker_model = None
    return _reranker_model


def rerank_results(
    query: str, 
    chunks: List[str], 
    top_k: int = 10,
    batch_size: int = 32
) -> List[str]:
    """
    Rerank retrieved chunks using cross-encoder for higher accuracy.
    
    Args:
        query: User's search query
        chunks: List of retrieved text chunks from hybrid search
        top_k: Number of top results to return after reranking
        batch_size: Batch size for cross-encoder inference
        
    Returns:
        List of reranked chunks (top_k most relevant)
    """
    if not chunks:
        return []
    
    # If fewer chunks than top_k, return all
    if len(chunks) <= top_k:
        return chunks
    
    model = _get_reranker_model()
    
    # Fallback: if model fails to load, return original order
    if model is None:
        print("⚠️ Cross-encoder unavailable, returning original order")
        return chunks[:top_k]
    
    try:
        # Prepare query-document pairs
        pairs = [[query, chunk] for chunk in chunks]
        
        # Get relevance scores from cross-encoder
        print(f"🔍 Reranking {len(chunks)} chunks with cross-encoder...")
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        
        # Sort by scores (descending)
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k reranked results
        reranked = [chunk for chunk, score in scored_chunks[:top_k]]
        
        # Log reranking impact
        top_score = scored_chunks[0][1]
        bottom_score = scored_chunks[-1][1] if len(scored_chunks) > 1 else top_score
        print(f"✅ Reranked to top {top_k} | Score range: [{bottom_score:.3f}, {top_score:.3f}]")
        
        return reranked
        
    except Exception as e:
        print(f"❌ Reranking failed: {e}")
        # Fallback to original order
        return chunks[:top_k]


def rerank_with_scores(
    query: str,
    chunks: List[str],
    top_k: int = 10,
    batch_size: int = 32
) -> List[Tuple[str, float]]:
    """
    Rerank chunks and return with relevance scores.
    
    Args:
        query: User's search query
        chunks: List of retrieved text chunks
        top_k: Number of top results to return
        batch_size: Batch size for inference
        
    Returns:
        List of tuples: (chunk, relevance_score)
    """
    if not chunks:
        return []
    
    if len(chunks) <= top_k:
        # Return all with dummy scores if fewer than top_k
        return [(chunk, 1.0) for chunk in chunks]
    
    model = _get_reranker_model()
    
    if model is None:
        # Fallback: return with dummy scores
        return [(chunk, 1.0) for chunk in chunks[:top_k]]
    
    try:
        pairs = [[query, chunk] for chunk in chunks]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        
        # Sort by scores
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks[:top_k]
        
    except Exception as e:
        print(f"❌ Reranking with scores failed: {e}")
        return [(chunk, 1.0) for chunk in chunks[:top_k]]


def get_reranker_info() -> dict:
    """Get information about the loaded reranker model"""
    model = _get_reranker_model()
    return {
        "model_name": _model_name,
        "is_loaded": model is not None,
        "model_type": "cross-encoder",
        "purpose": "Fine-grained relevance scoring for retrieved documents",
        "max_length": 512 if model else None
    }


def rerank_with_threshold(
    query: str,
    chunks: List[str],
    threshold: float = 0.5,
    batch_size: int = 32
) -> List[str]:
    """
    Rerank and filter chunks by relevance threshold.
    Only returns chunks with scores above threshold.
    
    Args:
        query: User's search query
        chunks: List of retrieved text chunks
        threshold: Minimum relevance score (0-1 scale after normalization)
        batch_size: Batch size for inference
        
    Returns:
        List of chunks with relevance scores above threshold
    """
    if not chunks:
        return []
    
    model = _get_reranker_model()
    
    if model is None:
        # Fallback: return all chunks
        return chunks
    
    try:
        pairs = [[query, chunk] for chunk in chunks]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        
        # Normalize scores to 0-1 range using sigmoid
        normalized_scores = 1 / (1 + np.exp(-np.array(scores)))
        
        # Filter by threshold
        filtered = [
            chunk for chunk, score in zip(chunks, normalized_scores)
            if score >= threshold
        ]
        
        print(f"✅ Filtered {len(chunks)} → {len(filtered)} chunks (threshold={threshold})")
        return filtered
        
    except Exception as e:
        print(f"❌ Threshold filtering failed: {e}")
        return chunks
