"""
Hypothetical Document Embeddings (HyDE)

Generate hypothetical answers/documents for a query to improve retrieval quality.
This addresses the mismatch between query language and document language.

Example:
  Query: "How to optimize database queries?"
  Hypothetical Docs:
    1. "Use indexing on frequently queried columns to speed up retrieval..."
    2. "Query optimization involves analyzing execution plans and avoiding full scans..."
    3. "Caching query results prevents redundant database accesses..."
"""

from typing import List, Dict, Any, Optional
from app.services.embeddings import embedding_model
import json


def _get_gemini_client():
    """Get Gemini client for hypothetical document generation"""
    from app.config import GEMINI_API_KEY, GEMINI_MODEL
    
    if GEMINI_API_KEY:
        try:
            from google import genai
            return ("genai", genai.Client(api_key=GEMINI_API_KEY), GEMINI_MODEL)
        except ImportError:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                return ("generativeai", genai.GenerativeModel(GEMINI_MODEL), GEMINI_MODEL)
            except ImportError:
                pass
    return None


def generate_hypothetical_documents(query: str, num_docs: int = 5, use_llm: bool = True) -> List[str]:
    """
    Generate hypothetical documents/answers that would be relevant to the query.
    
    Args:
        query: User query
        num_docs: Number of hypothetical documents to generate
        use_llm: Whether to use LLM (falls back to rule-based if fails)
        
    Returns:
        List of hypothetical document texts
    """
    if use_llm:
        docs = _generate_with_llm(query, num_docs)
        if docs:
            return docs
    
    return _generate_with_rules(query, num_docs)


def _generate_with_llm(query: str, num_docs: int) -> Optional[List[str]]:
    """Use LLM to generate hypothetical relevant documents"""
    client_info = _get_gemini_client()
    
    if not client_info:
        return None
    
    try:
        client_type, client, model = client_info
        
        prompt = f"""Generate {num_docs} short hypothetical answer snippets (2-3 sentences each) that would directly answer this user query.
        
Query: "{query}"

Generate realistic document snippets that could answer this query. Return as JSON array with "documents" key containing the snippets.
Example format:
{{"documents": ["Answer snippet 1 here...", "Answer snippet 2 here...", ...]}}

Important: Return ONLY valid JSON, no other text."""

        if client_type == "genai":
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0.7, "max_output_tokens": 1000}
            )
            response_text = getattr(response, "text", None)
        else:
            response = client.generate_content(
                prompt,
                generation_config={"temperature": 0.7, "max_output_tokens": 1000}
            )
            response_text = getattr(response, "text", None)
        
        if response_text:
            # Extract JSON
            json_match = response_text.find("{")
            if json_match != -1:
                json_end = response_text.rfind("}") + 1
                if json_end > json_match:
                    try:
                        result = json.loads(response_text[json_match:json_end])
                        docs = result.get("documents", [])
                        if docs:
                            return [str(d).strip() for d in docs[:num_docs]]
                    except:
                        pass
    except Exception as e:
        print(f"[WARNING] HyDE generation failed: {str(e)[:80]}")
    
    return None


def _generate_with_rules(query: str, num_docs: int) -> List[str]:
    """Rule-based hypothetical document generation"""
    query_lower = query.lower()
    documents = []
    
    # Analyze query to determine document types
    if any(word in query_lower for word in ['what', 'how', 'explain', 'describe']):
        # Explanatory documents
        documents.extend([
            f"The concept of {query} involves understanding the key principles and underlying mechanisms. "
            f"This is fundamental to comprehending how systems work together.",
            
            f"To understand {query}, we must consider the context and various factors that influence it. "
            f"Different perspectives provide valuable insights into this topic.",
            
            f"When studying {query}, it's important to break down complex ideas into simpler components. "
            f"This hierarchical approach helps build a solid foundation of knowledge.",
        ])
    
    if any(word in query_lower for word in ['best', 'improve', 'optimize', 'efficient']):
        # Optimization/best-practice documents
        documents.extend([
            f"To optimize {query}, consider implementing best practices that have proven effective. "
            f"Benchmarking against industry standards helps measure improvements.",
            
            f"Improving {query} requires analyzing current performance and identifying bottlenecks. "
            f"Incremental refinements often yield the most sustainable results.",
        ])
    
    if any(word in query_lower for word in ['example', 'case', 'scenario', 'real']):
        # Practical/example documents
        documents.extend([
            f"A practical example of {query} demonstrates how theoretical concepts apply in real situations. "
            f"This helps bridge the gap between theory and implementation.",
            
            f"In practice, {query} shows how principles are applied effectively. "
            f"Real-world scenarios reveal both challenges and solutions.",
        ])
    
    if any(word in query_lower for word in ['error', 'problem', 'issue', 'fix', 'debug']):
        # Problem-solving documents
        documents.extend([
            f"When encountering issues with {query}, systematic debugging helps isolate root causes. "
            f"Understanding common pitfalls prevents future problems.",
            
            f"Resolving {query} requires understanding the underlying mechanisms. "
            f"Documentation and testing are critical for validation.",
        ])
    
    # Generic fallback documents
    if not documents:
        documents.extend([
            f"Understanding {query} requires comprehensive knowledge of related concepts and their interactions.",
            
            f"The topic of {query} encompasses several important aspects that work together systematically.",
            
            f"Mastering {query} involves practice, understanding core principles, and applying knowledge practically.",
        ])
    
    # Pad with query-based variations if needed
    while len(documents) < num_docs:
        documents.append(
            f"Related to {query}, there are several important considerations and approaches to explore. "
            f"Each perspective offers unique insights and practical value."
        )
    
    return documents[:num_docs]


def get_hyde_embeddings(query: str, num_docs: int = 5) -> tuple[List[str], List[List[float]]]:
    """
    Generate hypothetical documents and their embeddings.
    
    Args:
        query: User query
        num_docs: Number of hypothetical documents to generate
        
    Returns:
        Tuple of (documents, embeddings)
    """
    # Generate hypothetical documents
    hyde_docs = generate_hypothetical_documents(query, num_docs)
    
    # Embed them
    embeddings = [embedding_model.embed_query(doc) for doc in hyde_docs]
    
    return hyde_docs, embeddings


def get_average_hyde_embedding(query: str, num_docs: int = 5) -> List[float]:
    """
    Get average embedding of hypothetical documents for the query.
    
    This averaged embedding can be used alongside the query embedding
    for more comprehensive retrieval.
    
    Args:
        query: User query
        num_docs: Number of hypothetical documents to generate
        
    Returns:
        Average embedding vector
    """
    import numpy as np
    
    hyde_docs, embeddings = get_hyde_embeddings(query, num_docs)
    
    # Convert to numpy array, compute mean, convert back
    avg_embedding = np.mean(np.array(embeddings), axis=0)
    return avg_embedding.tolist()


def hybrid_query_embedding(query: str, query_embedding: List[float], 
                          hyde_weight: float = 0.3) -> List[float]:
    """
    Combine query embedding with HyDE embedding for better retrieval.
    
    Args:
        query: User query
        query_embedding: Embedding of the original query
        hyde_weight: Weight for HyDE component (0.0-1.0)
        
    Returns:
        Combined embedding
    """
    import numpy as np
    
    # Get HyDE average embedding
    hyde_embedding = get_average_hyde_embedding(query, num_docs=5)
    
    # Combine with weights
    query_weight = 1.0 - hyde_weight
    combined = (
        np.array(query_embedding) * query_weight +
        np.array(hyde_embedding) * hyde_weight
    )
    
    # Normalize
    combined = combined / np.linalg.norm(combined)
    
    return combined.tolist()
