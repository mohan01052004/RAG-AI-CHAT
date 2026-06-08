"""
Query Expansion & Multi-Query Retrieval - Phase 5

Generates multiple query variations to improve retrieval recall.
Helps catch relevant documents that might not match the original query wording.
"""

from typing import List, Set
import re
from app.config import GEMINI_API_KEY, GEMINI_MODEL


_gemini_client = None


def _get_gemini_client():
    """Get or create Gemini client"""
    global _gemini_client
    if _gemini_client is None and GEMINI_API_KEY:
        try:
            from google import genai
            _gemini_client = ("genai", genai.Client(api_key=GEMINI_API_KEY))
        except ImportError:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                _gemini_client = ("generativeai", genai.GenerativeModel(GEMINI_MODEL))
            except ImportError:
                pass
    return _gemini_client


def _rule_based_expansion(query: str) -> List[str]:
    """
    Generate query variations using rule-based techniques.
    Fast fallback when LLM is unavailable.
    """
    variations = [query]
    query_lower = query.lower()
    
    # Add variations with common synonyms
    synonyms = {
        'algorithm': ['method', 'procedure', 'approach'],
        'explain': ['describe', 'what is', 'how does'],
        'difference': ['compare', 'distinguish', 'contrast'],
        'advantage': ['benefit', 'pros', 'strength'],
        'disadvantage': ['drawback', 'cons', 'limitation'],
        'time complexity': ['running time', 'efficiency', 'performance'],
        'space complexity': ['memory usage', 'space requirement'],
    }
    
    for term, alternatives in synonyms.items():
        if term in query_lower:
            for alt in alternatives:
                variation = query_lower.replace(term, alt)
                if variation not in [v.lower() for v in variations]:
                    variations.append(variation)
    
    # Add question variations
    if not any(q in query_lower for q in ['what', 'how', 'why', 'when', 'where']):
        if 'explain' in query_lower or 'describe' in query_lower:
            variations.append(f"What is {query}?")
            variations.append(f"How does {query} work?")
    
    # Limit to 5 variations for rule-based
    return variations[:5]


def expand_query_with_llm(query: str, num_variations: int = 3) -> List[str]:
    """
    Use LLM to generate diverse query variations.
    
    Args:
        query: Original user query
        num_variations: Number of variations to generate
        
    Returns:
        List of query variations (including original)
    """
    client_info = _get_gemini_client()
    
    if client_info is None:
        print("[WARNING] Gemini unavailable for query expansion, using rule-based fallback")
        return _rule_based_expansion(query)
    
    try:
        prompt = f"""Generate {num_variations} diverse paraphrases of this query for better document retrieval.

Original query: "{query}"

Requirements:
1. Keep the core meaning intact
2. Use different vocabulary and phrasing
3. Include technical and colloquial variations
4. Make each variation distinct
5. Focus on the domain of the query

Return ONLY the paraphrased queries, one per line, without numbering or explanation.

Paraphrased queries:"""

        client_type, client = client_info
        response_text = None
        
        if client_type == "genai":
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config={"temperature": 0.7, "max_output_tokens": 300}
                )
                response_text = getattr(response, "text", None)
            except Exception as e:
                print(f"[WARNING] New genai SDK failed, trying fallback: {str(e)[:80]}")
                return _rule_based_expansion(query)
        else:
            try:
                response = client.generate_content(
                    prompt,
                    generation_config={"temperature": 0.7, "max_output_tokens": 300}
                )
                response_text = getattr(response, "text", None)
            except Exception as e:
                print(f"[WARNING] Google generativeai SDK failed, trying fallback: {str(e)[:80]}")
                return _rule_based_expansion(query)
        
        if not response_text:
            return _rule_based_expansion(query)
        
        # Parse variations
        lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
        variations = [query]  # Always include original
        
        for line in lines:
            # Clean up (remove numbering, quotes, etc.)
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
            cleaned = cleaned.strip('"\'')
            
            if cleaned and cleaned.lower() != query.lower():
                variations.append(cleaned)
        
        # Limit to num_variations + 1 (original)
        variations = variations[:num_variations + 1]
        
        print(f"[INFO] Expanded query into {len(variations)} variations")
        return variations
        
    except Exception as e:
        print(f"[ERROR] Query expansion failed: {str(e)[:100]}")
        return _rule_based_expansion(query)


def expand_query(query: str, mode: str = "auto", num_variations: int = 3) -> List[str]:
    """
    Generate query variations with configurable mode.
    
    Args:
        query: Original query
        mode: "llm" (use LLM), "rules" (rule-based), "auto" (try LLM, fallback to rules)
        num_variations: Number of variations to generate
        
    Returns:
        List of query variations
    """
    if mode == "rules":
        return _rule_based_expansion(query)
    elif mode == "llm":
        return expand_query_with_llm(query, num_variations)
    else:  # auto
        return expand_query_with_llm(query, num_variations)


def multi_query_keywords(query: str) -> List[str]:
    """
    Extract key terms and generate keyword-focused variations.
    Useful for BM25 search.
    
    Args:
        query: Original query
        
    Returns:
        List of keyword variations
    """
    # Extract important terms
    stopwords = {'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    
    if not keywords:
        return [query]
    
    variations = [query]
    
    # All keywords together
    if len(keywords) > 1:
        variations.append(' '.join(keywords))
    
    # Important keyword subsets (2-3 most important)
    if len(keywords) >= 3:
        # Take first 3 and last 3 keywords (often most important)
        variations.append(' '.join(keywords[:3]))
        variations.append(' '.join(keywords[-3:]))
    
    # Individual important keywords as questions
    for kw in keywords[:3]:
        if len(kw) > 4:  # Skip very short keywords
            variations.append(f"What is {kw}")
            variations.append(f"{kw} explained")
    
    # Deduplicate
    seen = set()
    unique_variations = []
    for v in variations:
        if v.lower() not in seen:
            seen.add(v.lower())
            unique_variations.append(v)
    
    return unique_variations[:7]


def get_expansion_stats(original: str, variations: List[str]) -> dict:
    """Get statistics about query expansion"""
    return {
        "original_query": original,
        "num_variations": len(variations),
        "total_queries": len(variations),
        "avg_length": sum(len(v) for v in variations) / len(variations) if variations else 0,
        "variations": variations
    }
