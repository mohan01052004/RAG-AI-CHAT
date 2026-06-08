"""
Self-Query RAG - Extract structured filters from natural language queries

This module intelligently extracts metadata filters and constraints from user queries
to enable more precise retrieval targeting.

Example:
  Query: "Show me medium difficulty DSA questions from the algorithms chapter"
  Extracted: {
    "query": "DSA questions",
    "difficulty": "medium",
    "topic": "algorithms",
    "filters": {"difficulty": "medium", "topic": "algorithms"}
  }
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from app.config import GEMINI_API_KEY, GEMINI_MODEL


def _get_gemini_client():
    """Get Gemini client for self-query extraction"""
    if GEMINI_API_KEY:
        try:
            from google import genai
            return ("genai", genai.Client(api_key=GEMINI_API_KEY))
        except ImportError:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                return ("generativeai", genai.GenerativeModel(GEMINI_MODEL))
            except ImportError:
                pass
    return None


def _extract_filters_with_llm(query: str) -> Dict[str, Any]:
    """Use LLM to extract structured filters from natural language query"""
    client_info = _get_gemini_client()
    
    if not client_info:
        return {"query": query, "filters": {}}
    
    try:
        prompt = f"""Extract structured metadata filters from this user query.
        
User Query: "{query}"

Identify and extract the following if present:
1. difficulty_level (easy, medium, hard)
2. topic (chapter/section/topic name)
3. question_type (mcq, theory, numerical, coding)
4. time_limit (duration in minutes if mentioned)
5. chapter_name (specific chapter or module)
6. concepts (key concepts mentioned)

Return ONLY a JSON object like this (include only fields that are explicitly mentioned):
{{
  "core_query": "the main question without filters",
  "filters": {{
    "difficulty": "extracted difficulty or null",
    "topic": "extracted topic or null",
    "type": "extracted type or null",
    "chapter": "extracted chapter or null",
    "concepts": ["list", "of", "concepts"]
  }},
  "confidence": 0.0 to 1.0 (how confident are you about the extraction)
}}"""

        client_type, client = client_info
        
        if client_type == "genai":
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 500}
            )
            response_text = getattr(response, "text", None)
        else:
            response = client.generate_content(
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 500}
            )
            response_text = getattr(response, "text", None)
        
        if response_text:
            import json
            # Extract JSON from response
            json_match = response_text.find("{")
            if json_match != -1:
                json_end = response_text.rfind("}") + 1
                if json_end > json_match:
                    try:
                        result = json.loads(response_text[json_match:json_end])
                        return result
                    except:
                        pass
    except Exception as e:
        print(f"[WARNING] Self-query extraction failed: {str(e)[:80]}")
    
    return {"query": query, "filters": {}}


def _rule_based_filter_extraction(query: str) -> Dict[str, Any]:
    """Rule-based extraction of common patterns"""
    query_lower = query.lower()
    filters = {}
    core_query = query
    
    # Difficulty patterns
    difficulty_patterns = {
        'easy': r'\b(easy|basic|beginner|simple|fundamental)\b',
        'medium': r'\b(medium|intermediate|moderate)\b',
        'hard': r'\b(hard|difficult|advanced|challenging|complex)\b'
    }
    
    for level, pattern in difficulty_patterns.items():
        if re.search(pattern, query_lower):
            filters['difficulty'] = level
            core_query = re.sub(pattern, '', core_query, flags=re.IGNORECASE).strip()
            break
    
    # Question type patterns
    type_patterns = {
        'mcq': r'\b(mcq|multiple choice|objective)\b',
        'theory': r'\b(theory|explain|describe|concept)\b',
        'numerical': r'\b(numerical|calculate|compute|math problem)\b',
        'coding': r'\b(code|program|algorithm implementation|write code)\b'
    }
    
    for qtype, pattern in type_patterns.items():
        if re.search(pattern, query_lower):
            filters['type'] = qtype
            core_query = re.sub(pattern, '', core_query, flags=re.IGNORECASE).strip()
            break
    
    # Topic/Chapter patterns
    chapter_pattern = r'(?:chapter|module|section|unit|part)\s+(?:on\s+)?(\w+(?:\s+\w+)*)'
    chapter_match = re.search(chapter_pattern, query_lower)
    if chapter_match:
        filters['chapter'] = chapter_match.group(1)
        core_query = re.sub(chapter_pattern, '', core_query, flags=re.IGNORECASE).strip()
    
    # Time limit patterns
    time_pattern = r'(?:within|in|limit|minutes?|mins?)\s+(\d+)\s*(?:minutes?|mins?)'
    time_match = re.search(time_pattern, query_lower)
    if time_match:
        filters['time_limit'] = int(time_match.group(1))
        core_query = re.sub(time_pattern, '', core_query, flags=re.IGNORECASE).strip()
    
    return {
        "core_query": core_query.strip(),
        "filters": filters,
        "confidence": 0.6
    }


def extract_filters(query: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Extract structured filters from a natural language query.
    
    Args:
        query: User's natural language query
        use_llm: Whether to use LLM for extraction (falls back to rules if fails)
        
    Returns:
        Dict with:
        - core_query: Query without filters
        - filters: Extracted metadata filters
        - confidence: Confidence score of extraction
    """
    if use_llm:
        result = _extract_filters_with_llm(query)
        if result.get('filters') and result.get('confidence', 0) > 0.5:
            return result
    
    # Fallback to rule-based
    return _rule_based_filter_extraction(query)


def build_filter_dict(extracted_filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert extracted filters into database query format.
    
    Args:
        extracted_filters: Output from extract_filters()
        
    Returns:
        Filter dict for Pinecone/database queries
    """
    filters = extracted_filters.get('filters', {})
    db_filters = {}
    
    # Map extracted fields to database fields
    if 'difficulty' in filters and filters['difficulty']:
        db_filters['difficulty'] = filters['difficulty']
    
    if 'chapter' in filters and filters['chapter']:
        db_filters['chapter'] = filters['chapter']
    
    if 'topic' in filters and filters['topic']:
        db_filters['topic'] = filters['topic']
    
    if 'type' in filters and filters['type']:
        db_filters['question_type'] = filters['type']
    
    if 'concepts' in filters and filters['concepts']:
        db_filters['concepts'] = {'$in': filters['concepts']}
    
    return db_filters


def route_query(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Route a query by extracting filters and returning processed query + filters.
    
    Args:
        query: User query
        
    Returns:
        Tuple of (processed_query, filters_dict)
    """
    extraction = extract_filters(query, use_llm=True)
    core_query = extraction.get('core_query', query)
    filters = build_filter_dict(extraction)
    
    return core_query, filters
