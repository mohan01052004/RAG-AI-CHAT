"""
Response Enhancement with Citations & Source Attribution - Phase 6

Adds transparency and verifiability to RAG responses by:
- Tracking source documents for each answer
- Adding inline citations
- Providing confidence scores
- Enabling users to verify information
"""

from typing import List, Dict, Tuple, Optional
import re
from collections import defaultdict


def extract_sources_from_context(
    context_chunks: List[str],
    metadata_list: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Extract source information from retrieved chunks.
    
    Args:
        context_chunks: List of text chunks used for generation
        metadata_list: Optional metadata for each chunk
        
    Returns:
        List of source dictionaries with document info
    """
    sources = []
    seen_docs = set()
    
    for i, chunk in enumerate(context_chunks):
        meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
        
        doc_id = meta.get("doc_id") or meta.get("document_id") or "unknown"
        filename = meta.get("filename", "Unknown Document")
        subject = meta.get("subject", "General")
        page = meta.get("page")
        section = meta.get("section") or meta.get("heading")
        topic = meta.get("topic")
        subtopic = meta.get("subtopic")
        
        # Create unique identifier for document
        doc_key = f"{doc_id}_{filename}"
        
        if doc_key not in seen_docs:
            sources.append({
                "source_id": len(sources) + 1,
                "document_id": doc_id,
                "filename": filename,
                "subject": subject,
                "pages": set([page]) if page else set(),
                "sections": set([section]) if section else set(),
                "topics": set([topic]) if topic else set(),
                "subtopics": set([subtopic]) if subtopic else set(),
                "chunk_preview": chunk[:150] + "..." if len(chunk) > 150 else chunk
            })
            seen_docs.add(doc_key)
        else:
            for src in sources:
                if src["document_id"] == doc_id and src["filename"] == filename:
                    if page:
                        src["pages"].add(page)
                    if section:
                        src["sections"].add(section)
                    if topic:
                        src["topics"].add(topic)
                    if subtopic:
                        src["subtopics"].add(subtopic)
                    break
    
    return sources


def add_inline_citations(
    answer: str,
    sources: List[Dict],
    style: str = "numeric"
) -> str:
    """
    Add inline citations to generated answer.
    
    Args:
        answer: Generated answer text
        sources: List of source documents
        style: Citation style ("numeric" [1], "author" (Source 1), "none")
        
    Returns:
        Answer with inline citations
    """
    if style == "none" or not sources:
        return answer
    
    # Split answer into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    
    # Add citations to sentences (distribute evenly)
    cited_sentences = []
    source_idx = 0
    
    for i, sentence in enumerate(sentences):
        # Add citation every few sentences
        if i > 0 and i % 3 == 0 and sources:
            source_id = sources[source_idx % len(sources)]["source_id"]
            if style == "numeric":
                sentence = sentence + f" [{source_id}]"
            elif style == "author":
                filename = sources[source_idx % len(sources)]["filename"]
                sentence = sentence + f" (Source: {filename})"
            source_idx += 1
        
        cited_sentences.append(sentence)
    
    return " ".join(cited_sentences)


def format_sources_list(sources: List[Dict]) -> str:
    """
    Format sources as a reference list.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Formatted sources string
    """
    if not sources:
        return ""
    
    lines = ["\n\n📚 **Sources:**\n"]
    
    for source in sources:
        source_id = source["source_id"]
        filename = source["filename"]
        subject = source.get("subject", "General")
        pages = sorted(p for p in source.get("pages", set()) if p)
        sections = sorted(s for s in source.get("sections", set()) if s)
        topics = sorted(t for t in source.get("topics", set()) if t)
        subtopics = sorted(s for s in source.get("subtopics", set()) if s)

        details = []
        if pages:
            details.append(f"Pages: {', '.join(str(p) for p in pages[:8])}{'…' if len(pages) > 8 else ''}")
        if sections:
            details.append(f"Sections: {', '.join(sections[:3])}{'…' if len(sections) > 3 else ''}")
        if topics:
            details.append(f"Topics: {', '.join(topics[:3])}{'…' if len(topics) > 3 else ''}")
        if subtopics:
            details.append(f"Subtopics: {', '.join(subtopics[:3])}{'…' if len(subtopics) > 3 else ''}")

        detail_text = f" — {'; '.join(details)}" if details else ""
        lines.append(f"[{source_id}] {filename} (Subject: {subject}){detail_text}")
    
    return "\n".join(lines)


def calculate_answer_confidence(
    context_chunks: List[str],
    question: str,
    answer: str
) -> Dict[str, float]:
    """
    Calculate confidence metrics for the generated answer.
    
    Args:
        context_chunks: Retrieved context
        question: User's question
        answer: Generated answer
        
    Returns:
        Dictionary with confidence scores
    """
    metrics = {
        "context_coverage": 0.0,
        "answer_completeness": 0.0,
        "source_diversity": 0.0,
        "overall_confidence": 0.0
    }
    
    if not context_chunks or not answer:
        return metrics
    
    # Context coverage: How much context was used
    total_context_chars = sum(len(c) for c in context_chunks)
    answer_chars = len(answer)
    metrics["context_coverage"] = min(answer_chars / max(total_context_chars, 1) * 5, 1.0)
    
    # Answer completeness: Check if answer is substantial
    min_expected_length = len(question.split()) * 10  # Rough heuristic
    metrics["answer_completeness"] = min(len(answer.split()) / max(min_expected_length, 50), 1.0)
    
    # Source diversity: Number of different sources used
    metrics["source_diversity"] = min(len(context_chunks) / 10, 1.0)
    
    # Overall confidence (weighted average)
    metrics["overall_confidence"] = (
        metrics["context_coverage"] * 0.3 +
        metrics["answer_completeness"] * 0.4 +
        metrics["source_diversity"] * 0.3
    )
    
    return metrics


def enhance_response_with_metadata(
    answer: str,
    context_chunks: List[str],
    question: str,
    metadata_list: Optional[List[Dict]] = None,
    include_citations: bool = True,
    include_confidence: bool = True
) -> Dict:
    """
    Enhance answer with citations, sources, and confidence scores.
    
    Args:
        answer: Generated answer
        context_chunks: Retrieved context chunks
        question: Original question
        metadata_list: Metadata for each chunk
        include_citations: Whether to add inline citations
        include_confidence: Whether to calculate confidence
        
    Returns:
        Enhanced response dictionary
    """
    # Extract sources
    sources = extract_sources_from_context(context_chunks, metadata_list)
    
    # Add citations if requested
    enhanced_answer = answer
    if include_citations and sources:
        enhanced_answer = add_inline_citations(answer, sources, style="numeric")
        enhanced_answer += format_sources_list(sources)
    
    # Calculate confidence
    confidence = {}
    if include_confidence and context_chunks:
        confidence = calculate_answer_confidence(context_chunks, question, answer)
    else:
        confidence = {"overall_confidence": None}
    
    return {
        "answer": enhanced_answer,
        "original_answer": answer,
        "sources": sources,
        "confidence": confidence,
        "num_sources": len(sources),
        "num_chunks": len(context_chunks)
    }


def add_confidence_indicator(answer: str, confidence_score: float) -> str:
    """
    Add visual confidence indicator to answer.
    
    Args:
        answer: Answer text
        confidence_score: Overall confidence (0-1)
        
    Returns:
        Answer with confidence indicator
    """
    if confidence_score >= 0.8:
        indicator = "🟢 High confidence"
    elif confidence_score >= 0.6:
        indicator = "🟡 Moderate confidence"
    else:
        indicator = "🔴 Low confidence"
    
    percentage = int(confidence_score * 100)
    header = f"\n\n*{indicator} ({percentage}%)*\n"
    
    return answer + header


def generate_answer_summary(
    answer: str,
    max_length: int = 100
) -> str:
    """
    Generate a brief summary of the answer.
    
    Args:
        answer: Full answer text
        max_length: Maximum length of summary
        
    Returns:
        Brief summary
    """
    # Take first sentence or first max_length chars
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    
    if sentences and len(sentences[0]) <= max_length:
        return sentences[0]
    
    # Truncate to max_length
    if len(answer) <= max_length:
        return answer
    
    return answer[:max_length].rsplit(' ', 1)[0] + "..."


def track_chunk_usage(
    context_chunks: List[str],
    answer: str
) -> List[Tuple[int, float]]:
    """
    Estimate which chunks were most used in the answer.
    
    Args:
        context_chunks: Retrieved chunks
        answer: Generated answer
        
    Returns:
        List of (chunk_index, usage_score) tuples
    """
    usage_scores = []
    answer_lower = answer.lower()
    
    for i, chunk in enumerate(context_chunks):
        chunk_lower = chunk.lower()
        
        # Count overlapping words
        chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
        answer_words = set(re.findall(r'\b\w+\b', answer_lower))
        
        if not chunk_words:
            usage_scores.append((i, 0.0))
            continue
        
        overlap = len(chunk_words & answer_words)
        score = overlap / len(chunk_words)
        
        usage_scores.append((i, score))
    
    # Sort by usage score (descending)
    usage_scores.sort(key=lambda x: x[1], reverse=True)
    
    return usage_scores


def format_enhanced_response(response_data: Dict, style: str = "detailed") -> str:
    """
    Format enhanced response for display.
    
    Args:
        response_data: Enhanced response dictionary
        style: "detailed", "compact", or "minimal"
        
    Returns:
        Formatted response string
    """
    answer = response_data["answer"]
    
    if style == "minimal":
        return answer
    
    if style == "compact":
        sources = response_data.get("sources", [])
        if sources:
            return answer + f"\n\n*Based on {len(sources)} source(s)*"
        return answer
    
    # Detailed format (default)
    confidence = response_data.get("confidence", {})
    if confidence.get("overall_confidence"):
        answer = add_confidence_indicator(answer, confidence["overall_confidence"])
    
    return answer
