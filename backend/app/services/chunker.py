"""
Advanced Chunking Service - Phase 4
Implements semantic-aware chunking with improved strategies
"""

import re
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _is_heading(line: str) -> bool:
    """Detect if a line is likely a heading"""
    line = line.strip()
    if not line:
        return False
    
    # Check for common heading patterns
    patterns = [
        r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
        r'^[A-Z][A-Z\s]+:?$',  # "INTRODUCTION" or "INTRODUCTION:"
        r'^Chapter \d+',  # "Chapter 1"
        r'^Section \d+',  # "Section 3.2"
        r'^\d+\.\d+',  # "3.2 Subsection"
    ]
    
    for pattern in patterns:
        if re.match(pattern, line):
            return True
    
    # Short lines (< 60 chars) ending with colon or all caps
    if len(line) < 60 and (line.endswith(':') or line.isupper()):
        return True
    
    return False


def _split_by_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split text into sections with headings.
    Returns list of (heading, content) tuples.
    """
    lines = text.split('\n')
    sections = []
    current_heading = "Introduction"
    current_content = []
    
    for line in lines:
        if _is_heading(line):
            # Save previous section
            if current_content:
                sections.append((current_heading, '\n'.join(current_content)))
            current_heading = line.strip()
            current_content = []
        else:
            if line.strip():
                current_content.append(line)
    
    # Save last section
    if current_content:
        sections.append((current_heading, '\n'.join(current_content)))
    
    return sections if sections else [("Content", text)]


def _semantic_chunking(
    text: str,
    target_chunk_size: int = 1000,
    overlap_size: int = 200
) -> List[str]:
    """
    Semantic-aware chunking that respects document structure.
    Uses RecursiveCharacterTextSplitter with multiple separators.
    """
    # Define separators in order of priority
    separators = [
        "\n\n\n",  # Multiple blank lines (section breaks)
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentence endings
        "! ",      # Exclamation sentences
        "? ",      # Question sentences
        "; ",      # Semicolons
        ", ",      # Commas
        " ",       # Words
        ""         # Characters (last resort)
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_chunk_size,
        chunk_overlap=overlap_size,
        length_function=len,
        separators=separators,
        is_separator_regex=False
    )
    
    chunks = splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _extract_metadata_from_chunk(chunk: str, heading: str = "") -> dict:
    """
    Extract metadata from chunk for better retrieval.
    Returns dict with chunk characteristics.
    """
    topic, subtopic = _parse_heading_hierarchy(heading)
    metadata = {
        "heading": heading,
        "section": heading,
        "topic": topic,
        "subtopic": subtopic,
        "has_code": False,
        "has_formula": False,
        "has_list": False,
        "has_table": False,
        "word_count": len(chunk.split()),
        "char_count": len(chunk)
    }
    
    # Detect code blocks
    if "```" in chunk or re.search(r'(?:for|while|if|def|class|function)\s*\(', chunk):
        metadata["has_code"] = True
    
    # Detect formulas/equations
    if re.search(r'[=<>≤≥∑∫∂π]|\$.*\$|\\frac|\\sum|O\(.*\)', chunk):
        metadata["has_formula"] = True
    
    # Detect lists
    if re.search(r'^\s*[-*•]\s', chunk, re.MULTILINE) or re.search(r'^\s*\d+\.\s', chunk, re.MULTILINE):
        metadata["has_list"] = True
    
    # Detect tables
    if '|' in chunk and chunk.count('|') > 3:
        metadata["has_table"] = True
    
    return metadata


def _parse_heading_hierarchy(heading: str) -> tuple[str, str]:
    """Extract topic/subtopic from a heading string."""
    if not heading or heading.strip().lower() in {"introduction", "content"}:
        return "", ""

    text = heading.strip()

    # Split on colon or dash for topic/subtopic
    for delimiter in [":", "-"]:
        if delimiter in text:
            left, right = text.split(delimiter, 1)
            return left.strip(), right.strip()

    # Numeric sections like "3.2 Subsection Title"
    match = re.match(r"^(\d+(?:\.\d+)+)\s+(.*)$", text)
    if match:
        section_num, title = match.groups()
        return f"Section {section_num}", title.strip()

    # Chapters like "Chapter 3 - Trees"
    match = re.match(r"^Chapter\s+\d+\s*(.*)$", text, re.IGNORECASE)
    if match:
        remainder = match.group(1).strip()
        return "Chapter", remainder

    return text, ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Advanced chunking with semantic awareness.
    
    Phase 4 Enhancement:
    - Respects document structure (headings, paragraphs)
    - Uses recursive splitting with multiple separators
    - Maintains context across chunks with smart overlap
    - Preserves semantic boundaries
    
    Args:
        text: Input text to chunk
        chunk_size: Target size for each chunk (chars)
        overlap: Overlap between chunks (chars)
        
    Returns:
        List of text chunks
    """
    if not text or len(text.strip()) < 50:
        return [text] if text.strip() else []
    
    # Try to split by sections first
    sections = _split_by_sections(text)
    
    all_chunks = []
    
    for heading, content in sections:
        if not content.strip():
            continue
        
        # Apply semantic chunking to each section
        section_chunks = _semantic_chunking(content, chunk_size, overlap)
        
        # Add heading context to each chunk from this section
        for chunk in section_chunks:
            # Prepend heading if it's meaningful
            if heading and heading != "Introduction" and len(heading) < 100:
                enhanced_chunk = f"[{heading}]\n{chunk}"
            else:
                enhanced_chunk = chunk
            
            all_chunks.append(enhanced_chunk)
    
    # Fallback: if no sections detected, use direct semantic chunking
    if not all_chunks:
        all_chunks = _semantic_chunking(text, chunk_size, overlap)
    
    return all_chunks


def chunk_with_metadata(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Tuple[str, dict]]:
    """
    Chunk text and return with metadata for each chunk.
    
    Returns:
        List of (chunk_text, metadata_dict) tuples
    """
    sections = _split_by_sections(text)
    results = []
    
    for heading, content in sections:
        if not content.strip():
            continue
        
        section_chunks = _semantic_chunking(content, chunk_size, overlap)
        
        for chunk in section_chunks:
            enhanced_chunk = f"[{heading}]\n{chunk}" if heading and heading != "Introduction" else chunk
            metadata = _extract_metadata_from_chunk(chunk, heading)
            results.append((enhanced_chunk, metadata))
    
    return results if results else [(text, _extract_metadata_from_chunk(text, ""))]


def get_chunking_stats(chunks: List[str]) -> dict:
    """Get statistics about chunking quality"""
    if not chunks:
        return {"count": 0}
    
    sizes = [len(c) for c in chunks]
    word_counts = [len(c.split()) for c in chunks]
    
    return {
        "count": len(chunks),
        "avg_size": sum(sizes) / len(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "avg_words": sum(word_counts) / len(word_counts),
        "total_chars": sum(sizes)
    }