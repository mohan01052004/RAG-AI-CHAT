"""
Semantic Chunker Module
Implements intelligent chunking strategies with semantic boundaries,
variable chunk sizes, and context preservation.
"""

from typing import List, Tuple, Dict, Any, Optional
import re

try:
    from app.services.advanced_pdf_parser import PDFSection
except ImportError:
    # Fallback if PDFSection not available
    class PDFSection:
        def __init__(self, title, content, page_number, section_type, level, metadata=None):
            self.title = title
            self.content = content
            self.page_number = page_number
            self.section_type = section_type
            self.level = level
            self.metadata = metadata or {}

def hierarchical_chunk(
    sections: List[PDFSection],
    max_chunk_size: int = 800,
    min_chunk_size: int = 200,
    overlap: int = 100
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Chunk sections while respecting document hierarchy.
    
    Args:
        sections: List of PDFSection objects
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        overlap: Overlap between chunks
    
    Returns:
        List of (chunk_text, metadata) tuples with full metadata
    """
    chunks = []
    
    for section_idx, section in enumerate(sections):
        # Chunk each section
        section_chunks = _chunk_section(
            section,
            max_chunk_size,
            min_chunk_size,
            overlap
        )
        
        for chunk_idx, (chunk_text, chunk_meta) in enumerate(section_chunks):
            # Add section metadata (only serializable data for Pinecone)
            full_meta = {
                **chunk_meta,
                "section_index": section_idx,
                "chunk_in_section": chunk_idx,
                "page_number": section.page_number,
                "title": section.title,
                "section_type": section.section_type,
                "level": section.level
            }
            chunks.append((chunk_text, full_meta))
    
    return chunks

def _chunk_section(
    section: PDFSection,
    max_chunk_size: int,
    min_chunk_size: int,
    overlap: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """Chunk a single section while preserving metadata."""
    # Split into paragraphs
    paragraphs = _split_into_paragraphs(section.content)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if not para.strip():
            continue
        
        # Check if adding paragraph would exceed max size
        if len(current_chunk) + len(para) + 1 > max_chunk_size and len(current_chunk) >= min_chunk_size:
            chunks.append((current_chunk.strip(), {"position": len(chunks)}))
            # Create overlap
            current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk += "\n" + para
        else:
            current_chunk += "\n" + para if current_chunk else para
    
    # Add final chunk
    if len(current_chunk.strip()) >= min_chunk_size:
        chunks.append((current_chunk.strip(), {"position": len(chunks)}))
    
    return chunks

def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into semantic paragraphs."""
    # Split on double newline (or multiple newlines)
    paragraphs = re.split(r'\n\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]

def semantic_chunk(
    text: str,
    max_chunk_size: int = 800,
    min_chunk_size: int = 200,
    overlap: int = 100
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Semantic chunking with paragraph boundaries and sentence preservation.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        overlap: Overlap between chunks in characters
    
    Returns:
        List of (chunk_text, metadata) tuples
    """
    # Split into paragraphs (double newline = semantic boundary)
    paragraphs = _split_into_paragraphs(text)
    
    chunks = []
    current_chunk = ""
    chunk_metadata = {"chunk_index": 0}
    
    for para_idx, para in enumerate(paragraphs):
        if not para.strip():
            continue
        
        # Check if adding this paragraph would exceed max size
        if len(current_chunk) + len(para) + 1 > max_chunk_size:
            # If current chunk is too small, add paragraphs anyway
            if len(current_chunk) >= min_chunk_size:
                chunks.append((current_chunk.strip(), chunk_metadata.copy()))
                # Create overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n" + para
                chunk_metadata = {"chunk_index": len(chunks)}
            else:
                current_chunk += "\n" + para if current_chunk else para
        else:
            current_chunk += "\n" + para if current_chunk else para
    
    # Add final chunk
    if len(current_chunk.strip()) >= min_chunk_size:
        chunks.append((current_chunk.strip(), chunk_metadata))
    
    return chunks
