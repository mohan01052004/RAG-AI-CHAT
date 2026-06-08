"""
Advanced PDF Parser Module
Extracts structured content from PDFs with hierarchy preservation, heading detection,
and metadata tracking for better chunking and retrieval.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
from pypdf import PdfReader

@dataclass
class PDFSection:
    """Represents a section of a PDF with metadata."""
    title: str
    content: str
    page_number: int
    section_type: str  # 'heading', 'paragraph', 'list', 'table', 'code'
    level: int  # Hierarchy level (1=Chapter, 2=Section, 3+=Subsection)
    metadata: Dict[str, Any] = field(default_factory=dict)

def extract_structured_content(pdf_path: str) -> List[PDFSection]:
    """
    Extract structured content from PDF with hierarchy preservation.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        List of PDFSection objects with metadata
    """
    sections = []
    
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []
    
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if not text.strip():
            continue
        
        # Parse page structure
        page_sections = _parse_page_structure(text, page_num)
        sections.extend(page_sections)
    
    return sections

def _parse_page_structure(text: str, page_num: int) -> List[PDFSection]:
    """Parse a single page to extract structured content."""
    sections = []
    lines = text.split('\n')
    
    current_section = None
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # Check if this is a heading
        if _is_heading(stripped):
            # Save previous section
            if current_content and current_section:
                current_section.content = '\n'.join(current_content).strip()
                sections.append(current_section)
            
            # Start new section
            level = _detect_heading_level(stripped)
            current_section = PDFSection(
                title=stripped,
                content="",
                page_number=page_num,
                section_type="heading",
                level=level,
                metadata={"raw_title": stripped}
            )
            current_content = []
        
        # Check if this is a list item
        elif _is_list_item(stripped):
            if current_section is None or current_section.section_type != "list":
                # Save previous section
                if current_content and current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                # Start list section
                current_section = PDFSection(
                    title=f"List on page {page_num}",
                    content="",
                    page_number=page_num,
                    section_type="list",
                    level=2,
                    metadata={}
                )
                current_content = []
            
            current_content.append(stripped)
        
        # Regular paragraph
        else:
            if current_section is None or current_section.section_type == "list":
                # Save previous section
                if current_content and current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                # Start paragraph section
                current_section = PDFSection(
                    title=f"Content on page {page_num}",
                    content="",
                    page_number=page_num,
                    section_type="paragraph",
                    level=2,
                    metadata={}
                )
                current_content = []
            
            current_content.append(stripped)
    
    # Save final section
    if current_content and current_section:
        current_section.content = '\n'.join(current_content).strip()
        sections.append(current_section)
    
    return sections

def _is_heading(text: str) -> bool:
    """Check if text is a heading."""
    # Check for ALL CAPS
    if text.isupper() and len(text) > 3:
        return True
    
    # Check for numbered heading (1., 1.1, 1.1.1, etc.)
    if re.match(r'^[\d\.]+\s+', text):
        return True
    
    # Check for keyword headings
    keywords = ['Chapter', 'Section', 'Part', 'Module', 'Unit', 'Lesson', 'Topic']
    for keyword in keywords:
        if text.startswith(keyword):
            return True
    
    return False

def _detect_heading_level(text: str) -> int:
    """Detect hierarchy level of a heading."""
    # Count dots in numbered headings
    dots = text.count('.')
    if dots > 0:
        return min(dots + 1, 3)  # Max level 3
    
    # Check for keywords
    if text.startswith('Chapter'):
        return 1
    elif text.startswith('Section'):
        return 2
    else:
        return 2  # Default

def _is_list_item(text: str) -> bool:
    """Check if text is a list item."""
    # Bullet points
    if text.startswith(('•', '-', '*', '+', '○', '■')):
        return True
    
    # Numbered lists
    if re.match(r'^[\d]+\.\s+', text):
        return True
    
    # Lettered lists
    if re.match(r'^[a-z]\)\s+', text):
        return True
    
    return False

def get_section_hierarchy(sections: List[PDFSection]) -> Dict[int, List[PDFSection]]:
    """Organize sections by hierarchy level."""
    hierarchy = {}
    for section in sections:
        if section.level not in hierarchy:
            hierarchy[section.level] = []
        hierarchy[section.level].append(section)
    return hierarchy

def get_sections_by_page(sections: List[PDFSection], page: int) -> List[PDFSection]:
    """Get all sections from a specific page."""
    return [s for s in sections if s.page_number == page]

def get_sections_by_type(sections: List[PDFSection], section_type: str) -> List[PDFSection]:
    """Get all sections of a specific type."""
    return [s for s in sections if s.section_type == section_type]
