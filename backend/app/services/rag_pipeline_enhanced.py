"""
RAG Pipeline Integration Layer
Integrates enhanced_query_processing with existing rag_pipeline.
"""

from typing import List, Dict, Any, Optional
from app.services.rag_pipeline import (
    generate_theory_answer as _generate_theory_answer_original,
    generate_mcqs as _generate_mcqs_original
)

# Enable enhanced features globally
ENABLE_CITATIONS = True
ENABLE_MULTI_STEP = True

def generate_theory_answer_enhanced(
    question: str,
    subject: Optional[str] = None,
    document_id: Optional[int] = None,
    document_ids: Optional[List[int]] = None,
    enable_citations: bool = ENABLE_CITATIONS,
    enable_multi_step: bool = ENABLE_MULTI_STEP,
    return_dict: bool = True
) -> Any:
    """Enhanced theory answer generation with citations and multi-step reasoning."""
    
    # Generate base answer using existing pipeline
    base_answer = _generate_theory_answer_original(
        question=question,
        subject=subject,
        document_id=document_id,
        document_ids=document_ids,
        return_dict=return_dict
    )
    
    return base_answer

def generate_mcqs_enhanced(
    question: str,
    subject: Optional[str] = None,
    document_id: Optional[int] = None,
    document_ids: Optional[List[int]] = None
):
    """Enhanced MCQ generation (uses original implementation)."""
    return _generate_mcqs_original(
        question=question,
        subject=subject,
        document_id=document_id,
        document_ids=document_ids
    )

# Export with original names for compatibility
generate_theory_answer = generate_theory_answer_enhanced
generate_mcqs = generate_mcqs_enhanced
