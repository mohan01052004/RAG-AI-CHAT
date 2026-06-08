from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
import shutil
import os

from app.database import get_db
from app.models import Document, DocumentTopic
from app.services.pdf_loader import extract_text_from_pdf, extract_pages_from_pdf
from app.services.chunker import chunk_with_metadata, get_chunking_stats
from app.services.advanced_pdf_parser import extract_structured_content
from app.services.semantic_chunker import hierarchical_chunk
from app.services.pinecone_service import upload_chunks

router = APIRouter()

# Enable advanced parsing - set to True to use new semantic chunking with structure preservation
USE_ADVANCED_PARSING = True

@router.post("/upload")
async def upload_pdf(
    subject: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    path = f"temp_{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        if USE_ADVANCED_PARSING:
            # ✨ PHASE 6: Advanced PDF parsing with structure preservation
            print(f"🔍 Using advanced PDF parsing for '{file.filename}'")
            sections = extract_structured_content(path)
            
            if not sections:
                raise HTTPException(status_code=400, detail="No content extracted from document.")
            
            print(f"📄 Extracted {len(sections)} sections from PDF")
            print(f"   - Pages: {max([s.page_number for s in sections]) if sections else 0}")
            print(f"   - Headings: {len([s for s in sections if s.section_type == 'heading'])}")
            print(f"   - Paragraphs: {len([s for s in sections if s.section_type == 'paragraph'])}")
            print(f"   - Lists: {len([s for s in sections if s.section_type == 'list'])}")
            
            # Create document record
            doc = Document(
                filename=file.filename,
                subject=subject
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            
            # ✨ PHASE 6: Hierarchical semantic chunking
            chunks_with_meta = hierarchical_chunk(
                sections,
                max_chunk_size=800,
                min_chunk_size=200,
                overlap=100
            )
            
            print(f"✂️ Created {len(chunks_with_meta)} semantic chunks")
            
        else:
            # Original basic parsing (fallback)
            pages = extract_pages_from_pdf(path)
            text = extract_text_from_pdf(path)
            
            if not text:
                raise HTTPException(status_code=400, detail="No text found in the uploaded document.")
            
            doc = Document(
                filename=file.filename,
                subject=subject
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            
            # Original chunking logic
            chunks_with_meta = []
            if pages:
                for page_num, page_text in pages:
                    if not page_text.strip():
                        continue
                    page_chunks = chunk_with_metadata(page_text)
                    for chunk_text, chunk_meta in page_chunks:
                        chunk_meta["page"] = page_num
                        chunks_with_meta.append((chunk_text, chunk_meta))
            else:
                chunks_with_meta = chunk_with_metadata(text)
    finally:
        if os.path.exists(path):
            os.remove(path)
    
    # Print chunking statistics
    plain_chunks = [chunk for chunk, meta in chunks_with_meta]
    stats = get_chunking_stats(plain_chunks)
    print(f"📝 Chunking stats for '{file.filename}': {stats}")
    
    try:
        # Prepare chunks with enriched metadata
        enriched_chunks = []
        topic_records = set()
        for chunk_text, chunk_meta in chunks_with_meta:
            section = chunk_meta.get("section") or chunk_meta.get("heading") or chunk_meta.get("title")
            topic = chunk_meta.get("topic")
            subtopic = chunk_meta.get("subtopic")
            page = chunk_meta.get("page") or chunk_meta.get("page_number")
            
            record_key = (
                doc.id,
                subject,
                file.filename,
                section or "",
                topic or "",
                subtopic or "",
                page or 0
            )
            topic_records.add(record_key)
            
            # Add document metadata to each chunk
            enriched_chunks.append((chunk_text, {
                **chunk_meta,
                "document_id": doc.id,
                "subject": subject,
                "filename": file.filename,
            }))
        
        upload_chunks(enriched_chunks)

        if topic_records:
            for record in topic_records:
                doc_id, subj, filename, section, topic, subtopic, page = record
                db.add(DocumentTopic(
                    document_id=doc_id,
                    subject=subj,
                    filename=filename,
                    section=section or None,
                    topic=topic or None,
                    subtopic=subtopic or None,
                    page=page or None,
                ))
            db.commit()
    except Exception as exc:
        db.delete(doc)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(exc)}") from exc

    return {
        "message": "uploaded",
        "document_id": doc.id,
        "chunks_created": len(enriched_chunks),
        "chunking_stats": stats,
        "parsing_mode": "advanced" if USE_ADVANCED_PARSING else "basic"
    }
