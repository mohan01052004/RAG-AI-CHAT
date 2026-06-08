import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(override=True)

from sqlalchemy import create_engine, text
from app.database import SessionLocal
from app.models import Document
from app.services.practice_generator import generate_practice_problems
from app.services.rag_pipeline import generate_theory_answer

def main():
    print("=" * 60)
    print("1. QUERYING POSTGRES DATABASE FOR UPLOADED DOCUMENTS")
    db = SessionLocal()
    try:
        docs = db.query(Document).all()
        print(f"Found {len(docs)} documents in SQL database:")
        if not docs:
            print("No documents found! Please upload a document first.")
            return
        
        doc_ids = []
        for d in docs:
            # Count chunks
            conn = db.bind.connect()
            res = conn.execute(text("SELECT COUNT(*) FROM document_chunks WHERE doc_id = :doc_id"), {"doc_id": d.id})
            count = res.scalar()
            print(f"  - Document ID {d.id}: '{d.filename}' (Subject: {d.subject}) -> {count} chunks in DB")
            doc_ids.append(d.id)
            
        print("\n2. VERIFYING MCQ QUIZ GENERATION FOR THESE DOCUMENTS")
        print(f"Calling generate_practice_problems for document_ids={doc_ids}...")
        problems = generate_practice_problems(
            difficulty="medium",
            count=3,
            question_type="mcq",
            document_ids=doc_ids
        )
        print(f"Successfully generated {len(problems)} MCQ problems!")
        for i, p in enumerate(problems):
            print(f"\n  [Q{i+1}] {p.question}")
            if p.options:
                for opt in p.options:
                    print(f"    {opt.label}: {opt.text}")
                print(f"  Correct Answer: {p.correct_answer}")
                if p.solution:
                    print(f"  Explanation: {p.solution}")
                    
        print("\n3. VERIFYING THEORY ANSWER RAG GENERATION")
        print("Querying: 'Explain SQL index'...")
        answer_data = generate_theory_answer(
            question="Explain SQL index",
            document_ids=doc_ids,
            return_dict=True
        )
        print("\nStructured Response:")
        print(f"  Answer: {answer_data.get('answer')}")
        print(f"  Confidence: {answer_data.get('confidence')}")
        print(f"  Sources: {answer_data.get('sources')}")
        
    finally:
        db.close()
    print("=" * 60)

if __name__ == "__main__":
    main()
