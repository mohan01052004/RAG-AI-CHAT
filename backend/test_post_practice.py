import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(override=True)

import json
from app.services.practice_generator import generate_practice_problems, _generate_mcq_problems, _classify_difficulty_prompt
from app.services.rag_pipeline import _generate_with_gemini
from app.database import SessionLocal
from app.models import Document

def main():
    print("Running practice generation test on document ID 19...")
    
    # 1. Retrieve context first
    from app.services.hybrid_search import hybrid_search
    from app.services.multi_query_retrieval import multi_query_retrieve, smart_deduplication
    from app.services.reranker import rerank_results
    from app.services.query_expansion import expand_query
    
    document_ids = [19]
    filters = {"doc_id": {"$in": [str(d) for d in document_ids]}}
    
    query = "dlrl_m1 key concepts definitions explanations primary details"
    query_variations = expand_query(query, mode="auto", num_variations=2)
    
    def search_fn(q, k):
        return hybrid_search(q, top_k=k, filters=filters)
        
    candidates = multi_query_retrieve(
        query_variations,
        search_fn,
        top_k_per_query=25,
        final_top_k=25,
        fusion_method="rrf"
    )
    candidates = smart_deduplication(candidates, similarity_threshold=0.90)
    context = rerank_results(query, candidates, top_k=min(len(candidates), 25))
    context_text = "\n\n".join([c for c in context if c]).strip()
    
    print(f"Retrieved context length: {len(context_text)} chars")
    if not context_text:
        print("Context is empty! Check Pinecone indexing.")
        return
        
    # 2. Call LLM for MCQs and trace output
    difficulty = "medium"
    count = 5
    config = _classify_difficulty_prompt(difficulty, "mcq")
    
    prompt = f"""You are a practice problem generator. Generate {count} multiple-choice questions from the provided document context.

DIFFICULTY LEVEL: {difficulty.upper()}
{config['description']}

INSTRUCTIONS:
{chr(10).join(f"• {inst}" for inst in config['instructions'])}

EXAMPLE ({difficulty}):
{config['example_mcq']}

CONTEXT FROM STUDY MATERIAL:
{context_text[:4000]}

Generate exactly {count} questions in this JSON format:
[
  {{
    "question": "Clear, {difficulty}-level question",
    "options": {{
      "A": "First option",
      "B": "Second option",
      "C": "Third option",
      "D": "Fourth option"
    }},
    "correct_answer": "B",
    "explanation": "Detailed explanation with reasoning",
    "hints": ["Hint 1", "Hint 2"]
  }}
]

IMPORTANT: Return ONLY valid JSON array. No markdown, no extra text.
Generate {count} {difficulty}-difficulty MCQs now:"""

    print("\n--- Prompt length:", len(prompt))
    
    print("\nCalling _generate_with_gemini...")
    result = _generate_with_gemini(prompt, temperature=0.85, max_tokens=2000)
    print("\n--- LLM Result ---")
    print(result)
    print("------------------")
    
    if not result:
        print("Result is empty!")
        return
        
    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        json_match = clean.find("[")
        if json_match != -1:
            json_end = clean.rfind("]")
            if json_end != -1:
                json_str = clean[json_match:json_end + 1]
                data = json.loads(json_str)
                print(f"\nSuccessfully parsed JSON! Found {len(data)} problems.")
                for idx, p in enumerate(data):
                    print(f"  P{idx+1}: {p.get('question')}")
                    print(f"    Options: {p.get('options')}")
                    print(f"    Correct: {p.get('correct_answer')}")
            else:
                print("Could not find closing bracket ']'")
        else:
            print("Could not find opening bracket '['")
    except Exception as e:
        print(f"Parsing failed: {e}")

if __name__ == "__main__":
    main()
