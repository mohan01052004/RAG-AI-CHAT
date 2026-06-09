"""
Practice Problem Generator - Difficulty-Based RAG

Generates practice problems from uploaded content with adjustable difficulty:
- Easy: Basic recall, definitions, simple MCQs
- Medium: Application, understanding, moderate problem-solving
- Hard: Analysis, synthesis, complex multi-step problems
"""

from app.services.rag_pipeline import generate_mcqs, _generate_with_gemini
from app.services.hybrid_search import hybrid_search, build_bm25_index, _bm25_index
from app.services.reranker import rerank_results
from app.services.query_expansion import expand_query
from app.services.multi_query_retrieval import multi_query_retrieve, smart_deduplication
from app.schemas import PracticeProblem, MCQOption
from app.config import GROQ_API_KEY, GROQ_MODEL
from typing import List, Optional
import uuid
import random
import json

def _extract_json_array(text: str) -> list:
    """Robustly extract a JSON array from text, ignoring leading/trailing text/symbols."""
    if not text:
        return []
    clean = text.strip()
    
    # Try finding the first '['
    start_idx = clean.find("[")
    if start_idx == -1:
        start_obj = clean.find("{")
        if start_obj != -1:
            end_obj = clean.rfind("}")
            if end_obj != -1 and end_obj > start_obj:
                try:
                    obj = json.loads(clean[start_obj:end_obj+1])
                    return [obj]
                except Exception:
                    pass
        return []
    
    # Now find all ']' from the end
    end_indices = []
    curr = len(clean) - 1
    while True:
        idx = clean.rfind("]", 0, curr + 1)
        if idx == -1 or idx < start_idx:
            break
        end_indices.append(idx)
        curr = idx - 1
        
    # Try parsing candidates from longest to shortest
    for end_idx in end_indices:
        try:
            candidate = clean[start_idx:end_idx + 1]
            import re
            cleaned_candidate = re.sub(r',\s*([\]\}])', r'\1', candidate)
            data = json.loads(cleaned_candidate)
            if isinstance(data, list):
                return data
        except Exception:
            continue
                
    return []


def _classify_difficulty_prompt(difficulty: str, question_type: str) -> dict:
    """Get prompt instructions based on difficulty level"""
    
    if difficulty == "easy":
        return {
            "description": "Basic recall and understanding",
            "instructions": [
                "Focus on definitions and fundamental concepts",
                "Use direct, straightforward questions",
                "Test basic knowledge and terminology",
                "Options should be clearly distinct",
                "Avoid tricky or ambiguous phrasing"
            ],
            "example_mcq": "What is the time complexity of binary search? A) O(n) B) O(log n) C) O(n²) D) O(1)",
            "hint_level": "Give clear hints that guide toward the answer"
        }
    elif difficulty == "hard":
        return {
            "description": "Complex analysis and synthesis",
            "instructions": [
                "Require multi-step reasoning",
                "Combine multiple concepts",
                "Include edge cases and advanced scenarios",
                "Options should be subtle and require careful analysis",
                "Test deep understanding and application"
            ],
            "example_mcq": "A cache has 4-way set associativity with 256 sets and 64-byte blocks. If the address is 32 bits, how many tag bits are needed?",
            "hint_level": "Provide minimal hints that encourage independent thinking"
        }
    else:  # medium
        return {
            "description": "Application and moderate problem-solving",
            "instructions": [
                "Test understanding and application",
                "Include moderate problem-solving",
                "Balance between recall and analysis",
                "Options should require some thought",
                "Mix conceptual and practical aspects"
            ],
            "example_mcq": "Which sorting algorithm has the best average-case time complexity? A) Bubble Sort B) Merge Sort C) Quick Sort D) Selection Sort",
            "hint_level": "Provide moderate hints that help without giving away the answer"
        }


def _ensure_bm25_loaded(document_id: Optional[int] = None, document_ids: Optional[List[int]] = None):
    """
    If BM25 index is empty (e.g. after server restart), rebuild it from PostgreSQL document_chunks.
    This ensures newly uploaded documents are always searchable even without Pinecone.
    """
    from app.services.hybrid_search import _bm25_index as current_index, build_bm25_index
    if current_index is not None:
        return  # Already loaded

    try:
        from app.database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        try:
            if document_ids:
                placeholders = ", ".join([str(int(d)) for d in document_ids])
                rows = db.execute(
                    text(f"SELECT content FROM document_chunks WHERE doc_id IN ({placeholders}) AND content IS NOT NULL ORDER BY chunk_index LIMIT 500")
                ).fetchall()
            elif document_id:
                rows = db.execute(
                    text("SELECT content FROM document_chunks WHERE doc_id = :doc_id AND content IS NOT NULL ORDER BY chunk_index LIMIT 500"),
                    {"doc_id": int(document_id)}
                ).fetchall()
            else:
                rows = db.execute(
                    text("SELECT content FROM document_chunks WHERE content IS NOT NULL ORDER BY id DESC LIMIT 500")
                ).fetchall()

            chunks = [row[0] for row in rows if row[0] and row[0].strip()]
            if chunks:
                build_bm25_index(chunks)
                print(f"[PRACTICE] Rebuilt BM25 index from PostgreSQL with {len(chunks)} chunks")
        finally:
            db.close()
    except Exception as e:
        print(f"[PRACTICE] Warning: Could not rebuild BM25 from PostgreSQL: {e}")


def _fetch_context_from_db(
    document_id: Optional[int] = None,
    document_ids: Optional[List[int]] = None,
    max_chunks: int = 25
) -> str:
    """
    Fetch document text chunks directly from PostgreSQL as a fallback when
    Pinecone/vector search returns no results. Returns joined context string.
    """
    try:
        from app.database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        try:
            if document_ids:
                placeholders = ", ".join([str(int(d)) for d in document_ids])
                rows = db.execute(
                    text(f"SELECT content FROM document_chunks WHERE doc_id IN ({placeholders}) AND content IS NOT NULL ORDER BY chunk_index LIMIT :limit"),
                    {"limit": max_chunks}
                ).fetchall()
            elif document_id:
                rows = db.execute(
                    text("SELECT content FROM document_chunks WHERE doc_id = :doc_id AND content IS NOT NULL ORDER BY chunk_index LIMIT :limit"),
                    {"doc_id": int(document_id), "limit": max_chunks}
                ).fetchall()
            else:
                # No specific doc specified — get latest uploaded doc chunks
                rows = db.execute(
                    text("SELECT content FROM document_chunks WHERE content IS NOT NULL ORDER BY id DESC LIMIT :limit"),
                    {"limit": max_chunks}
                ).fetchall()

            chunks = [row[0] for row in rows if row[0] and row[0].strip()]
            if chunks:
                print(f"[PRACTICE] PostgreSQL fallback retrieved {len(chunks)} chunks")
            return "\n\n".join(chunks)
        finally:
            db.close()
    except Exception as e:
        print(f"[PRACTICE] PostgreSQL fallback failed: {e}")
        return ""


def generate_practice_problems(
    topic: Optional[str] = None,
    subject: Optional[str] = None,
    difficulty: str = "medium",
    count: int = 5,
    question_type: str = "mcq",
    document_id: Optional[int] = None,
    document_ids: Optional[List[int]] = None
) -> List[PracticeProblem]:
    """
    Generate practice problems with specified difficulty from uploaded content.

    Strategy (most reliable first):
    1. Always fetch context from PostgreSQL directly (no embeddings needed).
    2. Optionally enrich via vector search if the embedding service is healthy.
    3. Generate via Groq (reliable) → Gemini → rule-based fallback.
    """

    retrieve_count = {"easy": 15, "medium": 25, "hard": 40}.get(difficulty, 25)
    diff_config = _classify_difficulty_prompt(difficulty, question_type)

    # ── STEP 1: Fetch from PostgreSQL (primary, always works) ─────────────────
    context_text = _fetch_context_from_db(
        document_id=document_id,
        document_ids=document_ids,
        max_chunks=retrieve_count
    )
    print(f"[PRACTICE] PostgreSQL context: {len(context_text)} chars")

    # ── STEP 2: Try vector search to enrich (best-effort, non-fatal) ──────────
    try:
        search_terms = []
        if topic:
            search_terms.append(topic)
        if subject and subject.lower() != "general":
            search_terms.append(subject)
        if not search_terms:
            try:
                from app.database import SessionLocal
                from app.models import Document
                _db = SessionLocal()
                try:
                    if document_ids:
                        docs = _db.query(Document).filter(Document.id.in_(document_ids)).all()
                    elif document_id:
                        docs = _db.query(Document).filter(Document.id == document_id).all()
                    else:
                        docs = []
                    for doc in docs:
                        if doc.subject and doc.subject.lower() != "general":
                            search_terms.append(doc.subject)
                        fname = doc.filename.rsplit('.', 1)[0] if '.' in doc.filename else doc.filename
                        search_terms.append(fname.replace('_', ' ').replace('-', ' '))
                finally:
                    _db.close()
            except Exception:
                pass

        query = (" ".join(search_terms) + " key concepts definitions") if search_terms else "key concepts definitions main topics"

        filters = {}
        if document_ids:
            filters["doc_id"] = {"$in": [str(d) for d in document_ids]}
        elif document_id:
            filters["doc_id"] = {"$eq": str(document_id)}

        _ensure_bm25_loaded(document_id=document_id, document_ids=document_ids)
        # Use rule-based expansion only to avoid Gemini rate-limit during expansion
        query_variations = expand_query(query, mode="rules", num_variations=2)

        def search_fn(q, k):
            return hybrid_search(q, top_k=k, filters=filters if filters else None)

        candidates = multi_query_retrieve(
            query_variations, search_fn,
            top_k_per_query=retrieve_count,
            final_top_k=retrieve_count,
            fusion_method="rrf"
        )
        if candidates:
            candidates = smart_deduplication(candidates, similarity_threshold=0.90)
            reranked = rerank_results(query, candidates, top_k=min(len(candidates), retrieve_count))
            vector_context = "\n\n".join([c for c in reranked if c]).strip()
            if vector_context and len(vector_context) > len(context_text):
                context_text = vector_context
                print(f"[PRACTICE] Vector search enriched context to {len(context_text)} chars")
    except Exception as ve:
        print(f"[PRACTICE] Vector search skipped (non-fatal): {ve}")

    if not context_text:
        print("[PRACTICE] No context from any source — cannot generate problems")
        return []

    # ── STEP 3: Generate problems ──────────────────────────────────────────────
    if question_type == "mcq":
        return _generate_mcq_problems(context_text, count, difficulty, diff_config, subject, topic)
    elif question_type == "theory":
        return _generate_theory_problems(context_text, count, difficulty, diff_config, subject, topic)
    elif question_type == "numerical":
        return _generate_numerical_problems(context_text, count, difficulty, diff_config, subject, topic)
    else:
        return []


def _generate_mcq_problems(
    context: str,
    count: int,
    difficulty: str,
    config: dict,
    subject: Optional[str],
    topic: Optional[str]
) -> List[PracticeProblem]:
    """Generate MCQ problems with specified difficulty.
    Tries Groq first (reliable), then Gemini, then rule-based fallback.
    """

    prompt = f"""You are a practice problem generator. Generate {count} multiple-choice questions from the provided document context.

DIFFICULTY LEVEL: {difficulty.upper()}
{config['description']}

INSTRUCTIONS:
{chr(10).join(f"• {inst}" for inst in config['instructions'])}

EXAMPLE ({difficulty}):
{config['example_mcq']}

CONTEXT FROM STUDY MATERIAL:
{context[:4000]}

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

    result = None

    # ── Try Groq first (most reliable on Render free tier) ──────────────────
    if GROQ_API_KEY:
        try:
            from groq import Groq
            groq_client = Groq(api_key=GROQ_API_KEY)
            groq_resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=2000,
            )
            result = groq_resp.choices[0].message.content
            print(f"[PRACTICE MCQ] Generated via Groq")
        except Exception as e:
            print(f"[PRACTICE MCQ GROQ ERROR] {e}")
            result = None

    # ── Gemini fallback ───────────────────────────────────────────────────────
    if not result or result == "RATE_LIMIT_EXCEEDED":
        result = _generate_with_gemini(prompt, temperature=0.85, max_tokens=2000)

    problems = []
    if result and result != "RATE_LIMIT_EXCEEDED":
        try:
            data = _extract_json_array(result)
            if data:
                for item in data[:count]:
                    options_dict = item.get("options", {})
                    
                    # Normalize options_dict to keys A, B, C, D
                    normalized_options = {"A": "", "B": "", "C": "", "D": ""}
                    if isinstance(options_dict, dict):
                        for key in ["A", "B", "C", "D"]:
                            if key in options_dict:
                                normalized_options[key] = str(options_dict[key])
                            elif key.lower() in options_dict:
                                normalized_options[key] = str(options_dict[key.lower()])
                        
                        # If still empty, try numeric keys
                        if not any(normalized_options.values()):
                            for idx, key in enumerate(["A", "B", "C", "D"]):
                                for num_key in [str(idx+1), idx+1]:
                                    if num_key in options_dict:
                                        normalized_options[key] = str(options_dict[num_key])
                                        break
                                        
                        # If still empty, grab first 4 values
                        if not any(normalized_options.values()):
                            vals = list(options_dict.values())
                            for idx, key in enumerate(["A", "B", "C", "D"]):
                                if idx < len(vals):
                                    normalized_options[key] = str(vals[idx])
                    elif isinstance(options_dict, list):
                        for idx, key in enumerate(["A", "B", "C", "D"]):
                            if idx < len(options_dict):
                                normalized_options[key] = str(options_dict[idx])
                            else:
                                normalized_options[key] = ""

                    # Shuffle options randomly and map the correct answer label accordingly
                    raw_correct = str(item.get("correct_answer", "A")).upper().strip()
                    if raw_correct not in ["A", "B", "C", "D"]:
                        raw_correct = "A"
                    
                    options_list = []
                    for label in ["A", "B", "C", "D"]:
                        is_correct = (label == raw_correct)
                        options_list.append((normalized_options[label], is_correct))
                    
                    random.shuffle(options_list)
                    
                    shuffled_options = {}
                    new_correct_answer = "A"
                    for idx, label in enumerate(["A", "B", "C", "D"]):
                        text, is_correct = options_list[idx]
                        shuffled_options[label] = text
                        if is_correct:
                            new_correct_answer = label

                    problems.append(PracticeProblem(
                        id=str(uuid.uuid4()),
                        question=item.get("question", ""),
                        question_type="mcq",
                        difficulty=difficulty,
                        subject=subject,
                        topic=topic,
                        options=[
                            MCQOption(label="A", text=shuffled_options["A"]),
                            MCQOption(label="B", text=shuffled_options["B"]),
                            MCQOption(label="C", text=shuffled_options["C"]),
                            MCQOption(label="D", text=shuffled_options["D"])
                        ],
                        correct_answer=new_correct_answer,
                        solution=item.get("explanation", item.get("solution", "")),
                        hints=item.get("hints", [])
                    ))
        except Exception as e:
            print(f"[PRACTICE MCQ PARSE ERROR] {e}\nRaw result: {result[:500] if result else 'None'}")

    # Fallback: generate simple problems from context
    if not problems:
        print("[PRACTICE MCQ] Both LLMs failed — using context fallback")
        problems = _generate_fallback_mcqs(context, count, difficulty, subject, topic)

    return problems


def _generate_theory_problems(
    context: str,
    count: int,
    difficulty: str,
    config: dict,
    subject: Optional[str],
    topic: Optional[str]
) -> List[PracticeProblem]:
    """Generate theory/explanation problems"""
    
    from app.services.rag_pipeline import _generate_with_gemini
    
    prompt = f"""Generate {count} {difficulty}-level theory questions that require detailed explanations.

DIFFICULTY: {difficulty.upper()}
{chr(10).join(f"• {inst}" for inst in config['instructions'])}

CONTEXT:
{context[:4000]}

For each question, provide:
1. A clear question
2. The correct answer/explanation
3. {config['hint_level']}

Generate exactly {count} questions in this JSON format:
[
  {{
    "question": "Your theory question here",
    "correct_answer": "Detailed correct answer with explanation",
    "hints": ["Hint 1", "Hint 2"]
  }}
]

Generate {count} theory questions now (JSON only):"""
    
    result = _generate_with_gemini(prompt, temperature=0.8, max_tokens=2000)
    
    problems = []
    if result and result != "RATE_LIMIT_EXCEEDED":
        try:
            data = _extract_json_array(result)
            if data:
                for item in data[:count]:
                    problems.append(PracticeProblem(
                        id=str(uuid.uuid4()),
                        question=item.get("question", ""),
                        question_type="theory",
                        difficulty=difficulty,
                        subject=subject,
                        topic=topic,
                        correct_answer=item.get("correct_answer", ""),
                        solution=item.get("correct_answer", ""),
                        hints=item.get("hints", [])
                    ))
        except Exception as e:
            print(f"Error parsing theory problems: {e}")
            print(f"LLM result: {result[:500]}...")
            pass
    
    # Fallback if LLM generation failed
    if not problems:
        problems = _generate_fallback_theory(context, count, difficulty, subject, topic)
    
    return problems


def _generate_numerical_problems(
    context: str,
    count: int,
    difficulty: str,
    config: dict,
    subject: Optional[str],
    topic: Optional[str]
) -> List[PracticeProblem]:
    """Generate numerical/calculation problems"""
    
    from app.services.rag_pipeline import _generate_with_gemini
    
    prompt = f"""Generate {count} {difficulty}-level numerical problems requiring calculations.

DIFFICULTY: {difficulty.upper()}
{chr(10).join(f"• {inst}" for inst in config['instructions'])}

CONTEXT:
{context[:4000]}

For each problem:
1. Clear problem statement with given values
2. Step-by-step solution
3. Final numerical answer
4. {config['hint_level']}

Generate exactly {count} numerical problems in this JSON format:
[
  {{
    "question": "Problem statement with values",
    "correct_answer": "Final numerical answer (e.g., '42' or '3.14')",
    "solution": "Step-by-step solution process",
    "hints": ["Hint 1", "Hint 2"]
  }}
]

Generate {count} numerical problems now (JSON only):"""
    
    result = _generate_with_gemini(prompt, temperature=0.7, max_tokens=2000)
    
    problems = []
    if result and result != "RATE_LIMIT_EXCEEDED":
        try:
            data = _extract_json_array(result)
            if data:
                for item in data[:count]:
                    problems.append(PracticeProblem(
                        id=str(uuid.uuid4()),
                        question=item.get("question", ""),
                        question_type="numerical",
                        difficulty=difficulty,
                        subject=subject,
                        topic=topic,
                        correct_answer=str(item.get("correct_answer", "")),
                        solution=item.get("solution", ""),
                        hints=item.get("hints", [])
                    ))
        except Exception as e:
            print(f"Error parsing numerical problems: {e}")
            print(f"LLM result: {result[:500]}...")
            pass
    
    # Fallback if LLM generation failed
    if not problems:
        problems = _generate_fallback_numerical(context, count, difficulty, subject, topic)
    
    return problems


def _generate_fallback_mcqs(
    context: str,
    count: int,
    difficulty: str,
    subject: Optional[str],
    topic: Optional[str]
) -> List[PracticeProblem]:
    """Fallback MCQ generation when LLM fails"""
    
    sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 30]
    problems = []
    
    for i in range(min(count, len(sentences))):
        sentence = sentences[i]
        # Randomize correct answer instead of always "A"
        correct_option = random.choice(["A", "B", "C", "D"])
        
        problems.append(PracticeProblem(
            id=str(uuid.uuid4()),
            question=f"Q{i+1}: Based on the study material, which statement is correct about: {sentence[:80]}...?",
            question_type="mcq",
            difficulty=difficulty,
            subject=subject,
            topic=topic,
            options=[
                MCQOption(label="A", text="Option A based on context"),
                MCQOption(label="B", text="Option B based on context"),
                MCQOption(label="C", text="Option C based on context"),
                MCQOption(label="D", text="Insufficient information")
            ],
            correct_answer=correct_option,
            solution=f"Based on the provided context in the study material. The correct answer is option {correct_option}.",
            hints=["Review the relevant section", "Focus on key concepts"]
        ))
    
    return problems


def _generate_fallback_theory(
    context: str,
    count: int,
    difficulty: str,
    subject: Optional[str],
    topic: Optional[str]
) -> List[PracticeProblem]:
    """Fallback theory question generation when LLM fails - only if context is suitable"""
    
    import re as regex
    
    # Check if context has meaningful theory content (definitions, explanations, concepts)
    theory_indicators = [
        'definition', 'concept', 'explain', 'description', 'principle',
        'theory', 'algorithm', 'process', 'method', 'technique', 'approach',
        'property', 'characteristic', 'feature', 'purpose', 'function'
    ]
    
    context_lower = context.lower()
    theory_count = sum(1 for indicator in theory_indicators if indicator in context_lower)
    
    # Only generate if we have enough theory-related content
    if theory_count < 2:
        return []  # Return empty if not enough theory content
    
    sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 50 and len(s.strip()) < 500]
    
    if not sentences:
        return []
    
    problems = []
    
    for i in range(min(count, len(sentences))):
        sentence = sentences[i]
        
        # Only use sentences that contain theory indicators
        if any(indicator in sentence.lower() for indicator in theory_indicators):
            problems.append(PracticeProblem(
                id=str(uuid.uuid4()),
                question=f"Explain or describe the following concept based on the study material: {sentence[:100]}...",
                question_type="theory",
                difficulty=difficulty,
                subject=subject,
                topic=topic,
                correct_answer=sentence,
                solution=f"Detailed explanation: {sentence}",
                hints=["Identify key concepts", "Provide definitions and examples", "Explain the significance"]
            ))
    
    return problems


def _generate_fallback_numerical(
    context: str,
    count: int,
    difficulty: str,
    subject: Optional[str],
    topic: Optional[str]
) -> List[PracticeProblem]:
    """Fallback numerical question generation when LLM fails - only if context has relevant numbers"""
    
    import re as regex
    
    # Look for number patterns in context that suggest calculations or technical values
    # Examples: time complexity O(n), cache size 256KB, algorithm runs in 2.5ms, etc.
    number_patterns = [
        r'(\d+(?:\.\d+)?)\s*(ms|seconds?|minutes?|hours?|us|ns)',  # Time values
        r'(\d+(?:\.\d+)?)\s*(kb|mb|gb|bytes?)',  # Size values
        r'O\s*\(\s*([n\d+\-\*/^]+)\s*\)',  # Big O notation
        r'(\d+(?:\.\d+)?)\s*(%|percent)',  # Percentages
        r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)',  # Dimensions
    ]
    
    found_patterns = []
    for pattern in number_patterns:
        found_patterns.extend(regex.findall(pattern, context, regex.IGNORECASE))
    
    # If we didn't find meaningful numerical content, return empty
    if not found_patterns:
        return []
    
    # Try to extract meaningful numerical contexts
    sentences = [s.strip() for s in context.split('.') if any(c.isdigit() for c in s) and len(s.strip()) > 20]
    
    if not sentences:
        return []
    
    problems = []
    
    for i in range(min(count, len(sentences))):
        sentence = sentences[i]
        
        problems.append(PracticeProblem(
            id=str(uuid.uuid4()),
            question=f"Based on the study material, analyze or calculate: {sentence[:80]}...",
            question_type="numerical",
            difficulty=difficulty,
            subject=subject,
            topic=topic,
            correct_answer="[Calculate based on the given context in study material]",
            solution=f"Reference from study material: {sentence}",
            hints=["Carefully read the problem statement", "Identify given values and required calculation", "Apply relevant formulas or concepts from the study material"]
        ))
    
    return problems
