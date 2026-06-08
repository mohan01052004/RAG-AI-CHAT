from app.config import HUGGINGFACE_API_KEY, HUGGINGFACE_LLM_MODEL, GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL
from app.services.pinecone_service import search_similar
from app.services.hybrid_search import hybrid_search
from app.services.reranker import rerank_results
from app.services.query_expansion import expand_query
from app.services.multi_query_retrieval import multi_query_retrieve, smart_deduplication
from app.services.response_enhancement import enhance_response_with_metadata, format_enhanced_response
import json
import random
import re

_huggingface_client = None
_gemini_client = None
_groq_client = None

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


def _get_groq_client():
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=GROQ_API_KEY)
            print(f"[GROQ] Client initialized with model={GROQ_MODEL}")
        except Exception as e:
            print(f"[GROQ] Failed to initialize client: {e}")
    return _groq_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None and GEMINI_API_KEY:
        try:
            from google import genai
            _gemini_client = ("genai", genai.Client(api_key=GEMINI_API_KEY))
        except ImportError:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                _gemini_client = ("generativeai", genai.GenerativeModel(GEMINI_MODEL))
            except ImportError:
                pass
    return _gemini_client


def _get_huggingface_client():
    global _huggingface_client
    if _huggingface_client is None and HUGGINGFACE_API_KEY:
        try:
            from huggingface_hub import InferenceClient
            _huggingface_client = InferenceClient(token=HUGGINGFACE_API_KEY)
        except ImportError:
            pass
    return _huggingface_client


def _generate_with_gemini(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str | None:
    client_info = _get_gemini_client()
    result = None

    # ── Try Gemini ────────────────────────────────────────────────────────────
    if client_info is not None:
        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            client_type, client = client_info
            if client_type == "genai":
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=generation_config
                )
                result = getattr(response, "text", None)
            else:
                response = client.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                result = getattr(response, "text", None)
        except Exception as e:
            print(f"[GEMINI ERROR] Exception during generation: {e}")
            error_msg = str(e).lower()
            if "rate" in error_msg or "limit" in error_msg or "quota" in error_msg or "resource" in error_msg:
                result = "RATE_LIMIT_EXCEEDED"
            else:
                result = None

    # ── Groq fallback when Gemini failed or is rate-limited ───────────────────
    if result is None or result == "RATE_LIMIT_EXCEEDED":
        groq_client = _get_groq_client()
        if groq_client:
            try:
                groq_resp = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=min(max_tokens, 8000),
                )
                result = groq_resp.choices[0].message.content
                print(f"[GROQ FALLBACK] Successfully generated via Groq ({GROQ_MODEL})")
            except Exception as groq_e:
                print(f"[GROQ FALLBACK ERROR] {groq_e}")
                result = None

    return result


def _generate_with_huggingface(prompt: str) -> str | None:
    client = _get_huggingface_client()
    if client is None:
        print("[HF ERROR] _get_huggingface_client returned None")
        return None
    try:
        response = client.text_generation(
            prompt,
            model=HUGGINGFACE_LLM_MODEL,
            max_new_tokens=512,
            temperature=0.7
        )
        return response
    except Exception as e:
        print(f"[HF ERROR] Exception during generation: {e}")
        error_msg = str(e).lower()
        if "rate" in error_msg or "limit" in error_msg or "quota" in error_msg:
            return "RATE_LIMIT_EXCEEDED"
        return None


def _extractive_summary(text: str, max_sentences: int = 15) -> str:
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text)
        if len(s.strip()) > 25
    ]
    
    # Group sentences into logical paragraphs
    total_sentences = min(len(sentences), max_sentences)
    selected = sentences[:total_sentences]
    
    # Format as bullet points for better readability
    formatted = "\n\n".join([f"• {s}" for s in selected])
    return f"Summary (from context):\n\n{formatted}" if formatted else "Summary not available."


def _fallback_answer(context_text: str, question: str) -> str:
    return (
        "LLM not configured. Here is the most relevant context:\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_text[:2000]}"
    )


def _is_valid_query(query: str) -> bool:
    """Check if query is meaningful and not gibberish"""
    # Remove extra whitespace
    cleaned = query.strip()
    
    # Check minimum length (need at least reasonable length for meaningful query)
    if len(cleaned) < 5:
        return False
    
    # Split into words
    words = cleaned.split()
    if len(words) < 2:
        return False
    
    # Check for vowel/consonant pattern (gibberish like "lkhiu9i" has odd patterns)
    vowels = set('aeiouAEIOU')
    
    # Check for excessive non-letter characters
    alphanumeric_chars = re.findall(r'[a-zA-Z0-9]', cleaned)
    if len(alphanumeric_chars) == 0:
        return False
    
    # At least 70% should be alphanumeric
    ratio = len(alphanumeric_chars) / len(cleaned)
    if ratio < 0.7:
        return False
    
    # Check for word-like patterns (at least 2 words with 2+ letters each)
    word_pattern = re.findall(r'[a-zA-Z]{2,}', cleaned)
    if len(word_pattern) < 2:
        return False
    
    # Check that at least one word has vowels (meaningful words typically have vowels)
    meaningful_words = [w for w in word_pattern if any(c in vowels for c in w.lower())]
    if len(meaningful_words) < 1:
        return False
    
    return True


def _get_query_validation_message() -> str:
    """Return helpful message for invalid queries"""
    return (
        "❌ I didn't understand your query. Please provide a meaningful question.\n\n"
        "💡 **How to ask questions:**\n"
        "• Ask about concepts: 'Explain Dijkstra's algorithm'\n"
        "• Request practice questions: 'Give me 10 MCQs on sorting algorithms'\n"
        "• Ask for comparisons: 'Compare quicksort vs mergesort'\n"
        "• Request summaries: 'Summarize the content on dynamic programming'\n"
        "• Ask specific questions: 'What is time complexity of quicksort?'\n\n"
        "Try rephrasing your question with meaningful words."
    )


def _expand_abbreviations(query: str) -> str:
    """Expand common GATE abbreviations for better retrieval"""
    abbreviations = {
        r'\bds\b': 'data structures',
        r'\bdsa\b': 'data structures and algorithms',
        r'\bdbms\b': 'database management system',
        r'\bos\b': 'operating system',
        r'\bdn\b': 'digital networks',
        r'\bco\b': 'computer organization',
        r'\bcompilers\b': 'compiler design',
        r'\bcoa\b': 'computer organization and architecture',
        r'\btoc\b': 'theory of computation',
        r'\bcoa\b': 'computer organization',
        r'\bpq\b': 'priority queue',
        r'\bbst\b': 'binary search tree',
        r'\bavl\b': 'avl tree',
        r'\brbtree\b': 'red black tree',
        r'\bgraph\b': 'graph theory',
        r'\bstruct\b': 'structure',
        r'\barray\b': 'array data structure',
        r'\blinked\s+list\b': 'linked list',
        r'\bqueue\b': 'queue data structure',
        r'\bstack\b': 'stack data structure',
        r'\bhash\b': 'hash table',
        r'\bheap\b': 'heap data structure',
        r'\btree\b': 'tree data structure',
        r'\bgraph\b': 'graph data structure',
    }
    
    expanded = query.lower()
    for abbr, full_form in abbreviations.items():
        expanded = re.sub(abbr, full_form, expanded, flags=re.IGNORECASE)
    
    return expanded


def _classify_query(query: str) -> dict:
    """Classify query type and extract metadata for better retrieval"""
    q_lower = query.lower()
    
    classification = {
        "type": "general",
        "is_summary": False,
        "is_explanation": False,
        "is_example": False,
        "is_comparison": False,
        "is_formula": False,
        "is_numerical": False,
        "is_definition": False,
        "retrieve_count": 10,
    }
    
    # Detect query type
    if any(word in q_lower for word in ["summary", "summaries", "summarize", "summarise", "overview", "cover", "list all", "all topics", "entire"]):
        classification["type"] = "summary"
        classification["is_summary"] = True
        classification["retrieve_count"] = 20
    elif any(word in q_lower for word in ["explain", "what is", "how does", "describe", "tell me", "elaborate"]):
        classification["type"] = "explanation"
        classification["is_explanation"] = True
        classification["retrieve_count"] = 12
    elif any(word in q_lower for word in ["example", "show", "demonstrate", "illustrate", "use case"]):
        classification["type"] = "example"
        classification["is_example"] = True
        classification["retrieve_count"] = 10
    elif any(word in q_lower for word in ["compare", "difference", "vs", "versus", "contrast"]):
        classification["type"] = "comparison"
        classification["is_comparison"] = True
        classification["retrieve_count"] = 15
    elif any(word in q_lower for word in ["formula", "equation", "theorem", "law", "proof"]):
        classification["type"] = "formula"
        classification["is_formula"] = True
        classification["retrieve_count"] = 8
    elif any(word in q_lower for word in ["define", "definition", "what do you mean", "meaning"]):
        classification["type"] = "definition"
        classification["is_definition"] = True
        classification["retrieve_count"] = 6
    elif any(word in q_lower for word in ["calculate", "compute", "solve", "find", "numerical", "value"]):
        classification["type"] = "numerical"
        classification["is_numerical"] = True
        classification["retrieve_count"] = 10
    
    return classification


def generate_theory_answer(
    question: str,
    subject: str | None = None,
    document_id: int | None = None,
    document_ids: list[int] | None = None,
    return_dict: bool = False
):
    def make_response(ans_text, raw_ans=None, sources=None, confidence=None):
        if not return_dict:
            return ans_text
        return {
            "answer": ans_text,
            "original_answer": raw_ans or ans_text,
            "sources": sources or [],
            "confidence": confidence or {"overall_confidence": None}
        }

    # Validate query first
    if not _is_valid_query(question):
        return make_response(_get_query_validation_message())
    
    # If explicitly requested general AI chat (no documents selected/uploaded)
    if document_ids is not None and len(document_ids) == 0:
        general_prompt = f"""You are a professional, helpful AI assistant. Answer the user's question clearly and concisely.

User's Question: {question}

Provide a clear, focused answer:"""
        try:
            result = _generate_with_gemini(general_prompt, temperature=0.7, max_tokens=2500)
            if result == "RATE_LIMIT_EXCEEDED" or not result:
                result = _generate_with_huggingface(general_prompt)
            
            if result and result != "RATE_LIMIT_EXCEEDED":
                enhanced = enhance_response_with_metadata(
                    answer=result,
                    context_chunks=[],
                    question=question,
                    metadata_list=[],
                    include_citations=True,
                    include_confidence=True
                )
                if return_dict:
                    return enhanced
                return format_enhanced_response(enhanced, style="detailed")
            
            return make_response("❌ General AI response generation failed. Please try again.")
        except Exception as e:
            return make_response(f"❌ Error generating response: {str(e)}")
    
    # Classify query and expand abbreviations for better retrieval
    expanded_query = _expand_abbreviations(question)
    classification = _classify_query(question)
    
    filters = {}
    if subject:
        filters["subject"] = subject
    if document_ids:
        filters["doc_id"] = {"$in": [str(d) for d in document_ids]}
    elif document_id:
        filters["doc_id"] = {"$eq": str(document_id)}

    # Use classification to determine retrieval strategy
    top_k = classification["retrieve_count"]
    
    # Phase 5: Generate query variations for multi-query retrieval
    query_variations = expand_query(expanded_query, mode="auto", num_variations=2)
    
    # Multi-query retrieval with hybrid search (Phase 6: with metadata)
    def search_fn_with_meta(query, k):
        return hybrid_search(query, top_k=k, filters=filters if filters else None, return_metadata=True)
    
    # Retrieve more candidates using multiple queries
    initial_k = min(top_k * 3, 100)
    candidates_with_meta = multi_query_retrieve(
        query_variations,
        search_fn_with_meta,
        top_k_per_query=initial_k,
        final_top_k=initial_k,
        fusion_method="rrf"
    )
    
    # Separate chunks and metadata
    if candidates_with_meta and isinstance(candidates_with_meta[0], tuple):
        candidates = [chunk for chunk, meta in candidates_with_meta]
        metadata_list = [meta for chunk, meta in candidates_with_meta]
    else:
        candidates = candidates_with_meta
        metadata_list = [{} for _ in candidates]
    
    # Apply smart deduplication
    candidates = smart_deduplication(candidates, similarity_threshold=0.90)
    
    # Apply cross-encoder reranking for final top_k results
    is_out_of_domain = False
    
    # Check if the query is unrelated to the document content (out of domain)
    # Only perform domain check if user requested document search (not explicitly general chat)
    if document_ids is None or len(document_ids) > 0:
        if not candidates:
            is_out_of_domain = True
            context = []
            context_metadata = []
        else:
            from app.services.reranker import rerank_with_scores
            context_with_scores = rerank_with_scores(expanded_query, candidates, top_k=top_k)
            if context_with_scores:
                top_score = context_with_scores[0][1]
                # Scores below -5.0 indicate very low semantic similarity / unrelated topics
                if top_score < -5.0:
                    is_out_of_domain = True
                    context = []
                    context_metadata = []
                    print(f"[RAG] Out-of-domain query detected (top score: {top_score:.3f}). Switching to General AI.")
                else:
                    context = [chunk for chunk, score in context_with_scores]
                    # Keep metadata aligned with reranked context
                    context_metadata = []
                    for ctx_chunk in context:
                        for i, cand_chunk in enumerate(candidates):
                            if ctx_chunk == cand_chunk and i < len(metadata_list):
                                context_metadata.append(metadata_list[i])
                                break
                        else:
                            context_metadata.append({})
            else:
                context = []
                context_metadata = []
    else:
        # User requested General AI directly (empty document list)
        context = []
        context_metadata = []
    
    context_text = "\n\n".join([c for c in context if c]).strip()

    if not context_text:
        general_prompt = f"""You are a professional, helpful AI assistant. Answer the user's question clearly and concisely.

User's Question: {question}

Provide a clear, focused answer:"""
        try:
            result = _generate_with_gemini(general_prompt, temperature=0.7, max_tokens=2500)
            if result == "RATE_LIMIT_EXCEEDED" or not result:
                result = _generate_with_huggingface(general_prompt)
            
            if result and result != "RATE_LIMIT_EXCEEDED":
                if is_out_of_domain:
                    result = "⚠️ *Note: This question does not appear to be related to your uploaded documents. Answering using general knowledge.*\n\n" + result

                enhanced = enhance_response_with_metadata(
                    answer=result,
                    context_chunks=[],
                    question=question,
                    metadata_list=[],
                    include_citations=True,
                    include_confidence=True
                )
                if return_dict:
                    return enhanced
                return format_enhanced_response(enhanced, style="detailed")
            
            return make_response("❌ No relevant content found and general AI response generation failed. Please try again.")
        except Exception as e:
            return make_response(f"❌ Error generating response: {str(e)}")

    # Build targeted prompt based on query classification
    if classification["is_summary"]:
        prompt = f"""You are a professional, helpful document assistant. Create a comprehensive yet concise summary covering the key concepts.

Structure your answer with:
• Main concepts and definitions
• Important formulas or algorithms
• Key points to remember
• Multiple perspectives/approaches if available

Context from study material:
{context_text}

Student's Question: {question}

Provide a well-organized summary:"""

    elif classification["is_explanation"]:
        prompt = f"""You are a professional, helpful document assistant explaining a concept.

Instructions:
1. Start with a clear, simple explanation
2. Provide the technical definition
3. Explain step-by-step with examples
4. Highlight common mistakes/misunderstandings to avoid
5. Include formulas/diagrams if relevant (use LaTeX for formulas, ASCII diagrams if helpful)
6. End with "Key points to remember"
7. Provide multiple perspectives/approaches if available

Context from study material:
{context_text}

Student's Question: {question}

Provide a clear, detailed explanation:"""

    elif classification["is_comparison"]:
        prompt = f"""You are a professional, helpful document assistant explaining differences between concepts.

Instructions:
1. Clearly list the key differences
2. Provide comparison table if applicable
3. Give examples for each concept
4. Explain when to use which concept
5. Include formulas/diagrams if relevant (use LaTeX for formulas, ASCII diagrams if helpful)
6. End with "Key points to remember"
7. Provide multiple perspectives/approaches if available

Context from study material:
{context_text}

Student's Question: {question}

Provide a detailed comparison:"""

    elif classification["is_formula"]:
        prompt = f"""You are a professional, helpful document assistant explaining formulas and theorems.

Instructions:
1. State the formula/theorem clearly
2. Explain each component
3. Provide the derivation or proof if relevant
4. Show worked examples
5. Include formula in LaTeX and add diagram if useful
6. End with "Key points to remember"
7. Provide multiple perspectives/approaches if available

Context from study material:
{context_text}

Student's Question: {question}

Provide the formula with detailed explanation:"""

    elif classification["is_numerical"]:
        prompt = f"""You are a professional, helpful document assistant solving numerical problems.

Instructions:
1. Identify what is given and what to find
2. State the relevant formula/concept
3. Show step-by-step calculation
4. Provide final answer with units if applicable
5. Verify the solution
6. Include formulas in LaTeX and show intermediate steps clearly
7. End with "Key points to remember"
8. Provide multiple approaches if available

Context from study material:
{context_text}

Student's Question: {question}

Solve step-by-step:"""

    elif classification["is_definition"]:
        prompt = f"""You are a professional, helpful document assistant defining concepts.

Instructions:
1. Provide concise definition
2. Explain importance and core characteristics
3. Provide a real-world example
4. List related concepts
5. Include formulas/diagrams if relevant (use LaTeX for formulas, ASCII diagrams if helpful)
6. End with "Key points to remember"
7. Provide multiple perspectives/approaches if available

Context from study material:
{context_text}

Student's Question: {question}

Provide the definition:"""

    else:  # General explanation
        prompt = f"""You are a professional, helpful document assistant. Answer the student's question clearly and concisely.

Focus on:
• Key concepts from the context
• Practical examples where applicable
• Common mistakes to avoid
• Multiple perspectives/approaches if available
• Key points to remember
• Include formulas/diagrams if relevant (use LaTeX for formulas, ASCII diagrams if helpful)

Context from study material:
{context_text}

Student's Question: {question}

Provide a clear, focused answer:"""

    try:
        # Try Gemini first for high-quality responses
        result = _generate_with_gemini(prompt, temperature=0.7, max_tokens=2500)
        if result == "RATE_LIMIT_EXCEEDED" or not result:
            # Fallback to HuggingFace
            result = _generate_with_huggingface(prompt)
        
        if result and result != "RATE_LIMIT_EXCEEDED":
            # Phase 6: Enhance response with citations and confidence
            enhanced = enhance_response_with_metadata(
                answer=result,
                context_chunks=context,
                question=question,
                metadata_list=context_metadata,
                include_citations=True,
                include_confidence=True
            )
            if return_dict:
                return enhanced
            return format_enhanced_response(enhanced, style="detailed")
        
        # Fallback to extractive summary if both fail
        fallback_ans = _extractive_summary(context_text, max_sentences=20) if classification["is_summary"] else _fallback_answer(context_text, question)
        return make_response(fallback_ans)
        
    except Exception:
        fallback_ans = _extractive_summary(context_text, max_sentences=20) if classification["is_summary"] else _fallback_answer(context_text, question)
        return make_response(fallback_ans)

def generate_mcqs(
    question: str,
    subject: str | None = None,
    document_id: int | None = None,
    document_ids: list[int] | None = None
):
    """Generate MCQs in structured format using RAG with fallback"""
    # Validate query first
    if not _is_valid_query(question):
        from app.schemas import MCQQuestion, MCQOption, MCQResponse
        return MCQResponse(questions=[MCQQuestion(
            question=_get_query_validation_message(),
            options=[
                MCQOption(label="A", text="Please provide a meaningful question"),
                MCQOption(label="B", text="Example: 'Give me 10 MCQs on sorting'"),
                MCQOption(label="C", text="Example: 'What is time complexity?'"),
                MCQOption(label="D", text="Upload study materials first")
            ],
            correct_answer="A"
        )])
    
    filters = {}
    if subject:
        filters["subject"] = subject
    if document_ids:
        filters["doc_id"] = {"$in": [str(d) for d in document_ids]}
    elif document_id:
        filters["doc_id"] = {"$eq": str(document_id)}

    # Extract requested count
    from app.schemas import MCQQuestion, MCQOption, MCQResponse
    
    def _extract_count(text: str, default: int = 10, max_count: int = 500) -> int:
        match = re.search(r"(\d{1,3})", text.lower())
        if match:
            return max(1, min(int(match.group(1)), max_count))
        return default

    count = _extract_count(question)
    
    # Phase 5: Expand query for MCQ generation
    expanded_mcq_query = _expand_abbreviations(question)
    query_variations = expand_query(expanded_mcq_query, mode="auto", num_variations=2)
    
    # Multi-query retrieval for diverse MCQ content
    def search_fn(query, k):
        return hybrid_search(query, top_k=k, filters=filters if filters else None)
    
    # Retrieve MUCH more context for larger requests
    top_k = max(50, min(count * 2, 200))  # Dynamic retrieval based on request count
    initial_k = min(int(top_k * 1.5), 250)
    
    candidates = multi_query_retrieve(
        query_variations,
        search_fn,
        top_k_per_query=initial_k,
        final_top_k=initial_k,
        fusion_method="rrf"
    )
    
    # Apply smart deduplication
    candidates = smart_deduplication(candidates, similarity_threshold=0.90)
    
    # Apply cross-encoder reranking
    context = rerank_results(question, candidates, top_k=top_k)
    context_text = "\n\n".join([c for c in context if c]).strip()
    
    if not context_text:
        return MCQResponse(questions=[MCQQuestion(
            question="❌ No relevant documents found. Please upload materials first.",
            options=[
                MCQOption(label="A", text="Upload PDF documents"),
                MCQOption(label="B", text="Provide more context"),
                MCQOption(label="C", text="Try different keywords"),
                MCQOption(label="D", text="Check document format")
            ],
            correct_answer="A"
        )])

    def _extract_key_facts(text: str, num_facts: int = 50) -> list:
        """Extract key facts and concepts from context"""
        # Split into sentences more aggressively
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n|;", text) if len(s.strip()) > 10]
        
        # Prioritize sentences with technical keywords
        technical_keywords = [
            'algorithm', 'method', 'approach', 'technique', 'formula', 'equation',
            'theorem', 'property', 'characteristic', 'function', 'complexity',
            'result', 'value', 'calculate', 'determine', 'analysis', 'principle',
            'procedure', 'process', 'step', 'condition', 'statement', 'rule',
            'definition', 'concept', 'parameter', 'variable', 'computation',
            'efficiency', 'optimization', 'solution', 'implementation'
        ]
        
        technical_sents = [s for s in sentences if any(kw in s.lower() for kw in technical_keywords)]
        technical_sents = list(dict.fromkeys(technical_sents))  # Remove duplicates
        
        # If we don't have enough technical sentences, add regular ones
        if len(technical_sents) < num_facts:
            non_technical = [s for s in sentences if s not in technical_sents]
            technical_sents.extend(non_technical)
        
        return technical_sents[:num_facts]

    def _generate_questions_deterministic(num_questions: int) -> list:
        """Generate questions using deterministic extraction"""
        facts = _extract_key_facts(context_text, num_facts=max(num_questions * 4, 100))
        
        if len(facts) < 2:
            return []
        
        mcqs = []
        question_types = [
            "technical_analysis", "comparison", "assertion_reason", 
            "definition", "application", "calculation", "numerical", 
            "statement_combination", "scenario", "inference"
        ]
        
        for idx in range(num_questions):
            try:
                q_type = question_types[idx % len(question_types)]
                
                # Safe fact selection with modulo to avoid index errors
                fact_indices = [
                    (idx * 3 + i) % len(facts) for i in range(6)
                ]
                
                primary_facts = [facts[i] for i in fact_indices]
                
                if q_type == "technical_analysis":
                    correct_idx = idx % 4
                    q = MCQQuestion(
                        question=f"Q{idx + 1}: Based on the given context, which of the following accurately describes: \"{primary_facts[0][:75]}...\"?",
                        options=[
                            MCQOption(label="A", text=primary_facts[0][:180]),
                            MCQOption(label="B", text=primary_facts[1][:180]),
                            MCQOption(label="C", text=primary_facts[2][:180]),
                            MCQOption(label="D", text="None of the above")
                        ],
                        correct_answer=chr(65 + correct_idx),
                        explanation=f"This is the correct answer because: {primary_facts[correct_idx][:280]}. This directly addresses the question based on the provided study material."
                    )
                elif q_type == "comparison":
                    correct_idx = (idx + 1) % 4
                    q = MCQQuestion(
                        question=f"Q{idx + 1}: Compare: \"{primary_facts[0][:60]}\" vs \"{primary_facts[1][:60]}\". Which distinction is most accurate?",
                        options=[
                            MCQOption(label="A", text=f"First emphasizes: {primary_facts[0][:140]}"),
                            MCQOption(label="B", text=f"Second emphasizes: {primary_facts[1][:140]}"),
                            MCQOption(label="C", text=f"Both share: {primary_facts[2][:140]}"),
                            MCQOption(label="D", text="They are completely equivalent")
                        ],
                        correct_answer=chr(65 + correct_idx),
                        explanation=f"The key distinction is: {primary_facts[correct_idx][:300]}. This differentiates the concepts effectively as per the study material."
                    )
                elif q_type == "assertion_reason":
                    correct_idx = (idx + 2) % 4
                    q = MCQQuestion(
                        question=f"Q{idx + 1}: Assertion (A): {primary_facts[0][:100]}. Reason (R): {primary_facts[1][:100]}. Which is correct?",
                        options=[
                            MCQOption(label="A", text="Both true; R correctly explains A"),
                            MCQOption(label="B", text="Both true; R does NOT explain A"),
                            MCQOption(label="C", text="A true; R false"),
                            MCQOption(label="D", text="A false; R true")
                        ],
                        correct_answer=chr(65 + correct_idx),
                        explanation=f"Based on the study material: {primary_facts[2][:300]}. This validates the assertion and reason relationship."
                    )
                elif q_type == "definition":
                    correct_idx = (idx + 3) % 4
                    q = MCQQuestion(
                        question=f"Q{idx + 1}: What best defines: \"{primary_facts[0][:85]}...\"?",
                        options=[
                            MCQOption(label="A", text=primary_facts[0][:160]),
                            MCQOption(label="B", text=primary_facts[1][:160]),
                            MCQOption(label="C", text=primary_facts[2][:160]),
                            MCQOption(label="D", text=primary_facts[3][:160] if len(primary_facts) > 3 else "Alternative")
                        ],
                        correct_answer=chr(65 + correct_idx),
                        explanation=f"The precise definition is: {primary_facts[correct_idx][:300]}. This captures the essential meaning as presented in the course material."
                    )
                elif q_type == "application":
                    correct_idx = idx % 4
                    q = MCQQuestion(
                        question=f"Q{idx + 1}: When applying \"{primary_facts[0][:75]}...\", which approach is most suitable?",
                        options=[
                            MCQOption(label="A", text=f"Direct: {primary_facts[1][:120]}"),
                            MCQOption(label="B", text=f"Iterative: {primary_facts[2][:120]}"),
                            MCQOption(label="C", text=f"Conditional: {primary_facts[3][:120] if len(primary_facts) > 3 else 'Approach C'}"),
                            MCQOption(label="D", text=f"Recursive: {primary_facts[4][:120] if len(primary_facts) > 4 else 'Approach D'}")
                        ],
                        correct_answer=chr(65 + correct_idx),
                        explanation=f"This approach works best because: {primary_facts[correct_idx][:300]}. It aligns with the principles covered in your study material."
                    )
                elif q_type == "calculation":
                    correct_idx = (idx + 1) % 4
                    q = MCQQuestion(
                        question=f"Q{idx + 1}: Given \"{primary_facts[0][:85]}...\", what logically follows?",
                        options=[
                            MCQOption(label="A", text="First principle applies"),
                            MCQOption(label="B", text="Comparative analysis shows"),
                            MCQOption(label="C", text="Numerical computation yields"),
                            MCQOption(label="D", text="Cannot be determined")
                        ],
                        correct_answer=chr(65 + correct_idx),
                        explanation=f"From the provided context: {primary_facts[correct_idx][:300]}. This logical deduction is supported by the study material."
                    )
                elif q_type == "numerical":
                    correct_idx = (idx + 2) % 4
                    q = MCQQuestion(
                        question=f"Q{idx + 1}: Based on \"{primary_facts[0][:80]}...\", identify the correct numerical characteristic?",
                        options=[
                            MCQOption(label="A", text=f"Value from: {primary_facts[1][:130]}"),
                            MCQOption(label="B", text=f"Range: {primary_facts[2][:130]}"),
                            MCQOption(label="C", text=f"Ratio: {primary_facts[3][:130] if len(primary_facts) > 3 else 'Characteristic C'}"),
                            MCQOption(label="D", text="Insufficient data")
                        ],
                        correct_answer=chr(65 + correct_idx),
                        explanation=f"The correct numerical characteristic is: {primary_facts[correct_idx][:300]}. This is derived from the calculations in your study material."
                    )
                elif q_type == "statement_combination":
                    correct_idx = (idx + 3) % 4
                    q = MCQQuestion(
                        question=f"Q{idx + 1}: Consider: I. {primary_facts[0][:60]}... II. {primary_facts[1][:60]}... III. {primary_facts[2][:60]}... Which is valid?",
                        options=[
                            MCQOption(label="A", text="Only I"),
                            MCQOption(label="B", text="I and II"),
                            MCQOption(label="C", text="II and III"),
                            MCQOption(label="D", text="All three")
                        ],
                        correct_answer=chr(65 + correct_idx),
                        explanation=f"This combination is valid because: {primary_facts[correct_idx][:300]}. The other statements don't align with the principles covered."
                    )
                elif q_type == "scenario":
                    correct_idx = idx % 4
                    q = MCQQuestion(
                        question=f"Q{idx + 1}: In a scenario with \"{primary_facts[0][:70]}...\", the optimal approach considering \"{primary_facts[1][:70]}...\" would be?",
                        options=[
                            MCQOption(label="A", text=f"Strategy A: {primary_facts[2][:120]}"),
                            MCQOption(label="B", text=f"Strategy B: {primary_facts[3][:120] if len(primary_facts) > 3 else 'Strategy B'}"),
                            MCQOption(label="C", text=f"Strategy C: {primary_facts[4][:120] if len(primary_facts) > 4 else 'Strategy C'}"),
                            MCQOption(label="D", text=f"Strategy D: {primary_facts[5][:120] if len(primary_facts) > 5 else 'Strategy D'}")
                        ],
                        correct_answer=chr(65 + correct_idx),
                        explanation=f"The optimal approach in this scenario is: {primary_facts[correct_idx][:300]}. This strategy is best aligned with the course concepts."
                    )
                else:  # inference
                    correct_idx = (idx + 1) % 4
                    q = MCQQuestion(
                        question=f"Q{idx + 1}: From \"{primary_facts[0][:75]}...\", what can be inferred about \"{primary_facts[1][:75]}...\"?",
                        options=[
                            MCQOption(label="A", text=f"Inference 1: {primary_facts[2][:120]}"),
                            MCQOption(label="B", text=f"Inference 2: {primary_facts[3][:120] if len(primary_facts) > 3 else 'Inference 2'}"),
                            MCQOption(label="C", text=f"Inference 3: {primary_facts[4][:120] if len(primary_facts) > 4 else 'Inference 3'}"),
                            MCQOption(label="D", text="No valid inference")
                        ],
                        correct_answer=chr(65 + correct_idx),
                        explanation=f"This inference is valid because: {primary_facts[correct_idx][:300]}. The logical deduction follows from the provided concepts in the material."
                    )
                
                if q.question and all(opt.text for opt in q.options):
                    mcqs.append(q)
            except Exception as e:
                # Skip this question if any error occurs
                continue
        
        return mcqs

    # First try with Gemini
    mcqs_from_ai = None
    try:
        # Split large requests into chunks for Gemini (Gemini has limits)
        chunk_size = min(count, 50)  # Generate 50 at a time max from Gemini
        remaining = count
        all_mcqs_ai = []
        
        while remaining > 0 and len(all_mcqs_ai) < count:
            current_chunk = min(chunk_size, remaining)
            
            prompt = f"""You are an expert question paper setter with 15+ years of experience. Generate EXACTLY {current_chunk} high-quality multiple choice questions from the provided document context.

🎯 MCQ REQUIREMENTS:

QUESTION TYPES TO USE (vary across questions):
1. **Conceptual Understanding**: Test deep understanding of concepts
2. **Formula Application**: Apply formulas/theories with values/scenarios
3. **Analysis & Logic**: Analyze details, processes, or correctness
4. **Statement Verification**: True/False statement combinations
5. **Assertion-Reasoning**: Test logical relationships
6. **Scenario-Based**: Multi-step problem solving
7. **Comparison**: Compare concepts/approaches/methods
8. **Outcome Prediction**: What will be the result?
9. **Error Detection**: Find the incorrect statement
10. **Optimization**: Best choice for given constraints

CRITICAL RULES:
✅ Each question must test a DIFFERENT concept from the material
✅ Rotate correct answers: A, B, C, D (NOT all A or all B)
✅ Make wrong options plausible but clearly incorrect to an expert
✅ Include specific numbers, formulas, or technical terms from context
✅ Difficulty: 30% Easy, 40% Medium, 30% Hard
✅ Options should be concise (max 150 characters each)
✅ Question should be clear and unambiguous

❌ FORBIDDEN:
- Generic questions like "Which is correct?"
- All questions following same pattern
- Vague or ambiguous options
- Questions not based on provided context
- Repetitive testing of same concept

EXAMPLE QUESTIONS:

**Type 1 - Detail/Calculation:**
"An algorithm has time complexity T(n) = 2T(n/2) + n². Using Master's theorem, the complexity is:"
A) O(n²)  B) O(n² log n)  C) O(n³)  D) O(n log n)

**Type 2 - Statement Combination:**
"Consider these statements:
I. Quick sort has worst case O(n²)
II. Merge sort is always O(n log n)
III. Heap sort is unstable
Which are TRUE?"
A) I and II only  B) II and III only  C) All three  D) I and III only

**Type 3 - Scenario:**
"A database has table size 10. Keys 23, 43, 13, 27 are inserted in order. If 37 is inserted next, collisions occur at positions:"
A) 3 only  B) 3, 4  C) 3, 4, 5  D) 7, 8

Now generate {current_chunk} standard questions from this context:

CONTEXT:
{context_text[:6000]}

OUTPUT FORMAT (strict JSON, NO markdown):
[
  {{
    "question": "Clear question with specific details from the context",
    "options": {{
      "A": "First precise option",
      "B": "Second distinct option",
      "C": "Third unique option",
      "D": "Fourth different option"
    }},
    "correct_answer": "B",
    "type": "conceptual_understanding"
  }}
]

Generate {current_chunk} diverse MCQs now (JSON only):"""

            result = _generate_with_gemini(prompt, temperature=0.85, max_tokens=2000)
            
            if result and result != "RATE_LIMIT_EXCEEDED":
                try:
                    data = _extract_json_array(result)
                    if isinstance(data, list) and len(data) > 0:
                        for item in data:
                            try:
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

                                # Shuffle options randomly and map correct_answer accordingly
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

                                all_mcqs_ai.append(MCQQuestion(
                                    question=str(item.get("question", "")),
                                    options=[
                                        MCQOption(label="A", text=shuffled_options["A"]),
                                        MCQOption(label="B", text=shuffled_options["B"]),
                                        MCQOption(label="C", text=shuffled_options["C"]),
                                        MCQOption(label="D", text=shuffled_options["D"]),
                                    ],
                                    correct_answer=new_correct_answer,
                                    explanation=item.get("explanation", item.get("solution", ""))
                                ))
                            except Exception:
                                pass
                except Exception:
                    pass
            
            remaining -= current_chunk
            
            if len(all_mcqs_ai) >= count * 0.5:  # At least 50% success
                mcqs_from_ai = all_mcqs_ai[:count]
                return MCQResponse(questions=mcqs_from_ai)
    except Exception:
        pass
    
    # Fallback to deterministic generation from RAG context
    try:
        fallback_mcqs = _generate_questions_deterministic(count)
    except Exception as e:
        fallback_mcqs = []
    
    if not fallback_mcqs or len(fallback_mcqs) < count * 0.8:
        # If we don't have enough from fallback, generate additional ones
        additional_needed = count - len(fallback_mcqs) if fallback_mcqs else count
        if additional_needed > 0:
            try:
                extra_facts = _extract_key_facts(context_text, num_facts=max(additional_needed * 3, 50))
                for i in range(additional_needed):
                    if i < len(extra_facts):
                        try:
                            fallback_mcqs.append(MCQQuestion(
                                question=f"Q{len(fallback_mcqs) + 1}: Regarding \"{extra_facts[i][:100]}...\", which statement is most accurate?",
                                options=[
                                    MCQOption(label="A", text=extra_facts[i][:160]),
                                    MCQOption(label="B", text=extra_facts[(i + 1) % len(extra_facts)][:160] if len(extra_facts) > 1 else "Alternative view"),
                                    MCQOption(label="C", text=extra_facts[(i + 2) % len(extra_facts)][:160] if len(extra_facts) > 2 else "Another perspective"),
                                    MCQOption(label="D", text="None of the above")
                                ],
                                correct_answer=chr(65 + (i % 4))
                            ))
                        except Exception:
                            continue
            except Exception:
                pass
    
    if not fallback_mcqs:
        fallback_mcqs = [MCQQuestion(
            question="⚠️ Insufficient context. Please upload a comprehensive document.",
            options=[
                MCQOption(label="A", text="Upload PDF with more content"),
                MCQOption(label="B", text="Provide technical material"),
                MCQOption(label="C", text="Ensure document has formulas"),
                MCQOption(label="D", text="Check document readability")
            ],
            correct_answer="A"
        )]
    
    return MCQResponse(questions=fallback_mcqs[:count])