import sys
sys.path.insert(0, '.')

# Directly replicate generate_practice_problems logic
from app.services.hybrid_search import hybrid_search
from app.services.multi_query_retrieval import multi_query_retrieve, smart_deduplication
from app.services.reranker import rerank_results
from app.services.query_expansion import expand_query

# Same as practice_generator
subject = "Computer Science"
difficulty = "medium"
count = 2
question_type = "mcq"
document_id = None
document_ids = None
topic = None

query = f"Generate {difficulty} {question_type} questions on {subject}"
print(f'Query: {query}')

# Get config
from app.services.practice_generator import _classify_difficulty_prompt
config = _classify_difficulty_prompt(difficulty, question_type)

# Retrieve content (EXACT copy from practice_generator)
filters = {}
if subject:
    filters["subject"] = subject

query_variations = expand_query(query, mode='rules', num_variations=2)  # USE RULES TO DEBUG
print(f'Query variations: {query_variations}')

def search_fn(q, k):
    return hybrid_search(q, top_k=k, filters=filters if filters else None)

retrieve_count = {"easy": 15, "medium": 25, "hard": 40}.get(difficulty, 25)

candidates = multi_query_retrieve(
    query_variations,
    search_fn,
    top_k_per_query=retrieve_count,
    final_top_k=retrieve_count,
    fusion_method='rrf'
)
print(f'Candidates: {len(candidates)}')

candidates = smart_deduplication(candidates, similarity_threshold=0.90)
print(f'After dedup: {len(candidates)}')

context = rerank_results(query, candidates, top_k=min(len(candidates), retrieve_count))
print(f'After rerank: {len(context)}')

context_text = "\n\n".join([c for c in context if c]).strip()
print(f'Context length: {len(context_text)}')
print(f'Context empty? {not context_text}')

if not context_text:
    print("[ERROR] No context!")
else:
    print(f'[OK] Got context: {context_text[:100]}...')
    print(f'\nNow generating {question_type}...')
    
    from app.services.practice_generator import _generate_mcq_problems
    problems = _generate_mcq_problems(
        context_text, count, difficulty, config, subject, topic
    )
    print(f'Generated {len(problems)} problems')
    if problems:
        print(f'First problem: {problems[0].question[:80]}...')
