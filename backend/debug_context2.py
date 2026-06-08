import sys
sys.path.insert(0, '.')

from app.services.practice_generator import generate_practice_problems
from app.services.query_expansion import expand_query
from app.services.hybrid_search import hybrid_search
from app.services.multi_query_retrieval import multi_query_retrieve, smart_deduplication
from app.services.reranker import rerank_results

query = 'generate medium level mcq questions'
print(f'Query: {query}\n')

# Expand - FORCE rule-based since LLM is failing
variations = expand_query(query, mode='rules', num_variations=2)
print(f'Variations: {len(variations)}')
for v in variations:
    print(f'  - {v}')

# Search
def search_fn(q, k):
    results = hybrid_search(q, top_k=k)
    return results

print(f'\nSearching...')
candidates = multi_query_retrieve(
    variations,
    search_fn,
    top_k_per_query=25,
    final_top_k=25,
    fusion_method='rrf'
)
print(f'Candidates: {len(candidates)}')

# Dedup
candidates = smart_deduplication(candidates, similarity_threshold=0.90)
print(f'After dedup: {len(candidates)}')

# Rerank
context = rerank_results(query, candidates, top_k=min(len(candidates), 25))
print(f'After rerank: {len(context)}')
for i, c in enumerate(context[:3]):
    print(f'  [{i}] Type: {type(c)}, Length: {len(str(c)) if c else 0}')

# Join
context_text = "\n\n".join([c for c in context if c]).strip()
print(f'\nFinal context length: {len(context_text)}')
print(f'Empty? {len(context_text) == 0}')
print(f'Context preview: {context_text[:150]}...')
