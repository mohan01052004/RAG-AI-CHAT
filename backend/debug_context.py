import sys
sys.path.insert(0, '.')

from app.services.practice_generator import generate_practice_problems
from app.services.query_expansion import expand_query
from app.services.hybrid_search import hybrid_search
from app.services.multi_query_retrieval import multi_query_retrieve, smart_deduplication
from app.services.reranker import rerank_results

query = 'Generate medium-level mcq questions on CS'
print(f'Starting practice generation test...\n')

# Expand query
variations = expand_query(query, mode='auto', num_variations=2)
print(f'✅ Query variations: {len(variations)}')

# Define search function
def search_fn(q, k):
    return hybrid_search(q, top_k=k)

# Multi-query retrieve
retrieve_count = 25
candidates = multi_query_retrieve(
    variations,
    search_fn,
    top_k_per_query=retrieve_count,
    final_top_k=retrieve_count,
    fusion_method='rrf'
)
print(f'✅ Candidates retrieved: {len(candidates)}')

# Dedup
candidates = smart_deduplication(candidates, similarity_threshold=0.90)
print(f'✅ After dedup: {len(candidates)}')

# Rerank
context = rerank_results(query, candidates, top_k=min(len(candidates), retrieve_count))
print(f'✅ After rerank: {len(context)} items')

# Join context
context_text = "\n\n".join([c for c in context if c]).strip()
print(f'✅ Final context length: {len(context_text)} chars')
print(f'✅ Context empty? {len(context_text) == 0}')

if context_text:
    print(f'\nContext preview:\n{context_text[:200]}...')
else:
    print('\n❌ CONTEXT IS EMPTY!')
    print(f'context list: {context}')
    print(f'context types: {[type(c) for c in context[:3]]}')
