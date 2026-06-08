import sys
sys.path.insert(0, '.')

from app.services.query_expansion import expand_query
from app.services.hybrid_search import hybrid_search
from app.services.multi_query_retrieval import multi_query_retrieve

query = 'Generate medium-level mcq questions on CS'
print(f'Original query: "{query}"\n')

# Step 1: Expand
variations = expand_query(query, mode='auto', num_variations=2)
print(f'✅ Expanded to {len(variations)} variations:')
for i, v in enumerate(variations):
    print(f'  {i+1}. {v}')

# Step 2: Test hybrid search directly
print(f'\n✅ Testing hybrid search on each variation:')
def search_fn(q, k):
    results = hybrid_search(q, top_k=k)
    print(f'  Searching "{q}" → Found {len(results)} results')
    return results

# Get results from first variation
first_results = search_fn(variations[0], 5)
if first_results:
    print(f'  First result sample: {first_results[0][:100]}...')

# Step 3: Multi-query retrieve
print(f'\n✅ Running multi-query retrieval:')
candidates = multi_query_retrieve(
    variations,
    search_fn,
    top_k_per_query=15,
    final_top_k=15,
    fusion_method='rrf'
)
print(f'Result: {len(candidates)} candidates')
if candidates:
    print(f'Sample: {candidates[0][:100]}...')
