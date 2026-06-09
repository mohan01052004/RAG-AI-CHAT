try:
    from pinecone import Pinecone
    _PINECONE_AVAILABLE = True
except ImportError:
    Pinecone = None
    _PINECONE_AVAILABLE = False

from app.config import PINECONE_API_KEY, PINECONE_INDEX
from app.services.embeddings import embedding_model
from uuid import uuid4

_pc = None
_index = None


def _get_pinecone():
    global _pc
    if _pc is None:
        if not _PINECONE_AVAILABLE or Pinecone is None:
            raise ImportError("pinecone package is not installed. Add 'pinecone>=3.0.0' to requirements.txt.")
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def get_index():
    global _index
    if _index is None:
        pc = _get_pinecone()
        _index = pc.Index(PINECONE_INDEX)
    return _index

def upload_chunks(chunks: list[str], metadata: dict | None = None, namespace: str | None = None):
    """
    Upload chunks to Pinecone and build BM25 index for hybrid search.
    
    Phase 4 Enhancement: Now supports enriched metadata from semantic chunking
    """
    from app.services.hybrid_search import build_bm25_index
    from app.services.chunker import chunk_with_metadata
    
    vectors = []
    base_metadata = metadata or {}
    
    # Check if chunks already have metadata (from chunk_with_metadata)
    # If not, extract metadata on-the-fly
    has_metadata = isinstance(chunks, list) and len(chunks) > 0 and isinstance(chunks[0], tuple)
    
    if has_metadata:
        # Chunks are already (chunk_text, metadata) tuples
        chunk_data = chunks
    else:
        # Plain chunks - extract metadata
        chunk_data = [(chunk, {}) for chunk in chunks]

    for i, (chunk, chunk_meta) in enumerate(chunk_data):
        emb = embedding_model.embed_query(chunk)
        
        # Merge base metadata with chunk-specific metadata
        full_metadata = {
            **base_metadata,
            **chunk_meta,
            "text": chunk,
            "chunk_index": i,
        }

        vectors.append({
            "id": str(uuid4()),
            "values": emb,
            "metadata": full_metadata,
        })

    if not vectors:
        return

    index = get_index()
    if namespace:
        index.upsert(vectors=vectors, namespace=namespace)
    else:
        index.upsert(vectors=vectors)
    
    # Build BM25 index for hybrid search (extract just the text chunks)
    text_chunks = [chunk if isinstance(chunk, str) else chunk[0] for chunk in chunks]
    metadata_list = [v["metadata"] for v in vectors]
    build_bm25_index(text_chunks, metadata_list)
    print(f"✅ Built BM25 index with {len(text_chunks)} chunks")
    
    # Print chunking stats
    stats = {
        "total_chunks": len(vectors),
        "with_code": sum(1 for v in vectors if v["metadata"].get("has_code", False)),
        "with_formula": sum(1 for v in vectors if v["metadata"].get("has_formula", False)),
        "with_list": sum(1 for v in vectors if v["metadata"].get("has_list", False)),
        "with_table": sum(1 for v in vectors if v["metadata"].get("has_table", False)),
    }
    print(f"📊 Chunk stats: {stats}")


def search_similar(
    query: str, 
    top_k: int = 5, 
    filters: dict | None = None, 
    namespace: str | None = None,
    return_metadata: bool = False
):
    query_emb = embedding_model.embed_query(query)

    index = get_index()
    query_kwargs = {
        "vector": query_emb,
        "top_k": top_k,
        "include_metadata": True,
    }
    if filters:
        query_kwargs["filter"] = filters
    if namespace:
        query_kwargs["namespace"] = namespace

    results = index.query(**query_kwargs)
    if return_metadata:
        return [
            (match.metadata.get("text", ""), match.metadata)
            for match in results.matches
            if match.metadata
        ]
    return [match.metadata.get("text", "") for match in results.matches if match.metadata]