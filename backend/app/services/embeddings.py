import os
import logging
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

# Ensure env vars are loaded
load_dotenv(override=True)

# Suppress warnings
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Module-level singletons loaded on startup
_model = None


def _get_model():
    global _model
    if _model is None:
        logging.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logging.info("SentenceTransformer model loaded successfully.")
    return _model


_api_key = os.getenv("PINECONE_API_KEY")
_index_name = os.getenv("PINECONE_INDEX_NAME") or os.getenv("PINECONE_INDEX")

if not _api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set")
if not _index_name:
    raise ValueError("PINECONE_INDEX_NAME (or PINECONE_INDEX) environment variable is not set")

_pc = Pinecone(api_key=_api_key)
_index = _pc.Index(_index_name)


def get_pinecone_index():
    """Return the shared Pinecone index instance."""
    return _index


def embed_and_store(chunks: list[str], doc_id: str, filename: str) -> int:
    """
    For each chunk:
    - Generate embedding using sentence-transformers (all-MiniLM-L6-v2)
    - Upsert to Pinecone in batches of 100 to avoid rate limits
    - Return total number of chunks stored
    """
    if not chunks:
        return 0

    embeddings = _get_model().encode(chunks)
    vectors = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        emb_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        vectors.append({
            "id": f"{doc_id}_chunk_{i}",
            "values": emb_list,
            "metadata": {
                "doc_id": str(doc_id),
                "filename": filename,
                "chunk_index": i,
                "content": chunk,
                "text": chunk
            }
        })

    batch_size = 100
    total_stored = 0
    for idx in range(0, len(vectors), batch_size):
        batch = vectors[idx:idx + batch_size]
        _index.upsert(vectors=batch)
        total_stored += len(batch)

    return total_stored


def query_similar(query: str, top_k: int = 5, doc_id: str = None) -> list[dict]:
    """
    - Embed the query using sentence-transformers (all-MiniLM-L6-v2)
    - Query Pinecone for top_k nearest vectors
    - If doc_id is provided, filter by metadata: {"doc_id": {"$eq": doc_id}}
    - Return list of dicts with: content, filename, chunk_index, doc_id, score
    """
    query_vector = _get_model().encode(query).tolist()
    
    query_kwargs = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True
    }
    
    if doc_id is not None:
        if isinstance(doc_id, list):
            query_kwargs["filter"] = {
                "doc_id": {"$in": [str(d) for d in doc_id]}
            }
        else:
            query_kwargs["filter"] = {
                "doc_id": {"$eq": str(doc_id)}
            }

    response = _index.query(**query_kwargs)
    
    results = []
    for match in response.matches:
        meta = match.metadata or {}
        results.append({
            "content": meta.get("content") or meta.get("text") or "",
            "filename": meta.get("filename", ""),
            "chunk_index": meta.get("chunk_index"),
            "doc_id": meta.get("doc_id", ""),
            "score": match.score
        })
        
    return results


# Maintain backward compatibility with existing codebase
class EmbeddingModel:
    def embed_query(self, text):
        return _get_model().encode(text).tolist()

embedding_model = EmbeddingModel()