"""
Embeddings service — uses HuggingFace Inference API in production
(no local PyTorch/sentence-transformers model loaded).
Falls back to local SentenceTransformer for local development if HF API key is missing.
"""
import os
import logging
import requests
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv(override=True)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
_HF_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Normalize model name: prepend sentence-transformers/ if not specified
if _HF_MODEL and "/" not in _HF_MODEL:
    _HF_MODEL = f"sentence-transformers/{_HF_MODEL}"

_HF_API_URL = f"https://api-inference.huggingface.co/models/{_HF_MODEL}"

# Local model fallback (only used if HF API key is not set — i.e. local dev without key)
_local_model = None


def _embed_via_hf_api(texts: list) -> list:
    """Call HuggingFace Inference API to generate embeddings remotely."""
    headers = {"Authorization": f"Bearer {_HF_API_KEY}"}
    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True}
    }
    response = requests.post(_HF_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    return result  # list of embedding vectors


def _embed_via_local(texts: list) -> list:
    """Load and use local SentenceTransformer model (development fallback)."""
    global _local_model
    if _local_model is None:
        logging.info("Loading local SentenceTransformer model (dev fallback)...")
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = _local_model.encode(texts)
    return [e.tolist() for e in embeddings]


def _embed(texts: list) -> list:
    """
    Embed a list of texts.
    Uses HuggingFace Inference API if key is available (production),
    otherwise falls back to local model (development).
    """
    if isinstance(texts, str):
        texts = [texts]

    if _HF_API_KEY:
        try:
            return _embed_via_hf_api(texts)
        except Exception as e:
            logging.warning(f"HF API embedding failed ({e}), trying local fallback...")
            try:
                return _embed_via_local(texts)
            except Exception as local_e:
                logging.error(f"Local fallback also failed (e.g. missing sentence-transformers): {local_e}")
                # Return zero vector representation to prevent crash
                return [[0.0] * 384 for _ in texts]
    else:
        try:
            return _embed_via_local(texts)
        except Exception as local_e:
            logging.error(f"Local fallback failed: {local_e}")
            return [[0.0] * 384 for _ in texts]


# ─── Pinecone setup ─────────────────────────────────────────────────────────

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


# ─── Public API ─────────────────────────────────────────────────────────────

def embed_and_store(chunks: list, doc_id: str, filename: str) -> int:
    """
    Embed each chunk via HF Inference API (or local fallback),
    then upsert to Pinecone in batches of 100.
    Returns total number of chunks stored.
    """
    if not chunks:
        return 0

    embeddings = _embed(chunks)
    vectors = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        emb_list = emb if isinstance(emb, list) else list(emb)
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


def query_similar(query: str, top_k: int = 5, doc_id=None) -> list:
    """
    Embed query then search Pinecone for top_k nearest vectors.
    Optional doc_id (str or list) filters by metadata.
    """
    query_vector = _embed([query])[0]

    query_kwargs = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True
    }

    if doc_id is not None:
        if isinstance(doc_id, list):
            query_kwargs["filter"] = {"doc_id": {"$in": [str(d) for d in doc_id]}}
        else:
            query_kwargs["filter"] = {"doc_id": {"$eq": str(doc_id)}}

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


def embed_text(text: str) -> list:
    """Embed a single text string and return a float vector."""
    return _embed([text])[0]


def embed_batch(texts: list) -> list:
    """Embed a batch of texts and return a list of float vectors."""
    return _embed(texts)


# Backward compatibility shim
class EmbeddingModel:
    def embed_query(self, text):
        return embed_text(text)


embedding_model = EmbeddingModel()