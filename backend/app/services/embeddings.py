"""
Embeddings service — uses HuggingFace Inference API in production
(no local PyTorch/sentence-transformers model loaded).
Falls back to local SentenceTransformer for local development if HF API key is missing.
"""
import os
import logging
import requests
from dotenv import load_dotenv

# Safely import Pinecone — don't crash at startup if not installed
try:
    from pinecone import Pinecone as _PineconeClient
    _PINECONE_IMPORTABLE = True
except ImportError:
    _PineconeClient = None
    _PINECONE_IMPORTABLE = False
    logging.warning("[embeddings] pinecone package not installed — vector search unavailable")

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
    """Call HuggingFace Inference API to generate embeddings remotely.
    Sends in batches of 32 to avoid request timeouts on large documents.
    Timeout is 60 seconds to allow for model warm-up on cold start.
    """
    headers = {"Authorization": f"Bearer {_HF_API_KEY}"}
    all_embeddings = []

    # Process in batches of 32 to avoid timeouts
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "inputs": batch,
            "options": {"wait_for_model": True}
        }
        try:
            response = requests.post(_HF_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            batch_embeddings = response.json()
            # Handle both list-of-lists and list-of-dicts response formats
            if isinstance(batch_embeddings, list) and len(batch_embeddings) > 0:
                if isinstance(batch_embeddings[0], dict):
                    # Some HF models return {embedding: [...]}
                    batch_embeddings = [e.get("embedding", e) for e in batch_embeddings]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logging.warning(f"Primary HF API endpoint failed for batch {i//batch_size}: {e}. Trying mirror...")
            # Fallback to hf-mirror.com
            mirror_url = _HF_API_URL.replace("api-inference.huggingface.co", "api-inference.hf-mirror.com")
            response = requests.post(mirror_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            batch_embeddings = response.json()
            if isinstance(batch_embeddings, list) and len(batch_embeddings) > 0:
                if isinstance(batch_embeddings[0], dict):
                    batch_embeddings = [e.get("embedding", e) for e in batch_embeddings]
            all_embeddings.extend(batch_embeddings)

    return all_embeddings


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


# ─── Pinecone setup (lazy init so missing env vars don't crash at startup) ────

_pc = None
_index = None


def get_pinecone_index():
    """Return (and lazily initialise) the shared Pinecone index instance."""
    global _pc, _index
    if _index is not None:
        return _index

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME") or os.getenv("PINECONE_INDEX")

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY environment variable is not set. "
            "Please add it in your Render (or local .env) environment variables."
        )
    if not index_name:
        raise ValueError(
            "PINECONE_INDEX_NAME (or PINECONE_INDEX) environment variable is not set. "
            "Please add it in your Render (or local .env) environment variables."
        )
    if not _PINECONE_IMPORTABLE or _PineconeClient is None:
        raise ImportError(
            "pinecone package is not installed. "
            "Add 'pinecone>=3.0.0' to requirements.txt and redeploy."
        )

    _pc = _PineconeClient(api_key=api_key)
    _index = _pc.Index(index_name)
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
    index = get_pinecone_index()
    for idx in range(0, len(vectors), batch_size):
        batch = vectors[idx:idx + batch_size]
        index.upsert(vectors=batch)
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

    index = get_pinecone_index()
    response = index.query(**query_kwargs)

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