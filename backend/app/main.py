import sys
import io

# Force UTF-8 encoding for stdout/stderr to handle emojis
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import upload, query, documents, practice, chat
from app.database import engine
from app.models import Base

# Automatically create database tables if they do not exist
try:
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")
    
    # Synchronize primary key sequences (PostgreSQL only)
    if "postgresql" in engine.dialect.name:
        from sqlalchemy import text
        with engine.connect() as conn:
            for table in ["documents", "queries", "document_topics", "practice_attempts", "document_chunks"]:
                try:
                    conn.execute(text(f"""
                        SELECT setval(
                            pg_get_serial_sequence('{table}', 'id'),
                            COALESCE((SELECT MAX(id) FROM {table}), 1),
                            (SELECT MAX(id) FROM {table}) IS NOT NULL
                        );
                    """))
                    conn.commit()
                except Exception as seq_err:
                    print(f"Warning: Could not sync sequence for table {table}: {seq_err}")
        print("Database sequences synchronized successfully.")
except Exception as e:
    print(f"WARNING: Database table initialization failed (server will still start): {e}")

app = FastAPI()

# Add CORS middleware
# Allow all origins for flexibility across local dev and deployed Vercel frontend.
# The API is protected by server-side environment variable keys.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(practice.router)
app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "RAG AI Chat API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/debug/diagnostics")
async def debug_diagnostics():
    import os
    
    env_vars = {
        "DATABASE_URL_SET": bool(os.getenv("DATABASE_URL")),
        "PINECONE_API_KEY_SET": bool(os.getenv("PINECONE_API_KEY")),
        "PINECONE_INDEX": os.getenv("PINECONE_INDEX"),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME"),
        "HUGGINGFACE_API_KEY_SET": bool(os.getenv("HUGGINGFACE_API_KEY")),
        "GEMINI_API_KEY_SET": bool(os.getenv("GEMINI_API_KEY")),
    }
    
    pinecone_status = "Not tested"
    try:
        from app.services.embeddings import get_pinecone_index
        idx = get_pinecone_index()
        stats = idx.describe_index_stats()
        pinecone_status = f"Success: {stats}"
    except Exception as e:
        pinecone_status = f"Failed: {str(e)}"
        
    hf_status = "Not tested"
    try:
        from app.services.embeddings import _embed
        emb = _embed(["test chunk content"])
        hf_status = f"Success: embedding shape {len(emb[0]) if emb else 0}"
    except Exception as e:
        hf_status = f"Failed: {str(e)}"
        
    return {
        "env_vars": env_vars,
        "pinecone_status": pinecone_status,
        "huggingface_status": hf_status
    }

