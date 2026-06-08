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
