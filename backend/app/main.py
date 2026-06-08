import sys
import io

# Force UTF-8 encoding for stdout/stderr to handle emojis
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import upload, query, documents, practice, chat

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
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
