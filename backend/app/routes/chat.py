import os
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Union
from google import genai
from google.genai import types
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db



router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[dict]
    doc_id: Optional[Union[str, List[str]]] = None

class GeneralChatRequest(BaseModel):
    message: str


# ─── Groq Fallback Helper ────────────────────────────────────────────────────

def _call_groq(system_instruction: str, user_message: str, history: list = None) -> str:
    """Call Groq API as fallback when Gemini is exhausted."""
    from groq import Groq

    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not set in environment")

    client = Groq(api_key=groq_api_key)

    messages = [{"role": "system", "content": system_instruction}]

    # Add conversation history
    if history:
        for turn in history:
            role = turn.get("role", "user")
            if role == "model":
                role = "assistant"
            content = turn.get("content") or turn.get("text") or ""
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=groq_model,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0].message.content


def _is_quota_error(error_msg: str) -> bool:
    """Check if the error is a quota/rate limit error."""
    lower = error_msg.lower()
    return any(kw in lower for kw in ["quota", "429", "limit", "exhausted", "rate_limit", "resource_exhausted"])


# ─── RAG Chat Endpoint ───────────────────────────────────────────────────────

@router.post("/api/chat/rag")
async def chat_rag(request: ChatRequest, db: Session = Depends(get_db)):
    results = []

    # ── 1. Try Pinecone vector search first ───────────────────────────────────
    try:
        results = retrieve_relevant_chunks(request.message, top_k=5, doc_id=request.doc_id)
    except Exception as e:
        print(f"[CHAT/RAG] Pinecone retrieval failed (will use DB fallback): {e}")
        results = []

    # ── 2. PostgreSQL full-text fallback (always runs when Pinecone has no results) ──
    if not results:
        try:
            # Determine target doc_ids from request
            target_ids = []
            if request.doc_id:
                if isinstance(request.doc_id, list):
                    target_ids = [int(d) for d in request.doc_id if str(d).isdigit()]
                elif str(request.doc_id).isdigit():
                    target_ids = [int(request.doc_id)]

            # Build query — keyword search using ILIKE across PostgreSQL chunks
            query_words = [w for w in request.message.split() if len(w) > 3][:6]
            search_pattern = "%" + "%".join(query_words[:3]) + "%" if query_words else "%"

            if target_ids:
                db_chunks = db.execute(text("""
                    SELECT content, page_num, chunk_index, filename, doc_id
                    FROM document_chunks
                    WHERE doc_id = ANY(:doc_ids)
                      AND content ILIKE :pattern
                    ORDER BY chunk_index ASC
                    LIMIT 5
                """), {"doc_ids": target_ids, "pattern": search_pattern}).fetchall()

                # If keyword search got nothing, grab first chunks of the doc
                if not db_chunks:
                    db_chunks = db.execute(text("""
                        SELECT content, page_num, chunk_index, filename, doc_id
                        FROM document_chunks
                        WHERE doc_id = ANY(:doc_ids)
                        ORDER BY chunk_index ASC
                        LIMIT 5
                    """), {"doc_ids": target_ids}).fetchall()
            else:
                # No specific doc — keyword search across all chunks
                db_chunks = db.execute(text("""
                    SELECT content, page_num, chunk_index, filename, doc_id
                    FROM document_chunks
                    WHERE content ILIKE :pattern
                    ORDER BY id DESC
                    LIMIT 5
                """), {"pattern": search_pattern}).fetchall()

                # Still nothing — get latest chunks
                if not db_chunks:
                    db_chunks = db.execute(text("""
                        SELECT content, page_num, chunk_index, filename, doc_id
                        FROM document_chunks
                        ORDER BY id DESC
                        LIMIT 5
                    """)).fetchall()

            results = [{
                "content": c[0],
                "page_num": c[1],
                "chunk_index": c[2],
                "filename": c[3],
                "doc_id": str(c[4]),
                "score": 0.5
            } for c in db_chunks]

            if results:
                print(f"[CHAT/RAG] Using PostgreSQL full-text fallback — {len(results)} chunks found")
        except Exception as e:
            print(f"[CHAT/RAG] PostgreSQL fallback also failed: {e}")
            results = []

    # Build prompt based on retrieval results
    if results:
        system_instruction = """You are a document assistant. Answer using ONLY the 
context below. At the end, cite sources as [filename, chunk N].
Be detailed and accurate."""
        
        context_parts = []
        for idx, r in enumerate(results, start=1):
            context_parts.append(
                f"[{idx}] filename: {r['filename']} | chunk: {r['chunk_index']}\n"
                f"{r['content']}\n"
                f"---"
            )
        context = "\n".join(context_parts)
        
        user_message = f"Context:\n{context}\n\nQuestion: {request.message}"
        response_mode = "document"
    else:
        system_instruction = """You are a helpful general AI assistant."""
        user_message = request.message
        response_mode = "general"

    # Build Gemini history
    gemini_history = []
    for turn in request.conversation_history:
        role = turn.get("role")
        if role == "assistant":
            role = "model"
        content = turn.get("content") or turn.get("text") or ""
        gemini_history.append({
            "role": role,
            "parts": [{"text": content}]
        })

    answer = None
    used_groq = False

    # ── Try Gemini first ──────────────────────────────────────────────────────
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        try:
            client = genai.Client(api_key=gemini_api_key)
            model_name = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

            try:
                chat = client.chats.create(
                    model=model_name,
                    history=gemini_history,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )
                )
                response = chat.send_message(user_message)
                answer = response.text
            except Exception as inner_e:
                print(f"[CHAT/RAG] Primary Gemini model failed: {inner_e}. Trying fallback model...")
                try:
                    fallback_model = "gemini-1.5-flash"
                    chat = client.chats.create(
                        model=fallback_model,
                        history=gemini_history,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction
                        )
                    )
                    response = chat.send_message(user_message)
                    answer = response.text
                except Exception as fallback_e:
                    print(f"[CHAT/RAG] Fallback Gemini model also failed: {fallback_e}. Switching to Groq...")
                    answer = None
        except Exception as e:
            print(f"[CHAT/RAG] Gemini setup failed: {e}. Falling back to Groq...")
            answer = None

    # ── Groq fallback ─────────────────────────────────────────────────────────
    if answer is None:
        try:
            print("[CHAT/RAG] Using Groq fallback...")
            answer = _call_groq(
                system_instruction=system_instruction,
                user_message=user_message,
                history=request.conversation_history,
            )
            used_groq = True
        except Exception as groq_e:
            return {
                "answer": f"⚠️ Both Gemini and Groq APIs are unavailable: {str(groq_e)}",
                "mode": "general",
                "sources": []
            }

    if response_mode == "general":
        prefix = "⚠️ This question doesn't seem related to your uploaded documents.\nHere's a general answer:\n\n"
        if not answer.startswith(prefix):
            answer = prefix + answer

    if used_groq:
        answer = "🔄 *[Responded via Groq — Gemini quota exhausted]*\n\n" + answer

    # Format sources
    sources = []
    if response_mode == "document":
        for r in results:
            sources.append({
                "filename": r.get("filename"),
                "chunk_index": r.get("chunk_index"),
                "excerpt": r.get("content", "")[:150]
            })

    return {
        "answer": answer,
        "mode": response_mode,
        "sources": sources
    }


# ─── General Chat Endpoint ───────────────────────────────────────────────────

@router.post("/api/chat")
async def general_chat(request: GeneralChatRequest):
    system_instruction = "You are a helpful general AI assistant."
    answer = None
    used_groq = False

    # ── Try Gemini first ──────────────────────────────────────────────────────
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        try:
            client = genai.Client(api_key=gemini_api_key)
            model_name = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=request.message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )
                )
                answer = response.text
            except Exception as inner_e:
                print(f"[CHAT] Primary Gemini model failed: {inner_e}. Trying fallback model...")
                try:
                    fallback_model = "gemini-1.5-flash"
                    response = client.models.generate_content(
                        model=fallback_model,
                        contents=request.message,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction
                        )
                    )
                    answer = response.text
                except Exception as fallback_e:
                    print(f"[CHAT] Fallback Gemini model also failed: {fallback_e}. Switching to Groq...")
                    answer = None
        except Exception as e:
            print(f"[CHAT] Gemini setup failed: {e}. Falling back to Groq...")
            answer = None

    # ── Groq fallback ─────────────────────────────────────────────────────────
    if answer is None:
        try:
            print("[CHAT] Using Groq fallback...")
            answer = _call_groq(
                system_instruction=system_instruction,
                user_message=request.message,
            )
            used_groq = True
        except Exception as groq_e:
            return {
                "answer": f"⚠️ Both Gemini and Groq APIs are unavailable: {str(groq_e)}",
                "mode": "general",
                "sources": []
            }

    if used_groq:
        answer = "🔄 *[Responded via Groq — Gemini quota exhausted]*\n\n" + answer

    return {
        "answer": answer,
        "mode": "general",
        "sources": []
    }
