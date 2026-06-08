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

from app.services.retrieval import retrieve_relevant_chunks

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
    # Call retrieve_relevant_chunks
    try:
        results = retrieve_relevant_chunks(request.message, top_k=5, doc_id=request.doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    # Fallback to database chunk retrieval for general summary or meta queries when similarity score is low
    if not results:
        query_lower = request.message.lower()
        doc_keywords = ["summary", "summarize", "overview", "tell me about", "what is this", "about this", "this pdf", "this document", "this file", "what does this", "what is the pdf", "what is the document"]
        if any(kw in query_lower for kw in doc_keywords):
            try:
                # Determine target doc_ids
                target_ids = []
                if request.doc_id:
                    if isinstance(request.doc_id, list):
                        target_ids = [int(d) for d in request.doc_id if str(d).isdigit()]
                    elif str(request.doc_id).isdigit():
                        target_ids = [int(request.doc_id)]
                
                if not target_ids:
                    # Fallback to overall most recently uploaded doc
                    row = db.execute(text("""
                        SELECT doc_id FROM document_chunks 
                        ORDER BY id DESC LIMIT 1
                    """)).fetchone()
                    if row:
                        target_ids = [row[0]]
                
                if target_ids:
                    # Fetch first 5 chunks of the target documents
                    db_chunks = db.execute(text("""
                        SELECT content, page_num, chunk_index, filename, doc_id 
                        FROM document_chunks 
                        WHERE doc_id IN :doc_ids 
                        ORDER BY doc_id DESC, chunk_index ASC LIMIT 5
                    """), {"doc_ids": tuple(target_ids)}).fetchall()
                    
                    results = [{
                        "content": c[0],
                        "page_num": c[1],
                        "chunk_index": c[2],
                        "filename": c[3],
                        "doc_id": str(c[4]),
                        "score": 0.5
                    } for c in db_chunks]
            except Exception as e:
                print(f"[CHAT/RAG] Database fallback retrieval failed: {e}")

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
                inner_msg = str(inner_e)
                if _is_quota_error(inner_msg):
                    # Try fallback Gemini model first
                    try:
                        fallback_model = "gemini-flash-latest"
                        print(f"[CHAT/RAG GEMINI FALLBACK] Trying '{fallback_model}'...")
                        chat = client.chats.create(
                            model=fallback_model,
                            history=gemini_history,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction
                            )
                        )
                        response = chat.send_message(user_message)
                        answer = response.text
                    except Exception:
                        # Gemini fully exhausted — fall through to Groq
                        print("[CHAT/RAG] Gemini quota exhausted. Switching to Groq...")
                        answer = None
                else:
                    raise inner_e
        except Exception as e:
            err_msg = str(e)
            if not _is_quota_error(err_msg):
                raise HTTPException(status_code=500, detail=f"Gemini API call failed: {err_msg}")
            # Quota error — fall through to Groq
            print(f"[CHAT/RAG] Gemini error: {err_msg}. Falling back to Groq...")
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
                inner_msg = str(inner_e)
                if _is_quota_error(inner_msg):
                    try:
                        fallback_model = "gemini-flash-latest"
                        print(f"[CHAT GEMINI FALLBACK] Trying '{fallback_model}'...")
                        response = client.models.generate_content(
                            model=fallback_model,
                            contents=request.message,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction
                            )
                        )
                        answer = response.text
                    except Exception:
                        print("[CHAT] Gemini quota exhausted. Switching to Groq...")
                        answer = None
                else:
                    raise inner_e
        except Exception as e:
            err_msg = str(e)
            if not _is_quota_error(err_msg):
                raise HTTPException(status_code=500, detail=f"Gemini API call failed: {err_msg}")
            print(f"[CHAT] Gemini error: {err_msg}. Falling back to Groq...")
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
