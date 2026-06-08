# RAG AI Chat (DocuMind)

A premium, production-grade Retrieval-Augmented Generation (RAG) Document Assistant that allows users to upload PDF documents, chat with them interactively, view source citations with confidence gauges, and generate self-assessment practice quizzes.

Featuring a beautiful **sleek dark-glassmorphism UI**, the system is highly responsive, fully compatible with mobile screen viewports, and optimized for instant backend server boot times.

---

## 🚀 Key Features

* **Advanced PDF Structuring & Semantic Chunking**: Respects paragraph boundaries and document layouts, preserving page numbers, tags, and structure during chunking.
* **Hybrid Search (Dense + Sparse)**: Combines Pinecone dense vector embeddings (`all-MiniLM-L6-v2`) with rank-BM25 sparse keyword indexers.
* **Cross-Encoder Reranking**: Re-orders retrieved document chunks using `ms-marco-MiniLM-L-6-v2` for precise context grounding before query submission.
* **Smart LLM Grounding & Cascade Fallback**: Queries Google Gemini (`gemini-2.5-flash`) for general/document answers, falling back automatically to Groq (`llama-3.1-8b-instant`) and Hugging Face inference hubs.
* **Interactive Tooltip Citations**: Matches generated answer citations back to document cards. Hovering over citation pills shows source PDF filenames and page numbers.
* **Self-Assessment Quizzes**: Generates conceptual, scenario, and statement-based MCQs from uploaded documents. Option choices and correct answers are dynamically randomized to prevent pattern bias.
* **Instant Backend Startup (Lazy-loading)**: SentenceTransformer embedding models are lazy-loaded on the first request, resulting in instantaneous server boots and uvicorn hot-reloads.

---

## 📋 Technology Stack

### Backend
* **Web Framework**: FastAPI (Python 3.10+)
* **Relational Database**: PostgreSQL (Document registry and tags)
* **Vector Database**: Pinecone (Vector indexing & metadata search)
* **Embeddings Model**: SentenceTransformer (`all-MiniLM-L6-v2` — 384 dimensions)
* **Reranking Model**: CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
* **LLM Engine**: Google GenAI / Groq Cascade

### Frontend
* **UI Library**: React.js (ES6+)
* **Styling**: Vanilla CSS (Custom modern dark HSL color palettes, neon accents, and backdrop glassmorphism)

---

## 🛠️ Installation & Setup

### Prerequisites
* **Python 3.10 or higher**
* **Node.js 16 or higher**
* **PostgreSQL Server** (installed and running)
* **Pinecone Account** (with a configured index matching your env dimensions: 384)

---

### Step 1: Backend Configuration

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create and configure your `.env` file based on `.env.example`:
   ```env
   # Google Gemini API Configuration
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL=gemini-2.5-flash

   # Groq Fallback API Configuration
   GROQ_API_KEY=your_groq_api_key
   GROQ_MODEL=llama-3.1-8b-instant

   # HuggingFace (Optional Fallback)
   HUGGINGFACE_API_KEY=your_huggingface_key
   HUGGINGFACE_EMBEDDING_MODEL=all-MiniLM-L6-v2
   HUGGINGFACE_LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

   # Pinecone Vector DB Configuration
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENV=us-east-1
   PINECONE_INDEX=gate-rag

   # PostgreSQL Relational Database URL
   DATABASE_URL=postgresql://<username>:<password>@localhost:5432/rag_ai
   ```

5. Initialize the database schema and create local tables:
   ```bash
   python create_tables.py
   ```

6. Run the FastAPI server:
   ```bash
   python -m uvicorn app.main:app --reload --port 8000
   ```
   The backend documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

### Step 2: Frontend Configuration

1. Open a new terminal window and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install the frontend dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```
   The application will compile and launch automatically at [http://localhost:3000](http://localhost:3000).

---

## 📁 Project Directory Structure

```
RAG AI Chat/
├── backend/
│   ├── app/
│   │   ├── config.py           # Configuration and environment variables
│   │   ├── database.py         # SQLAlchemy connection engine
│   │   ├── main.py             # FastAPI entrypoint and CORS setup
│   │   ├── models.py           # PostgreSQL DB Schema models (Documents/Chunks)
│   │   ├── schemas.py          # Pydantic schemas (MCQs, Chat, Queries)
│   │   ├── routes/
│   │   │   ├── upload.py       # PDF ingest, structural parsing & chunking
│   │   │   ├── query.py        # Chat queries and RAG execution
│   │   │   ├── practice.py     # Practice problem generation triggers
│   │   │   └── documents.py    # PostgreSQL metadata list/delete endpoints
│   │   └── services/
│   │       ├── advanced_pdf_parser.py   # Hierarchical PDF structure reader
│   │       ├── chunker.py               # Context-aware text chunking
│   │       ├── embeddings.py            # Lazy-loaded SentenceTransformer client
│   │       ├── hybrid_search.py         # RRF fusion of Dense and BM25 search
│   │       ├── pinecone_service.py      # Upsert and retrieval vector calls
│   │       ├── practice_generator.py    # MCQ and practice generator pipelines
│   │       └── rag_pipeline.py          # Grounded LLM response processing
│   ├── create_tables.py        # Database schema initializer script
│   └── requirements.txt        # Python backend dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWithDocs.jsx  # Main container (Tabs, Document Center sidebar)
│   │   │   ├── chatbox.js        # Chat message entry component
│   │   │   ├── answerbox.js      # Citation mapping & confidence rating UI
│   │   │   ├── uploadbox.js      # Document category upload box
│   │   │   └── practicebox.js    # Self-Assessment MCQ panel
│   │   ├── App.js                # Core layout entrypoint
│   │   ├── App.css               # Main Glassmorphism design system
│   │   └── index.js              # ReactDOM renderer
│   └── package.json              # Frontend Node dependencies
└── README.md
```

---

## 🔄 Main API Routing

* **`POST /upload`**: Ingests document. Accepts multi-part Form: `file` (PDF) and `subject` (Category Tag). Semantic chunker parses text, uploads vectors to Pinecone, and writes records to PostgreSQL.
* **`POST /query/ask`**: Grounded chat query. Accepts `question` (string), `subject` (optional filter), and `document_ids` (array of numeric filters). Returns JSON payload:
  ```json
  {
    "answer": "Grounded answer text here...",
    "original_answer": "Raw response from LLM...",
    "sources": [
      {
        "content": "Context snippet text...",
        "filename": "document.pdf",
        "chunk_index": 1,
        "doc_id": "19",
        "score": 0.82
      }
    ],
    "confidence": {
      "overall_confidence": 85.0
    }
  }
  ```
* **`POST /practice/generate`**: Self-Assessment engine. Generates randomized conceptual, scenario, and calculation problems. Configures `difficulty` (`easy`, `medium`, `hard`) and `question_type` (`mcq`, `theory`, `numerical`).
* **`GET /documents`**: Returns JSON list of all uploaded documents including IDs, subjects/categories, filenames, and uploads timestamps.
* **`DELETE /documents/{id}`**: Purges metadata from PostgreSQL and vectors from Pinecone.

---

## ⚡ Development Diagnostics & Verification

To verify that embedding creation, database queries, and RAG fallbacks are configured properly, run the debug script inside the virtual environment:
```bash
cd backend
$env:PYTHONIOENCODING="utf-8"   # On Windows PowerShell
python test_practice_debug.py
```
This runs internal tests checking the active Groq fallback, local database connectivity, and Pydantic options normalizations.

## 🧠 Architectural Decisions & Design Trade-offs

During the design and implementation of RAG AI Chat, several architectural choices and product decisions were made to improve reliability, speed, and usability:

### 1. Lazy-Loading the SentenceTransformer Model
* **Choice**: Switched `SentenceTransformer` initialization in `embeddings.py` from module load-time to a lazy-loading getter function `_get_model()`.
* **Reasoning**: Loading the PyTorch/transformer weights (over 100MB) at startup delayed server boots and hot-reloads by 5–10 seconds. Deferring the load to when the first vector operation runs enables instant hot-reloads and rapid developer iterations.
* **Trade-off**: The very first query/upload request has a small "cold-start" latency penalty, which is highly acceptable compared to waiting for the server reload on every minor backend change.

### 2. Multi-Model Cascade Fallback (Gemini → Groq)
* **Choice**: Set up a robust fallback route hierarchy where API calls attempt Google Gemini (`gemini-2.5-flash`) first, and fall back to Groq (`llama-3.1-8b-instant`) under rate-limits (HTTP 429).
* **Reasoning**: Generative AI APIs (especially on free tiers) frequently experience quota exhaustion. An automated model cascade prevents application downtime, ensuring user prompts are successfully answered.
* **Trade-off**: Subtle styling and reasoning variations might exist between Gemini and Groq outputs, but this is controlled by enforcing highly structured prompt directives.

### 3. Reduced Token Reservation for Groq Completion
* **Choice**: Lowered the completion parameter `max_tokens` from 5,000/6,000 to 2,000 for quiz and chat generation queries.
* **Reasoning**: Groq enforces a 6,000 Tokens-Per-Minute (TPM) limit on its free-tier models. If an API call requests `max_tokens=6000`, Groq proactively blocks it as a rate limit violation even if the actual generated response would only be 50 tokens. Lowering the token window fits comfortably under the TPM rate limits while leaving ample room for detailed answers.

### 4. Backend Shuffling & Option Remapping for MCQs
* **Choice**: Shuffler helper inside `practice_generator.py` and `rag_pipeline.py` that shuffles the options list (A, B, C, D) and remaps the correct option pointer at runtime.
* **Reasoning**: Example templates inside prompts bias the LLM to output a specific choice (e.g. choice "B") as the correct answer. Dynamic shuffling at the backend guarantees a mathematically uniform probability distribution of correct options and eliminates prediction bias.

### 5. Local Cache Fallbacks for Hybrid Retrieval
* **Choice**: Implemented a graceful metadata extraction fallback inside hybrid search where the system queries Pinecone with metadata retrieval enabled if the in-memory BM25 index cache is cleared (e.g. after server restarts).
* **Reasoning**: In-memory caching is highly performant but transient. Ensuring vector database queries can stand alone ensures document sources and filenames never default to "Unknown Document".



**Version**: 1.0.0  
**License**: MIT
