# ⚖️ EU AI Act Assistant — RAG System

A production-style Retrieval-Augmented Generation (RAG) system for querying 
the EU Artificial Intelligence Act (Regulation 2024/1689).

Built by adapting the [Ray team's LLM applications architecture](https://github.com/ray-project/llm-applications), 
stripped of Ray cluster dependencies to run locally, with three additional 
improvements on top of their baseline.

## Live Demo
- Streamlit UI: `localhost:8501`
- REST API docs: `localhost:8000/docs`

## Architecture
**Pipeline flow:**

1. **Load** — PyMuPDF reads the EU AI Act PDF page by page
2. **Chunk** — RecursiveCharacterTextSplitter splits into 300-token chunks with 50-token overlap
3. **Embed** — all-MiniLM-L6-v2 converts each chunk into a 384-dimensional vector
4. **Index** — vectors stored in FAISS (dense) + tokenized text stored in BM25 (sparse)
5. **Query expansion** — Groq LLM rewrites the user query with legal terminology
6. **Hybrid retrieve** — FAISS semantic search (top-10) + BM25 keyword search (top-20) combined with Reciprocal Rank Fusion
7. **Rerank** — Flashrank cross-encoder rescores top-20 candidates, returns top-5
8. **Generate** — Llama 3.1 8B via Groq produces answer grounded in retrieved context
9. **Serve** — FastAPI REST endpoint + Streamlit UI

## My Improvements Over the Ray Baseline

| Improvement | Reason |
|---|---|
| Stripped Ray cluster dependency | Runs locally, no infrastructure cost |
| Hybrid BM25 + FAISS with RRF | BM25 catches exact legal terms (e.g. "Article 9") that semantic search misses |
| Cross-encoder reranking (Flashrank) | Re-scores top-20 candidates jointly with query — more accurate than cosine similarity alone |
| LLM query expansion | Enriches queries with legal terminology before retrieval |

## Chunk Size Experiment

Systematic experiment across 4 chunk sizes on the EU AI Act:

| Chunk size | Total chunks | Result |
|---|---|---|
| 200 | 4736 | Failed — chunks too small, context destroyed |
| 300 | 2544 | Best — correct article references, minimal hallucination |
| 500 | 1398 | Partial — missed definitions, honest "I don't know" |
| 1000 | 707 | Worst — hallucinated a definition of high-risk AI |

**Key finding:** 300 tokens worked best for legal documents. Larger chunks 
caused the LLM to hallucinate plausible-sounding but incorrect legal definitions.
Smaller chunks lost the context needed to answer definitional questions.

## Key Technical Findings

**What improved with hybrid search + reranking:**
- Provider obligations (Article 16) — correctly cited with structured list
- Transparency requirements (Article 13) — found correct page range
- Risk management references — correctly located cross-references

**Remaining challenge:**
- Definitions spread across Annex I and Annex III require multi-chunk 
  synthesis — a single retrieval step can't capture distributed information.
  Next step: metadata-aware retrieval that knows document structure.

**Query expansion failure mode discovered:**
- LLM hallucinated article numbers in expanded queries, degrading retrieval.
- Fix: constrained expansion prompt to legal terminology only, 
  preserving explicit article numbers from the original query.

## Tech Stack

| Component | Tool |
|---|---|
| Document loading | PyMuPDF (LangChain) |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Dense retrieval | FAISS |
| Lexical retrieval | BM25 (rank_bm25) |
| Fusion | Reciprocal Rank Fusion |
| Reranking | Flashrank (ms-marco-MiniLM-L-12-v2) |
| LLM | Llama 3.1 8B via Groq (free) |
| API serving | FastAPI + Uvicorn |
| UI | Streamlit |
| Orchestration | LangChain |

## How to Run

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/llm-applications
cd llm-applications/my_rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Set API key (free at console.groq.com)
export GROQ_API_KEY="your_key_here"

# 3. Add the EU AI Act PDF
# Download from: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689
# Save as: eu_ai_act.pdf in this folder

# 4a. Run pipeline directly
python3 improved_rag.py

# 4b. Or run as REST API
uvicorn api:app --reload
# Visit http://127.0.0.1:8000/docs

# 4c. Or run Streamlit UI (keep API running in another terminal)
streamlit run app.py
# Visit http://localhost:8501
```

## Project Files

| File | Purpose |
|---|---|
| `rag.py` | Basic LangChain RAG pipeline |
| `experiment.py` | Chunk size experiment (200/300/500/1000) |
| `improved_rag.py` | Full pipeline with hybrid search + reranking + query expansion |
| `api.py` | FastAPI REST endpoint |
| `app.py` | Streamlit UI |

## Business Context

The EU AI Act (effective August 2024) is the world's first comprehensive 
AI regulation. Every company deploying AI in the EU must understand their 
compliance obligations. This system makes the 459-page regulation queryable 
in natural language — directly relevant for German companies navigating 
compliance requirements.
