# RAG + Text2SQL Hybrid Assistant

This app answers business questions over both structured data (SQL) and unstructured documents (RAG). It routes each query to the most suitable tool, shows the exact SQL used for transparency, and can optionally validate long‑form answers with Cleanlab Codex trust scoring.

## Features

- Text2SQL over your database (SQLite/Postgres/MySQL/DuckDB)
- RAG over PDFs/DOCX/PPTX/TXT (Docling parsing + Milvus/FAISS vector store)
- Automatic tool routing (SQL vs Documents) with timeouts and fallbacks
- “SQL Used” display for auditability and faster iteration
- Optional Cleanlab Codex trust score for document answers
- Database explorer tailored to the selected database
- Vector store toggle: Milvus (Docker) or FAISS (local, in‑process)
- File-hash caching for fast re-uploads (reuses embeddings)

## Architecture

- UI: Streamlit (`app.py`)
  - Sidebar for keys, DB selection, doc uploads, and vector-store toggle
  - Progress indicators during parsing → embeddings → indexing
  - Chat interface shows SQL / trust score
  - Database explorer wired to the active DB
- Orchestration: LlamaIndex workflow (`workflow.py`)
  - Decides whether to call the SQL tool or Document tool
  - Tool timeouts and error handling
- Text2SQL (`tools.py` → `setup_sql_tool`)
  - SQLAlchemy connection (SQLite path or SQLAlchemy URL)
  - LlamaIndex NLSQL engine with:
    - SQL dialect hinting (SQLite/Postgres/MySQL/DuckDB)
    - Schema primer (reflected tables/columns)
    - Few‑shot examples (joins, aggregations, summary table usage)
    - Light entity normalization
  - Returns both the result and the exact SQL used
- Documents / RAG (`tools.py` → `setup_document_tool`)
  - Docling parsing → chunking → embeddings (`BAAI/bge-small-en-v1.5`, 384‑dim)
  - Vector store: Milvus (Docker) or FAISS (in‑process)
  - Filename-aware retrieval, tuned top‑k, summarization to avoid truncation
  - Optional Cleanlab Codex trust scoring
  - File hashing to skip re‑embedding unchanged uploads
- LLM: via OpenRouter (configurable model)
- Config: `.env` or `.streamlit/secrets.toml` for keys and model names

## Demo dataset (FEMA NFIP)

- Source: FEMA NFIP redacted claims → converted to SQLite
- Star schema in `fema_nfip_star.sqlite`:
  - Fact: `nfip_fact_claim` (payout metrics; keys to dimensions)
  - Dimensions: `dim_state`, `dim_event`, `dim_flood_zone`, `dim_time` (year/quarter)
  - Summary: `quarterly_flood_zone_trend` (pre‑aggregated trends)
- Performance:
  - Indexes on joins/time columns, tuned SQLite pragmas
  - Pre‑aggregated summary table prevents timeouts on complex trend questions

## Setup

### 1) Python deps
```bash
cd rag-sql-router
uv sync        # or: pip install -r requirements.txt
```

### 2) Milvus (optional; use FAISS if you prefer fully local)
```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

### 3) Environment variables
Provide via `.env` or `.streamlit/secrets.toml`:

- `OPENROUTER_API_KEY` (required for LLM)
- `OPENROUTER_MODEL` (optional, e.g., `qwen/qwen-turbo`)
- `CODEX_API_KEY` (optional, Cleanlab user key)
- `CODEX_PROJECT_ACCESS_KEY` (optional, Cleanlab project key)
- `SQLITE_DB_PATH` (optional, default SQLite path; e.g., `fema_nfip_star.sqlite`)
- `ENGINE_URL` (optional, SQLAlchemy URL for Postgres/MySQL/DuckDB)

## Run

```bash
streamlit run app.py
```

In the sidebar:
- Enter OpenRouter key; models initialize automatically
- Toggle Milvus on (Docker) or off (FAISS)
- Pick DB Type: SQLite path or SQLAlchemy URL
- Upload documents (shows parse → embed → index progress)

Ask questions in the chat. For SQL queries, the app shows the “SQL Used”. For RAG answers, a trust score is shown when Codex is configured.

## Text2SQL details

- LLM‑driven (LlamaIndex NLSQL engine). No custom model training.
- Prompt includes: dialect hint, schema primer, few‑shots, entity normalization.
- Executes via SQLAlchemy and returns both the answer and the exact SQL.

## Troubleshooting

- Milvus not reachable: ensure Docker Desktop is running; start with the script above. Or toggle off Milvus to use FAISS.
- Long doc processing: large PDFs can take minutes (Docling OCR + embeddings). Prefer smaller chapters or pre‑converted TXT; re‑uploads reuse cached embeddings.
- SQL timeouts: select the star‑schema DB; prefer the pre‑aggregated `quarterly_flood_zone_trend` table for trend queries.
- Keys/Models: verify `OPENROUTER_API_KEY`; set `OPENROUTER_MODEL` if needed. Codex keys are optional; without them, trust score is hidden.

## Local‑only mode

- Swap OpenRouter for a local OpenAI‑compatible server (e.g., Ollama/LM Studio)
- Disable Codex
- Use SQLite/Postgres/MySQL locally
- Use FAISS (in‑process) or Milvus (local Docker)

## Quick commands

```bash
uv sync
streamlit run app.py
# Milvus
bash standalone_embed.sh start
bash standalone_embed.sh stop
```

## License

This repository is provided as‑is under an open license. See the LICENSE file if present.
