# CLAUDE.md

Notes for Claude working in this repo. README.md is the user-facing doc;
this file is the working primer.

## What this is

Self-hosted Icelandic legal-research RAG. FastAPI + Postgres + pgvector,
all in one `docker compose` stack. Two corpora (statutes + regulations +
free-form documents) feed a tool-using LLM agent that answers in Icelandic
or English with citations.

## Stack

- Python 3.10, FastAPI, psycopg (v3), pgvector
- LangChain only as a thin tool-calling glue — the agent loop in `rag.py`
  is hand-rolled, not `AgentExecutor`
- OpenAI `text-embedding-3-large` (3072 dims) for all embeddings
- Chat LLM is selectable at runtime (Anthropic / Google / OpenAI) — see
  `rag.py:PROVIDERS`. Embeddings always OpenAI.
- Postgres 16 via `pgvector/pgvector:pg16` image
- No HNSW/IVFFlat — 3072 dims exceed pgvector's 2000-dim index cap, so
  search is sequential cosine. Intentional; corpus is small.

## File map

| File | Role |
|---|---|
| `main.py` | FastAPI app, all HTTP endpoints, auth glue |
| `rag.py` | Agent loop, tool definitions, model registry, SSE streaming |
| `ingestion.py` | Scrapers + parsers (althingi laws, island.is regulations, generic docs), embeddings, cross-ref resolver |
| `db.py` | psycopg connection pool + idempotent schema bootstrap |
| `auth.py` | bcrypt + DB-backed opaque session tokens (NOT signed cookies) |
| `create_user.py` | CLI to create/update users |
| `static/index.html` | Chat UI |
| `static/ingest.html` | Admin ingest UI (Laws / Regulations / Other tabs) |
| `static/admin.html` | Runtime model picker |
| `static/test.html` + `test_results.html` | A/B model rating page |
| `seed_municipalities.sql` | Canonical 62 Icelandic municipalities |

## Database tables

`laws → chapters → articles`, `regulations → regulation_articles`,
`documents → document_chunks`, plus `cross_references`, `municipalities`,
`users`, `sessions`, `model_votes`. Schema is created on app startup by
`db.py` — edit there, not via migrations.

URL is the unique key on `laws`, `regulations`, and used for
de-duplication. `laws.partial = true` means the row was created only to
satisfy a cross-ref — only some articles are stored.

## Ingestion specifics

Three URL formats supported, each via its own parser:

1. `https://www.althingi.is/lagas/nuna/<YYYYNNN>.html` → `parse_law` (laws)
2. `https://island.is/reglugerdir/nr/<num>-<year>` → `parse_regulation`
3. `https://island.is/stjornartidindi/nr/<uuid>` → `parse_regulation`
   (same parser, falls back to `<title>` tag and "Nr. NN/YYYY" page text
   for title + number; loop breaks on `class="signature"`)

Generic documents (PDF/TXT/HTML) go through `ingest_document` →
chunked (1500 chars, 200 overlap) → embedded → `document_chunks`.

Cross-refs: every `<a href>` inside a parsed law article that points to
another althingi law page is recorded. `POST /ingest/resolve-references`
walks unresolved refs one level at a time.

## Agent / tool loop

System prompt has a live catalog of laws + regulations + documents
injected each turn. Six tools across three corpora:

- statutes: `search_articles`, `get_article`, `get_chapter`
- regulations: `search_regulation_articles`, `get_regulation_article`,
  `get_regulation`
- documents: `search_documents`, `get_document`, `get_chunk(id, offset)`
  (offset walks neighboring chunks)

Pre-filter behavior: when the chat request includes `law_ids`,
`regulation_ids`, `municipality_ids`, or `document_ids`, the corresponding
search tools are auto-restricted and the *other* corpus's tools are locked
out for the turn (so a "documents" filter doesn't leak into statute
searches).

Streaming uses SSE with custom event names — see `_sse()` in `rag.py`.

## Model selection

`rag.py:PROVIDERS` is the registry. Active model = `DEFAULT_MODEL` env
(`provider/model`) at startup if its key is set, else first provider with
a key. `/admin` page changes it at runtime, in-memory only.

## Auth

- DB-backed opaque session tokens, not JWT
- `INGEST_SECRET` form field is the alternative to a session cookie for
  ingest endpoints (used for curl/cron)
- `SESSION_SECRET` env var is reserved, not currently consumed

## Conventions

- Python uses `from __future__ import annotations`, `list[...]`/`dict[...]`,
  `| None`. Match this style.
- `with conn() as c, c.cursor() as cur:` — every DB block uses the
  `db.conn()` context manager. Don't open raw connections.
- Always `register_vector(c)` before inserting/reading `vector` columns.
- Embeddings: `embed_texts` batches of 32, truncates to 28000 chars.
- Icelandic strings everywhere — never strip non-ASCII, never `.encode('ascii')`.
- Keep `db.py` schema idempotent — `CREATE TABLE IF NOT EXISTS`, `ALTER
  TABLE ... ADD COLUMN IF NOT EXISTS`. App restarts re-run it.

## Dev workflow

Python is in `.venv/bin/python3`; deps installed there. For ad-hoc
parsing checks without the DB:

```bash
DATABASE_URL=postgres://x .venv/bin/python3 -c "from ingestion import parse_regulation; ..."
```

(Importing `ingestion` triggers `db.py` import which requires
`DATABASE_URL` — a dummy value works for parse-only tests.)

App rebuild after code changes:

```bash
docker compose build app && docker compose up -d app
```

DB-only restart never needed — schema migrates on app boot.

## Gotchas

- The `vector` column needs an explicit `::vector` cast in some queries —
  if you see `operator does not exist: vector <=> double precision[]`,
  the running container predates the cast. Rebuild.
- pgvector dim cap is 2000; adding an index on `embedding` will fail.
  Don't add one.
- `docker compose down -v` wipes the volume. There's no auto-backup —
  see README's "Backup and restore" section.
- Ingest is synchronous — a 200-article law takes 10–30 s.
- The `partial` flag on laws is load-bearing: a law ingested as a
  cross-ref target has `partial=true` and only the linked articles
  exist. Don't assume `articles` contains the whole law.
- `_clean_body` collapses whitespace; cross-ref `position` offsets are
  computed *before* cleaning, so don't reorder those steps.
