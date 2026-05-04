# SocialLegal

A self-hosted RAG system for Icelandic legal research. Social workers ask
questions in plain Icelandic or English; an LLM uses tool calls to search a
local database of national statute articles and uploaded municipal documents,
then answers with citations.

---

## How it works

### Two corpora, one chat

- **Statutes** (national law from althingi.is) — scraped, parsed into laws →
  chapters → articles, with cross-references between articles tracked
  separately.
- **Documents** (PDF / TXT / HTML uploads) — extracted, chunked with overlap,
  optionally tagged to a municipality.

Both are embedded with OpenAI `text-embedding-3-large` (3072 dims) and stored
in a single Postgres + pgvector instance. There is no HNSW index — search is
exact cosine, fine for the small corpus this is built for.

### Tool-using agent

The chat endpoint runs a manual LangChain tool-calling loop. The LLM has six
tools available:

| Tool | Corpus | Purpose |
|---|---|---|
| `search_articles(query, k, law_ids?)` | statutes | Vector search articles |
| `get_article(article_id)` | statutes | Full article + breadcrumb + cross-refs |
| `get_chapter(chapter_id)` | statutes | All articles in a chapter |
| `search_documents(query, k, document_ids?, municipality_ids?)` | documents | Vector search chunks |
| `get_document(document_id)` | documents | Whole document, all chunks |
| `get_chunk(chunk_id, offset)` | documents | Walk chunks: `+1` = next, `-1` = previous |

A catalog of available laws and documents is injected into the system prompt
each turn, so the LLM can either narrow itself with `*_ids` arguments or ask
the user to pick when sources overlap (e.g. "pension" — general pension law vs.
public-sector pension law).

### Pre-filtering and lockout

If the user pre-filters the chat (e.g. picks "Reykjavíkurborg" in the
Municipalities mode), the relevant search tools are auto-restricted to that
subset and the catalog notes "PRE-FILTERED by the user — don't ask them to
filter again." The opposite tool is locked out for that turn so the model
doesn't go off-source.

### Cross-references between statutes

Every `<a href>` inside a parsed article body that points to another althingi
law page is recorded in `cross_references`, with the surrounding context
phrase. `POST /ingest/resolve-references` walks unresolved refs, fetches the
target laws, and *partially* ingests only the articles that were actually
linked to (`laws.partial = true`). Whole-law references (no anchor) are
reported but not auto-ingested.

### LLM selection

Active model is selected from the registry in `rag.py:PROVIDERS`
(currently: `anthropic/claude-sonnet-4-6`, `google/gemini-flash-3-preview`,
`openai/gpt-5.4-mini`). At startup it defaults to `DEFAULT_MODEL=provider/model`
(e.g. `anthropic/claude-sonnet-4-6`) when set and the matching API key is
present; otherwise it falls back to the first provider whose API key is
configured. A logged-in user can change it at runtime via `/admin`
(in-memory until restart). Embeddings always go through OpenAI.

---

## Project layout

```
Dockerfile, docker-compose.yml, requirements.txt, start.sh
.env.example
db.py             -- psycopg pool + idempotent schema init
auth.py           -- bcrypt + DB-backed session tokens
create_user.py    -- CLI for creating/updating users
ingestion.py      -- althingi scraper, document parser/chunker, cross-ref resolver
rag.py            -- tools, agent loop, SSE streaming
main.py           -- FastAPI app, all endpoints
static/
  index.html      -- chat UI
  ingest.html     -- admin ingest UI (Laws / Other tabs)
seed_municipalities.sql -- canonical municipality list
```

---

## Setup

1. `cp .env.example .env` and fill in:

   ```
   OPENAI_API_KEY=sk-...
   GEMINI_API_KEY=                   # optional
   ANTHROPIC_API_KEY=                # optional
   DEFAULT_MODEL=                    # optional, e.g. anthropic/claude-sonnet-4-6
   INGEST_SECRET=...                 # required for curl-based ingest
   SESSION_SECRET=...                # not currently used; reserved
   POSTGRES_PASSWORD=...
   ```

2. Bring it up:

   ```bash
   docker compose up -d --build
   ```

   The first start runs `initdb` with UTF-8 + `C.UTF-8` locale and the app
   creates its schema on startup.

3. Open <http://localhost:8000> — you'll be redirected to `/login`.

---

## Common commands

### Create or update a user

```bash
docker compose exec app python create_user.py alice s3cret
```

If `alice` already exists, the password is updated. Used for the chat login;
ingest endpoints accept either a logged-in cookie or `INGEST_SECRET`.

### Seed the canonical Icelandic municipalities

Pre-fills the 62 municipalities (Reykjavíkurborg, Kópavogsbær, etc.) with
their reserved ids:

```bash
docker compose exec -T db psql -U legal -d legal < seed_municipalities.sql
```

Idempotent — safe to re-run.

### Ingest law URLs (CLI)

```bash
curl -F "secret=$INGEST_SECRET" \
     --data-urlencode "urls=https://www.althingi.is/lagas/nuna/1997129.html
https://www.althingi.is/lagas/nuna/2003090.html" \
     http://localhost:8000/ingest
```

Or paste them in the **Ingest → Laws** tab in the browser.

### Resolve cross-references

After ingesting a law, expand the inbound refs by fetching just the articles
it points to:

```bash
curl -F "secret=$INGEST_SECRET" http://localhost:8000/ingest/resolve-references
```

Or click **Resolve pending references** in the Ingest UI. Re-run after
ingesting more laws or to chase newly-discovered refs (one level per call).

### Upload a document

Browser: **Ingest → Other** tab. Pick a file, give it a name, choose a
municipality (or check "No municipality").

CLI:

```bash
curl -F "secret=$INGEST_SECRET" \
     -F "name=Reglur Reykjavíkurborgar um félagslegt leiguhúsnæði" \
     -F "municipality_id=1" \
     -F "file=@rvk_felo.pdf" \
     http://localhost:8000/ingest/document
```

### Database introspection

```bash
docker compose exec db psql -U legal -d legal

# inside psql
\dt                                              -- list tables
SELECT id, law_number, title FROM laws;          -- ingested laws
SELECT id, name, source_type FROM documents;     -- uploaded docs
SELECT count(*) FROM cross_references
  WHERE resolved_article_id IS NULL;             -- pending refs
SHOW server_encoding;                            -- should be UTF8
```

### Restart / rebuild

```bash
docker compose restart app                       -- code-only restart
docker compose build app && docker compose up -d app   -- after deps change
docker compose down                              -- stop everything (keeps data)
docker compose down -v                           -- WIPE database volume
```

---

## API endpoints

Auth column: **session** = signed-in cookie (chat UI), **secret** = `INGEST_SECRET` form field, **either** = both accepted.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/login` | none | Login page |
| `POST` | `/login` | none | Form: `username`, `password` → set cookie |
| `GET` | `/logout` | none | Clear cookie |
| `GET` | `/me` | session | `{username}` |
| `GET` | `/` | session | Chat page (redirects to `/login` if no cookie) |
| `GET` | `/admin/ingest` | session | Ingest admin page |
| `POST` | `/chat` | session | SSE: `{message, law_ids?, municipality_ids?, document_ids?}` |
| `GET` | `/laws` | session | List ingested laws |
| `GET` | `/documents` | session | List uploaded documents |
| `GET` | `/municipalities` | session | List municipalities |
| `POST` | `/municipalities` | session | Form: `name` |
| `POST` | `/ingest` | either | Form: `urls`, `secret?` — ingest law URLs |
| `POST` | `/ingest/resolve-references` | either | Form: `secret?` — resolve cross-refs |
| `POST` | `/ingest/document` | either | Multipart: `file`, `name`, `municipality_id?`, `source_type?`, `secret?` |

`source_type` is auto-detected from content-type / extension when omitted
(`pdf` / `txt` / `html`).

---

## Troubleshooting

### Icelandic characters render as `?`

The cluster was probably initialized before the UTF-8 settings landed. Check:

```bash
docker compose exec db psql -U legal -d legal -c \
  "SHOW server_encoding; SHOW client_encoding;"
```

If `server_encoding` is anything other than `UTF8`:

```bash
# Back up users (everything else is re-ingestable)
docker compose exec db pg_dump -U legal -d legal \
  -t users -t sessions --data-only --inserts > users_backup.sql

docker compose down
docker volume rm "$(basename "$PWD")_pgdata"   # adjust to your project name
docker compose up -d --build

docker compose exec -T db psql -U legal -d legal < users_backup.sql
docker compose exec -T db psql -U legal -d legal < seed_municipalities.sql
# re-ingest laws and re-upload documents
```

### Vector type errors at search time

```
operator does not exist: vector <=> double precision[]
```

Means the running app container predates the `::vector` cast in the search
query. Rebuild the app:

```bash
docker compose build app && docker compose up -d app
```

### `/ingest` says "skipped" but I want to re-ingest

Currently the regular ingest skips any URL already present (full or partial).
To force a re-ingest, delete the row first:

```sql
DELETE FROM laws WHERE url = 'https://www.althingi.is/lagas/nuna/1997129.html';
```

The cascade drops chapters, articles, and cross-references for that law.

### "No text could be extracted from the document"

For PDFs this almost always means the PDF is image-based (scanned) and has no
selectable text. There's no OCR step in the pipeline. Convert with `ocrmypdf`
first, then upload.

---

## Notes

- `SESSION_SECRET` is listed in `.env.example` for forward-compat but the
  current implementation uses opaque DB-backed session tokens
  (`secrets.token_urlsafe(32)`), not signed cookies. Safe to leave blank.
- All article/chunk embeddings store `vector(3072)`. pgvector's `hnsw`/`ivfflat`
  indexes both cap at 2000 dimensions, so search runs as a sequential scan.
  This is intentional and fine at the corpus sizes this app targets.
- Ingest pipeline is synchronous. A large law (hundreds of articles) takes
  10–30 seconds to ingest because each article needs an embedding round-trip.
