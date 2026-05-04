"""LangChain tools + agent loop for the legal RAG chat."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Iterator

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from openai import OpenAI
from pgvector.psycopg import register_vector

from db import conn

EMBEDDING_MODEL = "text-embedding-3-large"
MAX_STEPS = 8
SEARCH_SNIPPET_CHARS = 300

SYSTEM_PROMPT = """You are a legal research assistant helping Icelandic social workers understand how specific laws apply to their cases.

Your users are social workers, not lawyers. They need clear, practical answers about whether a law applies to a situation and what it means for their client — not academic legal analysis.

## Retrieval rules
- You have three corpora: **statute articles** (national laws, parsed structurally with chapters, articles, cross-references), **regulation articles** (Icelandic reglugerðir from island.is, flat list of articles per regulation), and **municipal/other documents** (PDF/TXT/HTML uploads, retrieved by chunks). Each has its own search tool.
- Pick the right tool for the question:
  - National-law questions ("what does the pension act say...", "which articles govern...") → `search_articles`.
  - Regulation questions ("what does reglugerð X say...", "what are the 2026 income limits...") → `search_regulations`.
  - Municipal-rule, internal-policy or document-specific questions ("what are Reykjavík's social-housing rules...", "what does this PDF say about...") → `search_documents`.
  - When unsure, pick the most likely one first and broaden if it returns nothing relevant. Often a law and its implementing regulation both matter — search both.
- A catalog of available laws, regulations, and uploaded documents is provided below. Use the `law_ids` argument on `search_articles`, `regulation_ids` on `search_regulations`, or `document_ids`/`municipality_ids` on `search_documents` to scope to a subset when the question clearly belongs to one — different sources can use overlapping vocabulary.
- If the question is genuinely ambiguous about which source applies (e.g. "rules about housing" — could be national law, a regulation, or a Reykjavík rule), ask the user to pick one or more before searching, instead of guessing. Reference items by their human-readable name so the user can choose.
- If the user has pre-filtered, the catalog notes this and the relevant search tool is automatically restricted. Don't ask them to filter again.
- If a snippet looks relevant but is incomplete, call `get_article` (statute), `get_regulation_article` (regulation), or `get_document` (uploaded document) for the full text.
- If a document chunk from `search_documents` is on-topic but cuts off mid-thought, call `get_chunk(chunk_id, offset=+1)` for the next chunk or `offset=-1` for the previous one — cheaper than fetching the whole document. The response flags `is_first_chunk` / `is_last_chunk` so you know when to stop walking.
- If the question requires broader context within a statute, call `get_chapter`. For full regulation context, call `get_regulation`.
- When an article contains cross_references with ingested: true, call `get_article` on those article_ids before answering if the question depends on values or definitions found there.
- Do not invent legal content. If retrieval returns nothing relevant, say so plainly.
- Include key word search as well - f.ex. if the user asks "What rules apply to those on temporary visa?" search for whatever phrase you think is best and also include a second search for "temporary visa" as a keyword search in this case.

## Answer format
1. **Direct answer first.** Start with a plain-language conclusion — does the law apply, what does the client qualify for, what must happen next. No preamble.
2. **Brief explanation.** One short paragraph explaining the relevant rule in plain Icelandic or English.
3. **Citations.** At the end, list the specific articles you drew from. Format: "Lög nr. 129/1997, II. kafli, 7. gr." or "Act 129/1997, ch. II, art. 7" depending on response language.
4. **Flag uncertainty.** If the answer depends on facts you don't have (e.g. the client's income, employment status), say what information is needed before a conclusion can be drawn.

## Language
- Reply in Icelandic if the user wrote in Icelandic, English otherwise.
- Use plain language. Avoid legal jargon where possible; explain it briefly when unavoidable.
- Never quote entire articles verbatim. Summarize what matters and quote only the decisive phrase.
"""


def _embed_query(text: str) -> list[float]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def _format_cross_ref_row(row) -> dict:
    """Row columns:
    0 to_law_number, 1 to_article_anchor, 2 resolved_article_id,
    3 resolved_article_number, 4 resolved_law_title, 5 stub_law_title.
    """
    to_law_num, anchor, resolved_id, res_num, res_law_title, stub_law_title = row
    law_title = res_law_title or stub_law_title
    law_label = f"{law_title}, {to_law_num}" if law_title else to_law_num
    article_number = f"{res_num}. gr." if res_num else None
    entry: dict = {
        "law": law_label,
        "article_anchor": anchor,
        "article_number": article_number,
        "article_id": resolved_id,
        "ingested": resolved_id is not None,
    }
    if resolved_id is None:
        entry["note"] = (
            "whole-law reference, not ingested"
            if anchor is None
            else "anchored reference pending; run /ingest/resolve-references"
        )
    return entry


def _fetch_cross_refs(article_ids: list[int]) -> dict[int, list[dict]]:
    if not article_ids:
        return {}
    sql = """
        SELECT cr.from_article_id,
               cr.to_law_number,
               cr.to_article_anchor,
               cr.resolved_article_id,
               ra.number             AS resolved_article_number,
               rl.title              AS resolved_law_title,
               lstub.title           AS stub_law_title
          FROM cross_references cr
          LEFT JOIN articles ra    ON ra.id = cr.resolved_article_id
          LEFT JOIN laws rl        ON rl.id = ra.law_id
          LEFT JOIN laws lstub     ON lstub.url = cr.to_law_url
         WHERE cr.from_article_id = ANY(%s)
         ORDER BY cr.id
    """
    grouped: dict[int, list[dict]] = {}
    with conn() as c, c.cursor() as cur:
        cur.execute(sql, (article_ids,))
        for r in cur.fetchall():
            from_id = r[0]
            grouped.setdefault(from_id, []).append(_format_cross_ref_row(r[1:]))
    return grouped


def _normalize_law_ids(law_ids: list[int] | None) -> list[int] | None:
    if law_ids is None:
        return None
    cleaned = [int(x) for x in law_ids if x is not None]
    return cleaned or None


def _search_articles(query: str, k: int = 5, law_ids: list[int] | None = None) -> list[dict]:
    law_ids = _normalize_law_ids(law_ids)
    embedding = _embed_query(query)
    sql = """
        SELECT a.id,
               a.number,
               a.title,
               LEFT(a.content, %(snip)s) AS snippet,
               c.number AS chapter_number,
               c.title  AS chapter_title,
               c.id     AS chapter_id,
               l.title  AS law_title,
               l.law_number,
               l.id     AS law_id
          FROM articles a
          LEFT JOIN chapters c ON c.id = a.chapter_id
          JOIN laws l ON l.id = a.law_id
         WHERE (%(law_ids)s::int[] IS NULL OR a.law_id = ANY(%(law_ids)s::int[]))
         ORDER BY a.embedding <=> %(emb)s::vector
         LIMIT %(k)s
    """
    with conn() as c:
        register_vector(c)
        with c.cursor() as cur:
            cur.execute(
                sql,
                {
                    "snip": SEARCH_SNIPPET_CHARS,
                    "law_ids": law_ids,
                    "emb": embedding,
                    "k": max(1, min(k, 20)),
                },
            )
            rows = cur.fetchall()
    results = []
    for r in rows:
        results.append(
            {
                "article_id": r[0],
                "article_number": r[1],
                "article_title": r[2],
                "snippet": r[3],
                "chapter_number": r[4],
                "chapter_title": r[5],
                "chapter_id": r[6],
                "law_title": r[7],
                "law_number": r[8],
                "law_id": r[9],
            }
        )
    refs_by_article = _fetch_cross_refs([r["article_id"] for r in results])
    for entry in results:
        entry["cross_references"] = refs_by_article.get(entry["article_id"], [])
    return results


def _get_article(article_id: int) -> dict | None:
    sql = """
        SELECT a.id, a.number, a.title, a.content,
               c.id, c.number, c.title,
               l.id, l.law_number, l.title
          FROM articles a
          LEFT JOIN chapters c ON c.id = a.chapter_id
          JOIN laws l ON l.id = a.law_id
         WHERE a.id = %s
    """
    with conn() as c, c.cursor() as cur:
        cur.execute(sql, (article_id,))
        r = cur.fetchone()
    if r is None:
        return None
    refs_by_article = _fetch_cross_refs([r[0]])
    return {
        "article_id": r[0],
        "article_number": r[1],
        "article_title": r[2],
        "content": r[3],
        "chapter_id": r[4],
        "chapter_number": r[5],
        "chapter_title": r[6],
        "law_id": r[7],
        "law_number": r[8],
        "law_title": r[9],
        "cross_references": refs_by_article.get(r[0], []),
    }


def _get_chapter(chapter_id: int) -> dict | None:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.number, c.title, l.id, l.law_number, l.title
              FROM chapters c
              JOIN laws l ON l.id = c.law_id
             WHERE c.id = %s
            """,
            (chapter_id,),
        )
        ch = cur.fetchone()
        if ch is None:
            return None
        cur.execute(
            """
            SELECT id, number, title, content, ordinal
              FROM articles
             WHERE chapter_id = %s
             ORDER BY ordinal
            """,
            (chapter_id,),
        )
        rows = cur.fetchall()
    return {
        "chapter_id": ch[0],
        "chapter_number": ch[1],
        "chapter_title": ch[2],
        "law_id": ch[3],
        "law_number": ch[4],
        "law_title": ch[5],
        "articles": [
            {
                "article_id": r[0],
                "article_number": r[1],
                "article_title": r[2],
                "content": r[3],
                "ordinal": r[4],
            }
            for r in rows
        ],
    }


def list_laws() -> list[dict]:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT id, law_number, title FROM laws ORDER BY law_number"
        )
        rows = cur.fetchall()
    return [{"id": r[0], "law_number": r[1], "title": r[2]} for r in rows]


def list_regulations() -> list[dict]:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT id, regulation_number, title FROM regulations ORDER BY regulation_number"
        )
        rows = cur.fetchall()
    return [{"id": r[0], "regulation_number": r[1], "title": r[2]} for r in rows]


def _search_regulation_articles(
    query: str, k: int = 5, regulation_ids: list[int] | None = None
) -> list[dict]:
    regulation_ids = _normalize_law_ids(regulation_ids)
    embedding = _embed_query(query)
    sql = """
        SELECT ra.id,
               ra.number,
               ra.title,
               LEFT(ra.content, %(snip)s) AS snippet,
               r.title             AS regulation_title,
               r.regulation_number,
               r.id                AS regulation_id
          FROM regulation_articles ra
          JOIN regulations r ON r.id = ra.regulation_id
         WHERE (%(reg_ids)s::int[] IS NULL OR ra.regulation_id = ANY(%(reg_ids)s::int[]))
         ORDER BY ra.embedding <=> %(emb)s::vector
         LIMIT %(k)s
    """
    with conn() as c:
        register_vector(c)
        with c.cursor() as cur:
            cur.execute(
                sql,
                {
                    "snip": SEARCH_SNIPPET_CHARS,
                    "reg_ids": regulation_ids,
                    "emb": embedding,
                    "k": max(1, min(k, 20)),
                },
            )
            rows = cur.fetchall()
    return [
        {
            "regulation_article_id": r[0],
            "article_number": r[1],
            "article_title": r[2],
            "snippet": r[3],
            "regulation_title": r[4],
            "regulation_number": r[5],
            "regulation_id": r[6],
        }
        for r in rows
    ]


def _get_regulation_article(article_id: int) -> dict | None:
    sql = """
        SELECT ra.id, ra.number, ra.title, ra.content,
               r.id, r.regulation_number, r.title
          FROM regulation_articles ra
          JOIN regulations r ON r.id = ra.regulation_id
         WHERE ra.id = %s
    """
    with conn() as c, c.cursor() as cur:
        cur.execute(sql, (article_id,))
        r = cur.fetchone()
    if r is None:
        return None
    return {
        "regulation_article_id": r[0],
        "article_number": r[1],
        "article_title": r[2],
        "content": r[3],
        "regulation_id": r[4],
        "regulation_number": r[5],
        "regulation_title": r[6],
    }


def _get_regulation(regulation_id: int) -> dict | None:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT id, regulation_number, title, url FROM regulations WHERE id = %s",
            (regulation_id,),
        )
        reg = cur.fetchone()
        if reg is None:
            return None
        cur.execute(
            """
            SELECT id, number, title, content, ordinal
              FROM regulation_articles
             WHERE regulation_id = %s
             ORDER BY ordinal
            """,
            (regulation_id,),
        )
        rows = cur.fetchall()
    return {
        "regulation_id": reg[0],
        "regulation_number": reg[1],
        "regulation_title": reg[2],
        "url": reg[3],
        "articles": [
            {
                "regulation_article_id": r[0],
                "article_number": r[1],
                "article_title": r[2],
                "content": r[3],
                "ordinal": r[4],
            }
            for r in rows
        ],
    }


def _search_documents(
    query: str,
    k: int = 5,
    document_ids: list[int] | None = None,
    municipality_ids: list[int] | None = None,
) -> list[dict]:
    document_ids = _normalize_law_ids(document_ids)  # same int[] normalization
    municipality_ids = _normalize_law_ids(municipality_ids)
    embedding = _embed_query(query)
    sql = """
        SELECT dc.id           AS chunk_id,
               dc.ordinal,
               LEFT(dc.content, %(snip)s) AS snippet,
               d.id            AS document_id,
               d.name          AS document_name,
               d.source_type,
               m.id            AS municipality_id,
               m.name          AS municipality_name
          FROM document_chunks dc
          JOIN documents d ON d.id = dc.document_id
          LEFT JOIN municipalities m ON m.id = d.municipality_id
         WHERE (
                  (%(doc_ids)s::int[]  IS NULL AND %(muni_ids)s::int[] IS NULL)
                  OR d.id              = ANY(%(doc_ids)s::int[])
                  OR d.municipality_id = ANY(%(muni_ids)s::int[])
               )
         ORDER BY dc.embedding <=> %(emb)s::vector
         LIMIT %(k)s
    """
    with conn() as c:
        register_vector(c)
        with c.cursor() as cur:
            cur.execute(
                sql,
                {
                    "snip": SEARCH_SNIPPET_CHARS,
                    "doc_ids": document_ids,
                    "muni_ids": municipality_ids,
                    "emb": embedding,
                    "k": max(1, min(k, 20)),
                },
            )
            rows = cur.fetchall()
    return [
        {
            "chunk_id": r[0],
            "chunk_ordinal": r[1],
            "snippet": r[2],
            "document_id": r[3],
            "document_name": r[4],
            "source_type": r[5],
            "municipality_id": r[6],
            "municipality_name": r[7],
        }
        for r in rows
    ]


def _get_chunk_relative(chunk_id: int, offset: int = 0) -> dict:
    """Return the chunk at `chunk.ordinal + offset` within the same document, or an error dict."""
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT document_id, ordinal FROM document_chunks WHERE id = %s",
            (chunk_id,),
        )
        anchor = cur.fetchone()
        if anchor is None:
            return {"error": "anchor chunk_id not found", "chunk_id": chunk_id}
        doc_id, anchor_ord = anchor

        cur.execute(
            "SELECT MIN(ordinal), MAX(ordinal) FROM document_chunks WHERE document_id = %s",
            (doc_id,),
        )
        min_ord, max_ord = cur.fetchone()

        target_ord = anchor_ord + offset
        if target_ord < min_ord or target_ord > max_ord:
            return {
                "error": "no chunk at that offset (past document edge)",
                "anchor_chunk_id": chunk_id,
                "anchor_ordinal": anchor_ord,
                "requested_ordinal": target_ord,
                "min_ordinal": min_ord,
                "max_ordinal": max_ord,
            }

        cur.execute(
            """
            SELECT dc.id, dc.ordinal, dc.content,
                   d.id, d.name, d.source_type,
                   m.id, m.name
              FROM document_chunks dc
              JOIN documents d  ON d.id = dc.document_id
              LEFT JOIN municipalities m ON m.id = d.municipality_id
             WHERE dc.document_id = %s AND dc.ordinal = %s
            """,
            (doc_id, target_ord),
        )
        r = cur.fetchone()
    if r is None:
        return {
            "error": "neighbor chunk missing (document chunk ordinals are not contiguous)",
            "anchor_chunk_id": chunk_id,
            "requested_ordinal": target_ord,
        }
    return {
        "chunk_id": r[0],
        "ordinal": r[1],
        "content": r[2],
        "document_id": r[3],
        "document_name": r[4],
        "source_type": r[5],
        "municipality_id": r[6],
        "municipality_name": r[7],
        "is_first_chunk": r[1] == min_ord,
        "is_last_chunk": r[1] == max_ord,
        "anchor_chunk_id": chunk_id,
        "offset_applied": offset,
    }


def _get_document(document_id: int) -> dict | None:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d.name, d.source_type, d.char_length, d.created_at,
                   m.id, m.name
              FROM documents d
              LEFT JOIN municipalities m ON m.id = d.municipality_id
             WHERE d.id = %s
            """,
            (document_id,),
        )
        d = cur.fetchone()
        if d is None:
            return None
        cur.execute(
            "SELECT id, ordinal, content FROM document_chunks WHERE document_id = %s ORDER BY ordinal",
            (document_id,),
        )
        chunks = cur.fetchall()
    return {
        "document_id": d[0],
        "document_name": d[1],
        "source_type": d[2],
        "char_length": d[3],
        "created_at": d[4].isoformat() if d[4] else None,
        "municipality_id": d[5],
        "municipality_name": d[6],
        "chunks": [{"chunk_id": r[0], "ordinal": r[1], "content": r[2]} for r in chunks],
    }


def list_documents_with_municipality() -> list[dict]:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d.name, d.source_type, d.municipality_id, m.name
              FROM documents d
              LEFT JOIN municipalities m ON m.id = d.municipality_id
             ORDER BY COALESCE(m.name, ''), d.name
            """
        )
        return [
            {
                "id": r[0],
                "name": r[1],
                "source_type": r[2],
                "municipality_id": r[3],
                "municipality_name": r[4],
            }
            for r in cur.fetchall()
        ]


def list_municipalities_basic() -> list[dict]:
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT id, name FROM municipalities ORDER BY name")
        return [{"id": r[0], "name": r[1]} for r in cur.fetchall()]


def _make_tools(
    forced_law_ids: list[int] | None,
    forced_municipality_ids: list[int] | None,
    forced_document_ids: list[int] | None,
    forced_regulation_ids: list[int] | None,
):
    laws_locked = bool(forced_law_ids)
    docs_locked = bool(forced_municipality_ids) or bool(forced_document_ids)
    regs_locked = bool(forced_regulation_ids)

    def _locked_to_msg() -> str:
        if laws_locked:
            return "User pre-filtered to laws"
        if regs_locked:
            return "User pre-filtered to regulations"
        return "User pre-filtered to municipal/other documents"

    @tool
    def search_articles(
        query: str,
        k: int = 5,
        law_ids: list[int] = [],
    ) -> str:
        """Vector similarity search over Icelandic law articles.

        Args:
            query: Natural-language search query (Icelandic or English).
            k: Number of results to return (default 5, max 20).
            law_ids: Optional list of law ids (from the catalog in the system prompt).
                Restricts search to those laws. Use this to keep results from mixing
                across unrelated acts. Omit or pass an empty list to search everything.

        Returns:
            JSON list of {article_id, article_number, article_title, snippet, chapter_number, chapter_title, chapter_id, law_title, law_number, law_id, cross_references}. Use article_id with get_article to fetch the full text.
        """
        if docs_locked or regs_locked:
            return json.dumps(
                {"info": f"{_locked_to_msg()}; statute search is disabled this turn."}
            )
        effective = forced_law_ids if forced_law_ids else law_ids
        results = _search_articles(query, k=k, law_ids=effective)
        return json.dumps(results, ensure_ascii=False)

    @tool
    def get_article(article_id: int) -> str:
        """Fetch the full content of a single article along with its chapter and law metadata.

        Args:
            article_id: The integer article id (from search_articles results).

        Returns:
            JSON object with the full article content plus chapter and law breadcrumb.
        """
        result = _get_article(article_id)
        if result is None:
            return json.dumps({"error": "article not found"})
        return json.dumps(result, ensure_ascii=False)

    @tool
    def get_chapter(chapter_id: int) -> str:
        """Fetch all articles within a chapter, in order, with full content.

        Args:
            chapter_id: The integer chapter id (from search_articles results).

        Returns:
            JSON object with chapter metadata and a list of full articles.
        """
        result = _get_chapter(chapter_id)
        if result is None:
            return json.dumps({"error": "chapter not found"})
        return json.dumps(result, ensure_ascii=False)

    @tool
    def search_regulations(
        query: str,
        k: int = 5,
        regulation_ids: list[int] = [],
    ) -> str:
        """Vector similarity search over Icelandic regulation (reglugerð) articles.

        Args:
            query: Natural-language search query (Icelandic or English).
            k: Number of results to return (default 5, max 20).
            regulation_ids: Optional list of regulation ids (from the catalog).
                Restricts search to those regulations. Omit or pass an empty list
                to search every ingested regulation.

        Returns:
            JSON list of {regulation_article_id, article_number, article_title, snippet, regulation_title, regulation_number, regulation_id}. Use regulation_article_id with get_regulation_article to fetch the full text.
        """
        if laws_locked or docs_locked:
            return json.dumps(
                {"info": f"{_locked_to_msg()}; regulation search is disabled this turn."}
            )
        effective = forced_regulation_ids if forced_regulation_ids else regulation_ids
        results = _search_regulation_articles(query, k=k, regulation_ids=effective)
        return json.dumps(results, ensure_ascii=False)

    @tool
    def get_regulation_article(article_id: int) -> str:
        """Fetch the full content of a single regulation article with its regulation metadata.

        Args:
            article_id: The integer regulation_article_id (from search_regulations).

        Returns:
            JSON object with the full article content plus regulation breadcrumb.
        """
        result = _get_regulation_article(article_id)
        if result is None:
            return json.dumps({"error": "regulation article not found"})
        return json.dumps(result, ensure_ascii=False)

    @tool
    def get_regulation(regulation_id: int) -> str:
        """Fetch all articles within a regulation, in order, with full content.

        Args:
            regulation_id: The integer regulation id (from search_regulations results).

        Returns:
            JSON object with regulation metadata and a list of full articles.
        """
        result = _get_regulation(regulation_id)
        if result is None:
            return json.dumps({"error": "regulation not found"})
        return json.dumps(result, ensure_ascii=False)

    @tool
    def search_documents(
        query: str,
        k: int = 5,
        document_ids: list[int] = [],
        municipality_ids: list[int] = [],
    ) -> str:
        """Vector similarity search over uploaded municipal/other documents (PDF, TXT, HTML).

        Args:
            query: Natural-language search query (Icelandic or English).
            k: Number of results to return (default 5, max 20).
            document_ids: Optional list of document ids (from the catalog) to restrict to.
            municipality_ids: Optional list of municipality ids — narrows to all documents
                belonging to those municipalities. Combined with document_ids as OR (a chunk
                matches if it's in either set).

        Returns:
            JSON list of {chunk_id, chunk_ordinal, snippet, document_id, document_name, source_type, municipality_id, municipality_name}. Use document_id with get_document to fetch the full document.
        """
        if laws_locked or regs_locked:
            return json.dumps(
                {"info": f"{_locked_to_msg()}; municipal-document search is disabled this turn."}
            )
        eff_docs = forced_document_ids if forced_document_ids else document_ids
        eff_munis = forced_municipality_ids if forced_municipality_ids else municipality_ids
        results = _search_documents(query, k=k, document_ids=eff_docs, municipality_ids=eff_munis)
        return json.dumps(results, ensure_ascii=False)

    @tool
    def get_document(document_id: int) -> str:
        """Fetch the full content of an uploaded document, with all chunks in order.

        Args:
            document_id: The integer document id (from search_documents results).

        Returns:
            JSON object with document metadata and ordered chunks.
        """
        result = _get_document(document_id)
        if result is None:
            return json.dumps({"error": "document not found"})
        return json.dumps(result, ensure_ascii=False)

    @tool
    def get_chunk(chunk_id: int, offset: int = 0) -> str:
        """Fetch a document chunk relative to the given chunk within the same document.

        Use this to walk forward or backward when a chunk from `search_documents`
        looks relevant but is cut off at a chunk boundary.

        Args:
            chunk_id: The reference chunk id (from `search_documents` results).
            offset: Position relative to that chunk. 0 = the chunk itself,
                -1 = the previous chunk in the same document, +1 = the next.
                Larger magnitudes are allowed (e.g. +2) but typical use is ±1.

        Returns:
            JSON object with the requested chunk's content, document/municipality
            metadata, and `is_first_chunk` / `is_last_chunk` flags so you can tell
            when there is no further chunk to fetch. If the offset is past the
            document's edge, returns an error with the available ordinal range.
        """
        if laws_locked or regs_locked:
            return json.dumps(
                {"info": f"{_locked_to_msg()}; document chunk navigation is disabled this turn."}
            )
        return json.dumps(_get_chunk_relative(chunk_id, offset), ensure_ascii=False)

    return [
        search_articles,
        get_article,
        get_chapter,
        search_regulations,
        get_regulation_article,
        get_regulation,
        search_documents,
        get_document,
        get_chunk,
    ]


PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
        "display": "Anthropic Claude Sonnet 4.6",
    },
    "google": {
        "env": "GEMINI_API_KEY",
        "model": "gemini-3-flash-preview",
        "display": "Google Gemini Flash 3 Preview",
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        "model": "gpt-5.4-mini",
        "display": "OpenAI GPT-5.4 Mini",
    },
}


def _llm_for_provider(provider: str, *, temperature: float = 0.2):
    cfg = PROVIDERS[provider]
    key = os.environ[cfg["env"]]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=cfg["model"], api_key=key, temperature=temperature)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg["model"], google_api_key=key, temperature=temperature
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=cfg["model"], api_key=key, temperature=temperature)
    raise ValueError(f"unknown provider: {provider}")


_current_provider: str | None = None
_current_model: str | None = None


def parse_model_spec(spec: str) -> tuple[str, str] | None:
    """Parse a "provider/model" string. Returns (provider, model) or None."""
    if not spec or "/" not in spec:
        return None
    provider, model = spec.split("/", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        return None
    return provider, model


def _is_known_model(provider: str, model: str) -> bool:
    cfg = PROVIDERS.get(provider)
    return cfg is not None and cfg["model"] == model


def _provider_has_key(provider: str) -> bool:
    cfg = PROVIDERS.get(provider)
    return cfg is not None and bool(os.environ.get(cfg["env"]))


def init_current_model_from_env() -> None:
    """Set the active model at startup. Honor DEFAULT_MODEL=provider/model when valid;
    otherwise fall back to the first provider whose API key is configured."""
    global _current_provider, _current_model
    spec = (os.environ.get("DEFAULT_MODEL") or "").strip()
    parsed = parse_model_spec(spec) if spec else None
    if parsed and _is_known_model(*parsed) and _provider_has_key(parsed[0]):
        _current_provider, _current_model = parsed
        return
    for p in available_providers():
        _current_provider = p["provider"]
        _current_model = p["model"]
        return
    _current_provider = None
    _current_model = None


def get_current_model() -> dict | None:
    if _current_provider is None or _current_model is None:
        return None
    cfg = PROVIDERS.get(_current_provider)
    if cfg and cfg["model"] == _current_model:
        display = cfg["display"]
    else:
        display = f"{_current_provider} {_current_model}"
    return {"provider": _current_provider, "model": _current_model, "display": display}


def set_current_model(provider: str, model: str) -> dict:
    if not _is_known_model(provider, model):
        raise ValueError(f"unknown model: {provider}/{model}")
    if not _provider_has_key(provider):
        raise ValueError(f"provider {provider!r} has no API key configured")
    global _current_provider, _current_model
    _current_provider, _current_model = provider, model
    return get_current_model()  # type: ignore[return-value]


def list_known_models() -> list[dict]:
    """All registered models with availability flag (API key present)."""
    return [
        {
            "provider": p,
            "model": cfg["model"],
            "display": cfg["display"],
            "available": bool(os.environ.get(cfg["env"])),
        }
        for p, cfg in PROVIDERS.items()
    ]


def _get_llm():
    cur = get_current_model()
    if cur is None:
        raise RuntimeError("no LLM provider API key configured")
    return _llm_for_provider(cur["provider"])


def available_providers() -> list[dict]:
    """Providers whose API key is set in env. Each entry has provider/model/display."""
    out = []
    for provider, cfg in PROVIDERS.items():
        if os.environ.get(cfg["env"]):
            out.append(
                {"provider": provider, "model": cfg["model"], "display": cfg["display"]}
            )
    return out


def answer_single(provider: str, question: str) -> str:
    """One-shot answer from a single provider, no tools — used by /test page."""
    llm = _llm_for_provider(provider)
    msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    resp = llm.invoke(msgs)
    content = resp.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def record_model_vote(provider: str, model: str) -> None:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_votes (provider, model_name, votes)
            VALUES (%s, %s, 1)
            ON CONFLICT (provider, model_name)
            DO UPDATE SET votes = model_votes.votes + 1
            """,
            (provider, model),
        )
        c.commit()


def list_model_vote_leaderboard() -> list[dict]:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT provider, model_name, votes
              FROM model_votes
             ORDER BY votes DESC, provider, model_name
            """
        )
        rows = cur.fetchall()
    out = []
    for provider, model, votes in rows:
        cfg = PROVIDERS.get(provider)
        if cfg and cfg["model"] == model:
            display = cfg["display"]
        else:
            display = f"{provider} {model}"
        out.append(
            {"provider": provider, "model": model, "display": display, "votes": votes}
        )
    return out


def _extract_text_from_chunk(content) -> str:
    """LangChain stream chunks expose `content` as either a string (OpenAI/Gemini path)
    or a list of typed blocks (Anthropic path: [{"type":"text","text":"..."}, ...]).
    Pull plain text out of either shape; return "" for tool_use / non-text blocks."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif btype is None and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _sse(event: str, data) -> str:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


def _build_catalog_block(
    forced_law_ids: list[int] | None,
    forced_municipality_ids: list[int] | None,
    forced_document_ids: list[int] | None,
    forced_regulation_ids: list[int] | None,
) -> str:
    laws_locked = bool(forced_law_ids)
    docs_locked = bool(forced_municipality_ids) or bool(forced_document_ids)
    regs_locked = bool(forced_regulation_ids)

    sections: list[str] = []

    # --- Laws section ---
    if docs_locked or regs_locked:
        other = (
            "regulations" if regs_locked else "municipal/other documents"
        )
        sections.append(
            f"## Law catalog\n(Disabled: user pre-filtered to {other} this turn.)"
        )
    else:
        laws = list_laws()
        if not laws:
            sections.append("## Law catalog\n(No laws have been ingested yet.)")
        elif laws_locked:
            forced_set = set(forced_law_ids or [])
            selected = [l for l in laws if l["id"] in forced_set]
            if not selected:
                sections.append(
                    "## Law catalog\nThe user pre-filtered to law ids that no longer exist; treat as no filter."
                )
            else:
                lines = [f"- id={l['id']} | {l['law_number']} — {l['title']}" for l in selected]
                sections.append(
                    "## Law catalog (PRE-FILTERED by the user)\n"
                    "search_articles is automatically restricted to these laws. "
                    "Do not ask the user to filter further — they already did.\n"
                    + "\n".join(lines)
                )
        else:
            lines = [f"- id={l['id']} | {l['law_number']} — {l['title']}" for l in laws]
            sections.append(
                "## Law catalog (full)\n"
                "Pass relevant ids via `law_ids` on `search_articles` when the question clearly belongs to a subset, "
                "or ask the user to choose when ambiguous.\n"
                + "\n".join(lines)
            )

    # --- Regulations section ---
    if laws_locked or docs_locked:
        other = "laws" if laws_locked else "municipal/other documents"
        sections.append(
            f"## Regulation catalog\n(Disabled: user pre-filtered to {other} this turn.)"
        )
    else:
        regulations = list_regulations()
        if not regulations:
            sections.append("## Regulation catalog\n(No regulations have been ingested yet.)")
        elif regs_locked:
            forced_set = set(forced_regulation_ids or [])
            selected = [r for r in regulations if r["id"] in forced_set]
            if not selected:
                sections.append(
                    "## Regulation catalog\nThe user pre-filtered to regulation ids that no longer exist; treat as no filter."
                )
            else:
                lines = [
                    f"- id={r['id']} | {r['regulation_number']} — {r['title']}"
                    for r in selected
                ]
                sections.append(
                    "## Regulation catalog (PRE-FILTERED by the user)\n"
                    "search_regulations is automatically restricted to these regulations. "
                    "Do not ask the user to filter further — they already did.\n"
                    + "\n".join(lines)
                )
        else:
            lines = [
                f"- id={r['id']} | {r['regulation_number']} — {r['title']}"
                for r in regulations
            ]
            sections.append(
                "## Regulation catalog (full)\n"
                "Pass relevant ids via `regulation_ids` on `search_regulations` when the question clearly belongs to a subset, "
                "or ask the user to choose when ambiguous.\n"
                + "\n".join(lines)
            )

    # --- Documents section ---
    if laws_locked or regs_locked:
        other = "laws" if laws_locked else "regulations"
        sections.append(
            f"## Municipal/other documents catalog\n(Disabled: user pre-filtered to {other} this turn.)"
        )
    else:
        munis = list_municipalities_basic()
        docs = list_documents_with_municipality()
        if not docs and not munis:
            sections.append("## Municipal/other documents catalog\n(No documents uploaded yet.)")
        else:
            forced_muni = set(forced_municipality_ids or [])
            forced_doc = set(forced_document_ids or [])
            visible_docs = docs
            visible_munis = munis
            header = "## Municipal/other documents catalog (full)"
            footer_help = (
                "Pass `document_ids` and/or `municipality_ids` to `search_documents` "
                "to restrict to a subset, or ask the user to choose when ambiguous."
            )
            if docs_locked:
                visible_docs = [
                    d for d in docs
                    if d["id"] in forced_doc or (d["municipality_id"] and d["municipality_id"] in forced_muni)
                ]
                visible_muni_ids = {d["municipality_id"] for d in visible_docs if d["municipality_id"]}
                visible_muni_ids |= forced_muni
                visible_munis = [m for m in munis if m["id"] in visible_muni_ids]
                header = "## Municipal/other documents catalog (PRE-FILTERED by the user)"
                footer_help = (
                    "search_documents is automatically restricted to these. "
                    "Do not ask the user to filter further — they already did."
                )

            lines: list[str] = [header, footer_help]
            grouped: dict[int | None, list[dict]] = defaultdict(list)
            for d in visible_docs:
                grouped[d["municipality_id"]].append(d)
            # Render municipalities (with their docs), then a "(no municipality)" bucket
            muni_by_id = {m["id"]: m for m in visible_munis}
            seen_muni_ids: set[int] = set()
            for m in visible_munis:
                seen_muni_ids.add(m["id"])
                bucket = grouped.get(m["id"], [])
                lines.append(f"### {m['name']} (municipality_id={m['id']})")
                if not bucket:
                    lines.append("  (no documents yet)")
                else:
                    for d in bucket:
                        lines.append(
                            f"  - document_id={d['id']} | {d['name']} ({d['source_type']})"
                        )
            # Any docs whose municipality id isn't in muni_by_id (orphaned), and the (None) bucket
            stray = [
                d for d in visible_docs
                if d["municipality_id"] is None or d["municipality_id"] not in seen_muni_ids
            ]
            if stray:
                lines.append("### (no municipality)")
                for d in stray:
                    lines.append(
                        f"  - document_id={d['id']} | {d['name']} ({d['source_type']})"
                    )
            sections.append("\n".join(lines))

    return "\n\n".join(sections)


def chat_stream(
    message: str,
    law_ids: list[int] | None = None,
    municipality_ids: list[int] | None = None,
    document_ids: list[int] | None = None,
    regulation_ids: list[int] | None = None,
    provider: str | None = None,
) -> Iterator[str]:
    forced_laws = _normalize_law_ids(law_ids)
    forced_munis = _normalize_law_ids(municipality_ids)
    forced_docs = _normalize_law_ids(document_ids)
    forced_regs = _normalize_law_ids(regulation_ids)
    llm = _llm_for_provider(provider) if provider else _get_llm()
    tools = _make_tools(forced_laws, forced_munis, forced_docs, forced_regs)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    catalog = _build_catalog_block(forced_laws, forced_munis, forced_docs, forced_regs)
    system_text = SYSTEM_PROMPT + "\n\n" + catalog
    messages = [SystemMessage(content=system_text), HumanMessage(content=message)]

    try:
        for _ in range(MAX_STEPS):
            accumulated: AIMessageChunk | None = None
            for chunk in llm_with_tools.stream(messages):
                text_piece = _extract_text_from_chunk(chunk.content)
                if text_piece:
                    yield _sse("token", {"text": text_piece})
                accumulated = chunk if accumulated is None else accumulated + chunk

            if accumulated is None:
                break

            messages.append(accumulated)

            tool_calls = getattr(accumulated, "tool_calls", None) or []
            if not tool_calls:
                break

            for tc in tool_calls:
                name = tc.get("name")
                args = tc.get("args") or {}
                tc_id = tc.get("id") or ""
                yield _sse("tool", {"name": name, "args": args})
                fn = tools_by_name.get(name)
                if fn is None:
                    err = f"Unknown tool: {name}"
                    messages.append(ToolMessage(content=err, tool_call_id=tc_id))
                    yield _sse("tool_result", {"name": name, "ok": False, "error": err})
                    continue
                try:
                    result = fn.invoke(args)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))
                    yield _sse("tool_result", {"name": name, "ok": True})
                except Exception as e:
                    messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tc_id))
                    yield _sse("tool_result", {"name": name, "ok": False, "error": str(e)})
        yield _sse("done", {})
    except Exception as e:
        yield _sse("error", {"error": str(e)})
