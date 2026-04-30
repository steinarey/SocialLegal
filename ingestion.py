"""Scrape althingi.is law pages, parse into laws/chapters/articles, embed, and store."""
from __future__ import annotations

import io
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from openai import OpenAI
from pgvector.psycopg import register_vector

from db import conn

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS = 3072
EMBED_BATCH = 32
EMBED_MAX_CHARS = 28000  # rough headroom under text-embedding-3-large's 8192-token limit
HTTP_TIMEOUT = 30.0
USER_AGENT = "SocialLegal/1.0 (+legal RAG indexer)"

CHAPTER_RE = re.compile(r"^([IVXLCDM]+)\.\s*kafli\.?\s*$", re.IGNORECASE)
ARTICLE_HEADING_RE = re.compile(
    r"^(\d+)\s*\.?\s*gr\.?\s*([a-zA-ZáéíóúýþæöÁÉÍÓÚÝÞÆÖ]?)\.?$",
    re.IGNORECASE,
)
# Strip althingi's amendment-marker brackets ([ ]) before matching headings.
_BRACKET_STRIP_RE = re.compile(r"[\[\]]")


@dataclass
class ParsedChapter:
    number: str
    title: str
    ordinal: int


@dataclass
class ParsedCrossRef:
    to_law_url: str
    to_law_number: str
    to_article_anchor: str | None
    link_text: str
    position: int  # offset in raw (uncleaned) article body
    link_len: int
    context_text: str = ""


@dataclass
class ParsedArticle:
    chapter_ordinal: int | None
    number: str
    title: str
    body: str
    ordinal: int
    cross_refs: list[ParsedCrossRef] = field(default_factory=list)


@dataclass
class ParsedLaw:
    law_number: str
    title: str
    url: str
    chapters: list[ParsedChapter] = field(default_factory=list)
    articles: list[ParsedArticle] = field(default_factory=list)


def fetch(url: str) -> str:
    with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        r = client.get(url, follow_redirects=True)
        r.raise_for_status()
        return r.text


def _find_law_container(soup: BeautifulSoup) -> Tag | None:
    for div in soup.find_all("div", class_="boxbody"):
        h2 = div.find("h2")
        if h2 and h2.get_text(strip=True):
            return div
    return None


def _strip_amendment_brackets(raw: str) -> str:
    return _BRACKET_STRIP_RE.sub("", raw).strip()


def _normalize_article_number(raw: str) -> str:
    cleaned = _strip_amendment_brackets(raw)
    m = ARTICLE_HEADING_RE.match(cleaned)
    if not m:
        return cleaned.rstrip(".")
    num, suffix = m.group(1), m.group(2).lower()
    return f"{num}{suffix}" if suffix else num


def _normalize_chapter_number(raw: str) -> str:
    cleaned = _strip_amendment_brackets(raw)
    return re.sub(r"\s*kafli.*$", "", cleaned, flags=re.IGNORECASE).strip().rstrip(".").upper()


def _clean_body(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def _classify_law_link(href: str, base_url: str) -> dict | None:
    """If href points to an althingi /lagas/nuna/<YYYYNNN>.html law page, return its metadata.

    Returns None for non-law links (amendment pages under /altext/, external sites, etc.).
    """
    if not href:
        return None
    abs_url = urljoin(base_url, href)
    p = urlparse(abs_url)
    if p.netloc != "www.althingi.is":
        return None
    m = re.match(r"^/lagas/nuna/(\d{4})(\d+)\.html$", p.path)
    if not m:
        return None
    year, num_padded = m.group(1), m.group(2)
    canonical_url = f"https://www.althingi.is/lagas/nuna/{year}{num_padded}.html"
    anchor = p.fragment or None
    if anchor:
        anchor = re.sub(r"M\d+$", "", anchor)  # G7M3 -> G7 (drop paragraph suffix)
        anchor = anchor or None
    return {
        "url": canonical_url,
        "law_number": f"{int(num_padded)}/{year}",
        "anchor": anchor,
    }


def anchor_to_article_number(anchor: str | None) -> str | None:
    """Map a span anchor like 'G36C' or 'G7' to its article number ('36c', '7'). None on no match."""
    if not anchor:
        return None
    cleaned = re.sub(r"M\d+$", "", anchor)
    m = re.match(r"^G(\d+)([A-Za-z]?)$", cleaned)
    if not m:
        return None
    return m.group(1) + m.group(2).lower()


def parse_law(html: str, url: str) -> ParsedLaw:
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("title")
    title_text = title_tag.get_text(strip=True) if title_tag else ""
    m = re.match(r"(\d+/\d+)\s*:", title_text)
    law_number = m.group(1) if m else ""
    if not law_number:
        m2 = re.search(r"/lagas/[a-z]+/(\d{4})(\d+)\.html", url)
        if m2:
            law_number = f"{int(m2.group(2))}/{m2.group(1)}"

    container = _find_law_container(soup)
    if container is None:
        raise ValueError(f"Could not locate law body in {url}")

    h2 = container.find("h2")
    law_title = h2.get_text(" ", strip=True) if h2 else law_number

    parsed = ParsedLaw(law_number=law_number, title=law_title, url=url)

    chapter_ord_counter = 0
    article_ord_counter = 0
    current_chapter_ordinal: int | None = None
    expect_chapter_title = False
    current_article: ParsedArticle | None = None
    expect_article_title = False

    def commit_article(art: ParsedArticle | None) -> None:
        if art is None:
            return
        raw_body = art.body
        for cr in art.cross_refs:
            start = max(0, cr.position - 30)
            end = min(len(raw_body), cr.position + cr.link_len + 30)
            cr.context_text = re.sub(r"\s+", " ", raw_body[start:end]).strip()
        art.body = _clean_body(raw_body)
        if art.body or art.title:
            parsed.articles.append(art)

    for elem in list(container.children):
        if isinstance(elem, NavigableString):
            text = str(elem)
            if current_article is not None:
                if expect_article_title and text.strip():
                    expect_article_title = False
                current_article.body += text
            continue
        if not isinstance(elem, Tag):
            continue

        name = elem.name

        if name == "h2":
            continue

        if name == "b":
            txt = elem.get_text(" ", strip=True)
            cleaned = _strip_amendment_brackets(txt)
            stripped_dot = cleaned.rstrip(".").strip()

            if CHAPTER_RE.match(cleaned):
                commit_article(current_article)
                current_article = None
                chapter_ord_counter += 1
                ch = ParsedChapter(
                    number=_normalize_chapter_number(cleaned),
                    title="",
                    ordinal=chapter_ord_counter,
                )
                parsed.chapters.append(ch)
                current_chapter_ordinal = ch.ordinal
                expect_chapter_title = True
                continue

            if ARTICLE_HEADING_RE.match(stripped_dot):
                commit_article(current_article)
                article_ord_counter += 1
                current_article = ParsedArticle(
                    chapter_ordinal=current_chapter_ordinal,
                    number=_normalize_article_number(cleaned),
                    title="",
                    body="",
                    ordinal=article_ord_counter,
                )
                expect_chapter_title = False
                expect_article_title = True
                continue

            if expect_chapter_title and parsed.chapters:
                parsed.chapters[-1].title = stripped_dot
                expect_chapter_title = False
                continue

            if current_article is not None:
                current_article.body += txt
                if expect_article_title and txt.strip():
                    expect_article_title = False
            continue

        if name == "img":
            if current_article is not None:
                current_article.body += "\n"
            continue

        if name == "i":
            inner_small = elem.find("small")
            if inner_small is not None:
                continue
            text = elem.get_text(" ", strip=False)
            if current_article is not None:
                if expect_article_title and text.strip():
                    current_article.title = text.strip().rstrip(".")
                    expect_article_title = False
                else:
                    current_article.body += text
            continue

        if name == "small":
            if current_chapter_ordinal is None:
                continue
            if current_article is not None:
                current_article.body += elem.get_text(" ", strip=False)
            continue

        if name in ("br", "hr", "span", "sup"):
            continue

        if name == "a":
            link_text = elem.get_text(" ", strip=False)
            if current_article is not None:
                ref = _classify_law_link(elem.get("href", ""), url)
                if ref is not None:
                    current_article.cross_refs.append(
                        ParsedCrossRef(
                            to_law_url=ref["url"],
                            to_law_number=ref["law_number"],
                            to_article_anchor=ref["anchor"],
                            link_text=link_text.strip(),
                            position=len(current_article.body),
                            link_len=len(link_text),
                        )
                    )
                current_article.body += link_text
            continue

        if name == "p":
            if current_chapter_ordinal is None:
                continue
            if current_article is not None:
                current_article.body += elem.get_text(" ", strip=False) + "\n"
            continue

        if current_article is not None:
            current_article.body += elem.get_text(" ", strip=False)

    commit_article(current_article)
    return parsed


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    safe_texts = [t[:EMBED_MAX_CHARS] if t else " " for t in texts]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out: list[list[float]] = []
    for i in range(0, len(safe_texts), EMBED_BATCH):
        batch = safe_texts[i : i + EMBED_BATCH]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


def _article_embedding_text(law: ParsedLaw, art: ParsedArticle) -> str:
    chapter_title = ""
    chapter_number = ""
    if art.chapter_ordinal:
        for ch in law.chapters:
            if ch.ordinal == art.chapter_ordinal:
                chapter_title = ch.title
                chapter_number = ch.number
                break
    parts = [law.title]
    if chapter_number:
        suffix = f": {chapter_title}" if chapter_title else ""
        parts.append(f"{chapter_number}. kafli{suffix}")
    parts.append(f"{art.number}. gr.")
    if art.title:
        parts.append(art.title)
    return " | ".join(parts) + "\n" + art.body


def _insert_cross_refs(cur, article_id: int, refs: list[ParsedCrossRef]) -> None:
    for cr in refs:
        cur.execute(
            """
            INSERT INTO cross_references
                (from_article_id, to_law_url, to_law_number, to_article_anchor, context_text)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (article_id, cr.to_law_url, cr.to_law_number, cr.to_article_anchor, cr.context_text),
        )


def store_law(parsed: ParsedLaw) -> int:
    article_texts = [_article_embedding_text(parsed, a) for a in parsed.articles]
    embeddings = embed_texts(article_texts)

    with conn() as c:
        register_vector(c)
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO laws (law_number, title, url, partial) VALUES (%s, %s, %s, false) RETURNING id",
                (parsed.law_number, parsed.title, parsed.url),
            )
            law_id = cur.fetchone()[0]

            chapter_id_by_ordinal: dict[int, int] = {}
            for ch in parsed.chapters:
                cur.execute(
                    "INSERT INTO chapters (law_id, number, title, ordinal) VALUES (%s, %s, %s, %s) RETURNING id",
                    (law_id, ch.number, ch.title, ch.ordinal),
                )
                chapter_id_by_ordinal[ch.ordinal] = cur.fetchone()[0]

            for art, emb in zip(parsed.articles, embeddings):
                chapter_id = (
                    chapter_id_by_ordinal.get(art.chapter_ordinal)
                    if art.chapter_ordinal is not None
                    else None
                )
                cur.execute(
                    """
                    INSERT INTO articles (law_id, chapter_id, number, title, content, ordinal, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (law_id, chapter_id, art.number, art.title, art.body, art.ordinal, emb),
                )
                article_id = cur.fetchone()[0]
                _insert_cross_refs(cur, article_id, art.cross_refs)
        c.commit()
    return law_id


def url_already_ingested(url: str) -> bool:
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM laws WHERE url = %s", (url,))
        return cur.fetchone() is not None


def ingest_url(url: str) -> dict:
    if url_already_ingested(url):
        return {"url": url, "status": "skipped"}
    html = fetch(url)
    parsed = parse_law(html, url)
    if not parsed.articles:
        return {"url": url, "status": "error", "error": "no articles parsed"}
    law_id = store_law(parsed)
    return {
        "url": url,
        "status": "ingested",
        "law_id": law_id,
        "law_number": parsed.law_number,
        "title": parsed.title,
        "chapters": len(parsed.chapters),
        "articles": len(parsed.articles),
    }


def ingest_urls(urls: list[str]) -> list[dict]:
    results: list[dict] = []
    for raw in urls:
        url = raw.strip()
        if not url:
            continue
        try:
            results.append(ingest_url(url))
        except Exception as e:
            results.append({"url": url, "status": "error", "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# Cross-reference resolution
# ---------------------------------------------------------------------------


def _ensure_chapters(
    cur, law_id: int, parsed: ParsedLaw, articles: list[ParsedArticle]
) -> dict[int, int]:
    """Find or create chapter rows for the chapters referenced by the given articles."""
    needed_ordinals = {a.chapter_ordinal for a in articles if a.chapter_ordinal is not None}
    out: dict[int, int] = {}
    for ord_ in needed_ordinals:
        chapter = next((ch for ch in parsed.chapters if ch.ordinal == ord_), None)
        if chapter is None:
            continue
        cur.execute(
            "SELECT id FROM chapters WHERE law_id = %s AND ordinal = %s",
            (law_id, ord_),
        )
        existing = cur.fetchone()
        if existing:
            out[ord_] = existing[0]
        else:
            cur.execute(
                "INSERT INTO chapters (law_id, number, title, ordinal) VALUES (%s, %s, %s, %s) RETURNING id",
                (law_id, chapter.number, chapter.title, ord_),
            )
            out[ord_] = cur.fetchone()[0]
    return out


def _resolve_against_existing_law(law_id: int, refs: list[tuple]) -> list[dict]:
    """Target law is already fully ingested; look up articles by number and link the cross-ref."""
    output: list[dict] = []
    with conn() as c:
        with c.cursor() as cur:
            for cr_id, from_id, _to_url, to_law_num, anchor in refs:
                article_num = anchor_to_article_number(anchor)
                resolved_id: int | None = None
                if article_num:
                    cur.execute(
                        "SELECT id FROM articles WHERE law_id = %s AND number = %s",
                        (law_id, article_num),
                    )
                    row = cur.fetchone()
                    if row:
                        resolved_id = row[0]
                if resolved_id:
                    cur.execute(
                        "UPDATE cross_references SET resolved_article_id = %s WHERE id = %s",
                        (resolved_id, cr_id),
                    )
                    output.append(
                        {
                            "from_article_id": from_id,
                            "to_law": to_law_num,
                            "to_article": anchor,
                            "action": "already_existed",
                        }
                    )
                else:
                    output.append(
                        {
                            "from_article_id": from_id,
                            "to_law": to_law_num,
                            "to_article": anchor,
                            "action": "not_found_in_target",
                        }
                    )
        c.commit()
    return output


def _resolve_law_refs(to_law_url: str, refs: list[tuple]) -> list[dict]:
    """Resolve all anchored refs pointing at one target law: fetch+partial-ingest as needed."""
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT id, partial FROM laws WHERE url = %s", (to_law_url,))
        existing = cur.fetchone()

    if existing and not existing[1]:
        return _resolve_against_existing_law(existing[0], refs)

    html = fetch(to_law_url)
    parsed = parse_law(html, to_law_url)

    needed_numbers: set[str] = set()
    for r in refs:
        n = anchor_to_article_number(r[4])
        if n:
            needed_numbers.add(n)
    target_articles = [a for a in parsed.articles if a.number in needed_numbers]

    output: list[dict] = []
    with conn() as c:
        register_vector(c)
        with c.cursor() as cur:
            if existing:
                law_id = existing[0]
            else:
                cur.execute(
                    "INSERT INTO laws (law_number, title, url, partial) VALUES (%s, %s, %s, true) RETURNING id",
                    (parsed.law_number, parsed.title, parsed.url),
                )
                law_id = cur.fetchone()[0]

            article_id_by_number: dict[str, int] = {}
            articles_to_embed: list[ParsedArticle] = []
            for art in target_articles:
                cur.execute(
                    "SELECT id FROM articles WHERE law_id = %s AND number = %s",
                    (law_id, art.number),
                )
                row = cur.fetchone()
                if row:
                    article_id_by_number[art.number] = row[0]
                else:
                    articles_to_embed.append(art)

            newly_inserted_numbers: set[str] = set()
            if articles_to_embed:
                texts = [_article_embedding_text(parsed, a) for a in articles_to_embed]
                embs = embed_texts(texts)
                chapter_id_by_ordinal = _ensure_chapters(cur, law_id, parsed, articles_to_embed)
                for art, emb in zip(articles_to_embed, embs):
                    chapter_id = (
                        chapter_id_by_ordinal.get(art.chapter_ordinal)
                        if art.chapter_ordinal is not None
                        else None
                    )
                    cur.execute(
                        """
                        INSERT INTO articles (law_id, chapter_id, number, title, content, ordinal, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (law_id, chapter_id, art.number, art.title, art.body, art.ordinal, emb),
                    )
                    art_id = cur.fetchone()[0]
                    article_id_by_number[art.number] = art_id
                    newly_inserted_numbers.add(art.number)
                    # One-level recursion guard: persist new refs unresolved; operator re-runs as needed.
                    _insert_cross_refs(cur, art_id, art.cross_refs)

            for cr_id, from_id, _to_url, to_law_num, anchor in refs:
                article_num = anchor_to_article_number(anchor)
                resolved_id = article_id_by_number.get(article_num) if article_num else None
                if resolved_id is not None:
                    cur.execute(
                        "UPDATE cross_references SET resolved_article_id = %s WHERE id = %s",
                        (resolved_id, cr_id),
                    )
                    action = (
                        "ingested" if article_num in newly_inserted_numbers else "already_existed"
                    )
                    output.append(
                        {
                            "from_article_id": from_id,
                            "to_law": to_law_num,
                            "to_article": anchor,
                            "action": action,
                        }
                    )
                else:
                    output.append(
                        {
                            "from_article_id": from_id,
                            "to_law": to_law_num,
                            "to_article": anchor,
                            "action": "not_found_in_target",
                        }
                    )
        c.commit()
    return output


# ---------------------------------------------------------------------------
# Regulations (island.is /reglugerdir/nr/<num>-<year>)
# ---------------------------------------------------------------------------

REGULATION_URL_RE = re.compile(
    r"^https?://(?:www\.)?island\.is/reglugerdir/nr/(\d+)-(\d{4})/?$",
    re.IGNORECASE,
)
REGULATION_ARTICLE_HEADING_RE = re.compile(
    r"^(\d+)\s*\.?\s*gr\.\s*(.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ParsedRegulationArticle:
    number: str
    title: str
    body: str
    ordinal: int


@dataclass
class ParsedRegulation:
    regulation_number: str
    title: str
    url: str
    articles: list[ParsedRegulationArticle] = field(default_factory=list)


def _regulation_number_from_url(url: str) -> str | None:
    m = REGULATION_URL_RE.match(url.strip())
    if not m:
        return None
    return f"{int(m.group(1))}/{m.group(2)}"


def _parse_regulation_article_heading(raw: str) -> tuple[str, str]:
    """Heading text like "1. gr." or "1. gr. Gildissvið, markmið og framkvæmd." → (number, title)."""
    cleaned = _strip_amendment_brackets(raw)
    m = REGULATION_ARTICLE_HEADING_RE.match(cleaned)
    if not m:
        return cleaned.rstrip(".").strip(), ""
    number = m.group(1)
    title = m.group(2).strip().rstrip(".").strip()
    return number, title


def parse_regulation(html: str, url: str) -> ParsedRegulation:
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    title = title.rstrip(".").strip() or url

    regulation_number = _regulation_number_from_url(url) or ""

    parsed = ParsedRegulation(regulation_number=regulation_number, title=title, url=url)

    first_heading = soup.find("h3", class_="article__title")
    if first_heading is None:
        return parsed
    container = first_heading.parent

    current: ParsedRegulationArticle | None = None
    ordinal_counter = 0

    def commit(art: ParsedRegulationArticle | None) -> None:
        if art is None:
            return
        art.body = _clean_body(art.body)
        if art.body or art.title:
            parsed.articles.append(art)

    for child in list(container.children):
        if isinstance(child, NavigableString):
            if current is not None:
                current.body += str(child)
            continue
        if not isinstance(child, Tag):
            continue
        classes = child.get("class") or []
        if child.name == "h3" and "article__title" in classes:
            commit(current)
            ordinal_counter += 1
            number, art_title = _parse_regulation_article_heading(
                child.get_text(" ", strip=True)
            )
            current = ParsedRegulationArticle(
                number=number,
                title=art_title,
                body="",
                ordinal=ordinal_counter,
            )
            continue
        if current is None:
            continue
        text = child.get_text(" ", strip=False)
        if child.name in ("p", "ol", "ul", "div", "blockquote", "table"):
            current.body += text + "\n"
        else:
            current.body += text

    commit(current)
    return parsed


def _regulation_article_embedding_text(
    reg: ParsedRegulation, art: ParsedRegulationArticle
) -> str:
    parts = [reg.title, f"{art.number}. gr."]
    if art.title:
        parts.append(art.title)
    return " | ".join(parts) + "\n" + art.body


def regulation_url_already_ingested(url: str) -> bool:
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM regulations WHERE url = %s", (url,))
        return cur.fetchone() is not None


def store_regulation(parsed: ParsedRegulation) -> int:
    article_texts = [
        _regulation_article_embedding_text(parsed, a) for a in parsed.articles
    ]
    embeddings = embed_texts(article_texts)

    with conn() as c:
        register_vector(c)
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO regulations (regulation_number, title, url)
                VALUES (%s, %s, %s) RETURNING id
                """,
                (parsed.regulation_number, parsed.title, parsed.url),
            )
            regulation_id = cur.fetchone()[0]

            for art, emb in zip(parsed.articles, embeddings):
                cur.execute(
                    """
                    INSERT INTO regulation_articles
                        (regulation_id, number, title, content, ordinal, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (regulation_id, art.number, art.title, art.body, art.ordinal, emb),
                )
        c.commit()
    return regulation_id


def ingest_regulation_url(url: str) -> dict:
    if regulation_url_already_ingested(url):
        return {"url": url, "status": "skipped"}
    if not _regulation_number_from_url(url):
        return {
            "url": url,
            "status": "error",
            "error": "URL does not look like https://island.is/reglugerdir/nr/<num>-<year>",
        }
    html = fetch(url)
    parsed = parse_regulation(html, url)
    if not parsed.articles:
        return {"url": url, "status": "error", "error": "no articles parsed"}
    regulation_id = store_regulation(parsed)
    return {
        "url": url,
        "status": "ingested",
        "regulation_id": regulation_id,
        "regulation_number": parsed.regulation_number,
        "title": parsed.title,
        "articles": len(parsed.articles),
    }


def ingest_regulation_urls(urls: list[str]) -> list[dict]:
    results: list[dict] = []
    for raw in urls:
        url = raw.strip()
        if not url:
            continue
        try:
            results.append(ingest_regulation_url(url))
        except Exception as e:
            results.append({"url": url, "status": "error", "error": str(e)})
    return results


def list_regulations() -> list[dict]:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT id, regulation_number, title FROM regulations ORDER BY regulation_number"
        )
        return [
            {"id": r[0], "regulation_number": r[1], "title": r[2]}
            for r in cur.fetchall()
        ]


# ---------------------------------------------------------------------------
# Generic document ingestion (PDF / TXT / HTML)
# ---------------------------------------------------------------------------

DOC_CHUNK_SIZE = 1500
DOC_CHUNK_OVERLAP = 200
ALLOWED_DOC_TYPES = {"pdf", "txt", "html"}


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader  # local import keeps cold-start cheap if no PDFs ingested

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(p.strip() for p in pages if p and p.strip())


def _extract_html(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    soup = BeautifulSoup(text, "lxml")
    for bad in soup(["script", "style", "noscript"]):
        bad.decompose()
    return soup.get_text("\n", strip=True)


def _extract_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _extract_document_text(source_type: str, data: bytes) -> str:
    st = source_type.lower()
    if st == "pdf":
        return _extract_pdf(data)
    if st == "html":
        return _extract_html(data)
    if st == "txt":
        return _extract_txt(data)
    raise ValueError(f"Unsupported source_type: {source_type!r}")


def chunk_text(text: str, size: int = DOC_CHUNK_SIZE, overlap: int = DOC_CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window_floor = start + int(size * 0.5)
            br = text.rfind("\n\n", window_floor, end)
            if br == -1:
                br = text.rfind("\n", window_floor, end)
            if br == -1:
                br = text.rfind(". ", window_floor, end)
                if br != -1:
                    br += 1
            if br != -1 and br > start:
                end = br
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def list_municipalities() -> list[dict]:
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT id, name FROM municipalities ORDER BY name")
        return [{"id": r[0], "name": r[1]} for r in cur.fetchall()]


def create_municipality(name: str) -> dict:
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("Municipality name cannot be empty")
    with conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO municipalities (name) VALUES (%s)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id, name
                """,
                (cleaned,),
            )
            row = cur.fetchone()
        c.commit()
    return {"id": row[0], "name": row[1]}


def list_documents() -> list[dict]:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d.name, d.source_type, d.municipality_id, m.name, d.char_length
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
                "char_length": r[5],
            }
            for r in cur.fetchall()
        ]


def ingest_document(
    *,
    name: str,
    source_type: str,
    data: bytes,
    municipality_id: int | None = None,
) -> dict:
    """Extract text, chunk, embed, and store a document. Returns metadata + chunk count."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Document name cannot be empty")
    st = source_type.lower().strip()
    if st not in ALLOWED_DOC_TYPES:
        raise ValueError(f"source_type must be one of {sorted(ALLOWED_DOC_TYPES)}")

    text = _extract_document_text(st, data)
    if not text.strip():
        raise ValueError("No text could be extracted from the document")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Document produced no chunks after cleaning")

    embeddings = embed_texts(chunks)

    with conn() as c:
        register_vector(c)
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (name, source_type, municipality_id, char_length)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (name, st, municipality_id, len(text)),
            )
            doc_id = cur.fetchone()[0]
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    """
                    INSERT INTO document_chunks (document_id, ordinal, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (doc_id, i, chunk, emb),
                )
        c.commit()

    return {
        "document_id": doc_id,
        "name": name,
        "source_type": st,
        "municipality_id": municipality_id,
        "chunks": len(chunks),
        "char_length": len(text),
    }


def resolve_pending_references() -> dict:
    """Resolve all cross_references where resolved_article_id IS NULL.

    Whole-law refs (no anchor) are reported as unresolved, not ingested.
    One level of resolution per call: refs discovered in newly-ingested articles
    are persisted unresolved and surface on the next invocation.
    """
    resolved: list[dict] = []
    unresolved: list[dict] = []
    errors: list[dict] = []

    with conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT id, from_article_id, to_law_url, to_law_number, to_article_anchor
              FROM cross_references
             WHERE resolved_article_id IS NULL
            """
        )
        all_unresolved = cur.fetchall()

    by_url: dict[str, list[tuple]] = defaultdict(list)
    seen_no_anchor: set[str] = set()
    for r in all_unresolved:
        _cr_id, _from_id, to_url, to_law_num, anchor = r
        if not anchor:
            if to_url in seen_no_anchor:
                continue
            seen_no_anchor.add(to_url)
            unresolved.append(
                {
                    "to_law": to_law_num,
                    "to_law_url": to_url,
                    "reason": "no_anchor — manual ingestion required if needed",
                }
            )
            continue
        by_url[to_url].append(r)

    for to_url, refs in by_url.items():
        try:
            resolved.extend(_resolve_law_refs(to_url, refs))
        except Exception as e:
            errors.append({"to_law_url": to_url, "error": str(e)})

    return {"resolved": resolved, "unresolved": unresolved, "errors": errors}
