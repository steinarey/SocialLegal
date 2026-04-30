from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg  # noqa: F401  (imported for adapter registration in callers)
from psycopg_pool import ConnectionPool

DATABASE_URL = os.environ["DATABASE_URL"]

_pool: ConnectionPool | None = None


def _configure_connection(conn) -> None:
    """Run on every new connection so client encoding can never silently fall back."""
    with conn.cursor() as cur:
        cur.execute("SET client_encoding TO 'UTF8'")
    conn.commit()


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": False, "client_encoding": "UTF8"},
            configure=_configure_connection,
        )
    return _pool


@contextmanager
def conn():
    with get_pool().connection() as c:
        yield c


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS laws (
    id          SERIAL PRIMARY KEY,
    law_number  TEXT NOT NULL,
    title       TEXT NOT NULL,
    url         TEXT UNIQUE NOT NULL,
    scraped_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chapters (
    id          SERIAL PRIMARY KEY,
    law_id      INTEGER REFERENCES laws(id) ON DELETE CASCADE,
    number      TEXT NOT NULL,
    title       TEXT,
    ordinal     INTEGER
);

CREATE TABLE IF NOT EXISTS articles (
    id          SERIAL PRIMARY KEY,
    law_id      INTEGER REFERENCES laws(id) ON DELETE CASCADE,
    chapter_id  INTEGER REFERENCES chapters(id) ON DELETE SET NULL,
    number      TEXT NOT NULL,
    title       TEXT,
    content     TEXT NOT NULL,
    ordinal     INTEGER,
    embedding   vector(3072)
);

CREATE INDEX IF NOT EXISTS articles_law_idx ON articles(law_id);
CREATE INDEX IF NOT EXISTS articles_chapter_idx ON articles(chapter_id);

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE laws ADD COLUMN IF NOT EXISTS partial BOOLEAN DEFAULT false;

CREATE TABLE IF NOT EXISTS cross_references (
    id                   SERIAL PRIMARY KEY,
    from_article_id      INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    to_law_url           TEXT NOT NULL,
    to_law_number        TEXT NOT NULL,
    to_article_anchor    TEXT,
    context_text         TEXT,
    resolved_article_id  INTEGER REFERENCES articles(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS cross_references_from_idx ON cross_references(from_article_id);
CREATE INDEX IF NOT EXISTS cross_references_unresolved_idx
    ON cross_references(to_law_url) WHERE resolved_article_id IS NULL;

CREATE TABLE IF NOT EXISTS municipalities (
    id   SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    municipality_id INTEGER REFERENCES municipalities(id) ON DELETE SET NULL,
    char_length     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_muni_idx ON documents(municipality_id);

CREATE TABLE IF NOT EXISTS document_chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(3072)
);

CREATE INDEX IF NOT EXISTS document_chunks_doc_idx ON document_chunks(document_id);

CREATE TABLE IF NOT EXISTS regulations (
    id                 SERIAL PRIMARY KEY,
    regulation_number  TEXT NOT NULL,
    title              TEXT NOT NULL,
    url                TEXT UNIQUE NOT NULL,
    scraped_at         TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS regulation_articles (
    id              SERIAL PRIMARY KEY,
    regulation_id   INTEGER REFERENCES regulations(id) ON DELETE CASCADE,
    number          TEXT NOT NULL,
    title           TEXT,
    content         TEXT NOT NULL,
    ordinal         INTEGER,
    embedding       vector(3072)
);

CREATE INDEX IF NOT EXISTS regulation_articles_reg_idx ON regulation_articles(regulation_id);
"""


def init_schema() -> None:
    with conn() as c:
        with c.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        c.commit()
