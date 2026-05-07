import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from auth import (
    SESSION_COOKIE,
    authenticate,
    create_session,
    delete_session,
    lookup_session,
    optional_user,
    require_user,
)
from db import init_schema
from ingestion import (
    ALLOWED_DOC_TYPES,
    create_municipality,
    ingest_document,
    ingest_regulation_urls,
    ingest_urls,
    list_documents,
    list_municipalities,
    resolve_pending_references,
)
from rag import (
    available_providers,
    chat_stream,
    get_current_model,
    init_current_model_from_env,
    list_known_models,
    list_laws,
    list_model_vote_leaderboard,
    list_regulations,
    record_model_vote,
    set_current_model,
    usage_summary,
)

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    init_current_model_from_env()
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root(user=Depends(optional_user)):
    if user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return FileResponse(INDEX_FILE)


@app.get("/login")
def login_page():
    return FileResponse(INDEX_FILE)


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    user_id = authenticate(username, password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = create_session(user_id)
    response = JSONResponse({"ok": True, "username": username})
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * 60 * 24 * 30,
        path="/",
    )
    return response


@app.get("/logout")
def logout(session: str | None = Cookie(default=None)):
    if session:
        delete_session(session)
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get("/me")
def me(user=Depends(require_user)):
    return {"username": user["username"]}


@app.get("/laws")
def laws(user=Depends(require_user)):
    return list_laws()


@app.get("/regulations")
def regulations(user=Depends(require_user)):
    return list_regulations()


class ChatRequest(BaseModel):
    message: str
    law_ids: list[int] | None = None
    municipality_ids: list[int] | None = None
    document_ids: list[int] | None = None
    regulation_ids: list[int] | None = None


@app.post("/chat")
def chat(req: ChatRequest, user=Depends(require_user)):
    return StreamingResponse(
        chat_stream(
            req.message,
            law_ids=req.law_ids,
            municipality_ids=req.municipality_ids,
            document_ids=req.document_ids,
            regulation_ids=req.regulation_ids,
            user_id=user["id"],
            username=user["username"],
            kind="chat",
        ),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/municipalities")
def municipalities_list(user=Depends(require_user)):
    return list_municipalities()


@app.post("/municipalities")
def municipalities_create(name: str = Form(...), user=Depends(require_user)):
    try:
        return create_municipality(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/documents")
def documents_list(user=Depends(require_user)):
    return list_documents()


_DOC_TYPE_BY_CT = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/html": "html",
    "application/xhtml+xml": "html",
}
_DOC_TYPE_BY_EXT = {".pdf": "pdf", ".txt": "txt", ".html": "html", ".htm": "html"}


def _detect_source_type(upload: UploadFile) -> str | None:
    if upload.content_type and upload.content_type in _DOC_TYPE_BY_CT:
        return _DOC_TYPE_BY_CT[upload.content_type]
    name = (upload.filename or "").lower()
    for ext, st in _DOC_TYPE_BY_EXT.items():
        if name.endswith(ext):
            return st
    return None


def _authorize_ingest(session: str | None, secret: str | None) -> None:
    """Allow either a valid session cookie or a matching INGEST_SECRET form field."""
    if session and lookup_session(session) is not None:
        return
    expected = os.environ.get("INGEST_SECRET")
    if expected and secret and secret == expected:
        return
    raise HTTPException(status_code=403, detail="forbidden")


@app.get("/admin/ingest")
def admin_ingest_page(user=Depends(require_user)):
    return FileResponse(STATIC_DIR / "ingest.html")


@app.post("/ingest")
def ingest(
    urls: str = Form(...),
    secret: str = Form(default=""),
    session: str | None = Cookie(default=None),
):
    _authorize_ingest(session, secret)
    url_list = [line.strip() for line in urls.splitlines() if line.strip()]
    if not url_list:
        raise HTTPException(status_code=400, detail="no urls provided")
    results = ingest_urls(url_list)
    summary = {
        "total": len(results),
        "ingested": sum(1 for r in results if r.get("status") == "ingested"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "errors": sum(1 for r in results if r.get("status") == "error"),
    }
    return {"summary": summary, "results": results}


@app.post("/ingest/regulation")
def ingest_regulation(
    urls: str = Form(...),
    secret: str = Form(default=""),
    session: str | None = Cookie(default=None),
):
    _authorize_ingest(session, secret)
    url_list = [line.strip() for line in urls.splitlines() if line.strip()]
    if not url_list:
        raise HTTPException(status_code=400, detail="no urls provided")
    results = ingest_regulation_urls(url_list)
    summary = {
        "total": len(results),
        "ingested": sum(1 for r in results if r.get("status") == "ingested"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "errors": sum(1 for r in results if r.get("status") == "error"),
    }
    return {"summary": summary, "results": results}


@app.post("/ingest/resolve-references")
def ingest_resolve(
    secret: str = Form(default=""),
    session: str | None = Cookie(default=None),
):
    _authorize_ingest(session, secret)
    return resolve_pending_references()


@app.post("/ingest/document")
async def ingest_document_endpoint(
    file: UploadFile = File(...),
    name: str = Form(...),
    municipality_id: str = Form(default=""),
    source_type: str = Form(default=""),
    secret: str = Form(default=""),
    session: str | None = Cookie(default=None),
):
    _authorize_ingest(session, secret)

    st = (source_type or "").strip().lower() or _detect_source_type(file)
    if not st:
        raise HTTPException(
            status_code=400,
            detail=f"Could not detect source_type. Pass one of: {sorted(ALLOWED_DOC_TYPES)}",
        )
    if st not in ALLOWED_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source_type {st!r}. Allowed: {sorted(ALLOWED_DOC_TYPES)}",
        )

    muni_id: int | None = None
    if municipality_id and municipality_id.strip():
        try:
            muni_id = int(municipality_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="municipality_id must be an integer")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file upload")

    try:
        return ingest_document(name=name, source_type=st, data=data, municipality_id=muni_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


_TEST_SESSIONS: dict[str, dict] = {}
_TEST_SESSIONS_LOCK = Lock()
_TEST_SESSION_TTL_SECONDS = 60 * 30


def _prune_test_sessions() -> None:
    cutoff = time.time() - _TEST_SESSION_TTL_SECONDS
    stale = [sid for sid, s in _TEST_SESSIONS.items() if s["created_at"] < cutoff]
    for sid in stale:
        _TEST_SESSIONS.pop(sid, None)


class TestAskRequest(BaseModel):
    message: str
    law_ids: list[int] | None = None
    municipality_ids: list[int] | None = None
    document_ids: list[int] | None = None
    regulation_ids: list[int] | None = None


class TestVoteRequest(BaseModel):
    session_id: str
    key: str


@app.get("/test")
def test_page(user=Depends(require_user)):
    return FileResponse(STATIC_DIR / "test.html")


@app.post("/test/ask")
def test_ask(req: TestAskRequest, user=Depends(require_user)):
    """Set up a single comparison turn. Returns {session_id, keys}.
    Frontend then opens /test/stream/{session_id}/{key} per column."""
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="empty question")
    providers = available_providers()
    if not providers:
        raise HTTPException(status_code=503, detail="no LLM providers configured")

    items = list(providers)
    random.shuffle(items)

    sid = str(uuid.uuid4())
    mapping: dict[str, dict] = {}
    keys: list[str] = []
    for i, p in enumerate(items):
        key = chr(ord("A") + i)
        keys.append(key)
        mapping[key] = {
            "provider": p["provider"],
            "model": p["model"],
            "display": p["display"],
        }

    with _TEST_SESSIONS_LOCK:
        _prune_test_sessions()
        _TEST_SESSIONS[sid] = {
            "mapping": mapping,
            "question": msg,
            "filters": {
                "law_ids": req.law_ids,
                "municipality_ids": req.municipality_ids,
                "document_ids": req.document_ids,
                "regulation_ids": req.regulation_ids,
            },
            "voted": False,
            "created_at": time.time(),
        }

    return {"session_id": sid, "keys": keys}


@app.get("/test/stream/{session_id}/{key}")
def test_stream(session_id: str, key: str, user=Depends(require_user)):
    with _TEST_SESSIONS_LOCK:
        sess = _TEST_SESSIONS.get(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="session expired or not found")
        info = sess["mapping"].get(key)
        if info is None:
            raise HTTPException(status_code=400, detail="invalid key")
        message = sess["question"]
        filters = dict(sess["filters"])

    return StreamingResponse(
        chat_stream(
            message,
            law_ids=filters["law_ids"],
            municipality_ids=filters["municipality_ids"],
            document_ids=filters["document_ids"],
            regulation_ids=filters["regulation_ids"],
            provider=info["provider"],
            user_id=user["id"],
            username=user["username"],
            kind="test",
        ),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/test/vote")
def test_vote(req: TestVoteRequest, user=Depends(require_user)):
    with _TEST_SESSIONS_LOCK:
        sess = _TEST_SESSIONS.get(req.session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="session expired or not found")
        if sess.get("voted"):
            raise HTTPException(status_code=400, detail="already voted")
        info = sess["mapping"].get(req.key)
        if info is None:
            raise HTTPException(status_code=400, detail="invalid answer key")
        full_mapping = dict(sess["mapping"])
        sess["voted"] = True

    record_model_vote(info["provider"], info["model"])
    return {"voted": info, "mapping": full_mapping}


@app.get("/test/results")
def test_results_page(user=Depends(require_user)):
    return FileResponse(STATIC_DIR / "test_results.html")


@app.get("/test/leaderboard")
def test_leaderboard(user=Depends(require_user)):
    return list_model_vote_leaderboard()


class SetModelRequest(BaseModel):
    provider: str
    model: str


@app.get("/admin")
def admin_page(user=Depends(require_user)):
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/admin/models")
def admin_list_models(user=Depends(require_user)):
    return {
        "current": get_current_model(),
        "models": list_known_models(),
    }


@app.post("/admin/models")
def admin_set_model(req: SetModelRequest, user=Depends(require_user)):
    try:
        return {"current": set_current_model(req.provider, req.model)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/usage")
def usage_page(user=Depends(require_user)):
    return FileResponse(STATIC_DIR / "usage.html")


@app.get("/usage/data")
def usage_data(
    user=Depends(require_user),
    from_date: str = "",
    to_date: str = "",
):
    try:
        return usage_summary(from_date=from_date or None, to_date=to_date or None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
