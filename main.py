import os
from contextlib import asynccontextmanager
from pathlib import Path

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
    ingest_urls,
    list_documents,
    list_municipalities,
    resolve_pending_references,
)
from rag import chat_stream, list_laws

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
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


class ChatRequest(BaseModel):
    message: str
    law_ids: list[int] | None = None
    municipality_ids: list[int] | None = None
    document_ids: list[int] | None = None


@app.post("/chat")
def chat(req: ChatRequest, user=Depends(require_user)):
    return StreamingResponse(
        chat_stream(
            req.message,
            law_ids=req.law_ids,
            municipality_ids=req.municipality_ids,
            document_ids=req.document_ids,
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
