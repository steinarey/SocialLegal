import secrets

import bcrypt
from fastapi import Cookie, HTTPException, status

from db import conn

SESSION_COOKIE = "session"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def authenticate(username: str, password: str) -> int | None:
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
    if row is None:
        return None
    user_id, ph = row
    return user_id if verify_password(password, ph) else None


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (token, user_id) VALUES (%s, %s)",
                (token, user_id),
            )
        c.commit()
    return token


def delete_session(token: str) -> None:
    with conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE token = %s", (token,))
        c.commit()


def lookup_session(token: str) -> dict | None:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT u.id, u.username
              FROM sessions s
              JOIN users u ON u.id = s.user_id
             WHERE s.token = %s
            """,
            (token,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {"id": row[0], "username": row[1]}


def require_user(session: str | None = Cookie(default=None)) -> dict:
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not authenticated")
    user = lookup_session(session)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid session")
    return user


def optional_user(session: str | None = Cookie(default=None)) -> dict | None:
    if not session:
        return None
    return lookup_session(session)
