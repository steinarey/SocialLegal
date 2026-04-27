"""Usage: python create_user.py <username> <password>

Creates a new user. If the username already exists, updates the password.
"""
import sys

from auth import hash_password
from db import conn, init_schema


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    username, password = sys.argv[1], sys.argv[2]
    init_schema()
    ph = hash_password(password)
    with conn() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, password_hash) VALUES (%s, %s)
                ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash
                """,
                (username, ph),
            )
        c.commit()
    print(f"User {username!r} ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
