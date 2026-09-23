"""Application-managed SQLite databases; never infer ownership from old files."""

from contextlib import contextmanager
from datetime import UTC, datetime
import re
import sqlite3
from uuid import uuid4

from app.core.exceptions.custom import AIStudioException
from app.modules.sql.constants import DEFAULT_DATABASE_NAME, SQL_STORAGE_FOLDER


def fail(message: str, status: int = 400):
    raise AIStudioException(message, status_code=status, error_code="SQL_DATABASE_ERROR")


def parse_database_command(query: str):
    if not re.match(r"^\s*(CREATE|DROP)\s+DATABASE\b", query, re.I):
        return None
    match = re.fullmatch(
        r'\s*(CREATE|DROP)\s+DATABASE\s+([A-Za-z_][A-Za-z0-9_]{0,62}|"[A-Za-z_][A-Za-z0-9_]{0,62}"|`[A-Za-z_][A-Za-z0-9_]{0,62}`)\s*;?\s*',
        query, re.I,
    )
    if not match:
        fail("Use CREATE DATABASE name; or DROP DATABASE name; with a simple identifier (1–63 letters, digits or underscores, starting with a letter or underscore).")
    return match[1].upper(), match[2].strip('"`').lower()


@contextmanager
def registry():
    # A separate, inaccessible catalog keeps user SQL from changing ownership.
    connection = sqlite3.connect(SQL_STORAGE_FOLDER / "_ownership.sqlite3", timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""CREATE TABLE IF NOT EXISTS databases (
            name TEXT PRIMARY KEY COLLATE NOCASE,
            owner_user_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            organization_id TEXT,
            file_id TEXT NOT NULL UNIQUE
        )""")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def owned_database(user):
    with registry() as catalog:
        row = catalog.execute("SELECT * FROM databases WHERE owner_user_id = ?", (str(user.id),)).fetchone()
        return dict(row) if row else None


def owned_path(record):
    return SQL_STORAGE_FOLDER / "managed" / (record["file_id"] + ".db")


def database_state(user):
    record = owned_database(user)
    return {"active_database": record["name"] if record else DEFAULT_DATABASE_NAME,
            "databases": [DEFAULT_DATABASE_NAME] + ([record["name"]] if record else [])}


def execute_database_command(user, operation: str, name: str):
    if not user.id:
        fail("Authentication is required.", 401)
    protected = {"main", "temp", "master", "model", "msdb", "tempdb", "postgres",
                 "template0", "template1", "mysql", "information_schema", "performance_schema",
                 "sys", "sql_lab", DEFAULT_DATABASE_NAME.lower()}
    if name in protected or name.startswith("sqlite_"):
        fail("System and default databases are protected.", 403)
    with registry() as catalog:
        own = catalog.execute("SELECT * FROM databases WHERE owner_user_id = ?", (str(user.id),)).fetchone()
        target = catalog.execute("SELECT * FROM databases WHERE name = ?", (name,)).fetchone()
        if operation == "CREATE":
            if own:
                fail("You can create only one database. Delete your existing database before creating another.")
            if target:
                fail("A database with this name already exists.")
            # Old files are neither adopted nor overwritten, regardless of their apparent owner.
            if any(path.stem.lower() == name or path.name.lower() == name
                   for path in SQL_STORAGE_FOLDER.iterdir()):
                fail("This name belongs to an untracked legacy database and requires administrator review.", 403)
            record = {"file_id": uuid4().hex}
            path = owned_path(record)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also leaves crash-orphaned files untouched.
            with path.open("xb"):
                pass
            try:
                with sqlite3.connect(path) as database:
                    database.execute("PRAGMA user_version = 1")
                database.close()
                catalog.execute("INSERT INTO databases VALUES (?, ?, ?, ?, ?)",
                                (name, str(user.id), datetime.now(UTC).isoformat(),
                                 getattr(user, "organization_id", None), record["file_id"]))
            except Exception:
                path.unlink(missing_ok=True)
                raise
        else:
            if not target:
                fail("Database ownership is not verified. Missing or legacy databases cannot be deleted; contact an administrator.", 403)
            if target["owner_user_id"] != str(user.id):
                fail("You are not authorized to delete another user's database.", 403)
            # Delete only the generated file associated with verified ownership.
            # A retry can finish metadata cleanup after an interrupted deletion.
            owned_path(target).unlink(missing_ok=True)
            catalog.execute("DELETE FROM databases WHERE name = ? AND owner_user_id = ?", (name, str(user.id)))
    return {"columns": [], "rows": [], "database_changed": True,
            "message": f"Database '{name}' {'created' if operation == 'CREATE' else 'deleted'} successfully."}
