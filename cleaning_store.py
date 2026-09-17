"""
cleaning_store.py — SQLite persistence + controlled file store for the Data Cleaning Agent
=============================================================================================
Everything here is plain functions over a single SQLite database (nexus_data/nexus.db).
This module is the ONLY place that builds filesystem paths for cleaning sessions and
historical files — callers pass ids/bytes/filenames in, never raw paths.

Layout under NEXUS_DATA_DIR (default ./nexus_data/, override with the env var for
another machine/location — Render's disk is ephemeral, so this is meant for local use):

    nexus_data/
        nexus.db
        originals/<session_id>.<ext>      # never modified after upload
        working/<session_id>.<ext>        # openpyxl edits happen here
        output/<session_id>_cleaned.xlsx
        output/<session_id>_report.xlsx
        historical/<file_id>.<ext>
"""

import datetime
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Optional

MAX_HISTORICAL_FILES = 20

DATA_DIR = Path(os.getenv("NEXUS_DATA_DIR", Path(__file__).resolve().parent / "nexus_data"))
ORIGINALS_DIR = DATA_DIR / "originals"
WORKING_DIR = DATA_DIR / "working"
OUTPUT_DIR = DATA_DIR / "output"
HISTORICAL_DIR = DATA_DIR / "historical"
DB_PATH = DATA_DIR / "nexus.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    original_filename TEXT,
    original_path TEXT,
    working_path TEXT,
    file_type TEXT,
    sheet TEXT,
    all_sheets TEXT,
    selected_columns TEXT,
    state TEXT DEFAULT 'UNDERSTAND',
    profile TEXT,
    data_contract TEXT,
    client TEXT,
    dataset_type TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS anomaly_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    check_name TEXT,
    dimension TEXT,
    columns TEXT,
    count INTEGER,
    row_refs TEXT,
    examples TEXT,
    classification TEXT DEFAULT 'potential',
    detection_confidence TEXT,
    evidence_confidence TEXT,
    recommendation_confidence TEXT,
    evidence TEXT,
    status TEXT DEFAULT 'open',
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_groups_session ON anomaly_groups(session_id);
CREATE INDEX IF NOT EXISTS idx_groups_session_status ON anomaly_groups(session_id, status);

-- status: proposed | approved | applied | failed_stale | rejected | undone | reverted
CREATE TABLE IF NOT EXISTS operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    group_id INTEGER REFERENCES anomaly_groups(id),
    op_json TEXT NOT NULL,
    status TEXT DEFAULT 'proposed',
    cell_changes TEXT,
    reason TEXT,
    evidence TEXT,
    created_at TEXT,
    approved_at TEXT,
    applied_at TEXT,
    undone_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ops_session ON operations(session_id);
CREATE INDEX IF NOT EXISTS idx_ops_session_status ON operations(session_id, status);
CREATE INDEX IF NOT EXISTS idx_ops_group ON operations(group_id);

CREATE TABLE IF NOT EXISTS historical_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    path TEXT,
    file_type TEXT,
    sheets TEXT,
    columns TEXT,
    row_count INTEGER,
    size_bytes INTEGER,
    description TEXT,
    added_at TEXT
);

CREATE TABLE IF NOT EXISTS historical_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES historical_files(id),
    sheet TEXT,
    column TEXT,
    value TEXT,
    normalized_value TEXT,
    count INTEGER
);
CREATE INDEX IF NOT EXISTS idx_hv_lookup ON historical_values(column, normalized_value);
CREATE INDEX IF NOT EXISTS idx_hv_file ON historical_values(file_id);

-- scope: GLOBAL | DATASET_TYPE | CLIENT | COLUMN | CLIENT_COLUMN
-- status: approved (created directly via the UI form) | proposed (agent-suggested,
--         awaiting a person's approval) | rejected
CREATE TABLE IF NOT EXISTS learned_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    client TEXT,
    dataset_type TEXT,
    column TEXT,
    anomaly_type TEXT,
    original_pattern TEXT,
    approved_solution TEXT,
    reason TEXT,
    status TEXT DEFAULT 'approved',
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_rules_scope ON learned_rules(scope, client, column);

CREATE TABLE IF NOT EXISTS research (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    query TEXT,
    source_title TEXT,
    url TEXT,
    retrieved_at TEXT,
    fact TEXT,
    confidence TEXT
);
CREATE INDEX IF NOT EXISTS idx_research_session ON research(session_id);
"""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _ensure_dirs() -> None:
    for d in (DATA_DIR, ORIGINALS_DIR, WORKING_DIR, OUTPUT_DIR, HISTORICAL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    _ensure_dirs()
    conn = get_conn()
    try:
        conn.executescript(_SCHEMA)
        # Idempotent migrations for databases created before newer columns
        # were introduced. SQLite has no ADD COLUMN IF NOT EXISTS syntax.
        migrations = {
            "learned_rules": {
                "status": "TEXT NOT NULL DEFAULT 'approved'",
            },
            "historical_files": {
                "size_bytes": "INTEGER",
                "description": "TEXT",
            },
        }
        for table, columns in migrations.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for column, definition in columns.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rules_status ON learned_rules(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_historical_added ON historical_files(added_at)")
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    if row is None:
        return None
    return dict(row)


def _json_load(val: Optional[str], default: Any = None) -> Any:
    if val is None:
        return default
    try:
        return json.loads(val)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────
# File store
# ─────────────────────────────────────────────

def new_id() -> str:
    return uuid.uuid4().hex


def save_original(file_bytes: bytes, filename: str) -> tuple[str, Path, str]:
    """Copies an uploaded file into originals/ under a new session id. Returns
    (session_id, original_path, file_type). Never called again for that session —
    all edits happen on a separate working/ copy."""
    _ensure_dirs()
    ext = Path(filename).suffix.lower().lstrip(".") or "bin"
    session_id = new_id()
    original_path = ORIGINALS_DIR / f"{session_id}.{ext}"
    original_path.write_bytes(file_bytes)
    return session_id, original_path, ext


def working_path_for(session_id: str, file_type: str) -> Path:
    return WORKING_DIR / f"{session_id}.{file_type}"


def copy_to_working(session_id: str, original_path: Path, file_type: str) -> Path:
    working = working_path_for(session_id, file_type)
    working.write_bytes(Path(original_path).read_bytes())
    return working


def output_paths(session_id: str) -> tuple[Path, Path]:
    return (
        OUTPUT_DIR / f"{session_id}_cleaned.xlsx",
        OUTPUT_DIR / f"{session_id}_report.xlsx",
    )


def save_historical_file(file_bytes: bytes, filename: str) -> tuple[Path, str]:
    _ensure_dirs()
    ext = Path(filename).suffix.lower().lstrip(".") or "bin"
    file_id = new_id()
    path = HISTORICAL_DIR / f"{file_id}.{ext}"
    path.write_bytes(file_bytes)
    return path, ext


# ─────────────────────────────────────────────
# Sessions
# ─────────────────────────────────────────────

def create_session(
    session_id: str,
    original_filename: str,
    original_path: Path,
    working_path: Path,
    file_type: str,
    all_sheets: list[str],
) -> dict:
    conn = get_conn()
    try:
        now = _now()
        conn.execute(
            """INSERT INTO sessions
               (id, original_filename, original_path, working_path, file_type,
                all_sheets, state, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'UNDERSTAND', ?, ?)""",
            (session_id, original_filename, str(original_path), str(working_path),
             file_type, json.dumps(all_sheets), now, now),
        )
        conn.commit()
        return get_session(session_id)
    finally:
        conn.close()


def get_session(session_id: str) -> Optional[dict]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        d = _row_to_dict(row)
        if d is None:
            return None
        d["all_sheets"] = _json_load(d.get("all_sheets"), [])
        d["selected_columns"] = _json_load(d.get("selected_columns"), [])
        d["profile"] = _json_load(d.get("profile"), None)
        d["data_contract"] = _json_load(d.get("data_contract"), {})
        return d
    finally:
        conn.close()


def update_session(session_id: str, **fields) -> Optional[dict]:
    if not fields:
        return get_session(session_id)
    json_fields = {"all_sheets", "selected_columns", "profile", "data_contract"}
    cols, vals = [], []
    for k, v in fields.items():
        cols.append(f"{k} = ?")
        vals.append(json.dumps(v) if k in json_fields else v)
    cols.append("updated_at = ?")
    vals.append(_now())
    vals.append(session_id)
    conn = get_conn()
    try:
        conn.execute(f"UPDATE sessions SET {', '.join(cols)} WHERE id = ?", vals)
        conn.commit()
        return get_session(session_id)
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Messages
# ─────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str) -> None:
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def list_messages(session_id: str, limit: int = 200) -> list[dict]:
    """Returns up to `limit` messages, oldest-first. The limit keeps the NEWEST
    rows: cleaning_agent.run_agent_turn feeds this to the model as the recent
    conversation and relies on the last element being the current turn's message.
    (Ordering ASC directly under the LIMIT would pin the window to the start of
    the conversation, so the agent would stop seeing anything the user said.)"""
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT * FROM (
                   SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?
               ) ORDER BY id ASC""",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Anomaly groups
# ─────────────────────────────────────────────

def create_anomaly_group(session_id: str, **fields) -> dict:
    conn = get_conn()
    try:
        json_fields = {"columns", "row_refs", "examples", "evidence"}
        cols = ["session_id", "created_at"]
        vals = [session_id, _now()]
        for k, v in fields.items():
            cols.append(k)
            vals.append(json.dumps(v) if k in json_fields else v)
        placeholders = ", ".join("?" for _ in vals)
        cur = conn.execute(f"INSERT INTO anomaly_groups ({', '.join(cols)}) VALUES ({placeholders})", vals)
        conn.commit()
        return get_anomaly_group(cur.lastrowid)
    finally:
        conn.close()


def _group_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in ("columns", "row_refs", "examples", "evidence"):
        d[k] = _json_load(d.get(k), [] if k != "evidence" else {})
    return d


def get_anomaly_group(group_id: int) -> Optional[dict]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM anomaly_groups WHERE id = ?", (group_id,)).fetchone()
        return _group_row_to_dict(row) if row else None
    finally:
        conn.close()


def list_anomaly_groups(session_id: str) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM anomaly_groups WHERE session_id = ? ORDER BY id ASC", (session_id,)
        ).fetchall()
        return [_group_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def update_anomaly_group(group_id: int, **fields) -> Optional[dict]:
    if not fields:
        return get_anomaly_group(group_id)
    json_fields = {"columns", "row_refs", "examples", "evidence"}
    cols, vals = [], []
    for k, v in fields.items():
        cols.append(f"{k} = ?")
        vals.append(json.dumps(v) if k in json_fields else v)
    vals.append(group_id)
    conn = get_conn()
    try:
        conn.execute(f"UPDATE anomaly_groups SET {', '.join(cols)} WHERE id = ?", vals)
        conn.commit()
        return get_anomaly_group(group_id)
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Operations
# ─────────────────────────────────────────────

def create_operation(session_id: str, op: dict, group_id: Optional[int] = None,
                      reason: str = "", evidence: Optional[dict] = None) -> dict:
    conn = get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO operations
               (session_id, group_id, op_json, status, reason, evidence, created_at)
               VALUES (?, ?, ?, 'proposed', ?, ?, ?)""",
            (session_id, group_id, json.dumps(op), reason, json.dumps(evidence or {}), _now()),
        )
        conn.commit()
        return get_operation(cur.lastrowid)
    finally:
        conn.close()


def _op_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["op"] = _json_load(d.get("op_json"), {})
    d["cell_changes"] = _json_load(d.get("cell_changes"), [])
    d["evidence"] = _json_load(d.get("evidence"), {})
    return d


def get_operation(op_id: int) -> Optional[dict]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM operations WHERE id = ?", (op_id,)).fetchone()
        return _op_row_to_dict(row) if row else None
    finally:
        conn.close()


def list_operations(session_id: str, status: Optional[str] = None) -> list[dict]:
    conn = get_conn()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM operations WHERE session_id = ? AND status = ? ORDER BY id ASC",
                (session_id, status),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM operations WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
        return [_op_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def update_operation(op_id: int, **fields) -> Optional[dict]:
    if not fields:
        return get_operation(op_id)
    json_fields = {"cell_changes", "evidence"}
    cols, vals = [], []
    for k, v in fields.items():
        cols.append(f"{k} = ?")
        vals.append(json.dumps(v) if k in json_fields else v)
    vals.append(op_id)
    conn = get_conn()
    try:
        conn.execute(f"UPDATE operations SET {', '.join(cols)} WHERE id = ?", vals)
        conn.commit()
        return get_operation(op_id)
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Historical file library (max MAX_HISTORICAL_FILES)
# ─────────────────────────────────────────────

def count_historical_files() -> int:
    conn = get_conn()
    try:
        return conn.execute("SELECT COUNT(*) FROM historical_files").fetchone()[0]
    finally:
        conn.close()


def add_historical_file(name: str, path: Path, file_type: str, sheets: list,
                         columns: dict, row_count: int, size_bytes: Optional[int] = None,
                         description: Optional[str] = None) -> dict:
    if count_historical_files() >= MAX_HISTORICAL_FILES:
        raise ValueError(
            f"Historical file library is full ({MAX_HISTORICAL_FILES} files). "
            "Remove or replace one before adding another."
        )
    conn = get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO historical_files
               (name, path, file_type, sheets, columns, row_count, size_bytes, description, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, str(path), file_type, json.dumps(sheets), json.dumps(columns), row_count,
             size_bytes, description, _now()),
        )
        conn.commit()
        return get_historical_file(cur.lastrowid)
    finally:
        conn.close()


def _hist_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["sheets"] = _json_load(d.get("sheets"), [])
    d["columns"] = _json_load(d.get("columns"), {})
    return d


def get_historical_file(file_id: int) -> Optional[dict]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM historical_files WHERE id = ?", (file_id,)).fetchone()
        return _hist_row_to_dict(row) if row else None
    finally:
        conn.close()


def list_historical_files() -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM historical_files ORDER BY added_at DESC").fetchall()
        return [_hist_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def delete_historical_file(file_id: int) -> bool:
    conn = get_conn()
    try:
        row = conn.execute("SELECT path FROM historical_files WHERE id = ?", (file_id,)).fetchone()
        if row is None:
            return False
        path = Path(row["path"])
        conn.execute("DELETE FROM historical_values WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM historical_files WHERE id = ?", (file_id,))
        conn.commit()
        if path.exists():
            path.unlink()
        return True
    finally:
        conn.close()


def add_historical_values(file_id: int, rows: list[tuple[str, str, str, str, int]]) -> None:
    """rows: list of (sheet, column, value, normalized_value, count)"""
    if not rows:
        return
    conn = get_conn()
    try:
        conn.executemany(
            """INSERT INTO historical_values (file_id, sheet, column, value, normalized_value, count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(file_id, *r) for r in rows],
        )
        conn.commit()
    finally:
        conn.close()


def search_historical_values(column: str, normalized_value: Optional[str] = None, limit: int = 25) -> list[dict]:
    conn = get_conn()
    try:
        if normalized_value:
            rows = conn.execute(
                """SELECT hv.*, hf.name AS file_name FROM historical_values hv
                   JOIN historical_files hf ON hf.id = hv.file_id
                   WHERE hv.column = ? AND hv.normalized_value = ?
                   ORDER BY hv.count DESC LIMIT ?""",
                (column, normalized_value, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT hv.*, hf.name AS file_name FROM historical_values hv
                   JOIN historical_files hf ON hf.id = hv.file_id
                   WHERE hv.column = ?
                   ORDER BY hv.count DESC LIMIT ?""",
                (column, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def distinct_historical_values(column: str, limit: int = 500) -> list[dict]:
    """Used for near-match (fuzzy) search, which is done in Python with difflib."""
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT value, normalized_value, SUM(count) AS total, file_id
               FROM historical_values WHERE column = ?
               GROUP BY normalized_value ORDER BY total DESC LIMIT ?""",
            (column, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Learned rules
# ─────────────────────────────────────────────

VALID_RULE_SCOPES = {"GLOBAL", "DATASET_TYPE", "CLIENT", "COLUMN", "CLIENT_COLUMN"}


def add_learned_rule(scope: str, anomaly_type: str, original_pattern: str, approved_solution: str,
                      reason: str = "", client: Optional[str] = None,
                      dataset_type: Optional[str] = None, column: Optional[str] = None,
                      status: str = "approved") -> dict:
    """status='approved' is for the direct UI-form path (a person filling the
    form out themselves IS the approval). status='proposed' is for the agent's
    propose_learned_rule tool — such a rule is inert (find_learned_rules never
    returns it) until a person approves it via POST /clean/rules/<id>/approve."""
    if scope not in VALID_RULE_SCOPES:
        raise ValueError(f"Invalid rule scope: {scope}")
    if scope in ("CLIENT", "CLIENT_COLUMN") and not client:
        raise ValueError(f"Scope {scope} requires a client")
    if scope in ("COLUMN", "CLIENT_COLUMN") and not column:
        raise ValueError(f"Scope {scope} requires a column")
    if scope == "DATASET_TYPE" and not dataset_type:
        raise ValueError("Scope DATASET_TYPE requires a dataset_type")
    if status not in ("approved", "proposed"):
        raise ValueError(f"Invalid rule status: {status}")
    conn = get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO learned_rules
               (scope, client, dataset_type, column, anomaly_type, original_pattern,
                approved_solution, reason, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (scope, client, dataset_type, column, anomaly_type, original_pattern,
             approved_solution, reason, status, _now()),
        )
        conn.commit()
        return get_learned_rule(cur.lastrowid)
    finally:
        conn.close()


def update_learned_rule_status(rule_id: int, status: str) -> Optional[dict]:
    if status not in ("approved", "proposed", "rejected"):
        raise ValueError(f"Invalid rule status: {status}")
    conn = get_conn()
    try:
        conn.execute("UPDATE learned_rules SET status = ? WHERE id = ?", (status, rule_id))
        conn.commit()
        return get_learned_rule(rule_id)
    finally:
        conn.close()


def get_learned_rule(rule_id: int) -> Optional[dict]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM learned_rules WHERE id = ?", (rule_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_learned_rules(status: Optional[str] = None) -> list[dict]:
    conn = get_conn()
    try:
        if status:
            rows = conn.execute("SELECT * FROM learned_rules WHERE status = ? ORDER BY created_at DESC",
                                 (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM learned_rules ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def find_learned_rules(column: Optional[str] = None, client: Optional[str] = None,
                        dataset_type: Optional[str] = None) -> list[dict]:
    """Returns APPROVED rules whose scope matches the given context. A
    CLIENT/CLIENT_COLUMN rule only matches when `client` is given and equal —
    it never applies globally. A rule the agent proposed but no one has
    approved yet is never returned here, so it can never silently apply."""
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM learned_rules WHERE status = 'approved'").fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        scope = d["scope"]
        if scope == "GLOBAL":
            out.append(d)
        elif scope == "DATASET_TYPE" and dataset_type and d["dataset_type"] == dataset_type:
            out.append(d)
        elif scope == "CLIENT" and client and d["client"] == client:
            out.append(d)
        elif scope == "COLUMN" and column and d["column"] == column:
            out.append(d)
        elif scope == "CLIENT_COLUMN" and client and column and d["client"] == client and d["column"] == column:
            out.append(d)
    return out


def delete_learned_rule(rule_id: int) -> bool:
    conn = get_conn()
    try:
        cur = conn.execute("DELETE FROM learned_rules WHERE id = ?", (rule_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Research log
# ─────────────────────────────────────────────

def add_research(session_id: str, query: str, source_title: str, url: str,
                  fact: str, confidence: str) -> dict:
    conn = get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO research (session_id, query, source_title, url, retrieved_at, fact, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, query, source_title, url, _now(), fact, confidence),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM research WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_research(session_id: str) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM research WHERE session_id = ? ORDER BY id ASC", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
