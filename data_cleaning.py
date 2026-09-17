"""
data_cleaning.py — Flask Blueprint for the Data Cleaning Agent
==================================================================
Routes only. All logic lives in cleaning_store.py (persistence/files),
cleaning_tools.py (deterministic checks), cleaning_ops.py (structured,
human-approved workbook edits) and cleaning_agent.py (the one AGNO agent).

init_cleaning(model, model_id) keeps the same two-step register-then-init pattern
app.py already used for this blueprint. `model` (the AGNO Gemini model) is
passed so the agent can use it directly; `model_id` is stored for reference.
"""

import io
import json
import math
import re
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file

import cleaning_agent as agent
import cleaning_ops as ops
import cleaning_store as store
import cleaning_tools as ct

data_cleaning_bp = Blueprint("data_cleaning", __name__, url_prefix="/clean")

_model = None
_model_obj = None
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xlsm"}


def init_cleaning(model, model_id: str) -> None:
    global _model, _model_obj
    _model = model_id
    _model_obj = model
    store.init_db()


def _err(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _session_or_404(session_id: str):
    session = store.get_session(session_id)
    if session is None:
        return None, _err(f"no such session: {session_id}", 404)
    return session, None


def _op_or_404(op_id: int):
    op_row = store.get_operation(op_id)
    if op_row is None:
        return None, _err(f"no such operation: {op_id}", 404)
    return op_row, None


def _op_in_session_or_404(op_id: int, session_id: str):
    """DC-02: every operation-mutating route must verify the operation actually
    belongs to the session named in the URL, not just that the operation exists.
    Without this, an operation id from session A could be undone/approved/
    rejected/edited through session B's endpoint — since undo_operation() reads
    session B's working_path but session A's cell_changes, that would corrupt
    session B's workbook with session A's row/column data and wrongly mark
    session A's operation as resolved. This is a correctness/isolation check
    between cleaning sessions, not authentication — there is still no user
    concept, and every session on this shared/local installation remains
    equally reachable by whoever has its id."""
    op_row, err = _op_or_404(op_id)
    if err:
        return None, err
    if op_row["session_id"] != session_id:
        return None, _err(f"operation {op_id} does not belong to session {session_id}", 404)
    return op_row, None


def _group_in_session_or_404(group_id: int, session_id: str):
    group = store.get_anomaly_group(group_id)
    if group is None:
        return None, _err(f"no such group: {group_id}", 404)
    if group["session_id"] != session_id:
        return None, _err(f"group {group_id} does not belong to session {session_id}", 404)
    return group, None


# ─────────────────────────────────────────────
# Sessions
# ─────────────────────────────────────────────

@data_cleaning_bp.route("/upload", methods=["POST"])
def upload():
    """DC-06: the upload is fully validated in memory (io.BytesIO — nothing
    touches disk yet) before anything is persisted. A corrupt/malformed file
    is rejected without ever creating an original file, a working file, or a
    session record. If persistence itself fails after validation succeeds
    (e.g. a disk error), whatever was partially written is cleaned up too."""
    if "file" not in request.files:
        return _err("no file uploaded")
    f = request.files["file"]
    if not f.filename:
        return _err("no file selected")
    ext = Path(f.filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        return _err(f"unsupported file type '.{ext}' — use .csv, .xlsx, or .xlsm")

    file_bytes = f.read()
    if not file_bytes:
        return _err("uploaded file is empty")

    try:
        sheets = ct.list_sheets(io.BytesIO(file_bytes), ext)
        if not sheets:
            raise ValueError("the file has no worksheets")
        for sheet in sheets:
            df = ct.read_sheet_df(io.BytesIO(file_bytes), ext, sheet)
            ragged_error = ct.check_ragged_rows(df)
            if ragged_error:
                raise ValueError(f"sheet '{sheet}': {ragged_error}")
    except Exception as e:
        return _err(f"could not read the file — it may be corrupt or malformed: {e}")

    session_id, original_path, file_type = store.save_original(file_bytes, f.filename)
    try:
        working_path = store.copy_to_working(session_id, original_path, file_type)
        session = store.create_session(session_id, f.filename, original_path, working_path, file_type, sheets)
    except Exception:
        Path(original_path).unlink(missing_ok=True)
        store.working_path_for(session_id, file_type).unlink(missing_ok=True)
        current_app.logger.exception("failed to persist upload for session %s", session_id)
        return _err("could not save the uploaded file", 500)

    return jsonify({"session_id": session_id, "sheets": sheets, "file_type": file_type, "state": session["state"]})


@data_cleaning_bp.route("/session/<session_id>/sheet", methods=["POST"])
def set_sheet(session_id):
    session, err = _session_or_404(session_id)
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    sheet = body.get("sheet")
    if sheet not in session["all_sheets"]:
        return _err(f"sheet '{sheet}' not found; available sheets: {session['all_sheets']}")
    df = ct.read_sheet_df(session["working_path"], session["file_type"], sheet)
    columns = list(df.columns.astype(str))
    store.update_session(session_id, sheet=sheet, selected_columns=[])
    return jsonify({"sheet": sheet, "columns": columns, "n_rows": len(df)})


@data_cleaning_bp.route("/session/<session_id>/columns", methods=["POST"])
def set_columns(session_id):
    session, err = _session_or_404(session_id)
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    columns = body.get("columns") or []
    if not session.get("sheet"):
        return _err("select a sheet before selecting columns")
    available = ct.list_columns(session["working_path"], session["file_type"], session["sheet"])
    unknown = [c for c in columns if c not in available]
    if unknown:
        return _err(f"unknown columns: {unknown}")
    updated = store.update_session(session_id, selected_columns=columns)
    return jsonify({"selected_columns": updated["selected_columns"]})


# ─────────────────────────────────────────────
# Agent + state
# ─────────────────────────────────────────────

@data_cleaning_bp.route("/session/<session_id>/message", methods=["POST"])
def send_message(session_id):
    session, err = _session_or_404(session_id)
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return _err("message is required")
    if len(message) > agent.MAX_USER_MESSAGE_CHARS:
        return _err(f"message is too long (max {agent.MAX_USER_MESSAGE_CHARS} characters)")
    try:
        result = agent.run_agent_turn(session_id, message, _model)
    except Exception:
        current_app.logger.exception("agent turn failed for session %s", session_id)
        return _err("The Data Cleaning Agent is temporarily unavailable. Please try again.", 502)
    return jsonify(result)


@data_cleaning_bp.route("/session/<session_id>", methods=["GET"])
def get_session(session_id):
    """DC-05: also the session-restore endpoint the frontend calls after a page
    refresh (with the session id it kept in localStorage) — the response here
    is everything needed to rebuild the UI without re-uploading anything."""
    session, err = _session_or_404(session_id)
    if err:
        return err
    try:
        session["file_size"] = Path(session["original_path"]).stat().st_size
    except OSError:
        session["file_size"] = None
    return jsonify({
        "session": session,
        "messages": store.list_messages(session_id),
        "groups": store.list_anomaly_groups(session_id),
        "operations": store.list_operations(session_id),
    })


DEFAULT_ANOMALY_PAGE_SIZE = 100
MAX_ANOMALY_PAGE_SIZE = 500


def _jsonable(value):
    if value is None:
        return None
    try:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if hasattr(value, "item"):  # numpy scalar
            value = value.item()
    except Exception:
        pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _parse_int_list_param(name: str, raw: str):
    """API-01: validate before converting — e.g. group_ids=abc must return a
    controlled 400, not an uncaught ValueError -> 500. `raw is None` (the
    parameter was omitted entirely) means "no filter"; an explicit but empty
    value (`?group_ids=`) is treated as malformed input, not "no filter"."""
    if raw is None:
        return None
    if raw.strip() == "":
        raise ValueError(f"{name} must contain at least one numeric identifier")
    ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not re_fullmatch_int(part):
            raise ValueError(f"{name} must contain numeric identifiers only (got {part!r})")
        ids.append(int(part))
    if not ids:
        raise ValueError(f"{name} must contain at least one numeric identifier")
    return ids


def _parse_positive_int_param(name: str, raw: str, default: int, maximum: int = None):
    if raw is None or raw == "":
        return default
    if not re_fullmatch_int(raw):
        raise ValueError(f"{name} must be a positive integer (got {raw!r})")
    val = int(raw)
    if val < 1:
        raise ValueError(f"{name} must be a positive integer (got {raw!r})")
    if maximum is not None and val > maximum:
        val = maximum
    return val


def re_fullmatch_int(s: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", s.strip())) if s is not None else False


@data_cleaning_bp.route("/session/<session_id>/anomaly-rows", methods=["GET"])
def anomaly_rows(session_id):
    """Combined, row-level, paginated anomaly table: one entry per (group,
    affected row), each carrying the complete ORIGINAL row (DC-03 — read from
    the immutable original file, never the mutable working copy, so evidence
    never changes shape or value once a correction is applied) plus, where it
    differs, the current working value clearly labelled as such. Supports
    filtering by group, check type, column, and row (API-01: every parameter
    is validated before conversion; malformed input is a controlled 400, not
    an unhandled ValueError -> 500). Supports pagination (DC-07) instead of a
    hard row cutoff, so no anomaly evidence is silently dropped — it just
    takes more pages to see it all."""
    session, err = _session_or_404(session_id)
    if err:
        return err

    try:
        group_ids = _parse_int_list_param("group_ids", request.args.get("group_ids"))
        page = _parse_positive_int_param("page", request.args.get("page"), default=1)
        page_size = _parse_positive_int_param("page_size", request.args.get("page_size"),
                                               default=DEFAULT_ANOMALY_PAGE_SIZE, maximum=MAX_ANOMALY_PAGE_SIZE)
    except ValueError as e:
        return jsonify({"error": "Invalid query parameter", "message": str(e)}), 400

    check_name_filter = request.args.get("check_name") or None
    column_filter = request.args.get("column") or None
    sheet_filter = request.args.get("sheet") or None
    row_filter_raw = request.args.get("row")
    row_filter = None
    if row_filter_raw:
        if not re_fullmatch_int(row_filter_raw):
            return jsonify({"error": "Invalid query parameter", "message": "row must be a numeric Excel row number"}), 400
        row_filter = int(row_filter_raw)

    groups = store.list_anomaly_groups(session_id)
    if group_ids:
        wanted = set(group_ids)
        groups = [g for g in groups if g["id"] in wanted]
    if check_name_filter:
        groups = [g for g in groups if g["check_name"] == check_name_filter]
    if not groups:
        return jsonify({"columns": [], "rows": [], "page": page, "page_size": page_size,
                         "total_rows": 0, "total_pages": 0})

    try:
        original_df = ct.read_sheet_df(session["original_path"], session["file_type"], session.get("sheet"))
    except Exception:
        current_app.logger.exception("could not read original file for session %s", session_id)
        return _err("could not read the original file", 500)
    columns = list(original_df.columns.astype(str))

    working_df = None
    try:
        working_df = ct.read_sheet_df(session["working_path"], session["file_type"], session.get("sheet"))
    except Exception:
        pass  # current-value comparison is best-effort; original evidence never depends on it

    # Normalize to one evidence row per source row.  A row may belong to
    # several anomaly groups; retain each indicator in ``anomalies`` while
    # keeping the complete immutable original row only once.
    row_map = {}
    for g in groups:
        for ref in g["row_refs"]:
            excel_row = ref.get("row")
            col = ref.get("column")
            if sheet_filter and (session.get("sheet") or "") != sheet_filter:
                continue
            if column_filter and col != column_filter:
                continue
            if row_filter is not None and excel_row != row_filter:
                continue
            idx = excel_row - 2 if excel_row is not None else None
            if idx is None or idx < 0 or idx >= len(original_df):
                continue
            row_series = original_df.iloc[idx]
            row_data = {c: _jsonable(row_series[c]) for c in columns}
            original_value = row_data.get(col) if col else None

            current_value = None
            if working_df is not None and col and col in working_df.columns and idx < len(working_df):
                wv = _jsonable(working_df.iloc[idx][col])
                if wv != original_value:
                    current_value = wv

            key = (session.get("sheet"), excel_row)
            record = row_map.get(key)
            if record is None:
                record = {
                    "sheet": session.get("sheet"),
                    "row": excel_row,
                    "group_id": g["id"],
                    "check_name": g["check_name"],
                    "column": col,
                    "value": original_value,
                    "current_value": current_value,
                    "row_data": row_data,
                    "anomalies": [],
                }
                row_map[key] = record
            record["anomalies"].append({
                "group_id": g["id"],
                "check_name": g["check_name"],
                "column": col,
                "value": original_value,
                "current_value": current_value,
            })

    all_rows = list(row_map.values())
    for record in all_rows:
        first = record["anomalies"][0]
        record["group_id"] = first["group_id"]
        record["check_name"] = first["check_name"]
        record["column"] = first["column"]
        record["value"] = first["value"]

    total_rows = len(all_rows)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    page_rows = all_rows[start:start + page_size]

    return jsonify({
        "columns": columns,
        "rows": page_rows,
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
    })


# ─────────────────────────────────────────────
# Operations — approve/reject/edit are triggered ONLY by a person via these
# routes. No agent tool calls these. Every route is scoped under the owning
# session in the URL (DC-02): an operation id is only ever acted on after
# confirming it belongs to that exact session.
# ─────────────────────────────────────────────

@data_cleaning_bp.route("/session/<session_id>/operations/<int:op_id>/preview", methods=["GET"])
def preview_operation(session_id, op_id):
    session, err = _session_or_404(session_id)
    if err:
        return err
    op_row, err = _op_in_session_or_404(op_id, session_id)
    if err:
        return err
    try:
        preview = ops.dry_run_operation(op_row["op"], session)
    except ops.OperationError as e:
        return _err(str(e))
    return jsonify(preview)


@data_cleaning_bp.route("/session/<session_id>/operations/<int:op_id>/approve", methods=["POST"])
def approve_operation(session_id, op_id):
    session, err = _session_or_404(session_id)
    if err:
        return err
    op_row, err = _op_in_session_or_404(op_id, session_id)
    if err:
        return err
    if op_row["status"] != "proposed":
        return _err(f"operation is '{op_row['status']}', not 'proposed'")
    try:
        updated = ops.apply_operation(op_row, session)
    except ops.OperationError as e:
        store.update_operation(op_id, status="failed_stale")
        return _err(str(e))
    return jsonify(updated)


@data_cleaning_bp.route("/session/<session_id>/operations/<int:op_id>/reject", methods=["POST"])
def reject_operation(session_id, op_id):
    _, err = _session_or_404(session_id)
    if err:
        return err
    op_row, err = _op_in_session_or_404(op_id, session_id)
    if err:
        return err
    if op_row["status"] != "proposed":
        return _err(f"operation is '{op_row['status']}', not 'proposed'")
    updated = store.update_operation(op_id, status="rejected")
    return jsonify(updated)


@data_cleaning_bp.route("/session/<session_id>/operations/<int:op_id>/edit", methods=["POST"])
def edit_operation(session_id, op_id):
    session, err = _session_or_404(session_id)
    if err:
        return err
    op_row, err = _op_in_session_or_404(op_id, session_id)
    if err:
        return err
    if op_row["status"] != "proposed":
        return _err(f"operation is '{op_row['status']}', not 'proposed' — cannot edit")
    body = request.get_json(force=True, silent=True) or {}
    new_op = dict(op_row["op"])
    for key in ("operation", "sheet", "column", "targets", "flag_name", "flag_value"):
        if key in body:
            new_op[key] = body[key]
    try:
        ops.validate_operation(new_op, session)
    except ops.OperationError as e:
        return _err(str(e))
    store.update_operation(op_id, op_json=json.dumps(new_op))
    return jsonify(store.get_operation(op_id))


@data_cleaning_bp.route("/session/<session_id>/groups/<int:group_id>/approve", methods=["POST"])
def approve_group(session_id, group_id):
    """DC-04: transactional group approval. Every pending operation in the
    group is validated and applied independently; a stale/invalid operation
    is marked 'failed_stale' and left visible for the user to review — it
    never silently disappears and never causes the whole group to be
    misreported as resolved. The group is only marked 'resolved' when every
    operation that was pending at the start of this call actually applied."""
    session, err = _session_or_404(session_id)
    if err:
        return err
    group, err = _group_in_session_or_404(group_id, session_id)
    if err:
        return err
    pending = [op for op in store.list_operations(session_id, status="proposed")
               if op["group_id"] == group_id]
    body = request.get_json(force=True, silent=True) or {}
    if not body.get("confirm"):
        previews = []
        for op_row in pending:
            try:
                previews.append({"operation_id": op_row["id"], "preview": ops.dry_run_operation(op_row["op"], session)})
            except ops.OperationError as e:
                previews.append({"operation_id": op_row["id"], "error": str(e)})
        return jsonify({"pending_count": len(pending), "previews": previews, "confirmed": False})

    results = []
    any_failed = False
    for op_row in pending:
        try:
            results.append(ops.apply_operation(op_row, session))
        except ops.OperationError as e:
            store.update_operation(op_row["id"], status="failed_stale")
            results.append({"operation_id": op_row["id"], "status": "failed_stale", "error": str(e)})
            any_failed = True
    new_status = "resolved" if (pending and not any_failed) else group["status"]
    store.update_anomaly_group(group_id, status=new_status)
    return jsonify({"applied": results, "confirmed": True, "all_succeeded": not any_failed, "group_status": new_status})


# ─────────────────────────────────────────────
# Undo / revert / history / finalize / download
# ─────────────────────────────────────────────

@data_cleaning_bp.route("/session/<session_id>/undo", methods=["POST"])
def undo(session_id):
    session, err = _session_or_404(session_id)
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    op_id = body.get("op_id")
    group_id = body.get("group_id")

    if op_id:
        op_row, err = _op_in_session_or_404(op_id, session_id)
        if err:
            return err
        try:
            return jsonify(ops.undo_operation(op_row, session))
        except ops.OperationError as e:
            return _err(str(e))

    if group_id:
        applied = [op for op in store.list_operations(session_id, status="applied") if op["group_id"] == group_id]
        results = []
        for op_row in reversed(applied):
            try:
                results.append(ops.undo_operation(op_row, session))
            except ops.OperationError as e:
                results.append({"operation_id": op_row["id"], "error": str(e)})
        return jsonify({"undone": results})

    applied = store.list_operations(session_id, status="applied")
    if not applied:
        return _err("no applied operations to undo")
    latest = applied[-1]
    try:
        return jsonify(ops.undo_operation(latest, session))
    except ops.OperationError as e:
        return _err(str(e))


@data_cleaning_bp.route("/session/<session_id>/revert", methods=["POST"])
def revert(session_id):
    session, err = _session_or_404(session_id)
    if err:
        return err
    updated = ops.revert_session(session)
    return jsonify(updated)


@data_cleaning_bp.route("/session/<session_id>/history", methods=["GET"])
def history(session_id):
    session, err = _session_or_404(session_id)
    if err:
        return err
    return jsonify(store.list_operations(session_id))


@data_cleaning_bp.route("/session/<session_id>/finalize", methods=["POST"])
def finalize(session_id):
    session, err = _session_or_404(session_id)
    if err:
        return err
    try:
        before_df = ct.read_sheet_df(session["original_path"], session["file_type"], session.get("sheet"))
        after_df = ct.read_sheet_df(session["working_path"], session["file_type"], session.get("sheet"))
        before = ct.run_all_detections(before_df, session["selected_columns"], session.get("data_contract") or {},
                                        file_type=session["file_type"])
        after = ct.run_all_detections(after_df, session["selected_columns"], session.get("data_contract") or {},
                                       file_type=session["file_type"])
        ops.finalize_workbook(session)
        ops.generate_report(session, before, after)
    except Exception:
        current_app.logger.exception("finalize failed for session %s", session_id)
        return _err("finalize failed", 500)
    store.update_session(session_id, state="FINALIZE")
    comparison = ops.compare_before_after(before, after)
    return jsonify({"state": "FINALIZE", "comparison": comparison})


@data_cleaning_bp.route("/session/<session_id>/download/<kind>", methods=["GET"])
def download(session_id, kind):
    session, err = _session_or_404(session_id)
    if err:
        return err
    cleaned_path, report_path = store.output_paths(session_id)
    if kind == "cleaned":
        cleaned_path = cleaned_path.with_suffix(Path(session["working_path"]).suffix)
        if not cleaned_path.exists():
            return _err("finalize the session first", 404)
        return send_file(cleaned_path, as_attachment=True,
                          download_name=f"cleaned_{session['original_filename']}")
    if kind == "report":
        if not report_path.exists():
            return _err("finalize the session first", 404)
        return send_file(report_path, as_attachment=True,
                          download_name=f"cleaning_report_{session_id[:8]}.xlsx")
    return _err(f"unknown download kind '{kind}'")


# ─────────────────────────────────────────────
# Historical file library
# ─────────────────────────────────────────────

@data_cleaning_bp.route("/history-files", methods=["GET"])
def list_history_files():
    return jsonify(store.list_historical_files())


@data_cleaning_bp.route("/history-files", methods=["POST"])
def add_history_file():
    if "file" not in request.files:
        return _err("no file uploaded")
    f = request.files["file"]
    if not f.filename:
        return _err("no file selected")
    ext = Path(f.filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        return _err(f"unsupported file type '.{ext}'")

    file_bytes = f.read()
    if not file_bytes:
        return _err("uploaded file is empty")

    # DC-06: validate in memory first — an invalid historical file must never
    # be persisted (and must never consume one of the 20 library slots).
    try:
        sheets = ct.list_sheets(io.BytesIO(file_bytes), ext)
        if not sheets:
            raise ValueError("the file has no worksheets")
        columns = {}
        total_rows = 0
        for sheet in sheets:
            df = ct.read_sheet_df(io.BytesIO(file_bytes), ext, sheet)
            ragged_error = ct.check_ragged_rows(df)
            if ragged_error:
                raise ValueError(f"sheet '{sheet}': {ragged_error}")
            columns[sheet] = list(df.columns.astype(str))
            total_rows += len(df)
    except Exception as e:
        return _err(f"could not read file — it may be corrupt or malformed: {e}")

    path, file_type = store.save_historical_file(file_bytes, f.filename)
    try:
        name = request.form.get("name") or f.filename
        description = (request.form.get("description") or "").strip() or None
        record = store.add_historical_file(
            name, path, file_type, sheets, columns, total_rows,
            size_bytes=len(file_bytes), description=description,
        )
        for sheet in sheets:
            df = ct.read_sheet_df(path, file_type, sheet)
            rows = ct.build_historical_index_rows(df, sheet)
            store.add_historical_values(record["id"], rows)
    except ValueError as e:
        path.unlink(missing_ok=True)
        return _err(str(e))
    except Exception:
        path.unlink(missing_ok=True)
        current_app.logger.exception("failed to persist historical file %s", f.filename)
        return _err("could not save file", 500)
    return jsonify(record)


@data_cleaning_bp.route("/history-files/<int:file_id>", methods=["PUT"])
def replace_history_file(file_id):
    """Replace one historical reference only after the new file validates.
    The 20-file slot is retained and the old file is removed only after the
    replacement has been indexed successfully.
    """
    old = store.get_historical_file(file_id)
    if old is None:
        return _err(f"no such historical file: {file_id}", 404)
    if "file" not in request.files:
        return _err("no file uploaded")
    f = request.files["file"]
    if not f.filename:
        return _err("no file selected")
    ext = Path(f.filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        return _err(f"unsupported file type '.{ext}'")
    file_bytes = f.read()
    if not file_bytes:
        return _err("uploaded file is empty")
    try:
        candidate_sheets = ct.list_sheets(io.BytesIO(file_bytes), ext)
        candidate_columns = {}
        total_rows = 0
        for sheet in candidate_sheets:
            df = ct.read_sheet_df(io.BytesIO(file_bytes), ext, sheet)
            ragged_error = ct.check_ragged_rows(df)
            if ragged_error:
                raise ValueError(f"sheet '{sheet}': {ragged_error}")
            candidate_columns[sheet] = list(df.columns.astype(str))
            total_rows += len(df)
    except Exception:
        current_app.logger.exception("invalid historical replacement %s", f.filename)
        return _err("could not read file — it may be corrupt or malformed")

    new_path, new_type = store.save_historical_file(file_bytes, f.filename)
    try:
        name = request.form.get("name") or f.filename
        description = (request.form.get("description") or "").strip() or None
        # Build the complete index before touching the existing record.  This
        # keeps a failed replacement from leaving metadata and values out of
        # sync or deleting the previous valid file.
        indexed_rows = []
        for sheet in candidate_sheets:
            df = ct.read_sheet_df(new_path, new_type, sheet)
            indexed_rows.extend((sheet, *r) for r in ct.build_historical_index_rows(df, sheet))
        conn = store.get_conn()
        try:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM historical_values WHERE file_id = ?", (file_id,))
            conn.execute(
                "UPDATE historical_files SET name=?, path=?, file_type=?, sheets=?, columns=?, row_count=?, size_bytes=?, description=? WHERE id=?",
                (name, str(new_path), new_type, json.dumps(candidate_sheets),
                 json.dumps(candidate_columns), total_rows, len(file_bytes), description, file_id),
            )
            conn.executemany(
                "INSERT INTO historical_values (file_id, sheet, column, value, normalized_value, count) VALUES (?, ?, ?, ?, ?, ?)",
                [(file_id, *r) for r in indexed_rows],
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    except Exception:
        new_path.unlink(missing_ok=True)
        current_app.logger.exception("failed to replace historical file %s", file_id)
        return _err("could not save replacement file", 500)
    old_path = Path(old["path"])
    if old_path != new_path:
        old_path.unlink(missing_ok=True)
    return jsonify(store.get_historical_file(file_id))


@data_cleaning_bp.route("/history-files/<int:file_id>", methods=["DELETE"])
def delete_history_file(file_id):
    ok = store.delete_historical_file(file_id)
    if not ok:
        return _err(f"no such historical file: {file_id}", 404)
    return jsonify({"deleted": file_id})


# ─────────────────────────────────────────────
# Learned rules. POST /rules is the direct UI-form path — a person filling the
# form out themselves is the approval, so it's created already-approved. The
# Data Cleaning Agent's propose_learned_rule tool takes a different path: it
# can only create a status='proposed' rule (find_learned_rules never returns
# those), and only a person clicking approve here activates it. Rules are a
# shared resource across sessions by design (no per-user ownership in this
# app), so these routes are intentionally not session-scoped.
# ─────────────────────────────────────────────

@data_cleaning_bp.route("/rules", methods=["GET"])
def list_rules():
    status = request.args.get("status")
    if status and status not in ("approved", "proposed", "rejected"):
        return jsonify({"error": "Invalid query parameter", "message": "status must be approved, proposed, or rejected"}), 400
    return jsonify(store.list_learned_rules(status=status))


@data_cleaning_bp.route("/rules", methods=["POST"])
def add_rule():
    body = request.get_json(force=True, silent=True) or {}
    try:
        rule = store.add_learned_rule(
            scope=body.get("scope"),
            anomaly_type=body.get("anomaly_type"),
            original_pattern=body.get("original_pattern"),
            approved_solution=body.get("approved_solution"),
            reason=body.get("reason", ""),
            client=body.get("client"),
            dataset_type=body.get("dataset_type"),
            column=body.get("column"),
            status="approved",
        )
    except ValueError as e:
        return _err(str(e))
    return jsonify(rule)


@data_cleaning_bp.route("/rules/<int:rule_id>/approve", methods=["POST"])
def approve_rule(rule_id):
    rule = store.get_learned_rule(rule_id)
    if rule is None:
        return _err(f"no such rule: {rule_id}", 404)
    if rule["status"] != "proposed":
        return _err(f"rule is '{rule['status']}', not 'proposed'")
    return jsonify(store.update_learned_rule_status(rule_id, "approved"))


@data_cleaning_bp.route("/rules/<int:rule_id>/reject", methods=["POST"])
def reject_rule(rule_id):
    rule = store.get_learned_rule(rule_id)
    if rule is None:
        return _err(f"no such rule: {rule_id}", 404)
    if rule["status"] != "proposed":
        return _err(f"rule is '{rule['status']}', not 'proposed'")
    return jsonify(store.update_learned_rule_status(rule_id, "rejected"))


@data_cleaning_bp.route("/rules/<int:rule_id>", methods=["DELETE"])
def delete_rule(rule_id):
    ok = store.delete_learned_rule(rule_id)
    if not ok:
        return _err(f"no such rule: {rule_id}", 404)
    return jsonify({"deleted": rule_id})
