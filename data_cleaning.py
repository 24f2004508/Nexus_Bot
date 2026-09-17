"""data_cleaning.py — LLM-assisted data-cleaning service for the GenAI Lab.

Registers a Flask Blueprint  ``data_cleaning``  with two endpoints:

    POST /clean/analyse     Upload an XLSX file; LLM returns a structured
                            cleaning plan (list of proposed operations).

    POST /clean/apply       Re-upload the same file plus the accepted plan;
                            applies each operation and returns a cleaned XLSX.

The LLM is called through the same Gemini client used in app.py.
No API key is needed here — it is imported from app.py via the blueprint.
"""

from __future__ import annotations

import io
import json
import re
import datetime
from pathlib import Path

import pandas as pd
from flask import Blueprint, jsonify, request, send_file

data_cleaning_bp = Blueprint("data_cleaning", __name__)

# ── injected by app.py after blueprint registration ──────────────────────────
_gemini_client = None
_gemini_model  = None

def init_cleaning(client, model: str) -> None:
    """Called from app.py: supply the shared Gemini client and model name."""
    global _gemini_client, _gemini_model
    _gemini_client = client
    _gemini_model  = model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _profile_dataframe(df: pd.DataFrame) -> dict:
    """Return a compact profile suitable for the LLM prompt."""
    n_rows, n_cols = df.shape
    profile: list[dict] = []

    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        dtype_str  = str(s.dtype)

        info: dict = {
            "column":     col,
            "dtype":      dtype_str,
            "null_count": null_count,
            "null_pct":   round(null_count / max(n_rows, 1) * 100, 1),
        }

        if pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            if len(non_null) > 0:
                q1  = float(non_null.quantile(0.25))
                q3  = float(non_null.quantile(0.75))
                iqr = q3 - q1
                n_outliers = int(((non_null < q1 - 1.5 * iqr) |
                                  (non_null > q3 + 1.5 * iqr)).sum())
                info.update({
                    "mean":       round(float(non_null.mean()), 4),
                    "min":        round(float(non_null.min()), 4),
                    "max":        round(float(non_null.max()), 4),
                    "iqr_outliers": n_outliers,
                })
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            vc = s.value_counts()
            info["top_values"] = vc.head(5).index.tolist()
            info["unique_count"] = int(s.nunique(dropna=True))

        profile.append(info)

    return {
        "n_rows":   n_rows,
        "n_cols":   n_cols,
        "columns":  profile,
    }


def _find_col(df: pd.DataFrame, *candidates: str):
    """Case/space/underscore-insensitive column lookup."""
    norm = {re.sub(r"[\s_]+", "", c).lower(): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[\s_]+", "", cand).lower()
        if key in norm:
            return norm[key]
    return None


def _is_orange(rgb_hex) -> bool:
    """Broad orange-hue match so this isn't tied to one exact swatch."""
    if not rgb_hex or len(rgb_hex) < 6:
        return False
    hex6 = rgb_hex[-6:]
    try:
        r, g, b = int(hex6[0:2], 16), int(hex6[2:4], 16), int(hex6[4:6], 16)
    except ValueError:
        return False
    if r < 180 or b > 120:
        return False
    return b < g < r


def _cell_fill_rgb(cell):
    fill = cell.fill
    if fill is None or fill.fgColor is None:
        return None
    color = fill.fgColor
    if color.type == "rgb" and color.rgb and color.rgb != "00000000":
        return color.rgb
    return None


def _detect_orange_formatting(file_bytes: bytes, sheet_name: str) -> dict:
    """
    Reads actual cell fill colours from the workbook (no hardcoded column
    names or transition lists). Returns:
        {"orange_columns": [...], "orange_rows": {0-based data row idx, ...}, "supported": bool}
    """
    result = {"orange_columns": [], "orange_rows": set(), "supported": False}
    try:
        import openpyxl
    except ImportError:
        return result
    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
        ws = wb[sheet_name] if sheet_name in wb.sheetnames else wb.active
    except Exception:
        return result

    result["supported"] = True
    max_row, max_col = ws.max_row, ws.max_column
    if max_row < 1 or max_col < 1:
        return result

    headers = [
        str(ws.cell(row=1, column=c).value).strip() if ws.cell(row=1, column=c).value is not None
        else f"col_{c}"
        for c in range(1, max_col + 1)
    ]

    orange_columns = []
    for c_idx, col_name in enumerate(headers, start=1):
        if _is_orange(_cell_fill_rgb(ws.cell(row=1, column=c_idx))):
            orange_columns.append(col_name)
            continue
        total = orange_count = 0
        for r_idx in range(2, max_row + 1):
            total += 1
            if _is_orange(_cell_fill_rgb(ws.cell(row=r_idx, column=c_idx))):
                orange_count += 1
        if total and orange_count / total > 0.5:
            orange_columns.append(col_name)

    orange_rows = set()
    for r_idx in range(2, max_row + 1):
        for c_idx in range(1, max_col + 1):
            if _is_orange(_cell_fill_rgb(ws.cell(row=r_idx, column=c_idx))):
                orange_rows.add(r_idx - 2)  # 0-based data-row index
                break

    result["orange_columns"] = orange_columns
    result["orange_rows"] = orange_rows
    return result


_DEFAULT_VALID_TRANSITIONS = {
    "in force":       {"lapsed", "surrendered", "matured", "paid up", "paid up life", "claim"},
    "lapsed":         {"in force", "reinstated", "surrendered", "lapsed"},
    "reinstated":     {"in force"},
    "paid up":        {"in force", "surrendered", "matured", "claim"},
    "paid up life":   {"surrendered", "matured", "claim"},
    "surrendered":    set(),
    "matured":        set(),
    "claim":          set(),
}


def _parse_dob(val):
    if pd.isna(val):
        return None
    if isinstance(val, (datetime.datetime, datetime.date)):
        return pd.Timestamp(val)
    try:
        return pd.to_datetime(val, dayfirst=True, errors="coerce")
    except Exception:
        return None


def _build_rule_ops(df: pd.DataFrame, orange_info: dict) -> list[dict]:
    """
    The four explicitly-required checks, computed deterministically so they
    never depend on the LLM returning valid JSON:
      1. Non-mandatory columns (orange-highlighted in the workbook)
      2. Age < 18 computed from DOB
      3. Incorrect policy status movement (orange-highlighted rows and/or
         an unexpected PREV STATUSCODE -> current status transition)
      4. Missing values in DOB, SA, Gender, PREV STATUSCODE
    Each op uses the same schema as the LLM-proposed ops (id assigned by
    the caller) plus a "params" dict consumed by _apply_operation.
    """
    ops: list[dict] = []
    n = len(df)

    # 1) Non-mandatory (orange) columns
    orange_cols = [c for c in orange_info.get("orange_columns", []) if c in df.columns]
    if orange_cols:
        ops.append({
            "column": ", ".join(orange_cols),
            "issue": "Non-mandatory column(s), highlighted orange in the source file",
            "action": "drop_column",
            "params": {"columns": orange_cols},
            "rationale": "These columns are shaded orange in the workbook, indicating "
                         "they are not mandatory fields.",
        })

    # 2) Age < 18 via DOB
    dob_col = _find_col(df, "DOB", "Date of Birth", "DateOfBirth")
    if dob_col:
        today = pd.Timestamp(datetime.datetime.now().date())
        parsed = df[dob_col].apply(_parse_dob)
        ages = parsed.apply(lambda d: None if d is None or pd.isna(d) else (today - d).days // 365)
        underage_idx = [i for i, a in enumerate(ages) if a is not None and a < 18]
        if underage_idx:
            ops.append({
                "column": dob_col,
                "issue": f"{len(underage_idx)} row(s) computed as under 18 years old from {dob_col}",
                "action": "flag_rows",
                "params": {"row_indices": underage_idx, "flag_name": "UNDERAGE_FLAG"},
                "rationale": "Age is derived from DOB vs today's date; flagged for review "
                             "rather than auto-removed, since a data-entry error in DOB "
                             "(not an actual minor policyholder) may be the real cause.",
            })

    # 3) Incorrect policy status movement
    prev_status_col = _find_col(df, "PREV STATUSCODE", "Previous Status", "PrevStatus", "PREV_STATUSCODE")
    status_col = _find_col(df, "STATUSCODE", "Status", "PolicyStatus", "STATUS_CODE")
    bad_idx = set(orange_info.get("orange_rows", set()))

    if status_col and prev_status_col and status_col != prev_status_col:
        for i, (prev, cur) in enumerate(zip(df[prev_status_col], df[status_col])):
            if pd.isna(prev) or pd.isna(cur):
                continue
            p, c = str(prev).strip().lower(), str(cur).strip().lower()
            valid_next = _DEFAULT_VALID_TRANSITIONS.get(p)
            if valid_next is not None and c not in valid_next and c != p:
                bad_idx.add(i)

    if bad_idx:
        label = f"{prev_status_col or ''} -> {status_col or ''}".strip(" ->") or "policy status"
        ops.append({
            "column": label,
            "issue": f"{len(bad_idx)} row(s) show an incorrect policy status movement "
                     f"(workbook orange highlighting and/or an unexpected status transition)",
            "action": "flag_rows",
            "params": {"row_indices": sorted(bad_idx), "flag_name": "BAD_STATUS_MOVEMENT"},
            "rationale": "Flags rows where the source file highlights the status change in "
                         "orange, and/or where PREV STATUSCODE -> current status does not "
                         "match an expected forward progression.",
        })

    # 4) Missing values in DOB, SA, Gender, PREV STATUSCODE
    mandatory = {
        "DOB": dob_col,
        "SA": _find_col(df, "SA", "Sum Assured", "SumAssured"),
        "Gender": _find_col(df, "Gender", "Sex"),
        "PREV STATUSCODE": prev_status_col,
    }
    for label, col in mandatory.items():
        if not col:
            ops.append({
                "column": label,
                "issue": f"Mandatory field '{label}' was not found in this file",
                "action": "no_op",
                "params": {},
                "rationale": "Column not present in the uploaded file — flagged so it's "
                             "not silently missed.",
            })
            continue
        null_idx = df.index[df[col].isna()].tolist()
        if null_idx:
            ops.append({
                "column": col,
                "issue": f"{len(null_idx)} missing value(s) in mandatory field '{label}'",
                "action": "flag_rows",
                "params": {"row_indices": null_idx, "flag_name": f"MISSING_{label.replace(' ', '_')}"},
                "rationale": f"'{label}' is mandatory; blanks are flagged for manual "
                             f"completion rather than guessed/imputed.",
            })

    return ops


def _build_analysis_prompt(sheet_name: str, profile: dict, already_covered: list[str] | None = None) -> str:
    already_covered = already_covered or []
    skip_note = (
        f"\nThese columns are already handled by deterministic rule checks — do NOT "
        f"propose operations for them again: {json.dumps(already_covered)}\n"
        if already_covered else ""
    )
    return f"""You are a data-quality analyst reviewing a spreadsheet sheet called "{sheet_name}".

Dataset summary:
- Rows: {profile['n_rows']}
- Columns: {profile['n_cols']}
{skip_note}
Column details (JSON):
{json.dumps(profile['columns'], indent=2)}

Your task:
Produce a JSON array of proposed cleaning operations. Each element must have:
  "id"          : unique integer (1, 2, 3 …)
  "column"      : column name (or "__ALL__" for row-level operations)
  "issue"       : short description of the problem found
  "action"      : one of: "fill_forward", "fill_median", "fill_mode",
                  "fill_zero", "fill_value", "cap_outliers",
                  "flag_outliers", "drop_column", "drop_duplicates",
                  "strip_whitespace", "standardise_case"
  "params"      : object with any action-specific params
                  (e.g. for "fill_value": {{"value": "Unknown"}},
                   for "cap_outliers": {{"method": "iqr", "multiplier": 1.5}})
  "rationale"   : one sentence explaining why

Rules:
- Propose operations ONLY for columns that actually have issues.
- For numeric columns with >10% nulls, suggest "fill_median".
- For categorical columns with >10% nulls, suggest "fill_mode".
- For numeric columns with iqr_outliers > 0, suggest "cap_outliers".
- For columns where null_count == n_rows, suggest "drop_column".
- Always include a "drop_duplicates" operation on "__ALL__".
- Do NOT suggest dropping columns that have actual data.
- Return ONLY valid JSON — no markdown fences, no explanation outside the array.
"""


def _call_llm(prompt: str, max_output_tokens: int = 3000) -> str:
    if _gemini_client is None:
        raise RuntimeError("Cleaning service not initialised — call init_cleaning() first.")
    response = _gemini_client.models.generate_content(
        model=_gemini_model,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config={"max_output_tokens": max_output_tokens},
    )
    return response.text or ""


def _repair_json_array(raw: str):
    """
    Best-effort repair of a near-valid JSON array returned by the LLM.
    Handles: markdown fences, trailing commas, an unterminated trailing
    string, and a response truncated mid-object (falls back to the last
    complete object in the array, or drops the dangling partial field).
    Returns a parsed list, or None if nothing usable could be salvaged.
    """
    text = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    match = re.search(r"\[.*", text, re.DOTALL)  # from first [ to end (may be truncated)
    if match:
        text = match.group()

    def try_load(t):
        try:
            parsed = json.loads(t)
            return parsed if isinstance(parsed, list) else None
        except Exception:
            return None

    parsed = try_load(text)
    if parsed is not None:
        return parsed

    # Trailing commas before ] or }
    fixed = re.sub(r",\s*([\]}])", r"\1", text)
    parsed = try_load(fixed)
    if parsed is not None:
        return parsed

    # Odd number of unescaped quotes -> close the dangling string
    if len(re.findall(r'(?<!\\)"', fixed)) % 2 == 1:
        closed = fixed + '"'
        parsed = try_load(closed)
        if parsed is not None:
            return parsed

    # Truncated mid-object: cut back to the last complete "},", close the array
    last_complete = fixed.rfind("},")
    if last_complete != -1:
        candidate = fixed[: last_complete + 1] + "]"
        parsed = try_load(candidate)
        if parsed is not None:
            return parsed

    # Last resort: truncated inside the very first object with no earlier
    # complete item to fall back to. Drop the dangling partial "key": "val
    # fragment, close braces/brackets, try once more.
    candidate = fixed
    if len(re.findall(r'(?<!\\)"', candidate)) % 2 == 1:
        candidate += '"'
    candidate = re.sub(r',?\s*"[^"]*"\s*:\s*"[^"]*$', "", candidate)
    opens_cu = candidate.count("{") - candidate.count("}")
    opens_sq = candidate.count("[") - candidate.count("]")
    candidate += "}" * max(opens_cu, 0)
    candidate += "]" * max(opens_sq, 0)
    return try_load(candidate)


def _parse_plan(raw: str) -> list[dict]:
    """Extract a JSON array from the LLM response robustly."""
    parsed = _repair_json_array(raw)
    if parsed is None:
        raise ValueError(
            "Could not parse a JSON array out of the LLM response, even after repair."
        )
    return parsed


def _call_llm_plan_with_retry(prompt: str, retries: int = 1) -> tuple[list[dict], str | None]:
    """
    Calls the LLM and parses its plan, repairing near-valid JSON. If parsing
    still fails, retries once with a stricter follow-up prompt. Returns
    (plan, warning) — plan is [] and warning is set if nothing could be
    salvaged, so callers can degrade gracefully instead of hard-failing.
    """
    current_prompt = prompt
    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = _call_llm(current_prompt)
        except Exception as exc:
            last_err = str(exc)
            break
        try:
            return _parse_plan(raw), None
        except Exception as exc:
            last_err = str(exc)
            current_prompt = (
                prompt
                + "\n\nIMPORTANT: Your previous reply could not be parsed as JSON "
                  "(" + str(exc) + "). Reply with ONLY a single valid JSON array. "
                  "No markdown fences, no commentary before or after, no trailing "
                  "commas, and make sure every string and object is properly closed."
            )
    return [], last_err


def _apply_operation(df: pd.DataFrame, op: dict) -> tuple[pd.DataFrame, str]:
    """Apply one cleaning operation; return modified df and a log message."""
    col    = op["column"]
    action = op["action"]
    params = op.get("params", {})
    log    = f"[op {op['id']}] {action} on {col}"

    if action == "drop_duplicates":
        before = len(df)
        df = df.drop_duplicates()
        log += f": removed {before - len(df)} duplicate rows"

    elif action == "strip_whitespace":
        if col in df.columns and (pd.api.types.is_object_dtype(df[col]) or
                                   pd.api.types.is_string_dtype(df[col])):
            df[col] = df[col].str.strip()

    elif action == "standardise_case":
        if col in df.columns:
            case = params.get("case", "title")
            if case == "upper":
                df[col] = df[col].str.upper()
            elif case == "lower":
                df[col] = df[col].str.lower()
            else:
                df[col] = df[col].str.title()

    elif action == "fill_forward":
        if col in df.columns:
            df[col] = df[col].ffill()

    elif action == "fill_median":
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].fillna(median)
            log += f" (median={round(median, 4)})"

    elif action == "fill_mode":
        if col in df.columns:
            mode_vals = df[col].mode(dropna=True)
            if len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals.iloc[0])
                log += f" (mode={mode_vals.iloc[0]})"

    elif action == "fill_zero":
        if col in df.columns:
            df[col] = df[col].fillna(0)

    elif action == "fill_value":
        if col in df.columns:
            df[col] = df[col].fillna(params.get("value", "Unknown"))

    elif action == "cap_outliers":
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mult = params.get("multiplier", 1.5)
            q1   = df[col].quantile(0.25)
            q3   = df[col].quantile(0.75)
            iqr  = q3 - q1
            lower, upper = q1 - mult * iqr, q3 + mult * iqr
            before = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            log += f": capped {before} values to [{round(lower,2)}, {round(upper,2)}]"

    elif action == "flag_outliers":
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mult  = params.get("multiplier", 1.5)
            q1    = df[col].quantile(0.25)
            q3    = df[col].quantile(0.75)
            iqr   = q3 - q1
            flag_col = f"{col}_OUTLIER_FLAG"
            df[flag_col] = ((df[col] < q1 - mult * iqr) | (df[col] > q3 + mult * iqr))

    elif action == "drop_column":
        cols_to_drop = params.get("columns") or [c.strip() for c in str(col).split(",")]
        existing = [c for c in cols_to_drop if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            log += f": dropped {', '.join(existing)}"

    elif action == "flag_rows":
        flag_col = params.get("flag_name", f"{col}_FLAG")
        row_indices = params.get("row_indices", [])
        if flag_col not in df.columns:
            df[flag_col] = False
        valid_idx = [i for i in row_indices if i in df.index]
        df.loc[valid_idx, flag_col] = True
        log += f": flagged {len(valid_idx)} row(s) in new column '{flag_col}'"

    elif action == "no_op":
        log += f": no action taken ({op.get('issue', '')})"

    return df, log


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@data_cleaning_bp.route("/clean/analyse", methods=["POST"])
def analyse():
    """
    POST /clean/analyse
    Form-data:  file=<xlsx|xls|csv>  [sheet=<sheet_name>]
    Returns:    { sheet, profile, plan: [...], warnings: [...] }

    Read-only: nothing is written to the file here. The plan is only
    proposed for the user to review/tick; /clean/apply is what actually
    changes anything, and only for ops the user accepted.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded — send as multipart/form-data field 'file'"}), 400

    file_storage = request.files["file"]
    filename = file_storage.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    file_bytes = file_storage.read()

    warnings: list[str] = []

    if ext == "csv":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as exc:
            return jsonify({"error": f"Could not read CSV: {exc}"}), 400
        sheets = ["Sheet1"]
        sheet_name = "Sheet1"
        orange_info = {"orange_columns": [], "orange_rows": set(), "supported": False}
        warnings.append(
            "CSV files carry no cell formatting, so the orange-highlighted "
            "column/row checks were skipped for this file."
        )
    elif ext in ("xlsx", "xls", "xlsm"):
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
        except Exception as exc:
            return jsonify({"error": f"Could not read spreadsheet: {exc}"}), 400
        sheets = xls.sheet_names

        requested = request.form.get("sheet", "")
        sheet_name = requested if requested in sheets else sheets[-1]
        if not requested:
            data_sheets = [s for s in sheets if "agent" not in s.lower()
                           and "todo" not in s.lower() and "to do" not in s.lower()]
            sheet_name = data_sheets[0] if data_sheets else sheets[0]

        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
        orange_info = _detect_orange_formatting(file_bytes, sheet_name) if ext == "xlsx" else \
            {"orange_columns": [], "orange_rows": set(), "supported": False}
        if ext != "xlsx" or not orange_info.get("supported"):
            warnings.append(
                "Could not read cell formatting from this workbook; orange-based "
                "checks were skipped."
            )
    else:
        return jsonify({"error": f"Unsupported file type '.{ext}'. Please upload .csv, .xlsx or .xls."}), 400

    profile = _profile_dataframe(df)

    # Deterministic checks first — these never depend on the LLM.
    rule_ops = _build_rule_ops(df, orange_info)
    already_covered = sorted({op["column"] for op in rule_ops if op["action"] != "no_op"})

    prompt = _build_analysis_prompt(sheet_name, profile, already_covered)
    llm_ops, llm_warning = _call_llm_plan_with_retry(prompt, retries=1)
    if llm_warning:
        warnings.append(f"LLM suggestions unavailable ({llm_warning}); showing "
                         f"rule-based findings only.")

    # Assign sequential integer ids across both sources (frontend expects op.id: Number)
    plan = []
    for i, op in enumerate(rule_ops + llm_ops, start=1):
        op = dict(op)
        op["id"] = i
        op.setdefault("params", {})
        plan.append(op)

    return jsonify({
        "sheet":       sheet_name,
        "all_sheets":  sheets,
        "n_rows":      profile["n_rows"],
        "n_cols":      profile["n_cols"],
        "profile":     profile["columns"],
        "plan":        plan,
        "warnings":    warnings,
        "message":     "Here is what I found. Tick the changes you'd like me to make, "
                       "then click Apply.",
    })


@data_cleaning_bp.route("/clean/apply", methods=["POST"])
def apply_plan():
    """
    POST /clean/apply
    Form-data:  file=<xlsx>  sheet=<sheet_name>  plan=<JSON array str>
    Returns:    cleaned XLSX file as attachment + JSON summary header X-Clean-Log
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    if "plan" not in request.form:
        return jsonify({"error": "No plan provided"}), 400

    file_storage = request.files["file"]
    filename_in = file_storage.filename or ""
    ext = filename_in.rsplit(".", 1)[-1].lower() if "." in filename_in else ""
    file_bytes = file_storage.read()
    sheet_name = request.form.get("sheet", "") or "Sheet1"
    plan_str   = request.form.get("plan", "[]")

    try:
        plan: list[dict] = json.loads(plan_str)
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Invalid plan JSON: {exc}"}), 400

    is_csv = ext == "csv"
    if is_csv:
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)

    logs: list[str] = []
    for op in plan:
        try:
            df, msg = _apply_operation(df, op)
            logs.append(msg)
        except Exception as exc:
            logs.append(f"[op {op.get('id','?')}] FAILED: {exc}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_buffer = io.BytesIO()

    if is_csv:
        df.to_csv(out_buffer, index=False)
        out_buffer.seek(0)
        mimetype = "text/csv"
        filename = f"cleaned_{timestamp}.csv"
    else:
        with pd.ExcelWriter(out_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        out_buffer.seek(0)
        mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"cleaned_{sheet_name.replace(' ','_')}_{timestamp}.xlsx"

    response = send_file(
        out_buffer,
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename,
    )
    response.headers["X-Clean-Log"]   = " | ".join(logs)
    response.headers["X-Clean-Sheet"] = sheet_name
    response.headers["Access-Control-Expose-Headers"] = "X-Clean-Log, X-Clean-Sheet"
    return response