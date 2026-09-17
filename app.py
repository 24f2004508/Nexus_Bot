"""
app.py  —  GenAI Concepts Lab · Flask Backend
==============================================
Exposes the guardrail logic from 01_genai_basics.py as a REST API.
The frontend (index.html) calls POST /chat for every user message.

Run:
    pip install -r requirements.txt
    export GOOGLE_API_KEY=...
    python app.py

Endpoints:
    POST /chat          — main chat endpoint
    GET  /log           — download the CSV call log
    GET  /health        — liveness check
"""

import csv
import datetime
import os
from pathlib import Path

from data_cleaning import data_cleaning_bp, init_cleaning


from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from agno.agent import Agent
from agno.models.google import Gemini
from agno.run.base import RunStatus
from pricing_service import (
    calculate_ip_premium,
    calculate_pmi_premium,
    model_diagnostics,
    route_query,
    search_policy_docs,
)

# ─────────────────────────────────────────────
# §1 · Setup
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = Path(os.getenv("LOG_PATH", BASE_DIR / "genai_call_log.csv"))
PROJECT_ROOT = BASE_DIR.parent

# Project root

ENV_FILE = BASE_DIR / ".env"

# Load .env (do not override existing environment variables)
load_dotenv(ENV_FILE, override=False)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not set. "
        f"Add it to {ENV_FILE} or export it in your shell."
    )

# CFG-01: single source of truth for the active Gemini model. Chat Lab
# (guardrail_call) and the Data Cleaning Agent (init_cleaning -> cleaning_agent.
# build_agent) both receive this same MODEL value — there is nowhere else in
# the live app that names a model. Override via GEMINI_MODEL for local testing
# or a future model migration without a code change. The standalone teaching
# scripts (01_genai_basics.py, 02_models_as_tools.py, 05_pricing_team.py) are
# intentionally independent of this — they are seminar examples, not part of
# the running app, and may reference a different model on purpose.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# PERF-01: bounded operational limits, all overridable via env vars so a
# deployment can tune them without a code change.
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "8000"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "40"))
MAX_HISTORY_TURN_CHARS = int(os.getenv("MAX_HISTORY_TURN_CHARS", "8000"))
MODEL_TIMEOUT_MS = int(os.getenv("MODEL_TIMEOUT_MS", "30000"))
MODEL_MAX_ATTEMPTS = int(os.getenv("MODEL_MAX_ATTEMPTS", "3"))  # 1 try + 2 bounded retries

# Create the AGNO Gemini model, talking to the official Google Gemini API.
model = Gemini(
    id=MODEL,
    api_key=GOOGLE_API_KEY,
    timeout=MODEL_TIMEOUT_MS / 1000,
    retries=MODEL_MAX_ATTEMPTS - 1,
)

# ─────────────────────────────────────────────
# §1.1 · Concept labels  (same keys as the frontend MODES object)
# ─────────────────────────────────────────────
CONCEPT_LABELS: dict[str, str] = {
    "free":          "CHAT",
    "ccce":          "PROMPT CREATOR",
    "vague":         "VAGUE vs SPECIFIC",
    "audience":      "MULTI-AUDIENCE",
    "fewshot":       "FEW-SHOT FORMATTING",
    "cot":           "CHAIN-OF-THOUGHT",
    "hallucination": "HALLUCINATION RISK",
    "json":          "STRUCTURED OUTPUT",
}

# ─────────────────────────────────────────────
# §1.2 · Per-mode system prompts
#        These ARE the guardrails — they constrain exactly how the LLM
#        behaves for each sidebar mode, mirroring the JS MODES[mode].system
# ─────────────────────────────────────────────
SYSTEM_PROMPTS: dict[str, str] = {

    # §2 — basic call
    "free": (
        "You are an expert in both Generative AI and actuarial science "
        "(health insurance, PMI, IBNR reserves). Answer clearly and helpfully. "
        "When relevant, relate answers to Indian insurance concepts "
        "(IRDAI, INR, PMI, IBNR). Never invent regulatory figures."
    ),

    # §3 — CCCE
    "ccce": (
        "You are a CCCE Prompt-Builder — a general-purpose agent that helps anyone turn a "
        "rough task or question, in ANY domain, into a well-structured prompt using the "
        "CCCE framework (Clarity, Context, Constraints, Examples).\n"
        "When the user describes a task:\n"
        "1. If key details are missing (audience, format, tone, length, domain facts, "
        "desired output), ask up to 3 short, specific clarifying questions before drafting.\n"
        "2. Once you have enough to work with, output the finished prompt, clearly labelled "
        "with [Clarity], [Context], [Constraints], [Examples] sections, ready to paste into "
        "any LLM.\n"
        "3. After the prompt, add one short line noting any assumption you made to fill a gap.\n"
        "Do not answer the user's underlying task yourself — your job is to build the prompt, "
        "not to execute it."
    ),

    # §3.1 — vague vs specific
    "vague": (
        "You are demonstrating the impact of prompt specificity on output quality.\n"
        "When the user gives a topic, respond with TWO clearly separated sections:\n"
        "SECTION A — VAGUE PROMPT: show the vague prompt text, then the likely weak response.\n"
        "SECTION B — SPECIFIC PROMPT: show a detailed prompt, then a much stronger response.\n"
        "Use actuarial / health-insurance subject matter. "
        "Highlight in one sentence what makes the specific version better."
    ),

    # §3.2 — multi-audience
    "audience": (
        "You are an expert at translating actuarial facts for different audiences.\n"
        "When given a fact, produce TWO labelled responses:\n"
        "FOR THE BOARD: 2 sentences, business impact first, no jargon, INR figures prominent.\n"
        "FOR A NEW ACTUARIAL STUDENT: 4 sentences, explain the underlying mechanics, define technical terms.\n"
        "Make the contrast stark and educational."
    ),

    # §3.3 — few-shot
    "fewshot": (
        "You are demonstrating few-shot prompting for consistent output formatting.\n"
        "Always respond by first showing the few-shot prompt (with EXAMPLES section and IN/OUT pairs), "
        "then showing the formatted output.\n"
        "Use this exact output format for actuarial changes:\n"
        "  METRIC_CODE | segment | before -> after  (or +/- delta)\n"
        "Metric codes: HOSP_FREQ, SEV_TREND, IBNR_ULT, CI_INCIDENCE, NCB_FACTOR, PREM_RATE.\n"
        "Explain briefly why few-shot examples constrain the format."
    ),

    # §3.4 — chain-of-thought
    "cot": (
        "You are an actuarial calculation assistant using chain-of-thought (step-by-step) reasoning.\n"
        "When asked to calculate, ALWAYS:\n"
        "1. List all given inputs.\n"
        "2. Show each step with a label and running total.\n"
        "3. State the final answer clearly.\n"
        "4. Note any assumptions.\n"
        "Never skip steps. Use INR and Indian insurance terminology."
    ),

    # §4 — hallucination risk
    "hallucination": (
        "You are demonstrating hallucination risk in LLMs for actuarial users.\n"
        "When asked about specific regulatory figures, section numbers, or published data:\n"
        "1. Give the response the model would typically generate (which may be hallucinated).\n"
        "2. Add a clear WARNING block explaining: "
        "(a) this may not be accurate, "
        "(b) the model cannot access live IRDAI circulars, "
        "(c) how to verify — irdai.gov.in, official circulars, appointed actuary guidance.\n"
        "Be explicit: confident-sounding output is NOT the same as correct output. "
        "Never present invented section references as real without the warning."
    ),

    # §5 — structured JSON output
    "json": (
        "You are demonstrating structured (JSON) output from LLMs for actuarial applications.\n"
        "When the user asks for data:\n"
        "1. Give a one-sentence description of the schema.\n"
        "2. Return a valid, pretty-printed JSON block (wrap in ```json fences).\n"
        "3. Add a one-line note on how this JSON could be consumed "
        "(e.g. pandas DataFrame, dashboard, downstream API).\n"
        "Use realistic actuarial values — rating factors, IBNR components, claim metrics."
    ),
}


# ─────────────────────────────────────────────
# §6 · Call log  (mirrors 01_genai_basics.py §6)
# ─────────────────────────────────────────────
def log_call(
    mode: str,
    prompt_text: str,
    response_text: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    """Append one row to genai_call_log.csv."""
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["ts_utc", "mode", "model", "tokens_in", "tokens_out", "prompt", "response"]
            )
        writer.writerow([
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            mode,
            MODEL,
            tokens_in,
            tokens_out,
            prompt_text,
            response_text,
        ])


# ─────────────────────────────────────────────
# §2-§5 · Core call function
# ─────────────────────────────────────────────
def guardrail_call(
    prompt: str,
    mode: str,
    history: list[dict],
) -> tuple[str, int, int]:
    """
    Send a message through the mode-specific guardrail system prompt.
    Returns (reply_text, input_tokens, output_tokens).
    """
    if mode not in SYSTEM_PROMPTS:
        mode = "free"

    sys_prompt = SYSTEM_PROMPTS[mode]

    # Fold prior turns into the input text — `model` (agno.models.google.Gemini)
    # has no .run(); AGNO's Agent is the orchestration layer that provides it, so a
    # lightweight, tool-less Agent per call is used here (same run() call cleaning_agent
    # uses for the Data Cleaning Agent), with the mode's system prompt as instructions.
    history_block = ""
    if history:
        lines = [f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history]
        history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    chat_agent = Agent(model=model, instructions=sys_prompt, markdown=True)
    response = chat_agent.run(f"{history_block}User: {prompt}")

    # PROVIDER ERROR HANDLING: after AGNO's own bounded retries are exhausted on a
    # transient provider failure (rate limit, upstream overload, invalid model, etc.),
    # RunOutput.status is ERROR and .content holds the raw provider error text — that
    # must never be surfaced to the user as if it were the assistant's reply.
    if getattr(response, "status", None) == RunStatus.error:
        raise RuntimeError(f"model provider error: {response.content}")

    reply = response.content if hasattr(response, "content") else str(response)

    # Extract token usage if available
    tokens_in = 0
    tokens_out = 0
    metrics = getattr(response, "metrics", None)
    if metrics:
        tokens_in = getattr(metrics, "input_tokens", 0) or 0
        tokens_out = getattr(metrics, "output_tokens", 0) or 0

    return reply, tokens_in, tokens_out


# ─────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
CORS(app, origins=CORS_ORIGINS)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024


@app.errorhandler(413)
def _handle_too_large(_exc):
    return jsonify({"error": "Request too large."}), 413


@app.errorhandler(Exception)
def _handle_uncaught(exc):
    """ERROR HANDLING backstop: every route below has its own try/except with a
    safe message, but this guarantees that even an exception no one anticipated
    still reaches the client as a controlled JSON error, never a stack trace,
    file path, or other internal detail — full detail still goes to the
    server log."""
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.description or exc.name}), exc.code
    app.logger.exception("unhandled exception")
    return jsonify({"error": "Internal server error."}), 500


# right after `app = Flask(__name__)` and `CORS(app, ...)`
app.register_blueprint(data_cleaning_bp)
init_cleaning(model, MODEL)   # passes the AGNO model and model id string

@app.route("/")
def index():
    """Serve the chat UI from the same process as the API."""
    return send_file(Path(__file__).with_name("index.html"))


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL})


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Body (JSON):
        {
          "mode":    "ccce",           // one of CONCEPT_LABELS keys
          "prompt":  "user message",
          "history": [                 // optional prior turns
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
          ]
        }

    Response (JSON):
        {
          "reply":       "assistant text",
          "mode":        "ccce",
          "label":       "PROMPT CREATOR",
          "tokens_in":   312,
          "tokens_out":  87,
          "logged":      true
        }
    """
    data = request.get_json(force=True, silent=True) or {}

    mode    = data.get("mode", "free")
    prompt  = (data.get("prompt") or "").strip()
    history = data.get("history", [])

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # PERF-01: bound the prompt and the conversation history sent to the model —
    # both for cost/latency and so a single request can't grow unbounded.
    if len(prompt) > MAX_PROMPT_CHARS:
        return jsonify({"error": f"prompt is too long (max {MAX_PROMPT_CHARS} characters)"}), 400

    if mode not in CONCEPT_LABELS:
        return jsonify({"error": f"unknown mode '{mode}'"}), 400

    if not isinstance(history, list):
        return jsonify({"error": "history must be a list"}), 400

    # Sanitise history — only keep valid role/content pairs, cap each turn's
    # length, and keep only the most recent MAX_HISTORY_TURNS turns.
    clean_history = [
        {"role": h["role"], "content": h["content"][:MAX_HISTORY_TURN_CHARS]}
        for h in history
        if isinstance(h, dict)
        and h.get("role") in ("user", "assistant")
        and isinstance(h.get("content"), str)
    ][-MAX_HISTORY_TURNS:]

    try:
        reply, tok_in, tok_out = guardrail_call(prompt, mode, clean_history)
    except Exception:
        # ERROR HANDLING: log full detail server-side, never leak SDK/internal
        # exception text (which could include request internals) to the client.
        app.logger.exception("guardrail_call failed for mode=%s", mode)
        return jsonify({"error": "The assistant is temporarily unavailable. Please try again."}), 502

    log_call(
        mode=mode,
        prompt_text=prompt,
        response_text=reply,
        tokens_in=tok_in,
        tokens_out=tok_out,
    )

    return jsonify({
        "reply":      reply,
        "mode":       mode,
        "label":      CONCEPT_LABELS[mode],
        "tokens_in":  tok_in,
        "tokens_out": tok_out,
        "logged":     True,
    })


@app.route("/pricing/models")
def pricing_models():
    """GET /pricing/models - model cards, leakage guard, lift and fairness outputs."""
    return jsonify(model_diagnostics())


@app.route("/pricing/ip", methods=["POST"])
def pricing_ip():
    """POST /pricing/ip - governed Income Protection quote."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        quote = calculate_ip_premium(
            age=int(data.get("age")),
            monthly_income=float(data.get("monthly_income")),
            prior_episodes=int(data.get("prior_episodes", 0)),
            occupation=str(data.get("occupation", "desk")),
            deferred_weeks=int(data.get("deferred_weeks", 13)),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"quote": quote, "tool": "calculate_premium", "governed": True})


@app.route("/pricing/pmi", methods=["POST"])
def pricing_pmi():
    """POST /pricing/pmi - governed PMI frequency x severity quote."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        quote = calculate_pmi_premium(
            age=int(data.get("age")),
            sum_insured=int(data.get("sum_insured")),
            ncb_tier=int(data.get("ncb_tier", 0)),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"quote": quote, "tool": "calculate_pmi_premium", "governed": True})


@app.route("/pricing/team", methods=["POST"])
def pricing_team():
    """POST /pricing/team - route a pricing-desk question to a specialist or policy RAG."""
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    route = route_query(query)
    if route == "policy":
        tool_result = search_policy_docs(query)
        context = f"Policy retrieval result from {tool_result['source']}:\n{tool_result['passage']}"
        tool_name = "search_policy_docs"
    else:
        tool_result = {"route": route, "instruction": "Collect the required quote fields in the workbench form."}
        context = "No quote fields were supplied. Ask the user for the fields needed by the selected specialist."
        tool_name = "IP Pricing Agent" if route == "ip" else "PMI Pricing Agent"

    prompt = (
        "You are the ABC Health Pricing Desk. Route label: " + route.upper() + "\n"
        "Use only the governed tool context below. Do not invent figures. Explain that a qualified human signs pricing work.\n\n"
        + context + "\n\nUser question: " + query
    )
    try:
        reply, tok_in, tok_out = guardrail_call(prompt, "free", [])
    except Exception:
        reply = context
        tok_in, tok_out = 0, 0

    log_call("pricing_team", query, reply, tok_in, tok_out)
    return jsonify({
        "reply": reply,
        "route": route,
        "tool": tool_name,
        "tool_result": tool_result,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "logged": True,
    })


@app.route("/log")
def download_log():
    """GET /log — download the CSV call log."""
    if not LOG_PATH.exists():
        return jsonify({"error": "No log yet"}), 404
    return send_file(LOG_PATH, mimetype="text/csv", as_attachment=True,
                     download_name="genai_call_log.csv")


if __name__ == "__main__":
    print("=" * 60)
    print("GenAI Concepts Lab — Flask backend")
    print(f"Model  : {MODEL}")
    print(f"Log    : {LOG_PATH}")
    print("Serving: http://localhost:5000")
    print("=" * 60)
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
    )