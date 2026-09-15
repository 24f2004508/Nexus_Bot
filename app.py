"""
app.py  —  GenAI Concepts Lab · Flask Backend
==============================================
Exposes the guardrail logic from 01_genai_basics.py as a REST API.
The frontend (index.html) calls POST /chat for every user message.

Run:
    pip install flask flask-cors anthropic python-dotenv
    export ANTHROPIC_API_KEY=sk-ant-...
    python app.py

Endpoints:
    POST /chat          — main chat endpoint
    GET  /log           — download the CSV call log
    GET  /health        — liveness check
"""

import csv
import datetime
import json
import os
import pathlib
import re
from pathlib import Path

#import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from google import genai
from pricing_service import (
    calculate_ip_premium,
    calculate_pmi_premium,
    model_diagnostics,
    route_query,
    search_policy_docs,
)

# ─────────────────────────────────────────────
# §1 · Setup  (mirrors 01_genai_basics.py §1)
# ─────────────────────────────────────────────
#PROJECT_ROOT = Path(__file__).resolve().parent
#ENV_FILE     = PROJECT_ROOT / ".env"
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = Path(os.getenv("LOG_PATH", BASE_DIR / "genai_call_log.csv"))
PROJECT_ROOT = BASE_DIR.parent

# Project root

ENV_FILE = PROJECT_ROOT / ".env"

# Load .env
load_dotenv(ENV_FILE, override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#load_dotenv(ENV_FILE, override=True)

#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not set. "
        f"Add it to {ENV_FILE} or export it in your shell."
    )

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL  = "gemini-3.5-flash-lite"

# ─────────────────────────────────────────────
# §1.1 · Concept labels  (same keys as the frontend MODES object)
# ─────────────────────────────────────────────
CONCEPT_LABELS: dict[str, str] = {
    "free":          "FREE CHAT",
    "ccce":          "CCCE PROMPTING",
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
        "You are a prompt-engineering tutor specialising in actuarial and insurance topics. "
        "When the user gives a topic or rough question, do TWO things:\n"
        "1. Show a well-structured CCCE prompt for it, clearly labelling "
        "[Clarity], [Context], [Constraints], [Examples].\n"
        "2. Execute that prompt yourself and show the model response.\n"
        "Separate the two with a clear header. Keep the example voice actuarial/professional."
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
# §2-§5 · Core call function  (mirrors call() in 01_genai_basics.py)
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

    contents = []

    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Gemini uses "user" and "model"
        gemini_role = "model" if role == "assistant" else "user"

        contents.append({
            "role": gemini_role,
            "parts": [
                {"text": content}
            ],
        })

    # Add current user message
    contents.append({
        "role": "user",
        "parts": [
            {"text": prompt}
        ],
    })

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config={
            "system_instruction": sys_prompt,
            "max_output_tokens": 1200,
        },
    )

    reply = response.text or ""

    usage = response.usage_metadata

    tokens_in = (
        usage.prompt_token_count
        if usage and usage.prompt_token_count
        else 0
    )

    tokens_out = (
        usage.candidates_token_count
        if usage and usage.candidates_token_count
        else 0
    )
    return reply, tokens_in, tokens_out


# ─────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
CORS(app, origins=CORS_ORIGINS)


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
          "label":       "CCCE PROMPTING",
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

    if mode not in CONCEPT_LABELS:
        return jsonify({"error": f"unknown mode '{mode}'"}), 400

    # Sanitise history — only keep valid role/content pairs
    clean_history = [
        {"role": h["role"], "content": h["content"]}
        for h in history
        if isinstance(h, dict)
        and h.get("role") in ("user", "assistant")
        and isinstance(h.get("content"), str)
    ]

    try:
        reply, tok_in, tok_out = guardrail_call(prompt, mode, clean_history)
    #except anthropic.APIStatusError as exc:
    #    return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

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
