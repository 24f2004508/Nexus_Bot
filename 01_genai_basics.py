"""
01_genai_basics.py  —  GenAI Concepts Lab · Actuarial Edition
==============================================================
ROLE OF THIS FILE
-----------------
This is the *companion demo script* for the GenAI Concepts Lab.
It is NOT the web server.  The web server is app.py.

Architecture:
  ┌──────────────┐   HTTP POST /chat   ┌──────────────┐   Anthropic API
  │  index.html  │ ──────────────────► │   app.py     │ ──────────────►
  │  (browser)   │ ◄────────────────── │ (Flask + *guardrails*) │
  └──────────────┘                     └──────────────┘
                                              │  shares
                                              ▼
                                    01_genai_basics.py
                                    (SYSTEM_PROMPTS, call(),
                                     log_call() — the guardrails)

Each section below maps 1-to-1 to a sidebar mode in index.html
and to a key in SYSTEM_PROMPTS in app.py:

  §1   Setup
  §2   Free Chat           — basic generate_content call
  §3   CCCE Prompting      — Clarity · Context · Constraints · Examples
  §3.1 Vague vs Specific   — prompt-quality comparison
  §3.2 Multi-Audience      — same fact, two voices
  §3.3 Few-Shot Formatting — examples tame output format
  §3.4 Chain-of-Thought    — step-by-step reasoning
  §4   Hallucination Risk  — verify before you trust
  §5   Structured Output   — JSON schema mode
  §6   Call Log            — minimal audit trail (CSV)
  §7   Multi-turn          — conversation history

Run as a standalone demo:
  export ANTHROPIC_API_KEY=sk-ant-...
  python 01_genai_basics.py

Run the web app instead:
  python app.py          # starts Flask on :5000
  open index.html        # open in browser
"""

# ─────────────────────────────────────────────────────────────────────
# §1 · Setup
# ─────────────────────────────────────────────────────────────────────
import csv
import datetime
import json
import os
import pathlib
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE     = PROJECT_ROOT / ".env"
BASE_DIR     = Path(__file__).resolve().parent      # call log lives here

# ── Environment ────────────────────────────────────────────────────
load_dotenv(ENV_FILE, override=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY not set.\n"
        f"Add it to {ENV_FILE} or:  export ANTHROPIC_API_KEY=sk-ant-..."
    )

# ── Client & model — same as app.py and index.html ─────────────────
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL  = "claude-sonnet-4-6"

print("=" * 70)
print("GenAI Concepts Lab  —  Actuarial Edition  (standalone demo)")
print(f"Client ready · model pinned to: {MODEL}")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────
# §1.1 · Concept labels
#        Same keys as CONCEPT_LABELS in app.py  &  MODES in index.html
# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
# §1.2 · Per-mode system prompts  ← THE GUARDRAILS
#        Identical to SYSTEM_PROMPTS in app.py.
#        These constrain LLM behaviour for each concept mode.
#        When index.html calls POST /chat, app.py picks the right
#        system prompt from this same dictionary before forwarding
#        to the Anthropic API.
# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
# §1.3 · Utility helpers
# ─────────────────────────────────────────────────────────────────────
def show_response(text: str, mode: str = "free") -> None:
    """Print a labelled response block — mirrors the chat bubble in index.html."""
    label = CONCEPT_LABELS.get(mode, mode.upper())
    print(f"\n{'=' * 70}")
    print(f"📋  [{label}]")
    print("=" * 70)
    print(text)
    print(f"{'─' * 70}")
    print("END OF RESPONSE")
    print("=" * 70)


def call(
    prompt: str,
    *,
    mode: str = "free",
    history: list[dict] | None = None,
) -> str:
    """
    Send a prompt through the mode guardrail and return the reply text.
    This is the same logic that app.py's guardrail_call() uses —
    they both read from SYSTEM_PROMPTS above.
    """
    sys_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["free"])
    messages   = list(history or []) + [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        system=sys_prompt,
        messages=messages,
    )

    reply = response.content[0].text
    tok   = response.usage
    print(f"   ↑ {tok.input_tokens} in · ↓ {tok.output_tokens} out")
    return reply


# ─────────────────────────────────────────────────────────────────────
# §6 · Call log  — shared with app.py (same CSV file)
# ─────────────────────────────────────────────────────────────────────
LOG_PATH = BASE_DIR / "genai_call_log.csv"
#Promt: take understanding of full project create  flowchart of full project,create a plan to test and plan the project highlight areas where optimization is needed, identify the bugs in the project and identify all the edge cases for bugs in this project. For lower (small llm) can understand prompt that it can understand
#flaw wala data to be fileed by llm , flaws only 20%
#Generate data with anomalies , 20% o/w data qualty poor, tool to be added: data cleaning via llm
#pnitin037@gmail.com

def log_call(
    mode: str,
    prompt_text: str,
    response_text: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    path: pathlib.Path = LOG_PATH,
) -> None:
    """
    Append one row to genai_call_log.csv.
    Columns: ts_utc | mode | model | tokens_in | tokens_out | prompt | response

    app.py calls this same function for every /chat request from index.html.
    Running this script appends to the same log file, so both sources
    appear in the same audit trail (downloadable via GET /log in app.py).
    """
    path = pathlib.Path(path)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["ts_utc", "mode", "model", "tokens_in", "tokens_out", "prompt", "response"]
            )
        writer.writerow([
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            mode, MODEL, tokens_in, tokens_out, prompt_text, response_text,
        ])


# ─────────────────────────────────────────────────────────────────────
# §2 · Free Chat  →  sidebar: "Free Chat"
# ─────────────────────────────────────────────────────────────────────
print("\n\n### §2 · FREE CHAT ###")

prompt = "Define IBNR for a non-actuarial board member, in one line."
reply  = call(prompt, mode="free")
show_response(reply, mode="free")
log_call("free", prompt, reply)


# ─────────────────────────────────────────────────────────────────────
# §3 · CCCE Prompting  →  sidebar: "CCCE Prompting"
# ─────────────────────────────────────────────────────────────────────
print("\n\n### §3 · CCCE PROMPTING ###")

ccce_topic = "IBNR commentary for ABC Health Q3 2024 board deck"
reply = call(ccce_topic, mode="ccce")
show_response(reply, mode="ccce")
log_call("ccce", ccce_topic, reply)


# ─────────────────────────────────────────────────────────────────────
# §3.1 · Vague vs Specific  →  sidebar: "Vague vs. Specific"
# ─────────────────────────────────────────────────────────────────────
print("\n\n### §3.1 · VAGUE vs SPECIFIC ###")

topic = "IBNR reserves"
reply = call(topic, mode="vague")
show_response(reply, mode="vague")
log_call("vague", topic, reply)


# ─────────────────────────────────────────────────────────────────────
# §3.2 · Multi-Audience  →  sidebar: "Multi-Audience"
# ─────────────────────────────────────────────────────────────────────
print("\n\n### §3.2 · MULTI-AUDIENCE ###")

fact = (
    "We strengthened PMI hospitalisation reserves by INR 42 Cr "
    "following a sharp rise in empanelled-hospital tariffs."
)
reply = call(fact, mode="audience")
show_response(reply, mode="audience")
log_call("audience", fact, reply)


# ─────────────────────────────────────────────────────────────────────
# §3.3 · Few-Shot Formatting  →  sidebar: "Few-Shot Formatting"
# ─────────────────────────────────────────────────────────────────────
print("\n\n### §3.3 · FEW-SHOT FORMATTING ###")

fewshot_input = (
    "CI incidence for cardiac conditions, ages 40-55, moves from 0.45% to 0.52%."
)
reply = call(fewshot_input, mode="fewshot")
show_response(reply, mode="fewshot")
log_call("fewshot", fewshot_input, reply)


# ─────────────────────────────────────────────────────────────────────
# §3.4 · Chain-of-Thought  →  sidebar: "Chain-of-Thought"
# ─────────────────────────────────────────────────────────────────────
print("\n\n### §3.4 · CHAIN-OF-THOUGHT ###")

cot_prompt = (
    "A PMI policy has base premium INR 9,000 with relativities:\n"
    "  age band 46-55 = 1.45\n"
    "  sum insured 10L = 1.30\n"
    "  family floater = 1.10\n"
    "  NCB 30% = 0.70\n\n"
    "Walk through the premium calculation STEP BY STEP, showing the running "
    "total after each factor, then state the final premium."
)
reply = call(cot_prompt, mode="cot")
show_response(reply, mode="cot")
log_call("cot", cot_prompt, reply)


# ─────────────────────────────────────────────────────────────────────
# §4 · Hallucination Risk  →  sidebar: "Hallucination Risk"
# ─────────────────────────────────────────────────────────────────────
print("\n\n### §4 · HALLUCINATION RISK ###")

hallucination_prompt = (
    "What is the IRDAI-mandated co-payment factor for senior-citizen PMI "
    "policies with sum insured above INR 10 lakh? "
    "Give the exact factor value and the section reference."
)
reply = call(hallucination_prompt, mode="hallucination")
show_response(reply, mode="hallucination")
log_call("hallucination", hallucination_prompt, reply)

print(
    "\n⚠️  VERIFY BEFORE YOU TRUST — the guardrail forces a WARNING block, "
    "but the figures above may still be fabricated.\n"
    "   → Always cross-check at irdai.gov.in"
)


# ─────────────────────────────────────────────────────────────────────
# §5 · Structured Output  →  sidebar: "Structured Output"
# ─────────────────────────────────────────────────────────────────────
print("\n\n### §5 · STRUCTURED OUTPUT (JSON) ###")

json_prompt = (
    "For individual PMI (health) cover, list 5 rating factors. "
    "For each provide: name, direction (increase/decrease premium), "
    "one-line justification. Return a JSON array."
)
raw = call(json_prompt, mode="json")
show_response(raw, mode="json")
log_call("json", json_prompt, raw)

# Parse the JSON block out of the response
clean = raw.strip()
# Strip markdown fences if present
import re
m = re.search(r"```(?:json)?\s*([\s\S]+?)```", clean)
if m:
    clean = m.group(1).strip()
try:
    factors = json.loads(clean)
    print(f"\n✅  Parsed {len(factors)} rating factors:")
    for i, f in enumerate(factors, 1):
        print(f"  {i}. {json.dumps(f, ensure_ascii=False)}")
except json.JSONDecodeError as exc:
    print(f"⚠️  JSON parse error: {exc}")


# ─────────────────────────────────────────────────────────────────────
# §7 · Multi-turn conversation  →  mirrors chat history in index.html
# ─────────────────────────────────────────────────────────────────────
print("\n\n### §7 · MULTI-TURN CONVERSATION ###")
print("(index.html maintains history[] in JS; this script does the same in Python)")

history: list[dict] = []
turns = [
    "What is the claim frequency for a 52-year-old member with INR 10 lakh cover in a Tier 2 city?",
    "How would that change if the member moves to a Tier 1 city?",
]

for user_turn in turns:
    reply = call(user_turn, mode="free", history=history)
    show_response(reply, mode="free")
    log_call("free", user_turn, reply)
    history += [
        {"role": "user",      "content": user_turn},
        {"role": "assistant", "content": reply},
    ]


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'=' * 70}")
    print("✅  GenAI Concepts Lab — standalone demo complete.")
    print(f"    Modes run  : {', '.join(CONCEPT_LABELS.keys())}")
    print(f"    Call log   : {LOG_PATH}")
    print()
    print("To run the integrated web app instead:")
    print("    python app.py       # Flask backend on :5000")
    print("    open index.html     # open in browser")
    print("=" * 70)
