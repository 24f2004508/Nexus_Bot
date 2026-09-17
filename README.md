# Nexus_Bot
AI chatbot in the actuarial sense

## Run on Render

Create a Render Web Service from this directory, or use the included `render.yaml` blueprint.
Render will install `requirements.txt` and start the app with Gunicorn.

Required environment variable:

- `GOOGLE_API_KEY`: Google GenAI API key

The application serves the UI and API from the same origin. After deployment, open the Render URL
and use `/health` to verify the service is ready.

# 🧹 LLM Data Cleaning — GenAI Lab

An LLM-assisted data cleaning page inside the GenAI Lab dashboard. Upload a
CSV or Excel file, review a proposed cleaning plan, tick what you want,
apply it, and download the cleaned file.

Nothing is changed automatically — every operation is proposed first and
only runs after the user approves it.

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Backend framework | **Flask** (Python 3.10+) | `data_cleaning_bp` Blueprint, mounted into the existing `app.py` |
| Data handling | **pandas** | Reading CSV/Excel, profiling columns, applying cleaning operations |
| Excel I/O | **openpyxl** | Reading `.xlsx` cell formatting (fill colors) and writing cleaned `.xlsx` output |
| LLM | **Google Gemini** (`google-genai` client) | Shared client instance injected from `app.py`; used to propose extra column-level cleaning suggestions |
| Frontend | **Vanilla HTML / CSS / JavaScript** | No build step — a single page (`index.html`) with a dedicated "Data Cleaning" view, drag-and-drop upload, plan review table, and download flow |
| Data transport | **Fetch API + `multipart/form-data`** | File + JSON plan sent as form fields; cleaned file streamed back as a `Blob` |
| File formats supported | `.csv`, `.xlsx`, `.xls` | CSV has no cell formatting, so color-based checks are skipped for CSV with a UI warning |

## Architecture

```
Browser (index.html)
   │  drag/drop file
   ▼
POST /clean/analyse  ──►  pandas profiles the file
                           │
                           ├─► deterministic rule checks (Python only, no LLM):
                           │     • non-mandatory columns (orange-filled cells)
                           │     • Age < 18 computed from DOB
                           │     • incorrect policy status movement
                           │       (orange rows + status-transition map)
                           │     • missing DOB / SA / Gender / PREV STATUSCODE
                           │
                           └─► Gemini LLM ──► extra column-level suggestions
                                              (JSON, repaired/retried if malformed)
                           │
                           ▼
                     merged cleaning plan (JSON)
   ◄──────────────────────┘
   │  user ticks/unticks proposed operations
   ▼
POST /clean/apply  ──►  re-reads the file, applies only accepted ops
   ◄──────────────────────┘
   │  cleaned file (.xlsx or .csv) as a download
   ▼
Browser triggers download
```

## Key Design Decisions

- **Deterministic first, LLM second.** The four required checks (non-mandatory
  columns, age/DOB, status movement, missing mandatory fields) run in plain
  Python and never depend on the LLM returning valid JSON. The LLM is only
  asked for *additional* suggestions on the remaining columns.
- **Orange-highlight detection is dynamic**, not hardcoded. It reads the
  actual cell fill color from the uploaded workbook via `openpyxl`, using a
  broad orange-hue match — so it adapts to whichever file is uploaded rather
  than a fixed list of column names.
- **Robust LLM JSON parsing.** LLM responses are pre-processed to strip code
  fences, drop trailing commas, close unterminated strings, and truncate to
  the last complete item before parsing — with one retry using a stricter
  prompt if parsing still fails. If it still can't be parsed, the
  deterministic plan is returned on its own instead of erroring out.
- **Explicit human approval.** `/clean/analyse` is read-only. Nothing is
  written to the file until the user reviews the plan and clicks Apply.

## Endpoints

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/clean/analyse` | Upload a file, get back a profile + proposed cleaning plan |
| `POST` | `/clean/apply` | Upload the file again + the (edited) plan, get back the cleaned file |

## Local Setup

```bash
pip install flask pandas openpyxl google-genai

# in app.py:
from data_cleaning import data_cleaning_bp, init_cleaning
app.register_blueprint(data_cleaning_bp)
init_cleaning(client, MODEL)   # your existing Gemini client + model name

python app.py
```

Open the dashboard, click **🧹 Data Cleaning** in the sidebar, and upload a
file to try it.

## Repository Layout

```
.
├── app.py                 # existing Flask app (unchanged, just registers the blueprint)
├── data_cleaning.py        # Data Cleaning blueprint: profiling, rules, LLM plan, apply/download
└── index.html              # dashboard UI, including the Data Cleaning page
```
