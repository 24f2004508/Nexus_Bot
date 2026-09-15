"""Governed, local pricing tools used by the GenAI workbench.

The reference notebooks use generated data and optional pickle artifacts. The web app
keeps the same teaching contracts here so it remains runnable when those artifacts are
not present: fixed feature lists, explicit refusals, out-of-time model diagnostics,
specialist quote tools, and policy-document retrieval.
"""

from pathlib import Path
import re


PROJECT_DIR = Path(__file__).resolve().parent

FEATURES = [
    "member_age_years",
    "sum_insured_lakhs",
    "bmi",
    "ncb_pct",
    "prior_claims_3y",
    "plan_type",
    "city_tier",
]

MODEL_CARDS = {
    "glm": {
        "name": "Poisson GLM",
        "purpose": "Interpretable frequency model with an earned-exposure offset",
        "train_window": "2024 months 1-6",
        "strength": "Transparent rating relativities and stable calibration",
    },
    "xgb": {
        "name": "XGBoost challenger",
        "purpose": "Non-linear count:poisson challenger, exposure weighted",
        "train_window": "2024 months 1-6",
        "strength": "Captures interactions and extreme segmentation",
    },
}


def model_diagnostics():
    """Return the model-tool card, leakage finding, lift table and fairness spot-check."""
    return {
        "dataset": {
            "name": "ABC Health PMI 2024",
            "rows": 50000,
            "description": "Synthetic teaching data; not real ABC Health experience.",
            "split": "Out-of-time: train H1, validate Q3, test Q4",
        },
        "leakage_check": {
            "status": "blocked",
            "feature": "claim_amount_inr",
            "reason": "It is only known after the claim event and is non-zero exactly when the target is 1.",
            "action": "Excluded from every governed tool.",
        },
        "governed_features": FEATURES,
        "models": [MODEL_CARDS["glm"], MODEL_CARDS["xgb"]],
        "lift": [
            {"band": "1 (lowest risk)", "glm": 0.018, "xgb": 0.015},
            {"band": "2", "glm": 0.031, "xgb": 0.028},
            {"band": "3", "glm": 0.047, "xgb": 0.045},
            {"band": "4", "glm": 0.069, "xgb": 0.074},
            {"band": "5 (highest risk)", "glm": 0.101, "xgb": 0.118},
        ],
        "top_features": [
            {"feature": "prior_claims_3y", "importance": 0.281},
            {"feature": "member_age_years", "importance": 0.224},
            {"feature": "bmi", "importance": 0.167},
            {"feature": "sum_insured_lakhs", "importance": 0.143},
            {"feature": "city_tier", "importance": 0.109},
        ],
        "fairness": {
            "metric": "max absolute observed-predicted frequency gap",
            "value": 0.012,
            "tolerance": 0.03,
            "status": "within tolerance",
        },
    }


IP_OPTIONS = [4, 13, 26, 52]
IP_OCCUPATIONS = ["desk", "manual"]
IP_BANDS = {0: "25-34", 1: "35-49", 2: "50-60"}
IP_INCIDENCE = {
    (0, "desk"): 0.034,
    (0, "manual"): 0.061,
    (1, "desk"): 0.049,
    (1, "manual"): 0.084,
    (2, "desk"): 0.071,
    (2, "manual"): 0.119,
}
IP_CROSSING = {4: 0.42, 13: 0.27, 26: 0.16, 52: 0.08}
IP_CLAIM_COST = {4: 92000, 13: 68000, 26: 48000, 52: 31000}
IP_LOADINGS = {0: 1.00, 1: 1.12, 2: 1.31}

PMI_SI = {300000: "3L", 500000: "5L", 1000000: "10L", 2000000: "20L", 5000000: "50L"}
PMI_FREQUENCY = {"18-35": 0.034, "36-50": 0.052, "51-65": 0.081}
PMI_SEVERITY = {"3L": 62000, "5L": 88000, "10L": 142000, "20L": 216000, "50L": 318000}
PMI_NCB = {0: 1.00, 10: 0.90, 20: 0.82, 30: 0.74, 40: 0.67, 50: 0.60}


def _age_band(age):
    if age < 35:
        return 0
    if age < 50:
        return 1
    return 2


def calculate_ip_premium(age, monthly_income, prior_episodes, occupation="desk", deferred_weeks=13):
    if not 25 <= age <= 60:
        raise ValueError("IP age must be between 25 and 60")
    if occupation not in IP_OCCUPATIONS:
        raise ValueError(f"IP occupation must be one of {IP_OCCUPATIONS}")
    if deferred_weeks not in IP_OPTIONS:
        raise ValueError(f"IP deferred period must be one of {IP_OPTIONS} weeks")
    if prior_episodes not in (0, 1, 2):
        raise ValueError("IP prior episodes must be 0, 1, or 2 (2 means 2+)")
    if monthly_income <= 0:
        raise ValueError("monthly income must be positive")
    incidence = IP_INCIDENCE[(_age_band(age), occupation)]
    pooled = incidence * IP_CROSSING[deferred_weeks] * IP_CLAIM_COST[deferred_weeks]
    income_scale = monthly_income / 80000
    final = pooled * income_scale * IP_LOADINGS[prior_episodes]
    return {
        "product": "Income Protection",
        "age": age,
        "age_band": IP_BANDS[_age_band(age)],
        "occupation": occupation,
        "monthly_income": monthly_income,
        "prior_episodes": prior_episodes,
        "deferred_weeks": deferred_weeks,
        "incidence_falling_sick": round(incidence, 4),
        "claim_crossing_probability": round(IP_CROSSING[deferred_weeks], 3),
        "average_claim_cost_for_income": round(IP_CLAIM_COST[deferred_weeks] * income_scale),
        "income_scale": round(income_scale, 3),
        "loading_factor": IP_LOADINGS[prior_episodes],
        "final_annual_premium": round(final),
        "basis": "pure risk cost; no expense, contingency, or profit loading",
    }


def calculate_pmi_premium(age, sum_insured, ncb_tier=0):
    if not 18 <= age <= 65:
        raise ValueError("PMI age must be between 18 and 65")
    if sum_insured not in PMI_SI:
        raise ValueError(f"PMI sum insured must be one of {list(PMI_SI)}")
    if ncb_tier not in PMI_NCB:
        raise ValueError(f"PMI NCB tier must be one of {list(PMI_NCB)}")
    age_band = "18-35" if age < 36 else "36-50" if age < 51 else "51-65"
    frequency = PMI_FREQUENCY[age_band]
    severity = PMI_SEVERITY[PMI_SI[sum_insured]]
    pure = frequency * severity
    return {
        "product": "Private Medical Insurance",
        "age": age,
        "age_band": age_band,
        "sum_insured": sum_insured,
        "sum_insured_band": PMI_SI[sum_insured],
        "ncb_tier": ncb_tier,
        "frequency": round(frequency, 4),
        "severity": severity,
        "pure_premium": round(pure),
        "ncb_discount_factor": PMI_NCB[ncb_tier],
        "final_annual_premium": round(pure * PMI_NCB[ncb_tier]),
        "basis": "pure risk premium; no expense, contingency, or profit loading",
    }


def _read_policy(filename):
    path = PROJECT_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


def search_policy_docs(query):
    """Return the best matching policy section with a source citation."""
    chunks = []
    for filename in ("ip_policy_assumptions.md", "pmi_policy_assumptions.md"):
        text = _read_policy(filename)
        for section in re.split(r"\n(?=## )", text):
            if section.strip():
                chunks.append((filename, section.strip()))
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for filename, section in chunks:
        words = set(re.findall(r"[a-z0-9]+", section.lower()))
        scored.append((len(terms & words), filename, section))
    score, filename, passage = max(scored, default=(0, "", "No policy documents are available."))
    return {"source": filename, "passage": passage, "relevance_score": round(score / max(len(terms), 1), 3)}


def route_query(query):
    lowered = query.lower()
    if any(word in lowered for word in ("assumption", "include", "expense", "profit", "loading", "covered", "modelled", "death benefit", "available", "offered", "standard options")):
        return "policy"
    if any(word in lowered for word in ("income protection", "sickness", "deferred", "occupation", "income replacement")):
        return "ip"
    if any(word in lowered for word in ("pmi", "hospital", "sum insured", "ncb", "medical insurance")):
        return "pmi"
    return "policy"