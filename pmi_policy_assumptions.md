# ABC Health — Private Medical Insurance (PMI): Policy & Pricing Assumptions

**Status:** Illustrative / synthetic. This document describes the assumptions behind the simplified teaching model used in notebook 05 — it is not a real ABC Health product or filing, and it is deliberately simpler than the fuller PMI Pricing Logic Explainer used earlier in this seminar.

## What the coverage does

Private Medical Insurance reimburses (or pays cashless, via network hospitals) the cost of hospitalisation — a per-policy-year indemnity benefit up to the policyholder's chosen sum insured.

## Pricing structure — frequency × severity

This notebook's PMI tool prices coverage as a simple **pure premium**:

$$\text{Pure premium} = \text{Frequency (age band)} \times \text{Severity (sum-insured band)} \times \text{NCB discount factor}$$

- **Frequency** — the probability a policyholder makes at least one claim in a policy year, rated by **age band**: 18-35, 36-50, 51-65. Frequency rises with age.
- **Severity** — the average cost of a claim, given one occurs, rated by **sum-insured band**: ₹3L, ₹5L, ₹10L, ₹20L, ₹50L. Severity rises with sum insured (higher cover is associated with larger claims).
- **NCB (No-Claim Bonus) discount** — a discount ladder for consecutive claim-free years: 0%, 10%, 20%, 30%, 40%, 50% tiers, each with a fixed discount factor applied to the premium.

## Portfolio reference figures

Across the whole illustrative portfolio, these rating tables are calibrated to land close to:
- **Portfolio frequency**: approximately 6%
- **Portfolio severity**: approximately ₹85,000
- **Portfolio pure premium**: approximately ₹5,100 (6% × ₹85,000)

These are reference figures used consistently across this seminar's materials — exact numbers from any specific run of the synthetic data may differ slightly due to normal sampling variation, the same way real experience data never matches assumptions exactly.

## Why frequency and severity are estimated differently from NCB

- **Frequency (by age band)** and **severity (by sum-insured band)** are estimated directly from the synthetic claims experience — the same "estimate from data" actuarial approach used throughout this seminar.
- **NCB discount** is *not* estimated from a single year of claims data — a no-claim bonus is a **product design / governance rule** (a loyalty discount for staying claim-free), set directly as policy rather than fitted statistically. Real insurers set this ladder as a business decision, informed by but not mechanically derived from a single experience study.

## What is explicitly *not* modelled in this teaching version

- **Expenses and profit loading** — this tool prices **pure risk premium only**. No allowance for acquisition cost, administration expense, contingency margin, or profit is added. A real filed premium would be materially higher than this tool's output for that reason alone.
- **Plan type** (Individual vs. Family Floater) — not modelled as a separate rating factor in this simplified tool, though it is a real, common PMI rating dimension (a natural extension exercise).
- **Room-rent capping, co-payment, or sub-limits** — not modelled.
- **Medical underwriting, waiting periods for specific conditions, or pre-existing disease exclusions** — not modelled.
- **Claims trend or medical cost inflation** — the model prices off a single dataset snapshot only.

## Relationship to the fuller PMI worked example used elsewhere in this seminar

Other materials in this seminar reference a fully-loaded worked example (a 52-year-old, ₹10L sum insured, Family Floater, 30% NCB, resolving to a specific final premium). That example **includes** expense and profit loading and plan-type rating, which this simplified notebook-05 tool deliberately excludes. **Do not expect this tool's output to match that figure** — the gap is intentional and explained by the loadings this tool omits, not an inconsistency.

## Data basis

All figures are generated from a synthetic portfolio of illustrative policies and simulated claims (not real ABC Health experience), built to demonstrate the pricing methodology with reproducible, auditable numbers.
