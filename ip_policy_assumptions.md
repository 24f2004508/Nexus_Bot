# ABC Health — Income Protection (IP): Policy & Pricing Assumptions

**Status:** Illustrative / synthetic. This document describes the assumptions behind the teaching model used in this seminar's notebooks — it is not a real ABC Health product or filing.

## What the coverage does

Income Protection pays **80% of the policyholder's monthly income** as a replacement benefit while they are unable to work due to sickness or injury. It does **not** pay a lump sum or a death benefit — if the policyholder dies, income payments simply stop.

## States a policyholder can be in

1. **Healthy** — premium-paying, no benefit in payment.
2. **Sick (deferred)** — the policyholder has fallen sick but is still within the deferred (waiting) period. No benefit accrues here.
3. **Sick (claiming)** — the deferred period has elapsed while still sick. Income replacement now accrues.
4. **Death** — absorbing. No further premium, no further benefit.

## Deferred period

The deferred period is the waiting period a policyholder serves before benefit payments begin. Standard options offered: **4, 13, 26, or 52 weeks**, chosen by the policyholder at the start of the policy.

- A **shorter** deferred period means more sickness spells end up qualifying for benefit (since fewer of them resolve naturally before the waiting period ends) — this raises the premium.
- A **longer** deferred period filters out all but the more serious, longer-running spells — fewer claims, but each one that does occur tends to be more severe. This lowers the premium.

## Premium — how it's charged

- Premium is payable **annually**, at the start of each coverage year.
- **Waiver of premium**: no premium is charged while the policyholder is in the Sick (claiming) state. Premium is only charged during Healthy or Sick (deferred) time. This is why the incidence rate used for pricing is calculated using only Healthy + Sick(deferred) time in its exposure base — it must match exactly the period premium is actually collected over.
- The premium reflects **pure risk cost only** — **no loading for expenses, contingency margin, or profit** is applied in this teaching model. A real filed rate would add these on top.

## Rating factors — what changes the premium, and how

| Factor | How it's used | Values |
|---|---|---|
| **Age** | Rates the *frequency* of falling sick (how often), in the shared base table everyone is priced from | Banded: 25-34, 35-49, 50-60. Incidence rises with age. |
| **Occupation** | Also rates *frequency* only, same base table as age | Two classes: desk-based/sedentary, manual/physical. Manual work carries meaningfully higher incidence. |
| **Deferred period** | Rates both *frequency of a spell becoming a paid claim* and its *severity* (duration and cost) | 4 / 13 / 26 / 52 weeks, selected by the policyholder |
| **Prior sickness episodes** | A personal loading applied on top of the base rate, reflecting the policyholder's own claims history | Banded: 0, 1, 2+ prior episodes. More prior episodes → higher loading, both because a spell is more likely to reach a paid claim and because it tends to run longer once it does. |

Age and occupation sit in the **shared base rate table** everyone is priced from, because they're standard, broadly-applicable rating factors with plenty of population-level data behind them. Prior-episode history is a **personal adjustment** applied afterward, because only a subset of policyholders have any history to look up at all, and it needs credibility weighting (thin history shouldn't move the rate much; well-established history can).

## What is explicitly *not* modelled in this teaching version

- **Smoker status** — not a rating factor here. If asked, the correct answer is that this isn't currently priced, not an invented number.
- **Gender** — not modelled.
- **Benefit caps or maximum sum assured** — not modelled; the benefit is always exactly 80% of stated income.
- **Medical underwriting or pre-existing condition exclusions** — not modelled; this is a pure experience-based pricing exercise.
- **Claims trend or medical inflation** — the model prices off a single dataset snapshot, with no allowance for future cost trend.

## Data basis

All figures are generated from a synthetic population of illustrative policyholders and simulated sickness spells (not real ABC Health claims experience), built specifically to demonstrate the pricing methodology with reproducible, auditable numbers.
