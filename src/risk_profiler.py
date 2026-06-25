"""
Risk Profiling
--------------
Separates risk CAPACITY (objective ability to take risk) from
risk TOLERANCE (psychological willingness). Conflating them is
the classic retail-advisor mistake — capacity sets the ceiling,
tolerance sets the actual allocation.
"""

from dataclasses import dataclass


@dataclass
class RiskProfile:
    capacity_score:  float   # 0–10
    tolerance_score: float   # 0–10
    combined_score:  float   # weighted blend, 0–10
    persona:         str     # Conservative / Balanced / Aggressive
    target_return:   float   # annualised, feeds the optimizer


# ── Capacity questions (objective) ──────────────────────────────────────────

def score_capacity(
    age: int,
    income_stability: int,   # 1=unstable, 2=moderate, 3=stable
    dependents: int,         # number
    emergency_fund_months: int,
    investment_horizon_yrs: int,
) -> float:
    score = 0.0

    # Age: younger = more capacity
    if age < 30:   score += 2.5
    elif age < 40: score += 2.0
    elif age < 50: score += 1.5
    elif age < 60: score += 0.8
    else:          score += 0.2

    # Income stability
    score += {1: 0.5, 2: 1.5, 3: 2.5}.get(income_stability, 1.0)

    # Dependents: more dependents = less capacity
    if dependents == 0:   score += 2.0
    elif dependents <= 2: score += 1.0
    else:                 score += 0.0

    # Emergency fund
    if emergency_fund_months >= 6: score += 2.0
    elif emergency_fund_months >= 3: score += 1.0
    else: score += 0.0

    # Horizon: longer = more capacity
    if investment_horizon_yrs >= 15: score += 1.0
    elif investment_horizon_yrs >= 7: score += 0.7
    else: score += 0.3

    return min(score, 10.0)


# ── Tolerance questions (behavioural) ────────────────────────────────────────

def score_tolerance(answers: dict[str, int]) -> float:
    """
    answers keys and valid values:
      market_drop_reaction  : 1=sell, 2=hold, 3=buy_more
      past_investing_exp    : 1=none, 2=some, 3=experienced
      loss_sleep            : 1=cant_sleep, 2=uneasy, 3=fine
      volatility_comfort    : 1=low, 2=medium, 3=high
      goal_flexibility      : 1=rigid, 2=somewhat, 3=flexible
    """
    weights = {
        "market_drop_reaction": 3.0,
        "past_investing_exp":   1.5,
        "loss_sleep":           2.5,
        "volatility_comfort":   2.0,
        "goal_flexibility":     1.0,
    }
    max_score = sum(weights.values()) * 3
    raw = sum(weights[k] * answers.get(k, 2) for k in weights)
    return round((raw / max_score) * 10, 2)


# ── Combine & map to persona ─────────────────────────────────────────────────

# Capacity is the ceiling — tolerance can pull down but not push above capacity.
CAPACITY_WEIGHT  = 0.55
TOLERANCE_WEIGHT = 0.45

PERSONA_MAP = [
    (0,   4.0,  "Conservative", 0.08),
    (4.0, 6.5,  "Balanced",     0.12),
    (6.5, 10.1, "Aggressive",   0.15),
]


def build_profile(
    age: int,
    income_stability: int,
    dependents: int,
    emergency_fund_months: int,
    investment_horizon_yrs: int,
    tolerance_answers: dict[str, int],
) -> RiskProfile:
    cap  = score_capacity(age, income_stability, dependents,
                          emergency_fund_months, investment_horizon_yrs)
    tol  = score_tolerance(tolerance_answers)

    # Tolerance cannot exceed capacity — capacity is the hard ceiling
    effective_tol = min(tol, cap)
    combined = round(CAPACITY_WEIGHT * cap + TOLERANCE_WEIGHT * effective_tol, 2)

    for lo, hi, persona, target_ret in PERSONA_MAP:
        if lo <= combined < hi:
            return RiskProfile(
                capacity_score=round(cap, 2),
                tolerance_score=round(tol, 2),
                combined_score=combined,
                persona=persona,
                target_return=target_ret,
            )

    # Fallback
    return RiskProfile(cap, tol, combined, "Balanced", 0.12)
