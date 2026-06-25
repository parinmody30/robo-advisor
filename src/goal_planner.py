"""
Goal Planning
-------------
Handles multiple simultaneous goals. For each goal:
  - Inflates the target amount to future value
  - Back-calculates the required monthly SIP
  - Checks feasibility against current savings rate
  - Flags funding gap if SIP is unaffordable
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Goal:
    name:              str
    target_amount:     float   # today's rupees
    years_to_goal:     int
    priority:          int     # 1=highest
    inflation_rate:    float   = 0.06   # 6% default for India
    expected_return:   float   = 0.12   # filled from risk profile
    future_value:      float   = 0.0    # computed
    monthly_sip:       float   = 0.0    # computed
    feasibility_pct:   float   = 0.0    # % funded at current savings
    gap_monthly:       float   = 0.0    # extra monthly savings needed


def inflation_adjust(amount: float, rate: float, years: int) -> float:
    """Today's ₹X will cost this much in `years` years."""
    return amount * ((1 + rate) ** years)


def required_sip(future_value: float, annual_return: float, years: int) -> float:
    """
    Monthly SIP to accumulate future_value in `years` years.
    Uses standard future-value-of-annuity formula:
        FV = SIP * [((1+r)^n - 1) / r] * (1+r)
    where r = monthly rate, n = total months.
    """
    n = years * 12
    r = annual_return / 12
    if r == 0:
        return future_value / n
    fv_factor = ((1 + r) ** n - 1) / r * (1 + r)
    return future_value / fv_factor


def plan_goals(
    goals: List[dict],
    monthly_savings: float,
    expected_return: float,
) -> List[Goal]:
    """
    goals: list of dicts with keys:
        name, target_amount, years_to_goal, priority
        optionally: inflation_rate
    monthly_savings: total available per month across all goals
    expected_return: from risk profile (annualised)
    """
    planned: List[Goal] = []

    for g in sorted(goals, key=lambda x: x.get("priority", 99)):
        goal = Goal(
            name=g["name"],
            target_amount=g["target_amount"],
            years_to_goal=g["years_to_goal"],
            priority=g.get("priority", 99),
            inflation_rate=g.get("inflation_rate", 0.06),
            expected_return=expected_return,
        )

        goal.future_value = inflation_adjust(
            goal.target_amount, goal.inflation_rate, goal.years_to_goal
        )
        goal.monthly_sip = required_sip(
            goal.future_value, goal.expected_return, goal.years_to_goal
        )
        goal.feasibility_pct = min(100.0, round(
            (monthly_savings / goal.monthly_sip) * 100, 1
        )) if goal.monthly_sip > 0 else 100.0
        goal.gap_monthly = max(0.0, round(goal.monthly_sip - monthly_savings, 2))

        planned.append(goal)

    return planned


def summary_table(planned: List[Goal]) -> List[dict]:
    rows = []
    for g in planned:
        rows.append({
            "Goal":               g.name,
            "Priority":           g.priority,
            "Target Today (₹)":   f"{g.target_amount:,.0f}",
            "Future Value (₹)":   f"{g.future_value:,.0f}",
            "Years":              g.years_to_goal,
            "Monthly SIP (₹)":    f"{g.monthly_sip:,.0f}",
            "Feasibility":        f"{g.feasibility_pct:.1f}%",
            "Monthly Gap (₹)":    f"{g.gap_monthly:,.0f}" if g.gap_monthly > 0 else "—",
        })
    return rows
