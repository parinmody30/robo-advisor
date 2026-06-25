"""
Rebalancing & Glide Path
------------------------
Two distinct but related ideas:

1. GLIDE PATH
   As a goal date approaches, the portfolio should shift from
   high-risk (equity-heavy) to low-risk (debt/cash-heavy).
   This is the logic behind real target-date funds (e.g. HDFC
   Retirement Fund, SBI Retirement Benefit Fund).

   Mechanism: linearly interpolate between the user's current
   risk-based allocation and a "landing allocation" that is
   mostly debt/cash at T=0.

2. DRIFT DETECTION & REBALANCING
   Over time, assets that perform well grow beyond their target
   weight. A portfolio that started 60/40 equity/debt can drift
   to 75/25 after a bull run — taking on more risk than intended.

   We flag when any asset drifts beyond a threshold (default ±5%)
   and compute exactly what to buy/sell to return to target.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


# ── Landing allocation at goal date (T=0) ────────────────────────────────────
# Conservative end-state: capital preservation, minimal equity.
LANDING_ALLOCATION = {
    "Equity_LargeCap":  0.05,
    "Equity_MidCap":    0.00,
    "Equity_SmallCap":  0.00,
    "Equity_Intl":      0.00,
    "Gold":             0.08,
    "Silver":           0.00,
    "Debt_Gilt":        0.30,
    "Debt_Corporate":   0.25,
    "ELSS":             0.00,
    "FixedDeposit":     0.15,
    "PPF":              0.10,
    "RBI_Bond":         0.05,
    "LiquidFund":       0.02,
}


@dataclass
class GlidePath:
    years:        List[int]
    equity_pct:   List[float]   # total equity % each year
    debt_pct:     List[float]
    annual_weights: List[dict]  # full weight dict each year


@dataclass
class RebalanceAction:
    asset:          str
    current_weight: float
    target_weight:  float
    drift:          float       # current - target
    action:         str         # "BUY" / "SELL" / "HOLD"
    amount_inr:     float       # ₹ to buy or sell (given portfolio value)


def compute_glide_path(
    target_weights: dict,
    horizon_years: int,
    glide_start_year: int = None,
) -> GlidePath:
    """
    Interpolates from target_weights (today) to LANDING_ALLOCATION
    over the horizon. Glide starts at glide_start_year (default:
    halfway through horizon or 10 years before end, whichever is less).
    """
    if glide_start_year is None:
        glide_start_year = max(1, horizon_years - min(10, horizon_years // 2))

    # Normalise landing allocation to only include assets in target
    assets = list(target_weights.keys())
    landing = {a: LANDING_ALLOCATION.get(a, 0.0) for a in assets}
    # Renormalise landing to sum to 1
    total = sum(landing.values())
    landing = {a: v / total for a, v in landing.items()}

    years_list, equity_list, debt_list, weights_list = [], [], [], []

    equity_keys = {"Equity_LargeCap", "Equity_MidCap", "Equity_SmallCap",
                   "Equity_Intl", "ELSS"}
    debt_keys   = {"Debt_Gilt", "Debt_Corporate", "FixedDeposit",
                   "PPF", "RBI_Bond", "LiquidFund"}

    for yr in range(horizon_years + 1):
        if yr <= glide_start_year:
            # Before glide starts — hold original allocation
            t = 0.0
        else:
            # Linearly interpolate from start toward landing
            t = (yr - glide_start_year) / (horizon_years - glide_start_year)
            t = min(t, 1.0)

        w = {a: (1 - t) * target_weights.get(a, 0) + t * landing.get(a, 0)
             for a in assets}

        # Renormalise to sum to 1
        total = sum(w.values())
        w = {a: round(v / total, 4) for a, v in w.items()}

        equity_pct = sum(w.get(a, 0) for a in equity_keys) * 100
        debt_pct   = sum(w.get(a, 0) for a in debt_keys)   * 100

        years_list.append(yr)
        equity_list.append(round(equity_pct, 1))
        debt_list.append(round(debt_pct, 1))
        weights_list.append(w)

    return GlidePath(
        years=years_list,
        equity_pct=equity_list,
        debt_pct=debt_list,
        annual_weights=weights_list,
    )


def detect_drift(
    target_weights: dict,
    current_weights: dict,
    portfolio_value: float,
    threshold: float = 0.05,
) -> List[RebalanceAction]:
    """
    Compares current weights to target. Flags assets beyond threshold.
    Returns list of RebalanceAction with buy/sell instructions.
    """
    actions = []
    assets = set(list(target_weights.keys()) + list(current_weights.keys()))

    for asset in assets:
        target  = target_weights.get(asset, 0.0)
        current = current_weights.get(asset, 0.0)
        drift   = current - target

        if abs(drift) >= threshold:
            action_str = "SELL" if drift > 0 else "BUY"
        else:
            action_str = "HOLD"

        amount = abs(drift) * portfolio_value

        actions.append(RebalanceAction(
            asset=asset,
            current_weight=round(current, 4),
            target_weight=round(target, 4),
            drift=round(drift, 4),
            action=action_str,
            amount_inr=round(amount, 0),
        ))

    return sorted(actions, key=lambda x: abs(x.drift), reverse=True)


def simulate_drift(
    target_weights: dict,
    asset_returns: dict,
    years: int = 3,
) -> dict:
    """
    Simulates how weights drift over `years` given asset returns.
    asset_returns: {asset: annual_return}
    Returns dict of drifted weights after `years`.
    """
    weights = {k: v for k, v in target_weights.items()}

    for _ in range(years):
        grown = {a: w * (1 + asset_returns.get(a, 0.10))
                 for a, w in weights.items()}
        total = sum(grown.values())
        weights = {a: v / total for a, v in grown.items()}

    return {a: round(v, 4) for a, v in weights.items()}


def rebalance_summary(actions: List[RebalanceAction]) -> pd.DataFrame:
    rows = []
    for a in actions:
        if a.action != "HOLD":
            rows.append({
                "Asset":           a.asset,
                "Action":          a.action,
                "Current %":       f"{a.current_weight*100:.1f}%",
                "Target %":        f"{a.target_weight*100:.1f}%",
                "Drift":           f"{a.drift*100:+.1f}%",
                "Amount (₹)":      f"₹{a.amount_inr:,.0f}",
            })
    return pd.DataFrame(rows)
