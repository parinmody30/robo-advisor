"""
Asset Allocation Engine — Markowitz Mean-Variance Optimization
--------------------------------------------------------------
1. Loads historical returns for all 13 instruments
2. Computes expected returns + covariance matrix
3. Generates the efficient frontier
4. Selects the portfolio matching the user's risk target
5. Applies real-world constraints (min/max weights per asset)

Why MVO?
  Markowitz (1952) showed that for a given return target, there exists
  a portfolio with minimum variance — the efficient frontier. Every
  rational investor should hold a portfolio ON this frontier.

Where MVO breaks:
  - Assumes returns are normally distributed (fat tails exist in reality)
  - Assumes stable correlations (they spike in crises — the worst time)
  - Extremely sensitive to expected-return estimates (garbage in = garbage out)
  These are addressed in Module 4 (Monte Carlo) and the optional
  Black-Litterman extension.
"""

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all, SYNTHETIC_ASSETS


# ── Allocation constraints ────────────────────────────────────────────────────
# Prevents absurd corner solutions (100% in one asset).
# Informed by SEBI-registered advisory norms for Indian retail investors.

WEIGHT_BOUNDS = {
    "Conservative": {
        "Equity_LargeCap":  (0.05, 0.25),
        "Equity_MidCap":    (0.00, 0.05),
        "Equity_SmallCap":  (0.00, 0.00),
        "Equity_Intl":      (0.00, 0.05),
        "Gold":             (0.05, 0.15),
        "Silver":           (0.00, 0.05),
        "Debt_Gilt":        (0.15, 0.40),
        "Debt_Corporate":   (0.10, 0.30),
        "ELSS":             (0.00, 0.05),
        "FixedDeposit":     (0.05, 0.20),
        "PPF":              (0.05, 0.15),
        "RBI_Bond":         (0.00, 0.10),
        "LiquidFund":       (0.05, 0.15),
    },
    "Balanced": {
        "Equity_LargeCap":  (0.10, 0.35),
        "Equity_MidCap":    (0.05, 0.15),
        "Equity_SmallCap":  (0.00, 0.05),
        "Equity_Intl":      (0.00, 0.08),
        "Gold":             (0.05, 0.12),
        "Silver":           (0.00, 0.05),
        "Debt_Gilt":        (0.10, 0.25),
        "Debt_Corporate":   (0.08, 0.20),
        "ELSS":             (0.03, 0.10),
        "FixedDeposit":     (0.03, 0.12),
        "PPF":              (0.03, 0.10),
        "RBI_Bond":         (0.00, 0.08),
        "LiquidFund":       (0.02, 0.08),
    },
    "Aggressive": {
        "Equity_LargeCap":  (0.15, 0.40),
        "Equity_MidCap":    (0.10, 0.25),
        "Equity_SmallCap":  (0.05, 0.15),
        "Equity_Intl":      (0.05, 0.15),
        "Gold":             (0.03, 0.10),
        "Silver":           (0.00, 0.05),
        "Debt_Gilt":        (0.03, 0.12),
        "Debt_Corporate":   (0.03, 0.10),
        "ELSS":             (0.05, 0.15),
        "FixedDeposit":     (0.00, 0.05),
        "PPF":              (0.00, 0.05),
        "RBI_Bond":         (0.00, 0.03),
        "LiquidFund":       (0.02, 0.05),
    },
}

ASSET_LABELS = {
    "Equity_LargeCap":  "Large Cap Equity",
    "Equity_MidCap":    "Mid Cap Equity",
    "Equity_SmallCap":  "Small Cap Equity",
    "Equity_Intl":      "International Equity",
    "Gold":             "Gold",
    "Silver":           "Silver",
    "Debt_Gilt":        "Govt Bonds (Gilt)",
    "Debt_Corporate":   "Corporate Bonds",
    "ELSS":             "ELSS (Tax Saving)",
    "FixedDeposit":     "Fixed Deposit",
    "PPF":              "PPF",
    "RBI_Bond":         "RBI Floating Bond",
    "LiquidFund":       "Liquid Fund",
}

ASSET_COLORS = {
    "Equity_LargeCap":  "#1565C0",
    "Equity_MidCap":    "#1E88E5",
    "Equity_SmallCap":  "#42A5F5",
    "Equity_Intl":      "#0097A7",
    "Gold":             "#F9A825",
    "Silver":           "#90A4AE",
    "Debt_Gilt":        "#2E7D32",
    "Debt_Corporate":   "#66BB6A",
    "ELSS":             "#7B1FA2",
    "FixedDeposit":     "#EF6C00",
    "PPF":              "#FF8F00",
    "RBI_Bond":         "#E65100",
    "LiquidFund":       "#78909C",
}


# ── Core optimizer ────────────────────────────────────────────────────────────

def run_optimizer(
    mu: pd.Series,
    cov: pd.DataFrame,
    persona: str,
    target_return: float,
) -> dict:
    """
    Runs MVO for the given persona and target return.
    Returns a dict with weights, expected performance, and frontier data.
    """
    bounds = WEIGHT_BOUNDS[persona]
    weight_bounds = {asset: bounds.get(asset, (0, 0.3)) for asset in mu.index}

    # Build per-asset bounds list in order
    bounds_list = [weight_bounds[a] for a in mu.index]

    ef = EfficientFrontier(mu, cov, weight_bounds=bounds_list)

    try:
        ef.efficient_return(target_return=target_return)
    except Exception:
        # If exact target infeasible, fall back to max Sharpe
        ef = EfficientFrontier(mu, cov, weight_bounds=bounds_list)
        ef.max_sharpe(risk_free_rate=0.065)

    weights = ef.clean_weights()
    perf    = ef.portfolio_performance(verbose=False, risk_free_rate=0.065)

    return {
        "weights":         {k: v for k, v in weights.items() if v > 0.001},
        "expected_return": round(perf[0], 4),
        "volatility":      round(perf[1], 4),
        "sharpe_ratio":    round(perf[2], 4),
    }


def generate_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    persona: str,
    n_points: int = 40,
) -> pd.DataFrame:
    """
    Generates the efficient frontier by solving MVO at increasing return targets.
    Returns a DataFrame with columns: target_return, volatility, sharpe.
    """
    bounds = WEIGHT_BOUNDS[persona]
    bounds_list = [bounds.get(a, (0, 0.3)) for a in mu.index]

    min_ret = float(mu.min()) + 0.005
    max_ret = float(mu.max()) * 0.90
    targets = np.linspace(min_ret, max_ret, n_points)

    frontier = []
    for t in targets:
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=bounds_list)
            ef.efficient_return(target_return=t)
            p = ef.portfolio_performance(verbose=False, risk_free_rate=0.065)
            frontier.append({"return": p[0], "volatility": p[1], "sharpe": p[2]})
        except Exception:
            continue

    return pd.DataFrame(frontier)


def total_equity_weight(weights: dict) -> float:
    equity_keys = ["Equity_LargeCap", "Equity_MidCap", "Equity_SmallCap",
                   "Equity_Intl", "ELSS"]
    return sum(weights.get(k, 0) for k in equity_keys)
