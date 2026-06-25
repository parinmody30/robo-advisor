"""
Monte Carlo Simulation
----------------------
Runs N simulated return paths for the recommended portfolio.
Uses historical bootstrapping (block bootstrap) rather than
parametric normal draws — this preserves fat tails and
autocorrelation that a normal distribution would miss.

Outputs:
  - Wealth paths (percentile cone)
  - Probability of hitting each goal
  - Stress scenario (2008-style crash injected in year 1)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class MonteCarloResult:
    percentiles: pd.DataFrame      # index=year, cols=[p10, p25, p50, p75, p90]
    prob_success: float            # % of paths that hit the target corpus
    terminal_values: np.ndarray    # all N terminal wealth values
    stress_percentiles: pd.DataFrame  # same shape, with crash injected
    stress_prob_success: float


def _block_bootstrap_returns(
    annual_returns: np.ndarray,
    n_years: int,
    n_sims: int,
    block_size: int = 3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Block bootstrap: sample consecutive `block_size`-year blocks from history.
    Preserves serial correlation (momentum / mean-reversion cycles).
    Returns shape (n_sims, n_years).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T = len(annual_returns)
    paths = np.zeros((n_sims, n_years))

    for i in range(n_sims):
        sampled = []
        while len(sampled) < n_years:
            start = rng.integers(0, T - block_size + 1)
            sampled.extend(annual_returns[start: start + block_size].tolist())
        paths[i] = sampled[:n_years]

    return paths


def _portfolio_annual_returns(
    prices: pd.DataFrame,
    weights: dict,
    synthetic_assets: dict,
) -> np.ndarray:
    """
    Compute annual portfolio returns from price history + synthetic weights.
    """
    live_weights = {k: v for k, v in weights.items() if k in prices.columns}
    synth_weights = {k: v for k, v in weights.items() if k in synthetic_assets}

    # Daily returns for live assets, weighted sum
    daily_rets = np.log(prices / prices.shift(1)).dropna()
    live_port = sum(daily_rets[k] * w for k, w in live_weights.items()
                    if k in daily_rets.columns)

    # Resample to annual
    annual_live = (1 + live_port).resample('YE').prod() - 1

    # Add synthetic return (fixed, no historical series needed)
    synth_annual = sum(synthetic_assets[k] * w for k, w in synth_weights.items())
    annual_total = annual_live + synth_annual

    return annual_total.values


def run_simulation(
    prices: pd.DataFrame,
    weights: dict,
    synthetic_assets: dict,
    initial_investment: float,
    monthly_sip: float,
    horizon_years: int,
    target_corpus: float,
    n_sims: int = 10_000,
    crash_year: int = 1,
    crash_return: float = -0.45,
) -> MonteCarloResult:
    """
    Main Monte Carlo engine.

    Parameters
    ----------
    prices           : historical price DataFrame from data_loader
    weights          : allocation dict from optimizer
    synthetic_assets : SYNTHETIC_ASSETS dict from data_loader
    initial_investment: lump sum at t=0 (₹)
    monthly_sip      : monthly contribution (₹)
    horizon_years    : investment horizon
    target_corpus    : inflation-adjusted goal amount (₹)
    n_sims           : number of simulated paths
    crash_year       : which year to inject the stress crash (1-indexed)
    crash_return     : portfolio return during crash year (e.g. -0.45 = -45%)
    """
    annual_sip = monthly_sip * 12
    rng = np.random.default_rng(42)

    hist_returns = _portfolio_annual_returns(prices, weights, synthetic_assets)

    # ── Base simulation ───────────────────────────────────────────
    paths = _block_bootstrap_returns(hist_returns, horizon_years, n_sims, rng=rng)

    wealth = np.zeros((n_sims, horizon_years + 1))
    wealth[:, 0] = initial_investment

    for yr in range(1, horizon_years + 1):
        r = paths[:, yr - 1]
        wealth[:, yr] = wealth[:, yr - 1] * (1 + r) + annual_sip

    terminal = wealth[:, -1]
    prob_success = float(np.mean(terminal >= target_corpus) * 100)

    pcts = np.percentile(wealth, [10, 25, 50, 75, 90], axis=0)
    percentiles = pd.DataFrame(
        pcts.T,
        columns=["p10", "p25", "p50", "p75", "p90"],
        index=range(horizon_years + 1),
    )

    # ── Stress simulation (crash injected at crash_year) ──────────
    stress_paths = paths.copy()
    stress_paths[:, crash_year - 1] = crash_return  # override year with crash

    stress_wealth = np.zeros((n_sims, horizon_years + 1))
    stress_wealth[:, 0] = initial_investment

    for yr in range(1, horizon_years + 1):
        r = stress_paths[:, yr - 1]
        stress_wealth[:, yr] = stress_wealth[:, yr - 1] * (1 + r) + annual_sip

    stress_terminal = stress_wealth[:, -1]
    stress_prob = float(np.mean(stress_terminal >= target_corpus) * 100)

    stress_pcts = np.percentile(stress_wealth, [10, 25, 50, 75, 90], axis=0)
    stress_percentiles = pd.DataFrame(
        stress_pcts.T,
        columns=["p10", "p25", "p50", "p75", "p90"],
        index=range(horizon_years + 1),
    )

    return MonteCarloResult(
        percentiles=percentiles,
        prob_success=round(prob_success, 1),
        terminal_values=terminal,
        stress_percentiles=stress_percentiles,
        stress_prob_success=round(stress_prob, 1),
    )


def run_all_goals(
    prices: pd.DataFrame,
    weights: dict,
    synthetic_assets: dict,
    goals: list,
    monthly_savings: float,
    initial_investment: float = 0,
    n_sims: int = 10_000,
) -> List[dict]:
    """
    Runs Monte Carlo for each goal independently.
    Allocates monthly_savings proportionally to SIP required per goal.
    """
    results = []
    total_sip = sum(g.get("monthly_sip", 0) for g in goals)

    for g in goals:
        sip_share = (g.get("monthly_sip", 0) / total_sip) * monthly_savings \
                    if total_sip > 0 else monthly_savings / len(goals)

        mc = run_simulation(
            prices=prices,
            weights=weights,
            synthetic_assets=synthetic_assets,
            initial_investment=initial_investment,
            monthly_sip=sip_share,
            horizon_years=g["years_to_goal"],
            target_corpus=g["future_value"],
            n_sims=n_sims,
        )
        results.append({
            "goal":               g["name"],
            "target":             g["future_value"],
            "horizon":            g["years_to_goal"],
            "monthly_sip":        round(sip_share, 0),
            "prob_success":       mc.prob_success,
            "stress_prob":        mc.stress_prob_success,
            "mc":                 mc,
        })

    return results
