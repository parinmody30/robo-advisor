"""
Stock Screener
--------------
Curated universe of ~80 quality Indian stocks, organised into tiers.
Given a risk persona, screens the relevant tier(s) using live yfinance
fundamentals (beta, PE, D/E, 6-month momentum) and returns a ranked
basket to fill the equity allocation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field


# ── Curated universe ──────────────────────────────────────────────────────────

LARGE_CAP = {
    "Reliance Industries":   "RELIANCE.NS",
    "TCS":                   "TCS.NS",
    "HDFC Bank":             "HDFCBANK.NS",
    "Infosys":               "INFY.NS",
    "ICICI Bank":            "ICICIBANK.NS",
    "Hindustan Unilever":    "HINDUNILVR.NS",
    "ITC":                   "ITC.NS",
    "State Bank of India":   "SBIN.NS",
    "Bharti Airtel":         "BHARTIARTL.NS",
    "Kotak Mahindra Bank":   "KOTAKBANK.NS",
    "Axis Bank":             "AXISBANK.NS",
    "Larsen & Toubro":       "LT.NS",
    "Asian Paints":          "ASIANPAINT.NS",
    "Maruti Suzuki":         "MARUTI.NS",
    "Sun Pharma":            "SUNPHARMA.NS",
    "Titan Company":         "TITAN.NS",
    "Bajaj Finance":         "BAJFINANCE.NS",
    "HCL Technologies":      "HCLTECH.NS",
    "Wipro":                 "WIPRO.NS",
    "NTPC":                  "NTPC.NS",
    "Power Grid":            "POWERGRID.NS",
    "Coal India":            "COALINDIA.NS",
    "Nestle India":          "NESTLEIND.NS",
    "Britannia":             "BRITANNIA.NS",
    "Dr. Reddy's":           "DRREDDY.NS",
    "Cipla":                 "CIPLA.NS",
    "Tata Motors":           "TATAMOTORS.NS",
    "Tata Steel":            "TATASTEEL.NS",
    "JSW Steel":             "JSWSTEEL.NS",
    "UltraTech Cement":      "ULTRACEMCO.NS",
}

MID_CAP = {
    "Bajaj Auto":            "BAJAJ-AUTO.NS",
    "Godrej Consumer":       "GODREJCP.NS",
    "Voltas":                "VOLTAS.NS",
    "Muthoot Finance":       "MUTHOOTFIN.NS",
    "Page Industries":       "PAGEIND.NS",
    "Thermax":               "THERMAX.NS",
    "Persistent Systems":    "PERSISTENT.NS",
    "Coforge":               "COFORGE.NS",
    "Trent":                 "TRENT.NS",
    "Varun Beverages":       "VBL.NS",
    "Crompton Greaves":      "CROMPTON.NS",
    "Polycab India":         "POLYCAB.NS",
    "KEI Industries":        "KEI.NS",
    "Solar Industries":      "SOLARINDS.NS",
    "Tube Investments":      "TIINDIA.NS",
    "Chola Finance":         "CHOLAFIN.NS",
    "Max Healthcare":        "MAXHEALTH.NS",
    "Apollo Hospitals":      "APOLLOHOSP.NS",
    "Indraprastha Gas":      "IGL.NS",
    "Mahanagar Gas":         "MGL.NS",
    "Astral Ltd":            "ASTRAL.NS",
    "Havells India":         "HAVELLS.NS",
    "Cummins India":         "CUMMINSIND.NS",
    "Schaeffler India":      "SCHAEFFLER.NS",
    "Grindwell Norton":      "GRINDWELL.NS",
}

SMALL_CAP = {
    "Zomato":                "ZOMATO.NS",
    "Dixon Technologies":    "DIXON.NS",
    "Kaynes Technology":     "KAYNES.NS",
    "Ideaforge Technology":  "IDEAFORGE.NS",
    "Syrma SGS":             "SYRMA.NS",
    "Archean Chemical":      "ARCHEAN.NS",
    "Garware Tech Fibres":   "GARFIBRES.NS",
    "Senco Gold":            "SENCO.NS",
    "Nuvoco Vistas":         "NUVOCO.NS",
    "Sapphire Foods":        "SAPPHIRE.NS",
    "Campus Activewear":     "CAMPUS.NS",
    "Bikaji Foods":          "BIKAJI.NS",
    "Medplus Health":        "MEDPLUS.NS",
    "Latent View Analytics": "LATENTVIEW.NS",
    "Happiest Minds":        "HAPPSTMNDS.NS",
    "Affle India":           "AFFLE.NS",
    "Nazara Technologies":   "NAZARA.NS",
    "Nykaa":                 "NYKAA.NS",
    "PB Fintech":            "POLICYBZR.NS",
    "Delhivery":             "DELHIVERY.NS",
}

# Which tiers each persona can draw from
PERSONA_TIERS = {
    "Conservative": ["large"],
    "Balanced":     ["large", "mid"],
    "Aggressive":   ["large", "mid", "small"],
}

# Screening thresholds per persona
SCREEN_CONFIG = {
    "Conservative": {"max_beta": 0.85,  "max_pe": 35,  "max_de": 1.0,  "top_n": 10},
    "Balanced":     {"max_beta": 1.20,  "max_pe": 50,  "max_de": 1.5,  "top_n": 12},
    "Aggressive":   {"max_beta": 999,   "max_pe": 999, "max_de": 3.0,  "top_n": 15},
}


# ── Data fetching ─────────────────────────────────────────────────────────────

@dataclass
class StockInfo:
    name:       str
    ticker:     str
    tier:       str
    beta:       float = np.nan
    pe:         float = np.nan
    de:         float = np.nan
    momentum_6m: float = np.nan
    score:      float = np.nan


def _fetch_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "beta": info.get("beta", np.nan),
            "pe":   info.get("trailingPE", np.nan),
            "de":   info.get("debtToEquity", np.nan),
        }
    except Exception:
        return {"beta": np.nan, "pe": np.nan, "de": np.nan}


def _fetch_momentum(ticker: str, months: int = 6) -> float:
    try:
        hist = yf.download(ticker, period=f"{months}mo",
                           auto_adjust=True, progress=False)["Close"]
        if len(hist) < 2:
            return np.nan
        return float((hist.iloc[-1] / hist.iloc[0]) - 1)
    except Exception:
        return np.nan


def fetch_stock_universe(persona: str) -> list[StockInfo]:
    """Pull fundamentals for all stocks in the persona's eligible tiers."""
    tiers = PERSONA_TIERS.get(persona, ["large"])
    universe: list[StockInfo] = []

    tier_map = {"large": LARGE_CAP, "mid": MID_CAP, "small": SMALL_CAP}
    for tier in tiers:
        for name, ticker in tier_map[tier].items():
            universe.append(StockInfo(name=name, ticker=ticker, tier=tier))

    print(f"Fetching data for {len(universe)} stocks ({persona} universe)…")
    for s in universe:
        funda = _fetch_fundamentals(s.ticker)
        s.beta = funda["beta"]
        s.pe   = funda["pe"]
        s.de   = funda["de"] / 100 if not np.isnan(funda["de"]) else np.nan  # yfinance gives D/E * 100
        s.momentum_6m = _fetch_momentum(s.ticker)

    return universe


# ── Screening & ranking ───────────────────────────────────────────────────────

def _composite_score(s: StockInfo, cfg: dict) -> float:
    """
    Higher is better. Penalises high beta, high PE, high D/E.
    Rewards positive 6-month momentum.
    """
    score = 0.0

    # Beta: lower is better for conservative, neutral for aggressive
    if not np.isnan(s.beta):
        score += max(0, (2 - s.beta) * 20)

    # Momentum: positive momentum rewarded
    if not np.isnan(s.momentum_6m):
        score += s.momentum_6m * 50

    # PE: penalise extreme valuations
    if not np.isnan(s.pe) and s.pe > 0:
        score -= max(0, (s.pe - 25) * 0.3)

    # D/E: penalise high leverage
    if not np.isnan(s.de):
        score -= s.de * 5

    return round(score, 2)


def screen_stocks(persona: str, universe: list[StockInfo] = None) -> pd.DataFrame:
    """
    Returns a ranked DataFrame of recommended stocks for the equity bucket.
    If universe is None, fetches fresh data.
    """
    if universe is None:
        universe = fetch_stock_universe(persona)

    cfg = SCREEN_CONFIG[persona]

    filtered = []
    for s in universe:
        beta_ok = np.isnan(s.beta) or s.beta <= cfg["max_beta"]
        pe_ok   = np.isnan(s.pe)   or s.pe   <= cfg["max_pe"]
        de_ok   = np.isnan(s.de)   or s.de   <= cfg["max_de"]
        mom_ok  = np.isnan(s.momentum_6m) or s.momentum_6m > -0.30  # drop stocks down >30%

        if beta_ok and pe_ok and de_ok and mom_ok:
            s.score = _composite_score(s, cfg)
            filtered.append(s)

    df = pd.DataFrame([s.__dict__ for s in filtered])
    df = df.sort_values("score", ascending=False).head(cfg["top_n"])
    df = df.reset_index(drop=True)
    df.index += 1  # 1-based ranking
    return df


# ── Equal-weight basket for display ──────────────────────────────────────────

def build_equity_basket(screened_df: pd.DataFrame,
                        equity_allocation_pct: float) -> pd.DataFrame:
    """
    Takes the screened stocks and splits the equity_allocation_pct equally.
    Returns df with a 'portfolio_weight' column (% of total portfolio).
    """
    n = len(screened_df)
    screened_df = screened_df.copy()
    screened_df["equity_weight_pct"]    = round(100 / n, 2)
    screened_df["portfolio_weight_pct"] = round(equity_allocation_pct / n, 2)
    return screened_df
