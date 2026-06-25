"""
Stock Screener
--------------
Curated universe of ~80 quality Indian stocks, organised into tiers.
Given a risk persona, screens using an 8-factor composite model:

  Valuation  : P/E ratio, EV/EBITDA, Price-to-Book
  Quality    : ROE (return on equity), net profit margin
  Growth     : revenue growth (YoY)
  Leverage   : Debt/Equity ratio, current ratio
  Risk       : Beta (vs Nifty 50)
  Momentum   : 6-month price return

Factor weights differ per persona — conservative tilts toward
quality/low-beta; aggressive tilts toward growth/momentum.
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
    name:           str
    ticker:         str
    tier:           str
    # Factor 1 — Risk
    beta:           float = np.nan
    # Factor 2 — Valuation
    pe:             float = np.nan
    ev_ebitda:      float = np.nan
    pb:             float = np.nan
    # Factor 3 — Quality
    roe:            float = np.nan
    net_margin:     float = np.nan
    # Factor 4 — Growth
    revenue_growth: float = np.nan
    # Factor 5 — Leverage
    de:             float = np.nan
    current_ratio:  float = np.nan
    # Factor 6 — Momentum
    momentum_6m:    float = np.nan
    # Price data
    current_price:  float = np.nan
    # Output
    score:          float = np.nan


def _fetch_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "beta":           info.get("beta",                  np.nan),
            "pe":             info.get("trailingPE",            np.nan),
            "ev_ebitda":      info.get("enterpriseToEbitda",    np.nan),
            "pb":             info.get("priceToBook",           np.nan),
            "roe":            info.get("returnOnEquity",        np.nan),
            "net_margin":     info.get("profitMargins",         np.nan),
            "revenue_growth": info.get("revenueGrowth",        np.nan),
            "de":             info.get("debtToEquity",          np.nan),
            "current_ratio":  info.get("currentRatio",         np.nan),
        }
    except Exception:
        return {k: np.nan for k in ["beta","pe","ev_ebitda","pb","roe",
                                     "net_margin","revenue_growth","de","current_ratio"]}


def _fetch_current_price(ticker: str) -> float:
    try:
        hist = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if hist.empty:
            return np.nan
        return float(hist["Close"].squeeze().iloc[-1])
    except Exception:
        return np.nan


def _fetch_momentum(ticker: str, months: int = 6) -> float:
    try:
        hist = yf.download(ticker, period="6mo",
                           auto_adjust=True, progress=False)
        if hist.empty or len(hist) < 2:
            return np.nan
        close = hist["Close"].squeeze()
        return float((close.iloc[-1] / close.iloc[0]) - 1)
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
        f = _fetch_fundamentals(s.ticker)
        s.beta           = f["beta"]
        s.pe             = f["pe"]
        s.ev_ebitda      = f["ev_ebitda"]
        s.pb             = f["pb"]
        s.roe            = f["roe"]
        s.net_margin     = f["net_margin"]
        s.revenue_growth = f["revenue_growth"]
        s.de             = f["de"] / 100 if not np.isnan(f["de"]) else np.nan
        s.current_ratio  = f["current_ratio"]
        s.momentum_6m    = _fetch_momentum(s.ticker)
        s.current_price  = _fetch_current_price(s.ticker)

    return universe


# ── Screening & ranking ───────────────────────────────────────────────────────

# Per-persona factor weights — how much each factor contributes to the score
FACTOR_WEIGHTS = {
    #                      conservative  balanced  aggressive
    "beta_low":           [  3.0,         2.0,       0.5  ],  # reward low beta
    "pe_low":             [  2.5,         1.5,       0.5  ],  # reward low PE
    "ev_ebitda_low":      [  2.0,         1.5,       0.5  ],  # reward low EV/EBITDA
    "pb_low":             [  1.5,         1.0,       0.5  ],  # reward low PB (value)
    "roe_high":           [  3.0,         2.5,       2.0  ],  # reward high ROE
    "margin_high":        [  2.5,         2.0,       1.5  ],  # reward high net margin
    "revenue_growth_high":[  1.0,         2.0,       3.5  ],  # reward revenue growth
    "de_low":             [  3.0,         2.0,       1.0  ],  # reward low leverage
    "current_ratio_high": [  2.0,         1.5,       1.0  ],  # reward liquidity
    "momentum_high":      [  0.5,         2.0,       3.5  ],  # reward price momentum
}
PERSONA_IDX = {"Conservative": 0, "Balanced": 1, "Aggressive": 2}


def _composite_score(s: StockInfo, persona: str) -> float:
    """
    8-factor composite score with persona-specific weights.
    Each factor is normalised to a 0-10 signal before weighting.
    Higher final score = better stock for this persona.
    """
    idx = PERSONA_IDX.get(persona, 1)
    w   = {k: v[idx] for k, v in FACTOR_WEIGHTS.items()}
    score = 0.0

    # 1. Beta — reward low beta (signal: max(0, 2-beta) capped at 2)
    if not np.isnan(s.beta):
        score += w["beta_low"] * max(0, min(2.0, 2.0 - s.beta)) * 5

    # 2. PE — reward reasonable valuation (signal decays above 25)
    if not np.isnan(s.pe) and s.pe > 0:
        pe_signal = max(0, 10 - max(0, (s.pe - 15) * 0.2))
        score += w["pe_low"] * pe_signal

    # 3. EV/EBITDA — reward <15, penalise >25
    if not np.isnan(s.ev_ebitda) and s.ev_ebitda > 0:
        ev_signal = max(0, 10 - max(0, (s.ev_ebitda - 10) * 0.4))
        score += w["ev_ebitda_low"] * ev_signal

    # 4. Price-to-Book — reward low PB (value factor)
    if not np.isnan(s.pb) and s.pb > 0:
        pb_signal = max(0, 10 - min(10, s.pb * 1.5))
        score += w["pb_low"] * pb_signal

    # 5. ROE — reward high ROE (>15% is good, >25% is excellent)
    if not np.isnan(s.roe):
        roe_signal = min(10, max(0, s.roe * 100 * 0.4))
        score += w["roe_high"] * roe_signal

    # 6. Net profit margin — reward higher margins
    if not np.isnan(s.net_margin):
        margin_signal = min(10, max(0, s.net_margin * 100 * 0.5))
        score += w["margin_high"] * margin_signal

    # 7. Revenue growth — reward positive YoY growth
    if not np.isnan(s.revenue_growth):
        growth_signal = min(10, max(0, s.revenue_growth * 100 * 0.4))
        score += w["revenue_growth_high"] * growth_signal

    # 8. Debt/Equity — reward low leverage
    if not np.isnan(s.de):
        de_signal = max(0, 10 - min(10, s.de * 4))
        score += w["de_low"] * de_signal

    # 9. Current ratio — reward >1.5 (healthy liquidity)
    if not np.isnan(s.current_ratio):
        cr_signal = min(10, max(0, (s.current_ratio - 0.5) * 2.5))
        score += w["current_ratio_high"] * cr_signal

    # 10. 6-month momentum — reward positive price trend
    if not np.isnan(s.momentum_6m):
        mom_signal = min(10, max(0, s.momentum_6m * 100 * 0.4 + 5))
        score += w["momentum_high"] * mom_signal

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
            s.score = _composite_score(s, persona)
            filtered.append(s)

    df = pd.DataFrame([s.__dict__ for s in filtered])
    df = df.sort_values("score", ascending=False).head(cfg["top_n"])
    df = df.reset_index(drop=True)
    df.index += 1  # 1-based ranking
    return df


# ── Sector ETF alternatives ───────────────────────────────────────────────────
# When a stock is unaffordable, we suggest the closest sector ETF.
# All ETFs trade at ₹100–₹700, always affordable for SIP.

STOCK_TO_ETF = {
    # IT
    "TCS":                  ("Nifty IT ETF (ITBEES)",        "ITBEES.NS",    "Same IT sector exposure, ₹500 min SIP"),
    "Infosys":              ("Nifty IT ETF (ITBEES)",        "ITBEES.NS",    "Same IT sector exposure, ₹500 min SIP"),
    "HCL Technologies":     ("Nifty IT ETF (ITBEES)",        "ITBEES.NS",    "Same IT sector exposure, ₹500 min SIP"),
    "Wipro":                ("Nifty IT ETF (ITBEES)",        "ITBEES.NS",    "Same IT sector exposure, ₹500 min SIP"),
    "Persistent Systems":   ("Nifty IT ETF (ITBEES)",        "ITBEES.NS",    "Same IT sector exposure, ₹500 min SIP"),
    "Coforge":              ("Nifty IT ETF (ITBEES)",        "ITBEES.NS",    "Same IT sector exposure, ₹500 min SIP"),
    "Happiest Minds":       ("Nifty IT ETF (ITBEES)",        "ITBEES.NS",    "Same IT sector exposure, ₹500 min SIP"),
    "Latent View Analytics":("Nifty IT ETF (ITBEES)",        "ITBEES.NS",    "Same IT sector exposure, ₹500 min SIP"),
    # Banking & Finance
    "HDFC Bank":            ("Nifty Bank ETF (BANKBEES)",    "BANKBEES.NS",  "Banking sector, ₹500 min SIP"),
    "ICICI Bank":           ("Nifty Bank ETF (BANKBEES)",    "BANKBEES.NS",  "Banking sector, ₹500 min SIP"),
    "Kotak Mahindra Bank":  ("Nifty Bank ETF (BANKBEES)",    "BANKBEES.NS",  "Banking sector, ₹500 min SIP"),
    "Axis Bank":            ("Nifty Bank ETF (BANKBEES)",    "BANKBEES.NS",  "Banking sector, ₹500 min SIP"),
    "State Bank of India":  ("Nifty Bank ETF (BANKBEES)",    "BANKBEES.NS",  "Banking sector, ₹500 min SIP"),
    "Bajaj Finance":        ("Nifty Financial Svcs ETF",     "NETFNIFTY.NS", "Financial services exposure"),
    "Muthoot Finance":      ("Nifty Financial Svcs ETF",     "NETFNIFTY.NS", "Financial services exposure"),
    "Chola Finance":        ("Nifty Financial Svcs ETF",     "NETFNIFTY.NS", "Financial services exposure"),
    # Pharma
    "Sun Pharma":           ("Nifty Pharma ETF (PHARMABEES)","PHARMABEES.NS","Pharma sector, ₹500 min SIP"),
    "Cipla":                ("Nifty Pharma ETF (PHARMABEES)","PHARMABEES.NS","Pharma sector, ₹500 min SIP"),
    "Dr. Reddy's":          ("Nifty Pharma ETF (PHARMABEES)","PHARMABEES.NS","Pharma sector, ₹500 min SIP"),
    # FMCG
    "Hindustan Unilever":   ("Nifty FMCG ETF (FMCGIETF)",   "FMCGIETF.NS",  "FMCG sector, ₹500 min SIP"),
    "ITC":                  ("Nifty FMCG ETF (FMCGIETF)",   "FMCGIETF.NS",  "FMCG sector, ₹500 min SIP"),
    "Nestle India":         ("Nifty FMCG ETF (FMCGIETF)",   "FMCGIETF.NS",  "FMCG sector, ₹500 min SIP"),
    "Britannia":            ("Nifty FMCG ETF (FMCGIETF)",   "FMCGIETF.NS",  "FMCG sector, ₹500 min SIP"),
    # Auto
    "Maruti Suzuki":        ("Nifty Auto ETF (AUTOIETF)",   "AUTOIETF.NS",  "Auto sector exposure"),
    "Tata Motors":          ("Nifty Auto ETF (AUTOIETF)",   "AUTOIETF.NS",  "Auto sector exposure"),
    "Bajaj Auto":           ("Nifty Auto ETF (AUTOIETF)",   "AUTOIETF.NS",  "Auto sector exposure"),
    # Metals
    "Tata Steel":           ("Nifty Metal ETF (METALIETF)", "METALIETF.NS", "Metals & mining exposure"),
    "JSW Steel":            ("Nifty Metal ETF (METALIETF)", "METALIETF.NS", "Metals & mining exposure"),
    # Large cap fallback
    "Reliance Industries":  ("Nifty 50 ETF (NIFTYBEES)",    "NIFTYBEES.NS", "Broad large-cap exposure"),
    "Larsen & Toubro":      ("Nifty 50 ETF (NIFTYBEES)",    "NIFTYBEES.NS", "Broad large-cap exposure"),
    "Bharti Airtel":        ("Nifty 50 ETF (NIFTYBEES)",    "NIFTYBEES.NS", "Broad large-cap exposure"),
    "Asian Paints":         ("Nifty 50 ETF (NIFTYBEES)",    "NIFTYBEES.NS", "Broad large-cap exposure"),
    "Titan Company":        ("Nifty 50 ETF (NIFTYBEES)",    "NIFTYBEES.NS", "Broad large-cap exposure"),
    "UltraTech Cement":     ("Nifty 50 ETF (NIFTYBEES)",    "NIFTYBEES.NS", "Broad large-cap exposure"),
    # Mid cap fallback
    "Trent":                ("Nifty Next 50 ETF (JUNIORBEES)","JUNIORBEES.NS","Mid-large cap exposure"),
    "Voltas":               ("Nifty Next 50 ETF (JUNIORBEES)","JUNIORBEES.NS","Mid-large cap exposure"),
    "Apollo Hospitals":     ("Nifty Next 50 ETF (JUNIORBEES)","JUNIORBEES.NS","Mid-large cap exposure"),
    "Max Healthcare":       ("Nifty Next 50 ETF (JUNIORBEES)","JUNIORBEES.NS","Mid-large cap exposure"),
    "Polycab India":        ("Nifty Next 50 ETF (JUNIORBEES)","JUNIORBEES.NS","Mid-large cap exposure"),
    "Dixon Technologies":   ("Nifty Midcap 150 ETF",        "MID150BEES.NS","Midcap exposure"),
    "Kaynes Technology":    ("Nifty Midcap 150 ETF",        "MID150BEES.NS","Midcap exposure"),
}
ETF_FALLBACK = ("Nifty 50 ETF (NIFTYBEES)", "NIFTYBEES.NS", "Broad market exposure, ₹500 min SIP")


def get_etf_alternative(stock_name: str) -> tuple:
    """Returns (etf_name, etf_ticker, reason) for a given stock."""
    return STOCK_TO_ETF.get(stock_name, ETF_FALLBACK)


# ── Score-weighted basket ─────────────────────────────────────────────────────

def build_equity_basket(
    screened_df: pd.DataFrame,
    equity_allocation_pct: float,
    monthly_investment: float = 0,
) -> pd.DataFrame:
    """
    Score-weighted allocation: stocks with higher composite scores get
    proportionally larger allocations within the equity bucket.

    Args:
        screened_df          : output of screen_stocks()
        equity_allocation_pct: % of total portfolio in equity (e.g. 57.0)
        monthly_investment   : total monthly SIP in ₹ — used to show ₹ amounts
    """
    df = screened_df.copy()

    # Shift scores to be strictly positive before normalising
    min_score = df["score"].min()
    shift = abs(min_score) + 1 if min_score <= 0 else 0
    df["score_adj"] = df["score"] + shift

    # Normalise to get weights within equity bucket
    total_score = df["score_adj"].sum()
    df["equity_weight_pct"]    = (df["score_adj"] / total_score * 100).round(2)
    df["portfolio_weight_pct"] = (df["equity_weight_pct"] * equity_allocation_pct / 100).round(2)

    # ₹ amounts
    if monthly_investment > 0:
        df["monthly_amt_inr"] = (df["portfolio_weight_pct"] / 100 * monthly_investment).round(0).astype(int)
    else:
        df["monthly_amt_inr"] = 0

    # Shares per month + affordability + ETF alternative for unaffordable stocks
    def _affordability(row):
        price = row.get("current_price", np.nan)
        amt   = row.get("monthly_amt_inr", 0)
        name  = row.get("name", "")

        if pd.isna(price) or price <= 0:
            etf_name, etf_ticker, etf_reason = get_etf_alternative(name)
            return pd.Series({
                "shares_per_month":    0,
                "months_to_accumulate": 0,
                "actual_invest_inr":   amt,
                "affordable":          False,
                "etf_name":            etf_name,
                "etf_ticker":          etf_ticker,
                "etf_reason":          etf_reason,
                "action":              "⚠️ Price unavailable",
            })

        shares = int(amt // price)

        if shares >= 1:
            actual   = int(shares * price)
            leftover = int(amt - actual)
            return pd.Series({
                "shares_per_month":    shares,
                "months_to_accumulate": 0,
                "actual_invest_inr":   actual,
                "affordable":          True,
                "etf_name":            "",
                "etf_ticker":          "",
                "etf_reason":          "",
                "action":              f"✅ Buy {shares} share{'s' if shares>1 else ''}"
                                       + (f" · ₹{leftover:,} rolls over" if leftover > 0 else ""),
            })
        else:
            months_needed = int(np.ceil(price / amt)) if amt > 0 else 999
            etf_name, etf_ticker, etf_reason = get_etf_alternative(name)
            return pd.Series({
                "shares_per_month":    0,
                "months_to_accumulate": months_needed,
                "actual_invest_inr":   0,
                "affordable":          False,
                "etf_name":            etf_name,
                "etf_ticker":          etf_ticker,
                "etf_reason":          etf_reason,
                "action": f"⏳ {months_needed}mo to save  |  💡 See ETF alternative below",
            })

    extra = df.apply(_affordability, axis=1)
    df = pd.concat([df, extra], axis=1)
    df = df.drop(columns=["score_adj"])
    return df
