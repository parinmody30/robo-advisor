import pandas as pd
import numpy as np
import yfinance as yf

# ── Asset-class proxies ───────────────────────────────────────────────────────

ASSET_TICKERS = {
    "Equity_LargeCap":  "^NSEI",
    "Equity_MidCap":    "^NSMIDCP",
    "Equity_SmallCap":  "NIFTYSMLCAP250.NS",
    "Equity_Intl":      "MAFANG.NS",
    "Gold":             "GOLDBEES.NS",
    "Silver":           "SILVERIETF.NS",
    "Debt_Gilt":        "0P0000XVZM.BO",
    "Debt_Corporate":   "0P0001IGCN.BO",
    "ELSS":             "0P0000XWDM.BO",
}

# Synthetic instruments — fixed annual return, near-zero variance
SYNTHETIC_ASSETS = {
    "FixedDeposit": 0.070,
    "PPF":          0.071,
    "RBI_Bond":     0.0805,
    "LiquidFund":   0.065,
}


def fetch_prices(start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    prices = {}
    for name, ticker in ASSET_TICKERS.items():
        try:
            data = yf.download(ticker, start=start, end=end,
                               auto_adjust=True, progress=False)
            if not data.empty:
                prices[name] = data["Close"].squeeze()
        except Exception as e:
            print(f"Warning: could not fetch {ticker} ({e})")

    df = pd.DataFrame(prices).dropna()
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def annualised_stats(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    T = 252
    mu  = returns.mean() * T
    cov = returns.cov()  * T

    # Add synthetic assets
    for name, annual_ret in SYNTHETIC_ASSETS.items():
        mu[name]              = annual_ret
        cov[name]             = 0.0
        cov.loc[name]         = 0.0
        cov.loc[name, name]   = 1e-8  # near-zero variance, keeps matrix PSD

    return mu, cov


def load_all(start: str = "2015-01-01") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns (prices, mu, cov) — main entry point for the optimizer."""
    prices  = fetch_prices(start=start)
    returns = compute_returns(prices)
    mu, cov = annualised_stats(returns)
    return prices, mu, cov
