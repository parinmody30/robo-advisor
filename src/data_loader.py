import pandas as pd
import numpy as np
import yfinance as yf

ASSETS = {
    "Equity": "^NSEI",
    "Gold":   "GOLDBEES.NS",
    "Debt":   "0P0000XVZM.BO",
}

CASH_ANNUAL_RETURN = 0.06  # liquid fund proxy — ~6% p.a.


def fetch_prices(start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    """Download adjusted close prices for all asset proxies."""
    prices = {}
    for name, ticker in ASSETS.items():
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        prices[name] = data["Close"]

    df = pd.DataFrame(prices).dropna()
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns for each asset."""
    return np.log(prices / prices.shift(1)).dropna()


def annualised_stats(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Returns:
        mu  — annualised expected return per asset (Series)
        cov — annualised covariance matrix (DataFrame)
    """
    trading_days = 252
    mu  = returns.mean() * trading_days
    cov = returns.cov()  * trading_days

    # Add Cash as a synthetic asset (zero variance, fixed return)
    mu["Cash"]          = CASH_ANNUAL_RETURN
    cov["Cash"]         = 0.0
    cov.loc["Cash"]     = 0.0
    cov.loc["Cash", "Cash"] = 1e-8  # near-zero variance, not exactly 0 to keep matrix PSD

    return mu, cov


def load_all(start: str = "2015-01-01") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Convenience: returns (prices, mu, cov)."""
    prices  = fetch_prices(start=start)
    returns = compute_returns(prices)
    mu, cov = annualised_stats(returns)
    return prices, mu, cov
