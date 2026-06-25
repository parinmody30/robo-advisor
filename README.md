# Goal-Based Robo-Advisor

An end-to-end quantitative portfolio advisor for Indian retail investors.

## What it does
User inputs age, financial goals, risk appetite, and investment horizon →
system computes an optimal asset allocation and projects goal outcomes.

## Quant stack
| Layer | Method |
|-------|--------|
| Risk profiling | Weighted questionnaire → risk score → persona |
| Asset allocation | Markowitz Mean-Variance Optimization (efficient frontier) |
| Advanced allocation | Black-Litterman, CVaR optimization |
| Projections | Monte Carlo simulation (10k paths, probability cone) |
| Lifecycle | Glide path + drift-based rebalancing |

## Indian asset-class proxies
| Class | Proxy | Ticker |
|-------|-------|--------|
| Equity | Nifty 50 | `^NSEI` |
| Debt | ICICI Pru Gilt Fund | `0P0000XVZM.BO` |
| Gold | Nippon India Gold ETF | `GOLDBEES.NS` |
| Cash | Liquid fund proxy | 6% p.a. fixed |

## Modules
```
notebooks/
├── 01_risk_profiling.ipynb
├── 02_goal_planning.ipynb
├── 03_asset_allocation.ipynb
├── 04_monte_carlo.ipynb
├── 05_rebalancing_glidepath.ipynb
└── 06_full_demo.ipynb

src/
├── risk_profiler.py
├── goal_planner.py
├── optimizer.py
├── monte_carlo.py
├── rebalancer.py
└── data_loader.py

streamlit_app/
└── app.py
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

## Run on Colab
Open any notebook in `notebooks/` via Google Colab.
The first cell installs dependencies automatically.
