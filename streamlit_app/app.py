import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json, warnings
warnings.filterwarnings("ignore")

from risk_profiler import build_profile
from goal_planner import plan_goals, summary_table
from optimizer import run_optimizer, generate_frontier, total_equity_weight, ASSET_LABELS, ASSET_COLORS
from monte_carlo import run_simulation
from rebalancer import compute_glide_path, detect_drift, simulate_drift, rebalance_summary
from data_loader import load_all, SYNTHETIC_ASSETS
from stock_screener import fetch_stock_universe, screen_stocks, build_equity_basket

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Robo-Advisor | Goal-Based Portfolio Planner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 16px 20px; margin: 8px 0;
    }
    .persona-badge {
        display: inline-block; padding: 6px 18px;
        border-radius: 20px; font-weight: 700;
        font-size: 1.1rem; margin-top: 8px;
    }
    .conservative { background:#E8F5E9; color:#1B5E20; }
    .balanced     { background:#E3F2FD; color:#0D47A1; }
    .aggressive   { background:#FFF3E0; color:#E65100; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/investment-portfolio.png", width=60)
    st.title("Your Profile")

    st.subheader("📋 Capacity")
    age                   = st.slider("Age", 20, 65, 28)
    income_stability      = st.selectbox("Income stability",
                                          [1, 2, 3],
                                          format_func=lambda x: {1:"Unstable",2:"Moderate",3:"Stable"}[x],
                                          index=2)
    dependents            = st.slider("Financial dependents", 0, 5, 0)
    emergency_months      = st.slider("Emergency fund (months)", 0, 24, 6)
    horizon_years         = st.slider("Investment horizon (years)", 3, 35, 15)

    st.subheader("🧠 Behaviour")
    market_drop   = st.selectbox("If portfolio drops 20%, you:",
                                  [1,2,3],
                                  format_func=lambda x:{1:"Sell everything",2:"Hold",3:"Buy more"}[x],
                                  index=2)
    exp           = st.selectbox("Investing experience",
                                  [1,2,3],
                                  format_func=lambda x:{1:"None",2:"Some MFs",3:"Stocks/ETFs"}[x],
                                  index=1)
    loss_sleep    = st.selectbox("Portfolio -15%, your sleep:",
                                  [1,2,3],
                                  format_func=lambda x:{1:"Badly affected",2:"Uneasy",3:"Fine"}[x],
                                  index=2)
    vol_comfort   = st.selectbox("Comfortable with swings of:",
                                  [1,2,3],
                                  format_func=lambda x:{1:"<5%",2:"5–15%",3:">15%"}[x],
                                  index=1)
    goal_flex     = st.selectbox("Goal timeline flexibility:",
                                  [1,2,3],
                                  format_func=lambda x:{1:"Rigid",2:"Somewhat flexible",3:"Flexible"}[x],
                                  index=1)

    st.subheader("🎯 Goals & Savings")
    monthly_savings = st.number_input("Monthly savings (₹)", 1000, 500000, 25000, step=1000)

    st.subheader("📌 Primary Goal")
    goal_name    = st.text_input("Goal name", "Retirement Corpus")
    goal_amount  = st.number_input("Target amount today (₹)", 100000, 100000000, 10000000, step=100000)
    goal_inf     = st.slider("Inflation rate for this goal (%)", 4, 12, 6) / 100

    st.subheader("➕ Additional Goals")
    add_goal2 = st.checkbox("Add Goal 2")
    goal2_name, goal2_amount, goal2_years, goal2_inf = "", 0, 5, 0.06
    if add_goal2:
        goal2_name   = st.text_input("Goal 2 name", "House Down Payment")
        goal2_amount = st.number_input("Goal 2 amount (₹)", 100000, 50000000, 2000000, step=100000)
        goal2_years  = st.slider("Goal 2 horizon (years)", 1, 30, 7)
        goal2_inf    = st.slider("Goal 2 inflation (%)", 4, 12, 7) / 100

    run_btn = st.button("🚀 Generate My Plan", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📈 Goal-Based Robo-Advisor")
st.caption("Powered by Markowitz MVO · Monte Carlo Simulation · Indian Asset Markets")

if not run_btn:
    st.info("👈 Fill in your profile on the left and click **Generate My Plan**.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Asset Classes", "13", "equity, debt, gold, fixed")
    col2.metric("Stocks Universe", "80", "large, mid, small cap")
    col3.metric("MC Simulations", "10,000", "paths per goal")
    col4.metric("Horizon Support", "Up to 35yr", "with glide path")
    st.stop()

# ── Compute everything ────────────────────────────────────────────────────────
with st.spinner("Fetching market data…"):
    prices, mu, cov = load_all(start="2015-01-01")

tolerance_answers = {
    "market_drop_reaction": market_drop,
    "past_investing_exp":   exp,
    "loss_sleep":           loss_sleep,
    "volatility_comfort":   vol_comfort,
    "goal_flexibility":     goal_flex,
}
profile = build_profile(age, income_stability, dependents,
                         emergency_months, horizon_years, tolerance_answers)

goals_input = [{
    "name": goal_name, "target_amount": goal_amount,
    "years_to_goal": horizon_years, "priority": 1,
    "inflation_rate": goal_inf,
}]
if add_goal2:
    goals_input.append({
        "name": goal2_name, "target_amount": goal2_amount,
        "years_to_goal": goal2_years, "priority": 2,
        "inflation_rate": goal2_inf,
    })

planned_goals = plan_goals(goals_input, monthly_savings, profile.target_return)

with st.spinner("Running optimizer…"):
    opt = run_optimizer(mu, cov, profile.persona, profile.target_return)

with st.spinner("Running Monte Carlo (10,000 paths)…"):
    primary = planned_goals[0]
    mc = run_simulation(
        prices=prices, weights=opt["weights"],
        synthetic_assets=SYNTHETIC_ASSETS,
        initial_investment=0,
        monthly_sip=monthly_savings,
        horizon_years=primary.years_to_goal,
        target_corpus=primary.future_value,
        n_sims=10_000,
    )

glide = compute_glide_path(opt["weights"], horizon_years)
eq_pct = total_equity_weight(opt["weights"]) * 100

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "👤 Profile", "📊 Allocation", "🎯 Goals",
    "🎲 Projections", "🔄 Rebalancing"
])

# ─────────────────────────── TAB 1: PROFILE ──────────────────────────────────
with t1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Persona", profile.persona)
    c2.metric("Combined Score", f"{profile.combined_score} / 10")
    c3.metric("Target Return", f"{profile.target_return*100:.0f}% p.a.")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure(go.Bar(
            x=["Capacity", "Tolerance", "Combined"],
            y=[profile.capacity_score, profile.tolerance_score, profile.combined_score],
            marker_color=["#4CAF50", "#2196F3", "#FF9800"],
            text=[f"{v:.1f}" for v in [profile.capacity_score, profile.tolerance_score, profile.combined_score]],
            textposition="outside",
        ))
        fig.update_layout(title="Risk Score Breakdown", yaxis=dict(range=[0,11]),
                          template="plotly_white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if profile.tolerance_score > profile.capacity_score:
            st.warning("⚠️ Your **tolerance** exceeds your **capacity** — you *want* more risk than you can financially afford. Capacity is the hard ceiling.")
        else:
            st.success("✅ Capacity and tolerance are aligned.")

        persona_color = {"Conservative":"#E8F5E9","Balanced":"#E3F2FD","Aggressive":"#FFF3E0"}
        st.markdown(f"""
        <div style='background:{persona_color.get(profile.persona,"#eee")};
                    border-radius:12px; padding:20px; margin-top:12px;'>
            <h3 style='margin:0'>{profile.persona} Investor</h3>
            <p style='color:#555; margin:8px 0 0 0'>
                Target return: <b>{profile.target_return*100:.0f}% p.a.</b><br>
                Capacity: <b>{profile.capacity_score}/10</b> &nbsp;|&nbsp;
                Tolerance: <b>{profile.tolerance_score}/10</b>
            </p>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────── TAB 2: ALLOCATION ───────────────────────────────
with t2:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected Return", f"{opt['expected_return']*100:.1f}%")
    m2.metric("Volatility",      f"{opt['volatility']*100:.1f}%")
    m3.metric("Sharpe Ratio",    f"{opt['sharpe_ratio']:.2f}")
    m4.metric("Total Equity",    f"{eq_pct:.1f}%")

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        labels = [ASSET_LABELS.get(k,k) for k in opt["weights"]]
        values = list(opt["weights"].values())
        colors = [ASSET_COLORS.get(k,"#888") for k in opt["weights"]]
        fig2 = go.Figure(go.Pie(labels=labels, values=values,
                                 marker_colors=colors,
                                 textinfo="label+percent", hole=0.38))
        fig2.update_layout(title="Optimal Asset Allocation",
                           template="plotly_white", height=460)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Allocation breakdown")
        rows = [{
            "Asset":        ASSET_LABELS.get(k, k),
            "Weight":       f"{v*100:.1f}%",
            "Monthly (₹)":  f"₹{v*monthly_savings:,.0f}",
        } for k, v in sorted(opt["weights"].items(), key=lambda x: -x[1]) if v > 0.005]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Efficient frontier
    with st.spinner("Building efficient frontier…"):
        frontier = generate_frontier(mu, cov, profile.persona)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=frontier["volatility"]*100, y=frontier["return"]*100,
                               mode="lines", name="Efficient Frontier",
                               line=dict(color="#1565C0", width=3)))
    fig3.add_trace(go.Scatter(x=[opt["volatility"]*100], y=[opt["expected_return"]*100],
                               mode="markers+text", name="Your Portfolio",
                               marker=dict(size=14, color="#E53935", symbol="star"),
                               text=[profile.persona], textposition="top right"))
    fig3.update_layout(title="Efficient Frontier", xaxis_title="Volatility %",
                       yaxis_title="Expected Return %",
                       template="plotly_white", height=420)
    st.plotly_chart(fig3, use_container_width=True)

    # Stock basket
    st.subheader(f"📋 Recommended Stocks for Equity Bucket ({eq_pct:.0f}% of portfolio)")
    with st.spinner("Screening stocks…"):
        try:
            universe = fetch_stock_universe(profile.persona)
            screened = screen_stocks(profile.persona, universe)
            basket   = build_equity_basket(screened, eq_pct, monthly_savings)

            def fmt(val, pct=False, x2=False, na="—"):
                if pd.isna(val): return na
                if pct:  return f"{val*100:.1f}%"
                if x2:   return f"{val:.2f}"
                return f"{val:.1f}"

            display = basket.copy()
            for col, kw in [
                ("beta",           {"x2": True}),
                ("pe",             {}),
                ("ev_ebitda",      {}),
                ("pb",             {"x2": True}),
                ("roe",            {"pct": True}),
                ("net_margin",     {"pct": True}),
                ("revenue_growth", {"pct": True}),
                ("de",             {"x2": True}),
                ("current_ratio",  {"x2": True}),
                ("momentum_6m",    {"pct": True}),
            ]:
                if col in display.columns:
                    display[col] = display[col].apply(lambda x, k=kw: fmt(x, **k))

            display["monthly_amt_inr"] = display["monthly_amt_inr"].apply(
                lambda x: f"₹{int(x):,}"
            )

            show = ["name","tier","beta","pe","ev_ebitda","pb",
                    "roe","net_margin","revenue_growth",
                    "de","current_ratio","momentum_6m",
                    "portfolio_weight_pct","monthly_amt_inr"]
            show = [c for c in show if c in display.columns]

            st.dataframe(
                display[show].rename(columns={
                    "name":                "Company",
                    "tier":                "Cap",
                    "beta":                "Beta",
                    "pe":                  "P/E",
                    "ev_ebitda":           "EV/EBITDA",
                    "pb":                  "P/B",
                    "roe":                 "ROE",
                    "net_margin":          "Net Margin",
                    "revenue_growth":      "Rev Growth",
                    "de":                  "D/E",
                    "current_ratio":       "Curr Ratio",
                    "momentum_6m":         "6m Return",
                    "portfolio_weight_pct":"Portfolio %",
                    "monthly_amt_inr":     "Invest/Month",
                }),
                use_container_width=True,
                hide_index=False,
            )
        except Exception as e:
            st.warning(f"Stock screening skipped: {e}")

# ─────────────────────────── TAB 3: GOALS ────────────────────────────────────
with t3:
    df_goals = pd.DataFrame(summary_table(planned_goals))
    st.dataframe(df_goals, use_container_width=True, hide_index=True)

    g_names = [g.name for g in planned_goals]
    sips    = [g.monthly_sip for g in planned_goals]
    feas    = [g.feasibility_pct for g in planned_goals]
    colors_g = ["#2E7D32" if f>=100 else "#E65100" if f<50 else "#F9A825" for f in feas]

    col1, col2 = st.columns(2)
    with col1:
        fig4 = go.Figure(go.Bar(x=g_names, y=sips, marker_color=colors_g,
                                 text=[f"₹{v:,.0f}" for v in sips], textposition="outside"))
        fig4.add_hline(y=monthly_savings, line_dash="dash", line_color="#1565C0",
                       annotation_text=f"Your savings ₹{monthly_savings:,}/mo")
        fig4.update_layout(title="SIP Required vs Your Savings",
                           yaxis_title="₹/month", template="plotly_white", height=400)
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(name="Today's value",
                               x=g_names, y=[g.target_amount for g in planned_goals],
                               marker_color="#42A5F5"))
        fig5.add_trace(go.Bar(name="Future value (inflation-adj)",
                               x=g_names, y=[g.future_value for g in planned_goals],
                               marker_color="#EF5350"))
        fig5.update_layout(barmode="group", title="Inflation Impact",
                           yaxis_title="₹", template="plotly_white", height=400)
        st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────── TAB 4: PROJECTIONS ──────────────────────────────
with t4:
    col1, col2, col3 = st.columns(3)
    col1.metric("Probability of Success", f"{mc.prob_success}%",
                delta=f"{mc.prob_success-mc.stress_prob_success:+.1f}% vs crash")
    col2.metric("After 2008-style Crash", f"{mc.stress_prob_success}%")
    col3.metric("Median Terminal Value",
                f"₹{mc.percentiles['p50'].iloc[-1]/1e5:.1f}L")

    years = mc.percentiles.index
    p     = mc.percentiles

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=list(years)+list(years[::-1]),
        y=list(p["p90"])+list(p["p10"][::-1]),
        fill="toself", fillcolor="rgba(21,101,192,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="10–90th pct",
    ))
    fig6.add_trace(go.Scatter(
        x=list(years)+list(years[::-1]),
        y=list(p["p75"])+list(p["p25"][::-1]),
        fill="toself", fillcolor="rgba(21,101,192,0.22)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75th pct",
    ))
    fig6.add_trace(go.Scatter(x=years, y=p["p50"], mode="lines",
                               name="Median", line=dict(color="#1565C0", width=3)))
    fig6.add_trace(go.Scatter(x=years, y=mc.stress_percentiles["p50"],
                               mode="lines", name="Median (post-crash)",
                               line=dict(color="#E53935", width=2, dash="dot")))
    fig6.add_hline(y=primary.future_value, line_dash="dash", line_color="#388E3C",
                   annotation_text=f"Target ₹{primary.future_value/1e5:.0f}L")
    fig6.update_layout(
        title=f"{primary.name} — Monte Carlo Projection ({mc.prob_success}% success)",
        xaxis_title="Years", yaxis_title="Portfolio Value (₹)",
        template="plotly_white", height=480,
    )
    st.plotly_chart(fig6, use_container_width=True)

    # Histogram
    fig7 = go.Figure(go.Histogram(x=mc.terminal_values/1e5, nbinsx=80,
                                   marker_color="#1565C0", opacity=0.75))
    fig7.add_vline(x=primary.future_value/1e5, line_dash="dash", line_color="#E53935",
                   annotation_text="Target")
    fig7.update_layout(title="Terminal Wealth Distribution",
                       xaxis_title="₹ Lakhs", yaxis_title="# simulations",
                       template="plotly_white", height=360)
    st.plotly_chart(fig7, use_container_width=True)

# ─────────────────────────── TAB 5: REBALANCING ──────────────────────────────
with t5:
    col1, col2 = st.columns(2)
    with col1:
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=glide.years, y=glide.equity_pct,
                                   name="Equity %", mode="lines",
                                   line=dict(color="#1565C0", width=3),
                                   fill="tozeroy", fillcolor="rgba(21,101,192,0.12)"))
        fig8.add_trace(go.Scatter(x=glide.years, y=glide.debt_pct,
                                   name="Debt + Fixed %", mode="lines",
                                   line=dict(color="#2E7D32", width=3),
                                   fill="tozeroy", fillcolor="rgba(46,125,50,0.12)"))
        fig8.update_layout(title="Glide Path",
                           xaxis_title="Years", yaxis_title="Allocation %",
                           template="plotly_white", height=400)
        st.plotly_chart(fig8, use_container_width=True)

    with col2:
        st.subheader("Drift Simulator")
        bull_years = st.slider("Simulate bull run (years)", 1, 5, 3)
        portfolio_val = st.number_input("Portfolio value (₹)", 100000, 50000000, 1000000, step=50000)

        bull_returns = {
            "Equity_LargeCap":0.18,"Equity_MidCap":0.22,"Equity_SmallCap":0.25,
            "Equity_Intl":0.15,"ELSS":0.20,"Gold":0.08,"Silver":0.10,
            "Debt_Gilt":0.07,"Debt_Corporate":0.08,"FixedDeposit":0.07,
            "PPF":0.071,"RBI_Bond":0.0805,"LiquidFund":0.065,
        }
        drifted = simulate_drift(opt["weights"], bull_returns, years=bull_years)
        actions = detect_drift(opt["weights"], drifted, portfolio_val, threshold=0.05)
        df_reb  = rebalance_summary(actions)

        if df_reb.empty:
            st.success(f"✅ No rebalancing needed after {bull_years}-year bull run.")
        else:
            st.warning(f"⚠️ Rebalancing needed after {bull_years}-year bull run:")
            st.dataframe(df_reb, use_container_width=True, hide_index=True)

    # Drift bar chart
    assets_d = list(opt["weights"].keys())
    fig9 = go.Figure()
    fig9.add_trace(go.Bar(name="Target",
                           x=[ASSET_LABELS.get(a,a) for a in assets_d],
                           y=[opt["weights"].get(a,0)*100 for a in assets_d],
                           marker_color="#1565C0", opacity=0.85))
    fig9.add_trace(go.Bar(name=f"After {bull_years}yr bull",
                           x=[ASSET_LABELS.get(a,a) for a in assets_d],
                           y=[drifted.get(a,0)*100 for a in assets_d],
                           marker_color="#E53935", opacity=0.85))
    fig9.update_layout(barmode="group", title="Portfolio Drift",
                       xaxis_tickangle=-35, yaxis_title="Weight %",
                       template="plotly_white", height=420)
    st.plotly_chart(fig9, use_container_width=True)
