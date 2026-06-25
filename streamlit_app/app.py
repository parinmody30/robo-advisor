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
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #13112E 0%, #1E1B4B 40%, #1a1836 100%);
    border-right: 1px solid rgba(99,102,241,0.25);
}
[data-testid="stSidebar"] * { color: #C7D2FE !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #EEF2FF !important; }

/* Section group labels */
[data-testid="stSidebar"] .stMarkdown strong {
    color: #818CF8 !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* Input labels */
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stNumberInput > label,
[data-testid="stSidebar"] .stTextInput > label {
    color: #A5B4FC !important; font-size: 0.75rem !important;
    text-transform: uppercase !important; letter-spacing: 0.07em !important;
    font-weight: 600 !important;
}

/* Inputs & dropdowns */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(129,140,248,0.35) !important;
    border-radius: 8px !important; color: #EEF2FF !important;
    backdrop-filter: blur(4px) !important;
}
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(129,140,248,0.35) !important;
    border-radius: 8px !important; color: #EEF2FF !important;
}
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
    border-color: #818CF8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.2) !important;
}

/* Slider track */
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background: rgba(255,255,255,0.12) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: #818CF8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.3) !important;
}

/* Slider value text */
[data-testid="stSidebar"] [data-testid="stSlider"] p {
    color: #EEF2FF !important; font-weight: 600 !important;
}

/* CTA Button */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 50%, #14B8A6 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 0.92rem !important; padding: 0.65rem 1rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.5) !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.65) !important;
}

/* Dividers */
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(129,140,248,0.2) !important;
    margin: 4px 0 !important;
}

/* ── Main background ── */
[data-testid="stAppViewContainer"] > .main { background: #F8FAFC; }
[data-testid="stHeader"] { background: transparent; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1E1B4B 0%, #4F46E5 55%, #14B8A6 100%);
    border-radius: 16px; padding: 36px 40px; margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(79,70,229,0.25);
}
.hero h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.02em; }
.hero p  { color: #C7D2FE; margin: 0; font-size: 0.95rem; letter-spacing: 0.02em; }

/* ── Stat chips on landing ── */
.chip-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 20px; }
.chip {
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 100px; padding: 6px 18px; color: #fff;
    font-size: 0.82rem; font-weight: 500; backdrop-filter: blur(4px);
}

/* ── Stat cards (metric replacement) ── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 24px; }
.stat-card {
    background: #fff; border-radius: 12px; padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06); border: 1px solid #E2E8F0;
    transition: box-shadow 0.2s;
}
.stat-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.10); }
.stat-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 600; margin-bottom: 6px; }
.stat-value { font-size: 1.55rem; font-weight: 700; color: #0F172A; line-height: 1; }
.stat-sub   { font-size: 0.78rem; color: #64748B; margin-top: 4px; }

/* ── Section headers ── */
.section-title {
    font-size: 1.05rem; font-weight: 700; color: #111827;
    letter-spacing: -0.01em; margin: 28px 0 14px 0;
    padding-bottom: 8px; border-bottom: 2px solid #4F46E5;
}

/* ── Persona badge ── */
.persona-card {
    border-radius: 14px; padding: 22px 26px;
    border-left: 5px solid; margin-top: 8px;
}
.persona-conservative { background:#F0FDFA; border-color:#14B8A6; }
.persona-balanced     { background:#EEF2FF; border-color:#4F46E5; }
.persona-aggressive   { background:#FFF7ED; border-color:#F97316; }
.persona-name  { font-size: 1.3rem; font-weight: 700; color: #0F172A; margin: 0 0 6px 0; }
.persona-meta  { font-size: 0.85rem; color: #64748B; line-height: 1.7; }
.persona-meta b { color: #0F172A; }

/* ── Risk meter ── */
.risk-bar-wrap { background: linear-gradient(90deg,#22C55E,#EAB308,#EF4444); height: 6px; border-radius: 6px; margin: 8px 0 4px 0; }

/* ── Option cards ── */
.opt-card {
    border-radius: 12px; padding: 18px 20px;
    border: 1px solid; margin-bottom: 8px;
}
.opt-a { background:#EFF6FF; border-color:#BFDBFE; }
.opt-b { background:#F0FDF4; border-color:#BBF7D0; }
.opt-title { font-weight: 700; font-size: 0.9rem; color: #0F172A; margin-bottom: 6px; }
.opt-body  { font-size: 0.84rem; color: #475569; line-height: 1.6; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #F1F5F9; border-radius: 10px; padding: 4px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; font-weight: 500; font-size: 0.85rem;
    color: #64748B; padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #fff !important; color: #4F46E5 !important;
    box-shadow: 0 1px 6px rgba(79,70,229,0.12);
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 4px 16px 4px;'>
        <div style='display:flex;align-items:center;gap:10px;'>
            <div style='width:36px;height:36px;border-radius:10px;
                background:linear-gradient(135deg,#6366F1,#14B8A6);
                display:flex;align-items:center;justify-content:center;
                font-size:1.1rem;flex-shrink:0;'>◈</div>
            <div>
                <div style='font-size:1.1rem;font-weight:800;color:#EEF2FF;letter-spacing:-0.02em;line-height:1.1;'>Robo-Advisor</div>
                <div style='font-size:0.62rem;color:#818CF8;text-transform:uppercase;letter-spacing:0.1em;margin-top:1px;'>Goal-Based Portfolio Planner</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**PROFILE**")
    age              = st.slider("Age", 20, 65, 28)
    income_stability = st.selectbox("Income stability",
                                     [1, 2, 3],
                                     format_func=lambda x: {1:"Unstable",2:"Moderate",3:"Stable"}[x],
                                     index=2)
    dependents       = st.slider("Financial dependents", 0, 5, 0)
    horizon_years    = st.slider("Investment horizon (years)", 3, 35, 15)

    st.divider()
    st.markdown("**RISK APPETITE**")
    risk_appetite = st.slider(
        "Comfort with a 20% portfolio drop",
        min_value=1, max_value=10, value=5,
        help="1 = Sell immediately   ·   10 = Buy more"
    )
    _risk_label, _risk_color = (
        ("Low — prefer capital safety", "#EF4444") if risk_appetite <= 3 else
        ("Moderate — can handle swings", "#EAB308") if risk_appetite <= 6 else
        ("High — drops don't worry you", "#22C55E")
    )
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:8px;margin-top:-6px;margin-bottom:4px;'>
        <div style='width:8px;height:8px;border-radius:50%;background:{_risk_color};flex-shrink:0;'></div>
        <span style='font-size:0.78rem;color:#94A3B8;'>{_risk_label}</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**SAVINGS & GOALS**")
    monthly_savings = st.number_input("Monthly savings (₹)", 1000, 500000, 25000, step=1000)
    goal_name   = st.text_input("Primary goal", "Retirement Corpus")
    goal_amount = st.number_input("Target amount today (₹)", 100000, 100000000, 10000000, step=100000)
    goal_inf    = 0.06

    add_goal2 = st.checkbox("Add a second goal")
    goal2_name, goal2_amount, goal2_years, goal2_inf = "", 0, 5, 0.06
    if add_goal2:
        goal2_name   = st.text_input("Goal 2 name", "House Down Payment")
        goal2_amount = st.number_input("Goal 2 amount (₹)", 100000, 50000000, 2000000, step=100000)
        goal2_years  = st.slider("Goal 2 horizon (years)", 1, 30, 7)
        goal2_inf    = 0.06

    st.divider()
    run_btn = st.button("Generate My Plan", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Goal-Based Robo-Advisor</h1>
    <p>Markowitz MVO &nbsp;·&nbsp; Monte Carlo Simulation &nbsp;·&nbsp; Indian Asset Markets &nbsp;·&nbsp; 80-Stock Universe</p>
    <div class="chip-row">
        <span class="chip">13 Asset Classes</span>
        <span class="chip">80 Stocks Screened</span>
        <span class="chip">10,000 MC Paths</span>
        <span class="chip">8-Factor Scoring</span>
        <span class="chip">Glide Path &amp; Rebalancing</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
    <div style='background:#fff;border-radius:14px;padding:32px 36px;border:1px solid #E2E8F0;
                box-shadow:0 2px 12px rgba(0,0,0,0.05);text-align:center;margin-top:8px;'>
        <div style='font-size:2.5rem;margin-bottom:12px;'>←</div>
        <div style='font-size:1.1rem;font-weight:600;color:#0F172A;margin-bottom:6px;'>Fill in your profile to get started</div>
        <div style='font-size:0.88rem;color:#64748B;'>Enter your age, risk appetite, savings amount and goal — then click <b>Generate My Plan</b>.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Compute everything ────────────────────────────────────────────────────────
with st.spinner("Fetching market data…"):
    prices, mu, cov = load_all(start="2015-01-01")

# Map 1-10 slider to 1-3 scale used by risk_profiler for each tolerance question
_t = round(1 + (risk_appetite - 1) * 2 / 9)  # 1→1, 5→2, 10→3
tolerance_answers = {
    "market_drop_reaction": _t,
    "past_investing_exp":   _t,
    "loss_sleep":           _t,
    "volatility_comfort":   _t,
    "goal_flexibility":     _t,
}
profile = build_profile(age, income_stability, dependents,
                         6, horizon_years, tolerance_answers)

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
    "Profile", "Allocation", "Goals", "Projections", "Rebalancing"
])

# ─────────────────────────── TAB 1: PROFILE ──────────────────────────────────
with t1:
    _persona_cls = profile.persona.lower()
    _persona_icons = {"Conservative": "Shield", "Balanced": "Scale", "Aggressive": "Rocket"}
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">Risk Persona</div>
            <div class="stat-value" style="font-size:1.3rem;">{profile.persona}</div>
            <div class="stat-sub">Based on capacity + tolerance</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Combined Score</div>
            <div class="stat-value">{profile.combined_score}<span style="font-size:1rem;color:#94A3B8;font-weight:400"> / 10</span></div>
            <div class="stat-sub">Capacity {profile.capacity_score} · Tolerance {profile.tolerance_score}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Target Return</div>
            <div class="stat-value">{profile.target_return*100:.0f}<span style="font-size:1rem;color:#94A3B8;font-weight:400">% p.a.</span></div>
            <div class="stat-sub">Expected annualised</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Investment Horizon</div>
            <div class="stat-value">{horizon_years}<span style="font-size:1rem;color:#94A3B8;font-weight:400"> yrs</span></div>
            <div class="stat-sub">Glide path active</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 0.8])
    with col_a:
        fig = go.Figure(go.Bar(
            x=["Capacity", "Tolerance", "Combined"],
            y=[profile.capacity_score, profile.tolerance_score, profile.combined_score],
            marker_color=["#14B8A6", "#4F46E5", "#F97316"],
            text=[f"{v:.1f}" for v in [profile.capacity_score, profile.tolerance_score, profile.combined_score]],
            textposition="outside",
        ))
        fig.update_layout(
            title=dict(text="Risk Score Breakdown", font=dict(size=15, color="#0F172A")),
            yaxis=dict(range=[0,11], gridcolor="#F1F5F9"),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            font=dict(color="#64748B"), height=340,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        _pc = {"Conservative":"persona-conservative","Balanced":"persona-balanced","Aggressive":"persona-aggressive"}
        _alert = ""
        if profile.tolerance_score > profile.capacity_score:
            _alert = f"<div style='background:#FEF3C7;border:1px solid #FCD34D;border-radius:10px;padding:12px 14px;font-size:0.82rem;color:#92400E;margin-bottom:12px;'><b>Note:</b> Your risk appetite exceeds your financial capacity. Capacity sets the hard ceiling.</div>"
        st.markdown(f"""
        {_alert}
        <div class="persona-card {_pc.get(profile.persona,'persona-balanced')}">
            <div class="persona-name">{profile.persona} Investor</div>
            <div class="persona-meta">
                Target return &nbsp;<b>{profile.target_return*100:.0f}% p.a.</b><br>
                Capacity score &nbsp;<b>{profile.capacity_score} / 10</b><br>
                Tolerance score &nbsp;<b>{profile.tolerance_score} / 10</b><br>
                Monthly savings &nbsp;<b>₹{monthly_savings:,}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────── TAB 2: ALLOCATION ───────────────────────────────
with t2:
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">Expected Return</div>
            <div class="stat-value">{opt['expected_return']*100:.1f}<span style="font-size:1rem;color:#94A3B8;font-weight:400">%</span></div>
            <div class="stat-sub">Annualised</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Volatility</div>
            <div class="stat-value">{opt['volatility']*100:.1f}<span style="font-size:1rem;color:#94A3B8;font-weight:400">%</span></div>
            <div class="stat-sub">Annualised std dev</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Sharpe Ratio</div>
            <div class="stat-value">{opt['sharpe_ratio']:.2f}</div>
            <div class="stat-sub">Risk-adjusted return</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Equity Allocation</div>
            <div class="stat-value">{eq_pct:.1f}<span style="font-size:1rem;color:#94A3B8;font-weight:400">%</span></div>
            <div class="stat-sub">of total portfolio</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _chart_layout = dict(
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        font=dict(color="#64748B", size=12),
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0"),
    )

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        labels = [ASSET_LABELS.get(k,k) for k in opt["weights"]]
        values = list(opt["weights"].values())
        colors = [ASSET_COLORS.get(k,"#888") for k in opt["weights"]]
        fig2 = go.Figure(go.Pie(labels=labels, values=values,
                                 marker_colors=colors,
                                 textinfo="label+percent", hole=0.42,
                                 marker=dict(line=dict(color="#fff", width=2))))
        fig2.update_layout(title=dict(text="Optimal Asset Allocation", font=dict(size=15,color="#0F172A")),
                           paper_bgcolor="#fff", height=460,
                           margin=dict(t=50, b=20, l=20, r=20),
                           legend=dict(font=dict(size=11, color="#64748B")))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Allocation Breakdown</div>', unsafe_allow_html=True)
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
                               line=dict(color="#4F46E5", width=3)))
    fig3.add_trace(go.Scatter(x=[opt["volatility"]*100], y=[opt["expected_return"]*100],
                               mode="markers+text", name="Your Portfolio",
                               marker=dict(size=14, color="#EF4444", symbol="star"),
                               text=[profile.persona], textposition="top right"))
    fig3.update_layout(
        title=dict(text="Efficient Frontier — Where Your Portfolio Sits", font=dict(size=15,color="#0F172A")),
        xaxis_title="Volatility %", yaxis_title="Expected Return %",
        height=420, **_chart_layout
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Stock basket
    st.markdown(f'<div class="section-title">Recommended Stocks — Equity Bucket ({eq_pct:.0f}% of portfolio)</div>', unsafe_allow_html=True)
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

            if "current_price" in display.columns:
                display["current_price"] = display["current_price"].apply(
                    lambda x: f"₹{x:,.0f}" if not pd.isna(x) else "—"
                )

            show = ["name","tier","current_price","beta","pe","ev_ebitda","pb",
                    "roe","net_margin","revenue_growth",
                    "de","current_ratio","momentum_6m",
                    "portfolio_weight_pct","monthly_amt_inr","action"]
            show = [c for c in show if c in display.columns]

            st.dataframe(
                display[show].rename(columns={
                    "name":                "Company",
                    "tier":                "Cap",
                    "current_price":       "Price",
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
                    "monthly_amt_inr":     "Alloc/Month",
                    "action":              "Action / Alternative",
                }),
                use_container_width=True,
                hide_index=False,
                column_config={
                    "Action / Alternative": st.column_config.TextColumn(width="large"),
                }
            )

            # Summary metrics
            if "actual_invest_inr" in basket.columns:
                total_investable   = basket["actual_invest_inr"].sum()
                total_alloc        = basket["monthly_amt_inr"].sum()
                rollover           = int(total_alloc - total_investable)
                unaffordable_count = int((basket.get("affordable", pd.Series([True]*len(basket))) == False).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Investable this month",  f"₹{int(total_investable):,}")
                c2.metric("Accumulation pool",       f"₹{rollover:,}",
                          help="Rolls to liquid fund until enough for next share purchase")
                c3.metric("Stocks need a decision",  f"{unaffordable_count}",
                          help="Choose: accumulate over months OR switch to sector ETF")

            # Side-by-side decision cards for unaffordable stocks
            if "affordable" in basket.columns:
                unaffordable = basket[basket["affordable"] == False]
                if not unaffordable.empty:
                    with st.expander(f"📋 {len(unaffordable)} stock(s) where monthly allocation < share price — pick an option"):
                        for _, row in unaffordable.iterrows():
                            name   = row.get("name", "")
                            price  = row.get("current_price", 0)
                            amt    = row.get("monthly_amt_inr", 0)
                            months = row.get("months_to_accumulate", 0)
                            etf    = row.get("etf_name", "")
                            etf_t  = row.get("etf_ticker", "")
                            reason = row.get("etf_reason", "")

                            st.markdown(f"#### {name}")
                            st.caption(f"Share price: ₹{price:,.0f}  ·  Your monthly allocation: ₹{amt:,}")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.info(
                                    f"**⏳ Option A — Save up, then buy the stock directly**\n\n"
                                    f"Put ₹{amt:,} aside every month into a **Liquid Fund**.\n\n"
                                    f"After **{months} month{'s' if months>1 else ''}** you'll have ~₹{amt*months:,} — enough to buy **1 share of {name}** at ₹{price:,.0f}.\n\n"
                                    f"You own the actual stock. Full upside, full downside."
                                )
                            with col_b:
                                st.success(
                                    f"**💡 Option B — Invest in a sector ETF right now**\n\n"
                                    f"Start a ₹{amt:,}/month SIP in **{etf}** (ticker: `{etf_t}`) today — no waiting.\n\n"
                                    f"This ETF holds {name} + other top stocks in the same sector, so you get the same exposure at a fraction of the price.\n\n"
                                    f"Minimum investment: ₹500. Highly liquid, can sell anytime."
                                )
                            st.divider()
        except Exception as e:
            st.warning(f"Stock screening skipped: {e}")

# ─────────────────────────── TAB 3: GOALS ────────────────────────────────────
with t3:
    st.markdown('<div class="section-title">Goal Summary</div>', unsafe_allow_html=True)
    df_goals = pd.DataFrame(summary_table(planned_goals))
    st.dataframe(df_goals, use_container_width=True, hide_index=True)

    g_names = [g.name for g in planned_goals]
    sips    = [g.monthly_sip for g in planned_goals]
    feas    = [g.feasibility_pct for g in planned_goals]
    colors_g = ["#14B8A6" if f>=100 else "#E65100" if f<50 else "#F9A825" for f in feas]

    col1, col2 = st.columns(2)
    with col1:
        fig4 = go.Figure(go.Bar(x=g_names, y=sips, marker_color=colors_g,
                                 text=[f"₹{v:,.0f}" for v in sips], textposition="outside"))
        fig4.add_hline(y=monthly_savings, line_dash="dash", line_color="#4F46E5",
                       annotation_text=f"Your savings ₹{monthly_savings:,}/mo")
        fig4.update_layout(title="SIP Required vs Your Savings",
                           yaxis_title="₹/month", plot_bgcolor="#fff", paper_bgcolor="#fff", height=400)
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
                           yaxis_title="₹", plot_bgcolor="#fff", paper_bgcolor="#fff", height=400)
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
        fill="toself", fillcolor="rgba(79,70,229,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="10–90th pct",
    ))
    fig6.add_trace(go.Scatter(
        x=list(years)+list(years[::-1]),
        y=list(p["p75"])+list(p["p25"][::-1]),
        fill="toself", fillcolor="rgba(79,70,229,0.22)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75th pct",
    ))
    fig6.add_trace(go.Scatter(x=years, y=p["p50"], mode="lines",
                               name="Median", line=dict(color="#4F46E5", width=3)))
    fig6.add_trace(go.Scatter(x=years, y=mc.stress_percentiles["p50"],
                               mode="lines", name="Median (post-crash)",
                               line=dict(color="#F43F5E", width=2, dash="dot")))
    fig6.add_hline(y=primary.future_value, line_dash="dash", line_color="#14B8A6",
                   annotation_text=f"Target ₹{primary.future_value/1e5:.0f}L")
    fig6.update_layout(
        title=f"{primary.name} — Monte Carlo Projection ({mc.prob_success}% success)",
        xaxis_title="Years", yaxis_title="Portfolio Value (₹)",
        plot_bgcolor="#fff", paper_bgcolor="#fff", height=480,
    )
    st.plotly_chart(fig6, use_container_width=True)

    # Histogram
    fig7 = go.Figure(go.Histogram(x=mc.terminal_values/1e5, nbinsx=80,
                                   marker_color="#4F46E5", opacity=0.75))
    fig7.add_vline(x=primary.future_value/1e5, line_dash="dash", line_color="#F43F5E",
                   annotation_text="Target")
    fig7.update_layout(title="Terminal Wealth Distribution",
                       xaxis_title="₹ Lakhs", yaxis_title="# simulations",
                       plot_bgcolor="#fff", paper_bgcolor="#fff", height=360)
    st.plotly_chart(fig7, use_container_width=True)

# ─────────────────────────── TAB 5: REBALANCING ──────────────────────────────
with t5:
    col1, col2 = st.columns(2)
    with col1:
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=glide.years, y=glide.equity_pct,
                                   name="Equity %", mode="lines",
                                   line=dict(color="#4F46E5", width=3),
                                   fill="tozeroy", fillcolor="rgba(79,70,229,0.12)"))
        fig8.add_trace(go.Scatter(x=glide.years, y=glide.debt_pct,
                                   name="Debt + Fixed %", mode="lines",
                                   line=dict(color="#14B8A6", width=3),
                                   fill="tozeroy", fillcolor="rgba(46,125,50,0.12)"))
        fig8.update_layout(title="Glide Path",
                           xaxis_title="Years", yaxis_title="Allocation %",
                           plot_bgcolor="#fff", paper_bgcolor="#fff", height=400)
        st.plotly_chart(fig8, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Drift Simulator</div>', unsafe_allow_html=True)
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
            st.markdown(f"""<div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;
                padding:14px 18px;color:#166534;font-size:0.88rem;'>
                No rebalancing needed after a {bull_years}-year bull run — all assets within ±5% of target.</div>""",
                unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style='background:#FEF3C7;border:1px solid #FCD34D;border-radius:10px;
                padding:14px 18px;color:#92400E;font-size:0.88rem;margin-bottom:12px;'>
                Rebalancing required after {bull_years}-year bull run.</div>""",
                unsafe_allow_html=True)
            st.dataframe(df_reb, use_container_width=True, hide_index=True)

    # Drift bar chart
    assets_d = list(opt["weights"].keys())
    fig9 = go.Figure()
    fig9.add_trace(go.Bar(name="Target",
                           x=[ASSET_LABELS.get(a,a) for a in assets_d],
                           y=[opt["weights"].get(a,0)*100 for a in assets_d],
                           marker_color="#4F46E5", opacity=0.85))
    fig9.add_trace(go.Bar(name=f"After {bull_years}yr bull",
                           x=[ASSET_LABELS.get(a,a) for a in assets_d],
                           y=[drifted.get(a,0)*100 for a in assets_d],
                           marker_color="#F43F5E", opacity=0.85))
    fig9.update_layout(barmode="group", title="Portfolio Drift",
                       xaxis_tickangle=-35, yaxis_title="Weight %",
                       plot_bgcolor="#fff", paper_bgcolor="#fff", height=420)
    st.plotly_chart(fig9, use_container_width=True)
