import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bayesian Hierarchical Modeling",
    page_icon="📊",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f8f9fb; }
  [data-testid="stSidebar"] { background: #0d1b2a; }
  [data-testid="stSidebar"] * { color: #e8eaf0 !important; }
  /* Fix: input boxes and dropdowns need dark text on their light backgrounds */
  [data-testid="stSidebar"] input { color: #0d1b2a !important; background: #f0f2f6 !important; border-radius: 6px; }
  [data-testid="stSidebar"] [data-baseweb="select"] * { color: #0d1b2a !important; }
  [data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] * { color: #e8eaf0 !important; }
  [data-testid="stSidebar"] [role="listbox"] * { color: #0d1b2a !important; }
  [data-testid="stSidebar"] [data-baseweb="select"] div[class*="ValueContainer"] { background: #f0f2f6 !important; border-radius: 6px; }
  [data-testid="stSidebar"] [data-baseweb="select"] span { color: #0d1b2a !important; }
  .hero { background: #0d1b2a; color: white; padding: 2.5rem 2rem 2rem;
          border-radius: 12px; margin-bottom: 1.5rem; }
  .hero-eye { color: #f0b429; font-size: 0.8rem; font-weight: 700;
              letter-spacing: 3px; text-transform: uppercase; }
  .hero-title { font-size: 2rem; font-weight: 800; margin: 0.4rem 0 0.6rem; }
  .hero-sub { color: #9da8b7; font-size: 1rem; }
  .metric-card { background: white; border-radius: 10px; padding: 1.2rem 1.5rem;
                 box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; }
  .metric-value { font-size: 2rem; font-weight: 800; color: #0d1b2a; }
  .metric-label { font-size: 0.78rem; color: #6b7a90; text-transform: uppercase;
                  letter-spacing: 1px; margin-top: 0.2rem; }
  .section-label { font-size: 0.72rem; font-weight: 700; color: #f0b429;
                   letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.5rem; }
  .card { background: white; border-radius: 10px; padding: 1.5rem;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 1rem; }
  .finding-box { background: #f0f7ff; border-left: 4px solid #1e6fcf;
                 border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
  h3 { color: #0d1b2a; }
</style>
""", unsafe_allow_html=True)

# ── Survey Data ─────────────────────────────────────────────────────────────────
SURVEY_DATA = {
    "Democrats":    {"Increase Aid": 35, "Maintain Aid": 39, "Not Sure": 16, "Decrease Aid": 10},
    "Independents": {"Increase Aid": 19, "Maintain Aid": 23, "Not Sure": 26, "Decrease Aid": 33},
    "Republicans":  {"Increase Aid": 10, "Maintain Aid": 24, "Not Sure": 21, "Decrease Aid": 45},
}
TOTAL_N = 1603
GROUP_N = round(TOTAL_N / 3)
STANCE_COLORS = {
    "Increase Aid": "#27ae60", "Maintain Aid": "#2980b9",
    "Not Sure": "#f39c12",    "Decrease Aid": "#e74c3c",
}
PARTY_COLORS = {"Democrats": "#2980b9", "Independents": "#8e44ad", "Republicans": "#e74c3c"}

# ── Helper: conjugate normal-normal posterior ───────────────────────────────────
def posterior_normal(y_hat, n, sigma2=0.0025, mu0=0.5, tau2=1.0):
    post_var  = 1 / (n / sigma2 + 1 / tau2)
    post_mean = post_var * (y_hat * n / sigma2 + mu0 / tau2)
    return post_mean, np.sqrt(post_var)

# ── Helper: Gibbs sampler ───────────────────────────────────────────────────────
def run_gibbs(y, n, sigma2=0.0025, n_iter=2000, seed=42):
    np.random.seed(seed)
    J = len(y)
    mu, tau2, theta = 0.5, 0.01, y.copy()
    theta_samp = np.zeros((n_iter, J))
    mu_samp    = np.zeros(n_iter)
    tau2_samp  = np.zeros(n_iter)
    for t in range(n_iter):
        for j in range(J):
            V = 1 / (n[j] / sigma2 + 1 / tau2)
            m = V * (y[j] * n[j] / sigma2 + mu / tau2)
            theta[j] = np.random.normal(m, np.sqrt(V))
        V_mu = 1 / (J / tau2 + 1)
        m_mu = V_mu * (theta.sum() / tau2 + 0.5)
        mu   = np.random.normal(m_mu, np.sqrt(V_mu))
        shape = (J - 1) / 2
        scale = np.sum((theta - mu) ** 2) / 2
        if scale > 0:
            tau2 = 1 / np.random.gamma(shape, 1 / scale)
        theta_samp[t] = theta
        mu_samp[t]    = mu
        tau2_samp[t]  = tau2
    return theta_samp, mu_samp, tau2_samp

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("---")
    st.markdown("**Gibbs Sampler Settings**")
    n_iter  = st.slider("Iterations", 500, 5000, 2000, step=500)
    sigma2  = st.select_slider("Known Variance (σ²)",
                                options=[0.001, 0.0025, 0.005, 0.01], value=0.0025)
    seed    = st.number_input("Random Seed", 0, 9999, 42)
    st.markdown("---")
    st.markdown("**Hyperpriors**")
    mu0   = st.slider("Prior Mean (μ₀)", 0.0, 1.0, 0.5, 0.05)
    tau2  = st.slider("Prior Variance (τ²)", 0.1, 5.0, 1.0, 0.1)
    st.markdown("---")
    focus_stance = st.selectbox("Focus Stance (Hierarchical Model)",
                                 ["Increase Aid", "Maintain Aid", "Not Sure", "Decrease Aid"],
                                 index=0)
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Survey of 1,603 U.S. adults on military aid to Ukraine. Compares three Bayesian modeling strategies.")

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
  <div class='hero-eye'>MDS · Bayesian Statistics</div>
  <div class='hero-title'>Bayesian Hierarchical Modeling</div>
  <div class='hero-sub'>
    Comparing Separate, Pooled &amp; Hierarchical models on U.S. political opinion data
    with a custom Gibbs sampler — n = 1,603 respondents
  </div>
</div>
""", unsafe_allow_html=True)

# KPIs
k1, k2, k3, k4 = st.columns(4)
for col, val, lbl in zip(
    [k1, k2, k3, k4],
    ["1,603", "3", "12", str(n_iter)],
    ["Survey Respondents", "Political Groups", "Group × Stance Pairs", "Gibbs Iterations"],
):
    col.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value'>{val}</div>
      <div class='metric-label'>{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Survey Data",
    "🔵 Separate Model",
    "🔗 Pooled Model",
    "🏛️ Hierarchical + Gibbs",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Survey Data
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-label'>Raw Survey Data</div>", unsafe_allow_html=True)
    st.markdown("### Political Opinions on Military Aid to Ukraine")

    # Grouped bar chart
    stances = ["Increase Aid", "Maintain Aid", "Not Sure", "Decrease Aid"]
    parties = ["Democrats", "Independents", "Republicans"]
    fig = go.Figure()
    for party in parties:
        fig.add_trace(go.Bar(
            name=party,
            x=stances,
            y=[SURVEY_DATA[party][s] for s in stances],
            marker_color=PARTY_COLORS[party],
            text=[f"{SURVEY_DATA[party][s]}%" for s in stances],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group", height=420,
        title="Survey Responses by Party Affiliation (%)",
        yaxis_title="% Support", xaxis_title="",
        legend_title="Party",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw table
    rows = []
    for party in parties:
        for stance in stances:
            pct = SURVEY_DATA[party][stance]
            count = round(pct / 100 * GROUP_N)
            rows.append({"Group": party, "Stance": stance,
                          "% Support": pct, "Est. Count": count,
                          "Proportion": round(count / GROUP_N, 4)})
    df_raw = pd.DataFrame(rows)
    st.dataframe(df_raw, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Separate Model
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-label'>Model 1</div>", unsafe_allow_html=True)
    st.markdown("### Separate Bayesian Model")
    st.markdown("""
    <div class='card'>
    Each of the 12 group–stance combinations is modeled <strong>independently</strong>.
    We apply a conjugate Normal-Normal model:<br><br>
    &nbsp;&nbsp;<em>Likelihood:</em> y | θ ~ N(θ, σ²) &nbsp;&nbsp;&nbsp;
    <em>Prior:</em> θ ~ N(μ₀, τ²) &nbsp;&nbsp;&nbsp;
    <em>Posterior:</em> θ | y ~ N(post_mean, post_var)
    <br><br>No information is shared between groups.
    </div>
    """, unsafe_allow_html=True)

    sep_rows = []
    for party in parties:
        for stance in stances:
            pct   = SURVEY_DATA[party][stance]
            count = round(pct / 100 * GROUP_N)
            prop  = count / GROUP_N
            pm, ps = posterior_normal(prop, GROUP_N, sigma2, mu0, tau2)
            sep_rows.append({
                "Group": party, "Stance": stance,
                "Observed %": pct, "Proportion": round(prop, 4),
                "Posterior Mean": round(pm, 4),
                "Posterior SD": round(ps, 5),
                "CI Lower (95%)": round(pm - 1.96 * ps, 4),
                "CI Upper (95%)": round(pm + 1.96 * ps, 4),
            })
    df_sep = pd.DataFrame(sep_rows)
    st.dataframe(df_sep, use_container_width=True, hide_index=True)

    # Plot posterior means for selected stance
    st.markdown(f"#### Posterior Estimates by Party — *{focus_stance}*")
    subset = df_sep[df_sep["Stance"] == focus_stance].reset_index(drop=True)
    fig2 = go.Figure()
    for _, row in subset.iterrows():
        fig2.add_trace(go.Scatter(
            x=[row["CI Lower (95%)"], row["Posterior Mean"], row["CI Upper (95%)"]],
            y=[row["Group"]] * 3,
            mode="lines+markers",
            name=row["Group"],
            marker=dict(size=[6, 12, 6], color=PARTY_COLORS[row["Group"]]),
            line=dict(color=PARTY_COLORS[row["Group"]], width=2),
            showlegend=True,
        ))
    fig2.update_layout(
        height=280, title=f"Posterior Mean ± 95% CI — {focus_stance}",
        xaxis_title="Support Proportion", yaxis_title="",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Pooled Model
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-label'>Model 2</div>", unsafe_allow_html=True)
    st.markdown("### Pooled Bayesian Model")
    st.markdown("""
    <div class='card'>
    Political affiliation is <strong>ignored</strong>. We pool all respondents and estimate
    one posterior per stance across the entire sample.
    This maximises data use but assumes no group differences exist.
    </div>
    """, unsafe_allow_html=True)

    pool_rows = []
    for stance in stances:
        total_count = sum(
            round(SURVEY_DATA[p][stance] / 100 * GROUP_N) for p in parties
        )
        total_n = GROUP_N * 3
        prop = total_count / total_n
        pm, ps = posterior_normal(prop, total_n, sigma2, mu0, tau2)
        pool_rows.append({
            "Stance": stance,
            "Pooled Proportion": round(prop, 4),
            "Posterior Mean": round(pm, 4),
            "Posterior SD": round(ps, 5),
            "CI Lower (95%)": round(pm - 1.96 * ps, 4),
            "CI Upper (95%)": round(pm + 1.96 * ps, 4),
        })
    df_pool = pd.DataFrame(pool_rows)
    st.dataframe(df_pool, use_container_width=True, hide_index=True)

    fig3 = go.Figure(go.Bar(
        x=df_pool["Stance"], y=df_pool["Posterior Mean"],
        error_y=dict(type="data", array=(df_pool["CI Upper (95%)"] - df_pool["Posterior Mean"]).tolist(),
                     arrayminus=(df_pool["Posterior Mean"] - df_pool["CI Lower (95%)"]).tolist(),
                     visible=True),
        marker_color=[STANCE_COLORS[s] for s in df_pool["Stance"]],
        text=[f"{v:.1%}" for v in df_pool["Posterior Mean"]],
        textposition="outside",
    ))
    fig3.update_layout(
        height=380, title="Pooled Posterior Means by Stance (all parties combined)",
        yaxis_title="Posterior Mean Support", xaxis_title="",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(tickformat=".0%"),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Hierarchical + Gibbs
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-label'>Model 3 — Most Advanced</div>", unsafe_allow_html=True)
    st.markdown("### Hierarchical Model + Gibbs Sampler")
    st.markdown(f"""
    <div class='card'>
    Group-specific means θⱼ are drawn from a shared population distribution N(μ, τ²).
    A custom <strong>Gibbs sampler</strong> jointly estimates θⱼ, μ, and τ² via MCMC.
    Currently analysing: <strong>{focus_stance}</strong> across the three political groups.
    <br><br>
    <em>σ² = {sigma2}</em> &nbsp;|&nbsp; <em>μ₀ = {mu0}</em> &nbsp;|&nbsp;
    <em>τ² = {tau2}</em> &nbsp;|&nbsp; <em>Iterations = {n_iter}</em>
    </div>
    """, unsafe_allow_html=True)

    # Run sampler
    y_vals = np.array([
        round(SURVEY_DATA[p][focus_stance] / 100 * GROUP_N) / GROUP_N
        for p in parties
    ])
    n_vals = np.array([GROUP_N] * 3, dtype=float)

    with st.spinner(f"Running Gibbs sampler ({n_iter} iterations)…"):
        theta_samp, mu_samp, tau2_samp = run_gibbs(
            y_vals, n_vals, sigma2=sigma2, n_iter=n_iter, seed=int(seed)
        )

    warmup = n_iter // 4
    theta_post = theta_samp[warmup:]
    mu_post    = mu_samp[warmup:]

    # ── Posterior distributions ──────────────────────────────────────────────
    st.markdown("#### Posterior Distributions of Group Support (θⱼ)")
    fig4 = go.Figure()
    for j, party in enumerate(parties):
        vals = theta_post[:, j]
        fig4.add_trace(go.Histogram(
            x=vals, name=party,
            marker_color=PARTY_COLORS[party],
            opacity=0.65, nbinsx=60,
            histnorm="probability density",
        ))
    # Overall mean
    fig4.add_trace(go.Histogram(
        x=mu_post, name="Population Mean (μ)",
        marker_color="#f0b429", opacity=0.5, nbinsx=60,
        histnorm="probability density",
    ))
    fig4.update_layout(
        barmode="overlay", height=380,
        title=f"Posterior Distributions — {focus_stance}",
        xaxis_title="Support Proportion", yaxis_title="Density",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ── Posterior summary table ──────────────────────────────────────────────
    st.markdown("#### Posterior Summary")
    summary_rows = []
    for j, party in enumerate(parties):
        vals = theta_post[:, j]
        summary_rows.append({
            "Group": party,
            "Posterior Median": f"{np.median(vals):.3f}",
            "Posterior Mean":   f"{np.mean(vals):.3f}",
            "2.5%":  f"{np.percentile(vals, 2.5):.3f}",
            "97.5%": f"{np.percentile(vals, 97.5):.3f}",
        })
    summary_rows.append({
        "Group": "Population Mean (μ)",
        "Posterior Median": f"{np.median(mu_post):.3f}",
        "Posterior Mean":   f"{np.mean(mu_post):.3f}",
        "2.5%":  f"{np.percentile(mu_post, 2.5):.3f}",
        "97.5%": f"{np.percentile(mu_post, 97.5):.3f}",
    })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── Trace plots ──────────────────────────────────────────────────────────
    st.markdown("#### MCMC Trace Plots (post-warmup)")
    fig5 = make_subplots(rows=len(parties), cols=1, shared_xaxes=True,
                          subplot_titles=parties)
    for j, party in enumerate(parties):
        fig5.add_trace(
            go.Scatter(y=theta_post[:, j], mode="lines",
                       line=dict(color=PARTY_COLORS[party], width=0.8),
                       name=party, showlegend=False),
            row=j+1, col=1
        )
        fig5.add_hline(y=np.mean(theta_post[:, j]),
                       line_dash="dash", line_color="black",
                       line_width=1, row=j+1, col=1)
    fig5.update_layout(height=500, title="θⱼ Trace Plots — Convergence Check",
                        plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig5, use_container_width=True)

    # ── Key findings ─────────────────────────────────────────────────────────
    st.markdown("#### Key Findings")
    medians = {p: np.median(theta_post[:, j]) for j, p in enumerate(parties)}
    ranked  = sorted(medians.items(), key=lambda x: x[1], reverse=True)
    for i, (party, med) in enumerate(ranked):
        icon = ["🥇", "🥈", "🥉"][i]
        st.markdown(f"""
        <div class='finding-box'>
        {icon} <strong>{party}</strong> — posterior median support for <em>{focus_stance}</em>:
        <strong>{med:.1%}</strong>
        &nbsp; (95% CI: {np.percentile(theta_post[:, parties.index(party)], 2.5):.1%}
        – {np.percentile(theta_post[:, parties.index(party)], 97.5):.1%})
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card' style='background:#f0f7ff;'>
    <strong>Why Hierarchical?</strong> The hierarchical model borrows strength across groups —
    shrinking extreme estimates toward the overall mean (μ) when group sample sizes are small.
    With only J = 3 political groups, this regularisation provides more stable and
    realistic uncertainty estimates than the Separate model, while preserving genuine group differences
    that the Pooled model erases.
    </div>
    """, unsafe_allow_html=True)
