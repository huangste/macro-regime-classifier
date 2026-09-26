"""Market regime research application.

Run with:  streamlit run app/streamlit_app.py

The app keeps two things visually and verbally separate, because conflating
them is what made the original model hard to trust:

* the **historical classification**, fitted on the whole sample, which is a
  description of what happened and is allowed to use hindsight; and
* the **forward forecast**, which uses only causal inputs and is shown next to
  the out-of-sample skill that method actually achieved.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from regimelab.config import ARTIFACT_DIR, FORECAST_HORIZONS
from regimelab.evaluation import reliability_table, summarise_predictions
from regimelab.pipeline import build_state

st.set_page_config(page_title="Market Regime Research",
                   page_icon="chart", layout="wide")

REGIME_COLORS = ["#b2182b", "#ef8a62", "#92c5de", "#2166ac"]
OFF_COLOR, ON_COLOR = "#b2182b", "#2166ac"


def regime_palette(k: int) -> list[str]:
    if k == len(REGIME_COLORS):
        return REGIME_COLORS
    import plotly.colors as pc
    return pc.sample_colorscale("RdBu", np.linspace(0.08, 0.92, k))


# --------------------------------------------------------------------------
# Model state
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_state(sample: str, k: int, method: str, refresh: bool):
    return build_state(sample=sample, n_regimes=k, labeler_method=method,
                       refresh_data=refresh)


@st.cache_data(show_spinner=False)
def get_benchmark() -> pd.DataFrame | None:
    """Out-of-sample skill tables from the experiment scripts.

    07 supersedes 03 for the two regime targets because it adds the
    combinations; 06 contributes the forward-volatility target.
    """
    frames = []
    for name in ("07_combination.csv", "03_benchmark_full.csv",
                 "06_volatility_benchmark.csv"):
        p = ARTIFACT_DIR / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["target", "h", "model"], keep="first")


@st.cache_data(show_spinner=False)
def get_oos_predictions() -> pd.DataFrame | None:
    p = ARTIFACT_DIR / "03_oos_predictions.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index.names = ["run", "date"]
    return df.reset_index()


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def regime_timeline(state, start=None, end=None) -> go.Figure:
    spx = state.panel["SPX"].reindex(state.labels.index)
    labels = state.labels
    if start is not None:
        m = (labels.index >= pd.Timestamp(start)) & (labels.index <= pd.Timestamp(end))
        spx, labels = spx[m], labels[m]

    colors = regime_palette(state.n_regimes)
    fig = go.Figure()

    v = labels.values
    cuts = [0] + list(np.where(v[1:] != v[:-1])[0] + 1) + [len(v)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 3:
            continue
        fig.add_vrect(
            x0=labels.index[a], x1=labels.index[b - 1],
            fillcolor=colors[int(v[a])], opacity=0.25, line_width=0, layer="below",
        )

    fig.add_trace(go.Scatter(
        x=spx.index, y=spx.values, mode="lines", name="S&P 500",
        line=dict(color="#111111", width=1.2),
        hovertemplate="%{x|%Y-%m-%d}<br>S&P %{y:,.0f}<extra></extra>",
    ))
    for j in range(state.n_regimes):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=state.regime_names[j],
            marker=dict(size=12, symbol="square", color=colors[j]),
        ))

    fig.update_yaxes(type="log", title="S&P 500 (log scale)")
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.14))
    return fig


def risk_off_history(state, start=None, end=None) -> go.Figure:
    p_off = (state.proba_filtered.values[:, state.risk_off_mask].sum(axis=1))
    s = pd.Series(p_off, index=state.proba_filtered.index)
    sm = pd.Series(state.proba_smoothed.values[:, state.risk_off_mask].sum(axis=1),
                   index=state.proba_smoothed.index)
    if start is not None:
        m = (s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))
        s, sm = s[m], sm[m]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sm.index, y=sm.values, name="smoothed (uses hindsight)",
                             line=dict(color="#999999", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name="filtered (real time)",
                             line=dict(color=OFF_COLOR, width=1.4)))
    fig.add_hline(y=0.5, line=dict(color="#666", width=1, dash="dash"))
    fig.update_yaxes(title="P(risk-off state)", range=[-0.02, 1.02])
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.25))
    return fig


def transition_heatmap(T: pd.DataFrame, names: list[str], title: str) -> go.Figure:
    short = [n.split(" (")[0] for n in names]
    fig = go.Figure(go.Heatmap(
        z=T.values, x=short, y=short, colorscale="Blues", zmin=0, zmax=1,
        text=np.round(T.values * 100, 1), texttemplate="%{text}%",
        colorbar=dict(title="P"),
    ))
    fig.update_layout(title=title, height=420, xaxis_title="state at t+h",
                      yaxis_title="state at t", yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def forecast_chart(fc: pd.DataFrame, target: str, models: list[str]) -> go.Figure:
    d = fc[fc["target"] == target]
    fig = go.Figure()
    for m in models:
        sub = d[d["model"] == m].sort_values("horizon")
        if sub.empty:
            continue
        dash = "dash" if m in ("uncond", "persist_markov") else "solid"
        fig.add_trace(go.Scatter(
            x=sub["horizon"], y=sub["prob"], mode="lines+markers", name=m,
            line=dict(dash=dash, width=2),
            hovertemplate="h=%{x}d<br>P=%{y:.1%}<extra>" + m + "</extra>",
        ))
    fig.update_yaxes(title="probability of risk-off", tickformat=".0%",
                     range=[0, 1])
    fig.update_xaxes(title="horizon (trading days)",
                     tickmode="array", tickvals=list(FORECAST_HORIZONS))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    return fig


def skill_chart(bench: pd.DataFrame, target: str) -> go.Figure:
    # The unconditional baseline is excluded: it scores several hundred per
    # cent below the benchmark and would flatten everything else to a line.
    d = bench[(bench["target"] == target)
              & (~bench["model"].isin(["persist_markov", "uncond"]))]
    fig = go.Figure()
    for m, sub in d.groupby("model"):
        sub = sub.sort_values("h")
        fig.add_trace(go.Scatter(x=sub["h"], y=sub["skill_vs_base"],
                                 mode="lines+markers", name=m))
    fig.add_hline(y=0, line=dict(color="#333", width=1.5))
    lim = float(np.nanmax(np.abs(d["skill_vs_base"]))) * 1.1 if len(d) else 0.5
    fig.update_yaxes(title="skill vs persistence benchmark", tickformat=".0%",
                     range=[-lim, lim])
    fig.update_xaxes(title="horizon (trading days)",
                     tickmode="array", tickvals=list(FORECAST_HORIZONS))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Regime model")
sample = st.sidebar.selectbox(
    "Sample", ["core_1990", "full_2000"], index=0,
    help="core_1990 drops gold to buy ten extra years of history; "
         "full_2000 keeps every series but starts in August 2000.")
k = st.sidebar.slider("Number of regimes (K)", 2, 5, 4)
method = st.sidebar.selectbox(
    "Labelling model", ["hmm", "gmm", "score"], index=0,
    help="hmm models persistence and is the default; gmm reproduces the "
         "original notebook's estimator; score is a transparent quantile "
         "baseline on the risk-appetite index.")
refresh = st.sidebar.button("Refresh market data")

with st.spinner("Fitting regime model..."):
    state = get_state(sample, k, method, bool(refresh))

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**{state.meta['n_days']:,}** trading days  \n"
    f"{state.meta['start']} to {state.meta['end']}  \n"
    f"built {state.built_at.replace('T', ' ')}")

bench = get_benchmark()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Market regime research")

p_off_now = float(state.proba_filtered.values[-1][state.risk_off_mask].sum())
hard_f = state.proba_filtered.values.argmax(axis=1)
cur_f = int(hard_f[-1])
cur_name = state.regime_names[cur_f].split(" (")[0]
cur_side = "Risk-off" if state.risk_off_mask[cur_f] else "Risk-on"

run_start = len(hard_f) - 1
while run_start > 0 and hard_f[run_start - 1] == cur_f:
    run_start -= 1
days_in = len(hard_f) - run_start
since = state.proba_filtered.index[run_start].date()
typical = float(state.expected_durations.iloc[cur_f])

st.subheader(f"Current regime: {cur_name}  ·  {cur_side}")

c1, c2, c3, c4, c5 = st.columns([2.2, 1, 1, 1, 1.2])
c1.metric("Current regime (real time)", cur_name,
          delta=cur_side, delta_color="inverse" if cur_side == "Risk-off" else "normal")
c2.metric("Days in regime", f"{days_in}", help=f"In this state since {since}. "
          f"Typical episode length for this state: about {typical:.0f} trading days.")
c3.metric("P(risk-off) today",
          "<0.1%" if p_off_now < 0.001 else f"{p_off_now:.1%}")
c4.metric("Risk-appetite index", f"{state.risk_score.iloc[-1]:+.2f}",
          help="Block-weighted z-score of equity trend, volatility, credit and "
               "carry. Zero is the sample average.")
c5.metric("As of", state.meta["end"])

st.markdown(f"**Baseline transition probability from {cur_name}**")
bt = state.baseline_transition.copy()
bt.index = [f"{h} day" if h == 1 else f"{h} days" for h in bt.index]
st.dataframe(
    bt.style.format("{:.1%}")
      .background_gradient(cmap="Blues", axis=None,
                           subset=[c for c in bt.columns if c != "P(risk-off)"])
      .background_gradient(cmap="Reds", vmin=0, vmax=1, subset=["P(risk-off)"]),
    use_container_width=True)
st.caption(
    "Row h: where the market was h trading days later, historically, starting "
    "from today's regime. Estimated from real-time (filtered) states over the "
    "full sample. This is the persistence benchmark every forecast in the app "
    "is measured against; the Forecasts tab shows what the models add to it.")

st.markdown("")

tabs = st.tabs([
    "Historical regimes", "Characteristics", "Transitions",
    "Forecasts", "Diagnostics", "Method",
])

# --------------------------------------------------------------------------
# 1. Historical regimes
# --------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Regime classification through time")
    st.caption(
        "Fitted on the full sample. This is a *description* of history, not a "
        "backtest: it is allowed to use hindsight, and it does.")

    idx = state.labels.index
    yr0, yr1 = idx[0].year, idx[-1].year
    rng = st.slider("Period", yr0, yr1, (max(yr0, 1995), yr1))
    lo, hi = f"{rng[0]}-01-01", f"{rng[1]}-12-31"

    st.plotly_chart(regime_timeline(state, lo, hi), use_container_width=True)
    st.plotly_chart(risk_off_history(state, lo, hi), use_container_width=True)

    st.markdown("**Regime episodes**")
    ep = state.episodes.copy()
    ep = ep[(ep["start"] >= pd.Timestamp(lo)) & (ep["end"] <= pd.Timestamp(hi))]
    ep["regime"] = ep["regime"].map(lambda j: state.regime_names[j])
    ep["start"] = ep["start"].dt.date
    ep["end"] = ep["end"].dt.date
    st.dataframe(ep.sort_values("days", ascending=False),
                 use_container_width=True, hide_index=True, height=300)

# --------------------------------------------------------------------------
# 2. Characteristics
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("What each regime looks like economically")
    ch = state.characteristics.copy()
    fmt = {
        "risk_score": "{:+.2f}", "share_of_days": "{:.1%}",
        "SPX_ann_return": "{:+.1%}", "SPX_ann_vol": "{:.1%}",
        "SPX_hit_rate": "{:.1%}", "SPX_sharpe": "{:+.2f}",
        "VIX_level": "{:.1f}", "Baa_spread": "{:.2f}",
        "Curve_10y2y": "{:+.2f}", "US10y": "{:.2f}",
    }
    st.dataframe(ch.style.format({c: f for c, f in fmt.items() if c in ch.columns}),
                 use_container_width=True)
    st.caption(
        "Return statistics are *next-day* S&P outcomes conditional on being in "
        "the state today, annualised. Canonical index 0 is always the most "
        "risk-averse state and K-1 the most risk-seeking, by construction.")

    st.markdown("---")
    st.markdown("**Next-day asset behaviour by regime**")
    ar = state.asset_returns.copy()
    ar.index = [n.split(" (")[0] for n in state.regime_names]
    st.dataframe(
        ar.style.format("{:+.1f}").background_gradient(cmap="RdYlGn", axis=None),
        use_container_width=True)
    st.caption(
        "Equity and commodity columns are annualised percentage returns; rate "
        "and spread columns are annualised changes in basis points. This is "
        "the check that the ordering means what it claims: defensive "
        "behaviour should strengthen as the canonical index falls.")

    st.markdown("---")
    st.markdown("**Standardised feature profile by regime**")
    L = state.features[[c for c in state.labeler.columns_]]
    z = (L - L.mean()) / L.std()
    prof = z.groupby(state.labels.values).mean().T
    prof.columns = [n.split(" (")[0] for n in state.regime_names]
    fig = go.Figure(go.Heatmap(
        z=prof.values, x=list(prof.columns), y=list(prof.index),
        colorscale="RdBu", zmid=0, colorbar=dict(title="z"),
        text=np.round(prof.values, 2), texttemplate="%{text}"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Risk-appetite index**")
    rs = state.risk_score
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=rs.index, y=rs.rolling(5).mean().values,
                            line=dict(color="#333", width=1), name="risk score"))
    f2.add_hline(y=0, line=dict(color="#999", dash="dash"))
    f2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                     yaxis_title="risk appetite (z)")
    st.plotly_chart(f2, use_container_width=True)

# --------------------------------------------------------------------------
# 3. Transitions
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Transition behaviour")
    h_sel = st.selectbox("Horizon (trading days)", [1] + list(FORECAST_HORIZONS),
                         index=0, key="trans_h")
    T = state.transitions[h_sel]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(
            transition_heatmap(T, state.regime_names,
                               f"P(state at t+{h_sel} | state at t)"),
            use_container_width=True)
    with c2:
        st.markdown("**Expected time in state**")
        dur = state.expected_durations.copy()
        dur.index = [n.split(" (")[0] for n in state.regime_names]
        st.dataframe(dur.round(0).to_frame("trading days"),
                     use_container_width=True)

        ev = np.linalg.eig(state.transitions[1].values.T)
        i = int(np.argmin(np.abs(ev[0] - 1)))
        pi = np.real(ev[1][:, i]); pi = pi / pi.sum()
        st.markdown("**Long-run share of time**")
        st.dataframe(pd.DataFrame(
            {"share": pi}, index=[n.split(" (")[0] for n in state.regime_names]
        ).style.format({"share": "{:.1%}"}), use_container_width=True)

        st.caption(
            "Rows of the one-step matrix are dominated by the diagonal, which "
            "is exactly why a persistence forecast is hard to beat. These "
            "counts come from the full-sample descriptive labelling, so they "
            "describe history; the live forecast uses a matrix estimated on "
            "real-time filtered states instead, which is slightly less "
            "persistent.")

# --------------------------------------------------------------------------
# 4. Forecasts
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Forward probabilities")

    target = st.radio(
        "What is being forecast",
        ["any_in_window", "point_in_time", "mkt_high_vol"], horizontal=True,
        format_func=lambda s: {
            "any_in_window": "At least one risk-off day in the next h days",
            "point_in_time": "Risk-off exactly h days from now",
            "mkt_high_vol": "High realised volatility over the next h days",
        }[s])

    st.info({
        "any_in_window":
            "P(at least one risk-off day in (t, t+h]). This rises with the "
            "horizon by construction: a longer window has more chances to "
            "contain a risk-off day. It is the quantity most useful for risk "
            "management.",
        "point_in_time":
            "P(the regime is risk-off on day t+h). This decays towards the "
            "unconditional base rate as the horizon grows, because a single "
            "distant day is close to a draw from the stationary distribution. "
            "A flat or rising curve here would be a red flag.",
        "mkt_high_vol":
            "P(realised S&P volatility over (t, t+h] exceeds its historical "
            "median). Unlike the two regime targets this event is directly "
            "observable and shares no data with the features, and it is the "
            "target on which the model has by far the strongest out-of-sample "
            "evidence.",
    }[target])

    fc = state.forecasts
    available = list(dict.fromkeys(fc["model"]))

    HEADLINE = ("compact|logit_reg" if target == "mkt_high_vol"
                else "blend50|compact|logit_reg")
    BLURB = {
        "mkt_high_vol":
            "A regularised logistic model on twelve economically chosen "
            "features plus the current state posterior. On this target it "
            "beats the persistence benchmark by around a quarter of its log "
            "loss at every horizon, with p-values below 0.0001 — no "
            "combination needed.",
        "regime":
            "A regularised logistic model on twelve economically chosen "
            "features plus the current state posterior, averaged equally with "
            "the persistence benchmark. This is the specification that showed "
            "positive out-of-sample skill significant at the 5% level at "
            "*every* horizon and under *both* regime target definitions.",
    }
    if HEADLINE in available:
        st.markdown("**Headline forecast**")
        st.caption(BLURB.get(target, BLURB["regime"]))
        head = (fc[(fc["target"] == target) & (fc["model"] == HEADLINE)]
                .set_index("horizon")["prob"])
        cols = st.columns(len(FORECAST_HORIZONS))
        for col, h in zip(cols, FORECAST_HORIZONS):
            if h in head.index:
                col.metric(f"{h} days", f"{head[h]:.1%}")
        st.markdown("")
    default = [m for m in ["blend50|compact|logit_reg", "persist_markov",
                           "hmm_analytic", "uncond"] if m in available]
    chosen = st.multiselect("Models", available, default=default)
    st.plotly_chart(forecast_chart(fc, target, chosen), use_container_width=True)

    tab = (fc[fc["target"] == target]
           .pivot_table(index="horizon", columns="model", values="prob"))
    keep = [c for c in chosen if c in tab.columns]
    st.dataframe(tab[keep].style.format("{:.1%}"), use_container_width=True)

    if bench is not None:
        st.markdown("**Out-of-sample skill of each of these methods**")
        b = bench[bench["target"] == target].pivot_table(
            index="h", columns="model", values="skill_vs_base")
        keep = [c for c in chosen if c in b.columns]
        st.dataframe(
            b[keep].style.format("{:+.1%}").background_gradient(
                cmap="RdYlGn", vmin=-0.15, vmax=0.15),
            use_container_width=True)
        st.caption(
            "Skill is the fraction of the persistence benchmark's log loss "
            "removed, measured on purged walk-forward folds. Values at or "
            "below zero mean the method added nothing over knowing today's "
            "state. Read the forecast above in that light.")
    else:
        st.warning("Run `python experiments/03_forecast_benchmark.py` to "
                   "populate the out-of-sample skill table.")

# --------------------------------------------------------------------------
# 5. Diagnostics
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Model diagnostics")

    if bench is None:
        st.warning("No benchmark artifact found. Run "
                   "`python experiments/03_forecast_benchmark.py`.")
    else:
        tgt = st.selectbox("Target", sorted(bench["target"].unique()), key="diag_t")
        st.plotly_chart(skill_chart(bench, tgt), use_container_width=True)

        st.markdown("**Full out-of-sample table**")
        show = bench[bench["target"] == tgt].drop(columns=["target"])
        st.dataframe(
            show.style.format({
                "log_loss": "{:.4f}", "brier": "{:.4f}", "auc": "{:.3f}",
                "skill_vs_base": "{:+.2%}", "dm_t": "{:+.2f}", "dm_p": "{:.3f}"}),
            use_container_width=True, hide_index=True, height=430)
        st.caption(
            "`dm_t` is a Diebold-Mariano statistic on daily log-loss "
            "differentials with a Newey-West variance at bandwidth 2h; "
            "negative means lower loss than the benchmark. `dm_p` is the "
            "one-sided p-value. Overlapping windows mean the effective sample "
            "is roughly the number of daily observations divided by h.")

        oos = get_oos_predictions()
        if oos is not None:
            st.markdown("---")
            st.markdown("**Calibration (out-of-sample)**")
            runs = sorted(oos["run"].unique())
            run = st.selectbox("Run", runs,
                               index=runs.index(f"{tgt}_h25") if f"{tgt}_h25" in runs else 0)
            d = oos[oos["run"] == run]
            mcols = [c for c in d.columns
                     if c not in ("run", "date", "y", "state_at_t", "fold")]
            m = st.selectbox("Model", mcols,
                             index=mcols.index("persist_markov")
                             if "persist_markov" in mcols else 0)
            rel = reliability_table(d["y"].values, d[m].values, bins=8)
            f = go.Figure()
            f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                   line=dict(color="#999", dash="dash"),
                                   name="perfect calibration"))
            f.add_trace(go.Scatter(x=rel["mean_predicted"], y=rel["observed_freq"],
                                   mode="markers+lines", name=m,
                                   marker=dict(size=10)))
            f.update_layout(height=400, xaxis_title="predicted probability",
                            yaxis_title="observed frequency",
                            margin=dict(l=10, r=10, t=10, b=10))
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(f, use_container_width=True)
            c2.dataframe(rel.round(3), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Label stability**")
    st.caption(
        "Regime names here are not the estimator's component indices. Each "
        "state is scored on a risk-appetite functional whose signs are fixed "
        "in advance, and states are re-indexed in ascending score order, so "
        "index 0 is always the most risk-averse state for any seed, any K and "
        "any data vintage.")
    stab = ROOT / "reports" / "02_label_stability.md"
    if stab.exists():
        with st.expander("Stability report"):
            st.markdown(stab.read_text(encoding="utf-8"))

# --------------------------------------------------------------------------
# 6. Method
# --------------------------------------------------------------------------
with tabs[5]:
    md = ROOT / "reports" / "METHODOLOGY.md"
    if md.exists():
        st.markdown(md.read_text(encoding="utf-8"))
    else:
        st.write("See reports/ for the research write-ups.")
