"""
Streamlit Production Monitoring Dashboard

Run with: streamlit run src/dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Install plotly: pip install plotly")

st.set_page_config(
    page_title="Churn Model — Production Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Churn Prediction Model — Production Dashboard")
st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))


@st.cache_data(ttl=300)
def load_metrics(n_days: int = 90):
    """Load production metrics (replace with real DB query)."""
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(n_days)][::-1]
    auc = np.clip(0.82 + np.cumsum(np.random.normal(-0.001, 0.008, n_days)), 0.65, 0.92)

    return pd.DataFrame({
        "date": dates,
        "auc": auc,
        "daily_requests": np.random.poisson(1000, n_days),
        "p50_ms": np.random.normal(18, 3, n_days).clip(5),
        "p95_ms": np.random.exponential(50, n_days).clip(20, 500),
        "error_rate": np.random.beta(1, 200, n_days),
        "churn_rate": (0.25 + np.cumsum(np.random.normal(0.001, 0.005, n_days))).clip(0.15, 0.40),
        "psi_tenure": np.random.exponential(0.05, n_days) * (1 + np.arange(n_days)/60),
        "psi_monthly": np.random.exponential(0.04, n_days),
    })


# Sidebar
with st.sidebar:
    st.header("Settings")
    days = st.slider("Days to display", 7, 90, 30)
    auc_threshold = st.slider("AUC alert threshold", 0.70, 0.85, 0.78, 0.01)
    st.divider()
    st.header("Model Info")
    st.write("**Version:** 1.2.0")
    st.write("**Deployed:** 2024-07-01")
    st.write("**Algorithm:** GradientBoosting")

df = load_metrics().tail(days)
current = df.iloc[-1]

# KPI row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Model AUC", f"{current['auc']:.4f}",
            delta=f"{current['auc'] - df.iloc[0]['auc']:.4f}")
col2.metric("Daily Requests", f"{int(current['daily_requests']):,}")
col3.metric("P95 Latency", f"{current['p95_ms']:.0f}ms",
            delta=f"{current['p95_ms'] - df['p95_ms'].mean():.0f}ms", delta_color="inverse")
col4.metric("Error Rate", f"{current['error_rate']*100:.2f}%", delta_color="inverse")
col5.metric("Churn Rate", f"{current['churn_rate']*100:.1f}%")

# Alerts
alerts = []
if current['auc'] < auc_threshold:
    alerts.append(f"AUC ({current['auc']:.4f}) below threshold ({auc_threshold})")
if current['p95_ms'] > 200:
    alerts.append(f"P95 latency ({current['p95_ms']:.0f}ms) above 200ms")
if current['psi_tenure'] > 0.2:
    alerts.append("Tenure drift (PSI > 0.2) — consider retraining")

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.success("All systems healthy")

# Charts
tab1, tab2, tab3 = st.tabs(["Performance", "Infrastructure", "Drift"])

with tab1:
    if PLOTLY_AVAILABLE:
        fig = px.line(df, x="date", y="auc", title="Model AUC")
        fig.add_hline(y=auc_threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(df, x="date", y="churn_rate", title="Actual Churn Rate")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.line_chart(df.set_index('date')[['auc']])

with tab2:
    if PLOTLY_AVAILABLE:
        fig3 = px.area(df, x="date", y="daily_requests", title="Daily Requests")
        fig4 = px.line(df, x="date", y=["p50_ms", "p95_ms"], title="API Latency (ms)")
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.line_chart(df.set_index('date')[['daily_requests']])

with tab3:
    drift_df = df[['date', 'psi_tenure', 'psi_monthly']].copy()
    drift_df['status_tenure'] = drift_df['psi_tenure'].apply(
        lambda x: 'HIGH' if x > 0.2 else 'MODERATE' if x > 0.1 else 'OK'
    )
    st.dataframe(drift_df.tail(14))
    if PLOTLY_AVAILABLE:
        fig5 = px.line(df, x="date", y=["psi_tenure", "psi_monthly"], title="Feature Drift (PSI)")
        fig5.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Monitor")
        fig5.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Retrain")
        st.plotly_chart(fig5, use_container_width=True)
