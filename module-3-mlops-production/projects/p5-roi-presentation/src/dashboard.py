"""Streamlit dashboard for ML model monitoring and ROI analysis.

Run with: streamlit run src/dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def generate_mock_metrics(n_days=90):
    """Generate simulated production metrics for the dashboard."""
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=n_days - i) for i in range(n_days)]

    # Model AUC: starts 0.85, degrades after day 60
    base_auc = 0.85
    auc = [base_auc + np.random.normal(0, 0.005) - max(0, (i - 60) * 0.002) for i in range(n_days)]

    # Latency: ~50ms normally, spike on days 44-46
    latency_p50 = [50 + np.random.normal(0, 5) + (100 if 44 <= i <= 46 else 0) for i in range(n_days)]
    latency_p95 = [p * 2.5 + np.random.normal(0, 10) for p in latency_p50]

    # Request volume: weekly seasonal pattern
    requests = [int(1000 + 200 * np.sin(i * 2 * np.pi / 7) + np.random.normal(0, 50)) for i in range(n_days)]

    # Error rate: mostly low, occasional spikes
    error_rate = [0.001 + np.random.exponential(0.0005) for _ in range(n_days)]

    return pd.DataFrame({
        'date': dates, 'auc': auc, 'latency_p50': latency_p50,
        'latency_p95': latency_p95, 'requests': requests, 'error_rate': error_rate
    })


def calculate_roi(n_customers, churn_rate, precision, recall, retention_cost, customer_ltv, infra_monthly):
    """Calculate ROI of the churn prediction model."""
    actual_churners = int(n_customers * churn_rate)
    predicted_churners = int(actual_churners * recall / max(precision, 0.01))
    true_positives = int(predicted_churners * precision)

    # Value: saved customers * LTV * save_rate (assume 40% of targeted are actually saved)
    save_rate = 0.40
    saved = int(true_positives * save_rate)
    value = saved * customer_ltv

    # Cost: retention offers + infrastructure
    cost = predicted_churners * retention_cost + infra_monthly * 12

    net = value - cost
    roi_pct = (net / cost * 100) if cost > 0 else 0

    return {
        'actual_churners': actual_churners,
        'predicted_churners': predicted_churners,
        'true_positives': true_positives,
        'saved_customers': saved,
        'annual_value': value,
        'annual_cost': cost,
        'net_value': net,
        'roi_pct': roi_pct,
    }


# ==================== STREAMLIT APP ====================

st.set_page_config(page_title="ML Monitoring Dashboard", layout="wide")
st.title("🎯 Churn Model — Production Dashboard")

# --- Sidebar ---
st.sidebar.header("Settings")
n_days = st.sidebar.slider("Days to display", 30, 90, 90)

st.sidebar.header("ROI Parameters")
n_customers = st.sidebar.number_input("Total customers", value=5000, step=500)
churn_rate = st.sidebar.slider("Churn rate", 0.05, 0.50, 0.25)
precision = st.sidebar.slider("Model precision", 0.30, 0.95, 0.72)
recall = st.sidebar.slider("Model recall", 0.30, 0.95, 0.65)
retention_cost = st.sidebar.number_input("Retention offer ($)", value=50, step=10)
customer_ltv = st.sidebar.number_input("Customer LTV ($)", value=1200, step=100)
infra_monthly = st.sidebar.number_input("Infra cost ($/month)", value=500, step=100)

# --- Generate Data ---
metrics = generate_mock_metrics(n_days)

# --- KPI Row ---
st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
current_auc = metrics['auc'].iloc[-1]
prev_auc = metrics['auc'].iloc[-7] if len(metrics) > 7 else metrics['auc'].iloc[0]
col1.metric("Model AUC", f"{current_auc:.3f}", f"{current_auc - prev_auc:.3f}")
col2.metric("Latency p50", f"{metrics['latency_p50'].iloc[-1]:.0f}ms")
col3.metric("Daily Requests", f"{metrics['requests'].iloc[-1]:,}")
col4.metric("Error Rate", f"{metrics['error_rate'].iloc[-1]:.3%}")

# --- Charts ---
st.subheader("Performance Trends")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics['date'], metrics['auc'], color='#2ecc71', linewidth=1.5)
    ax.axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_ylabel('AUC')
    ax.set_title('Model AUC Over Time')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with chart_col2:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics['date'], metrics['latency_p50'], color='#3498db', label='p50', linewidth=1.5)
    ax.plot(metrics['date'], metrics['latency_p95'], color='#e74c3c', label='p95', linewidth=1.5)
    ax.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='SLA')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('API Latency')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# --- ROI Section ---
st.subheader("ROI Analysis")
roi = calculate_roi(n_customers, churn_rate, precision, recall, retention_cost, customer_ltv, infra_monthly)

roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
roi_col1.metric("Saved Customers", f"{roi['saved_customers']:,}")
roi_col2.metric("Annual Value", f"${roi['annual_value']:,.0f}")
roi_col3.metric("Annual Cost", f"${roi['annual_cost']:,.0f}")
roi_col4.metric("ROI", f"{roi['roi_pct']:.0f}%", f"${roi['net_value']:,.0f} net")

# --- Footer ---
st.markdown("---")
st.caption("Dashboard built with Streamlit | Data is simulated for demonstration")
