"""A/B test analysis utilities for model comparison."""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats


def sample_size_calculator(
    baseline_rate: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Calculate required sample size per group for a two-proportion Z-test.

    Args:
        baseline_rate: Current conversion/churn rate (e.g., 0.25 for 25%)
        min_detectable_effect: Minimum effect size to detect (e.g., 0.03 for 3pp improvement)
        alpha: Significance level (default 0.05)
        power: Statistical power (default 0.80)

    Returns:
        Required sample size per group

    Example:
        >>> sample_size_calculator(0.25, 0.03)
        2781
    """
    treatment_rate = baseline_rate - min_detectable_effect
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p_bar = (baseline_rate + treatment_rate) / 2

    n = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
         z_beta * np.sqrt(baseline_rate * (1 - baseline_rate) + treatment_rate * (1 - treatment_rate))) ** 2
    n /= (baseline_rate - treatment_rate) ** 2

    return int(np.ceil(n))


def analyze_ab_test(
    control: pd.DataFrame,
    treatment: pd.DataFrame,
    outcome_col: str,
    metric_type: str = "binary",
    alpha: float = 0.05,
) -> dict:
    """
    Analyze A/B test results.

    Args:
        control: DataFrame for control group (Model A)
        treatment: DataFrame for treatment group (Model B)
        outcome_col: Column with the outcome metric
        metric_type: "binary" (proportion test) or "continuous" (t-test)
        alpha: Significance level

    Returns:
        Dict with test results, confidence intervals, and recommendation
    """
    n_ctrl = len(control)
    n_trt = len(treatment)

    mean_ctrl = control[outcome_col].mean()
    mean_trt = treatment[outcome_col].mean()

    if metric_type == "binary":
        # Z-test for proportions
        p_pool = (control[outcome_col].sum() + treatment[outcome_col].sum()) / (n_ctrl + n_trt)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_trt))
        z_stat = (mean_trt - mean_ctrl) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        test_name = "Two-proportion Z-test"

        # CI for difference
        se_diff = np.sqrt(mean_ctrl*(1-mean_ctrl)/n_ctrl + mean_trt*(1-mean_trt)/n_trt)
    else:
        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(control[outcome_col], treatment[outcome_col])
        z_stat = t_stat
        test_name = "Welch's T-test"
        se_diff = np.sqrt(control[outcome_col].var()/n_ctrl + treatment[outcome_col].var()/n_trt)

    diff = mean_trt - mean_ctrl
    ci_low = diff - 1.96 * se_diff
    ci_high = diff + 1.96 * se_diff

    significant = p_value < alpha
    relative_change = (diff / mean_ctrl) * 100 if mean_ctrl != 0 else 0

    return {
        "test": test_name,
        "control_mean": round(mean_ctrl, 6),
        "treatment_mean": round(mean_trt, 6),
        "absolute_diff": round(diff, 6),
        "relative_change_pct": round(relative_change, 2),
        "ci_95": (round(ci_low, 6), round(ci_high, 6)),
        "p_value": round(p_value, 6),
        "significant": significant,
        "recommendation": "Deploy treatment" if significant and diff < 0 and metric_type == "binary"
                         else "Deploy treatment" if significant and diff > 0 and metric_type == "continuous"
                         else "Keep control (no significant improvement)",
        "n_control": n_ctrl,
        "n_treatment": n_trt,
    }


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index."""
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf; bins[-1] = np.inf
    ref_pct = np.histogram(reference, bins=bins)[0] / len(reference) + 1e-8
    cur_pct = np.histogram(current, bins=bins)[0] / len(current) + 1e-8
    ref_pct /= ref_pct.sum(); cur_pct /= cur_pct.sum()
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(ref_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Generate a drift report for all numeric columns."""
    results = []
    for col in ref_df.select_dtypes(include=np.number).columns:
        ks_stat, ks_p = stats.ks_2samp(ref_df[col].dropna(), current_df[col].dropna())
        psi = compute_psi(ref_df[col].dropna().values, current_df[col].dropna().values)
        results.append({
            'feature': col,
            'ref_mean': ref_df[col].mean(),
            'current_mean': current_df[col].mean(),
            'ks_statistic': round(ks_stat, 4),
            'ks_p_value': round(ks_p, 6),
            'psi': round(psi, 4),
            'status': '🔴 RETRAIN' if psi > 0.2 else '🟡 MONITOR' if psi > 0.1 else '🟢 OK',
        })
    return pd.DataFrame(results).sort_values('psi', ascending=False)
