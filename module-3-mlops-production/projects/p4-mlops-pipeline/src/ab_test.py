"""A/B testing framework for comparing ML models in production."""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


class ABTestSimulator:
    """Simulate and analyze A/B tests for ML model comparison."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def simulate(self, n_users=10000, model_a_churn_rate=0.25, model_b_churn_rate=0.22,
                 split_ratio=0.5, customer_ltv=1200, retention_cost=50):
        """Simulate A/B test with two churn models."""
        n_a = int(n_users * split_ratio)
        n_b = n_users - n_a

        group_a = pd.DataFrame({
            'user_id': range(1, n_a+1),
            'group': 'A',
            'churned': self.rng.binomial(1, model_a_churn_rate, n_a),
            'customer_ltv': customer_ltv,
            'retention_cost': retention_cost,
        })
        group_b = pd.DataFrame({
            'user_id': range(n_a+1, n_users+1),
            'group': 'B',
            'churned': self.rng.binomial(1, model_b_churn_rate, n_b),
            'customer_ltv': customer_ltv,
            'retention_cost': retention_cost,
        })

        results = pd.concat([group_a, group_b], ignore_index=True)
        # Simulate daily enrollment
        results['day'] = self.rng.randint(1, 31, len(results))
        results = results.sort_values('day').reset_index(drop=True)
        return results

    def analyze(self, results_df):
        """Run statistical analysis on A/B results."""
        a = results_df[results_df['group'] == 'A']
        b = results_df[results_df['group'] == 'B']

        rate_a = a['churned'].mean()
        rate_b = b['churned'].mean()
        n_a, n_b = len(a), len(b)

        # Chi-squared test
        contingency = pd.crosstab(results_df['group'], results_df['churned'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Confidence intervals (Wilson score)
        def wilson_ci(p, n, z=1.96):
            denom = 1 + z**2/n
            center = (p + z**2/(2*n)) / denom
            spread = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
            return (center - spread, center + spread)

        ci_a = wilson_ci(rate_a, n_a)
        ci_b = wilson_ci(rate_b, n_b)

        # Effect size (Cohen's h)
        h = 2 * (np.arcsin(np.sqrt(rate_a)) - np.arcsin(np.sqrt(rate_b)))

        # Business impact
        ltv = results_df['customer_ltv'].iloc[0]
        ret_cost = results_df['retention_cost'].iloc[0]
        saved_customers = (rate_a - rate_b) * n_b
        value_saved = saved_customers * ltv
        extra_cost = saved_customers * ret_cost

        return {
            'rate_a': rate_a, 'rate_b': rate_b,
            'n_a': n_a, 'n_b': n_b,
            'absolute_diff': rate_a - rate_b,
            'relative_diff': (rate_a - rate_b) / rate_a if rate_a > 0 else 0,
            'chi2': chi2, 'p_value': p_value,
            'significant': p_value < 0.05,
            'ci_a': ci_a, 'ci_b': ci_b,
            'effect_size_h': h,
            'saved_customers': saved_customers,
            'value_saved': value_saved,
            'net_value': value_saved - extra_cost,
        }

    def required_sample_size(self, baseline_rate, min_effect, alpha=0.05, power=0.80):
        """Calculate required sample size per group."""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        p1 = baseline_rate
        p2 = baseline_rate - min_effect
        p_avg = (p1 + p2) / 2
        n = ((z_alpha * np.sqrt(2*p_avg*(1-p_avg)) + z_beta * np.sqrt(p1*(1-p1)+p2*(1-p2)))**2) / (p1-p2)**2
        return int(np.ceil(n))

    def visualize(self, results_df, analysis):
        """Create 3-panel A/B test visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Churn rates with CI
        groups = ['A (Control)', 'B (Treatment)']
        rates = [analysis['rate_a'], analysis['rate_b']]
        ci_lower = [analysis['ci_a'][0], analysis['ci_b'][0]]
        ci_upper = [analysis['ci_a'][1], analysis['ci_b'][1]]
        errors = [[r-l for r, l in zip(rates, ci_lower)], [u-r for r, u in zip(rates, ci_upper)]]

        colors = ['#e74c3c', '#2ecc71']
        axes[0].bar(groups, rates, yerr=errors, color=colors, capsize=10, alpha=0.8)
        axes[0].set_ylabel('Churn Rate')
        axes[0].set_title(f"Churn Rates (p={analysis['p_value']:.4f})")

        # Panel 2: Cumulative churn over time
        for grp, color, label in [('A', '#e74c3c', 'Model A'), ('B', '#2ecc71', 'Model B')]:
            g = results_df[results_df['group'] == grp].sort_values('day')
            cum_churn = g['churned'].expanding().mean()
            axes[1].plot(range(len(cum_churn)), cum_churn, color=color, label=label, alpha=0.8)
        axes[1].set_xlabel('Users enrolled')
        axes[1].set_ylabel('Cumulative churn rate')
        axes[1].set_title('Cumulative Churn Over Enrollment')
        axes[1].legend()

        # Panel 3: Business impact
        impact = ['Saved Customers', 'Value Saved ($)', 'Net Value ($)']
        values = [analysis['saved_customers'], analysis['value_saved'], analysis['net_value']]
        bar_colors = ['#3498db', '#2ecc71', '#27ae60']
        axes[2].barh(impact, values, color=bar_colors, alpha=0.8)
        axes[2].set_title('Business Impact')
        for i, v in enumerate(values):
            axes[2].text(v + max(values)*0.02, i, f'{v:,.0f}', va='center')

        plt.tight_layout()
        plt.show()
