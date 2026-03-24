"""Plotting helpers wrapping Matplotlib/Seaborn."""

import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    """Set a clean default plotting style."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100


def save_fig(fig, path: str, tight: bool = True):
    """Save a figure to disk."""
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved figure to {path}")
