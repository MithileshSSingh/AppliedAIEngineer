"""Common data loading and cleaning helpers."""

import pandas as pd
from pathlib import Path


def load_csv(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file with sensible defaults."""
    return pd.read_csv(filepath, **kwargs)


def describe_df(df: pd.DataFrame) -> None:
    """Print a quick summary of a DataFrame."""
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
