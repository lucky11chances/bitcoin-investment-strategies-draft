"""
Utility Functions
Common helper functions for data loading, formatting, etc.
"""

from pathlib import Path
from typing import Any
import pandas as pd


def load_bitcoin_data(csv_path: Path) -> pd.DataFrame:
    """
    Load Bitcoin CSV data with automatic date column detection.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        DataFrame with standardized 'Date' column and sorted by date
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If no valid date column is found
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Auto-detect date column
    date_col = None
    for candidate in ("Date", "Start"):
        if candidate in df.columns:
            date_col = candidate
            break
    
    if date_col is None:
        raise ValueError("No date column found. Expected one of: Date, Start")

    # Standardize date column
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "Date"})
    
    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)
    
    return df


def format_value(key: str, value: Any, decimal_places: int = 4) -> str:
    """
    Format metric values for display.
    
    Args:
        key: Metric name (used to determine formatting)
        value: Value to format
        decimal_places: Number of decimal places for floats
    
    Returns:
        Formatted string representation
    """
    if isinstance(value, float):
        # Currency values
        if any(keyword in key.lower() for keyword in ['value', 'profit', 'invested', 'price', 'capital', 'cost']):
            return f"${value:,.2f}"
        # Ratio/percentage values
        elif any(keyword in key.lower() for keyword in ['ratio', 'drawdown', 'return', 'volatility']):
            return f"{value:.{decimal_places}f}"
        # Default float formatting
        else:
            return f"{value:.{decimal_places}f}"
    elif isinstance(value, list):
        # For arrays like weights
        return str(value)
    else:
        return str(value)


def print_separator(title: str = "", width: int = 80):
    """
    Print a formatted separator line.
    
    Args:
        title: Optional title to center in the separator
        width: Width of the separator line
    """
    if title:
        print(f"\n{'=' * width}")
        print(f"{title:^{width}}")
        print(f"{'=' * width}")
    else:
        print(f"{'=' * width}")


def print_metrics_table(metrics: dict, title: str = "", width: int = 80):
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metric name -> value
        title: Optional table title
        width: Width of the table
    """
    if title:
        print_separator(title, width)
    
    for key, value in metrics.items():
        formatted_value = format_value(key, value)
        print(f"{key.replace('_', ' ').title():.<40} {formatted_value:>38}")
    
    print_separator("", width)


def validate_required_columns(df: pd.DataFrame, required_cols: list):
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
    
    Raises:
        ValueError: If any required column is missing
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
