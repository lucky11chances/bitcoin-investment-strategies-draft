import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    TRAIN_DATA_PATH,
    DCA_MONTHLY_AMOUNT,
    INITIAL_CAPITAL
)
from metrics import calculate_all_metrics
from utils import load_bitcoin_data


def build_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly contribution schedule for DCA strategy.
    Invests DCA_MONTHLY_AMOUNT per month until reaching INITIAL_CAPITAL total.
    """
    max_months = int(np.ceil(INITIAL_CAPITAL / DCA_MONTHLY_AMOUNT))
    monthly_first = df.resample("MS", on="Date").first().reset_index()
    if len(monthly_first) < max_months:
        raise ValueError(f"Not enough monthly data to invest {max_months} months; only {len(monthly_first)} months present.")

    monthly_first = monthly_first.head(max_months)
    monthly_first["btc_bought"] = DCA_MONTHLY_AMOUNT / monthly_first["Close"]
    monthly_first["cum_btc"] = monthly_first["btc_bought"].cumsum()
    return monthly_first[["Date", "cum_btc"]]


def compute_metrics(df: pd.DataFrame, contribs: pd.DataFrame) -> dict:
    """
    Compute DCA strategy metrics.
    Merges contribution schedule with price data and calculates performance.
    """
    merged = df.merge(contribs, on="Date", how="left")
    merged["cum_btc"] = merged["cum_btc"].ffill().fillna(0.0)
    merged["portfolio"] = merged["Close"] * merged["cum_btc"]

    # Only calculate returns after first investment
    invested = merged[merged["cum_btc"] > 0].copy()
    
    start_date = invested["Date"].iloc[0]
    end_date = invested["Date"].iloc[-1]
    
    # Use unified metrics calculation
    portfolio_values = invested["portfolio"]
    total_invested = DCA_MONTHLY_AMOUNT * len(contribs)
    
    metrics = calculate_all_metrics(
        portfolio_values=portfolio_values,
        initial_investment=total_invested,
        start_date=start_date,
        end_date=end_date
    )
    
    # Add DCA-specific fields
    metrics.update({
        "months_invested": len(contribs),
        "total_invested": total_invested,
    })
    
    return metrics


def main():
    """Run DCA strategy on training data."""
    df = load_bitcoin_data(TRAIN_DATA_PATH)
    contribs = build_contributions(df)
    metrics = compute_metrics(df, contribs)

    for key, val in metrics.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
