import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import (
    TRAIN_DATA_PATH,
    INITIAL_CAPITAL,
    CAPITAL_COST_RATE_ANNUAL,
    DAYS_PER_YEAR
)
from metrics import calculate_all_metrics
from utils import load_bitcoin_data


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute HODL strategy metrics.
    Buy and hold from first day with all initial capital.
    """
    close = df["Close"]
    start_date = df.loc[0, "Date"]
    end_date = df.loc[df.index[-1], "Date"]

    buy_price = close.iloc[0]
    btc_held = INITIAL_CAPITAL / buy_price

    # Portfolio value over time
    portfolio_values = close * btc_held
    
    # Use unified metrics calculation
    metrics = calculate_all_metrics(
        portfolio_values=portfolio_values,
        initial_investment=INITIAL_CAPITAL,
        start_date=start_date,
        end_date=end_date
    )
    
    # Add HODL-specific metrics
    duration_years = metrics["duration_years"]
    final_value = metrics["final_value"]
    
    # Capital cost: 7% annual opportunity cost
    capital_cost = INITIAL_CAPITAL * (1.0 + CAPITAL_COST_RATE_ANNUAL) ** duration_years
    net_profit_after_cost = final_value - capital_cost
    
    # Add HODL-specific fields
    metrics.update({
        "buy_price": float(buy_price),
        "btc_held": float(btc_held),
        "net_profit_after_7pct_cost": net_profit_after_cost,
    })
    
    return metrics


def main():
    """Run HODL strategy on training data."""
    df = load_bitcoin_data(TRAIN_DATA_PATH)
    metrics = compute_metrics(df)

    for key, val in metrics.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
