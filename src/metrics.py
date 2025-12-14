"""
Financial Metrics Calculation Module
Unified functions for computing Sharpe ratio, Sortino ratio, Max Drawdown, etc.
"""

import math
import numpy as np
import pandas as pd
from typing import Optional

from config import RISK_FREE_RATE_ANNUAL, DAYS_PER_YEAR


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
    periods_per_year: int = DAYS_PER_YEAR
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of daily returns (not excess returns)
        risk_free_rate: Annual risk-free rate (e.g., 0.03 for 3%)
        periods_per_year: Number of periods per year for annualization
    
    Returns:
        Annualized Sharpe ratio, or np.nan if cannot be calculated
    """
    if len(returns) == 0:
        return np.nan
    
    # Convert annual risk-free rate to daily
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    
    # Calculate excess returns
    excess_returns = returns - rf_daily
    
    # Calculate volatility
    vol = excess_returns.std(ddof=1)
    
    if vol > 0 and not np.isnan(vol):
        sharpe = (excess_returns.mean() / vol) * math.sqrt(periods_per_year)
        return float(sharpe)
    
    return np.nan


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
    periods_per_year: int = DAYS_PER_YEAR
) -> float:
    """
    Calculate annualized Sortino ratio (focuses on downside volatility).
    
    Args:
        returns: Series of daily returns (not excess returns)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year for annualization
    
    Returns:
        Annualized Sortino ratio, or np.nan if cannot be calculated
    """
    if len(returns) == 0:
        return np.nan
    
    # Convert annual risk-free rate to daily
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    
    # Calculate excess returns
    excess_returns = returns - rf_daily
    
    # Only consider negative excess returns for downside volatility
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.nan
    
    down_vol = downside_returns.std(ddof=1)
    
    if down_vol > 0 and not np.isnan(down_vol):
        sortino = (excess_returns.mean() / down_vol) * math.sqrt(periods_per_year)
        return float(sortino)
    
    return np.nan


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Calculate maximum drawdown (as a negative number).
    
    Args:
        portfolio_values: Series of portfolio values over time
    
    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.5 means 50% drawdown)
    """
    if len(portfolio_values) == 0:
        return np.nan
    
    running_max = portfolio_values.cummax()
    drawdown = portfolio_values / running_max - 1.0
    max_dd = float(drawdown.min())
    
    return max_dd


def calculate_cumulative_return(
    initial_value: float,
    final_value: float
) -> float:
    """
    Calculate cumulative return percentage.
    
    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value
    
    Returns:
        Cumulative return as decimal (e.g., 1.5 means 150% gain)
    """
    if initial_value <= 0:
        return np.nan
    
    return (final_value - initial_value) / initial_value


def calculate_annualized_return(
    initial_value: float,
    final_value: float,
    years: float
) -> float:
    """
    Calculate annualized return (CAGR).
    
    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value
        years: Investment duration in years
    
    Returns:
        Annualized return as decimal
    """
    if initial_value <= 0 or years <= 0:
        return np.nan
    
    return (final_value / initial_value) ** (1.0 / years) - 1.0


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = DAYS_PER_YEAR
) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Series of period returns
        periods_per_year: Number of periods per year for annualization
    
    Returns:
        Annualized volatility (standard deviation)
    """
    if len(returns) == 0:
        return np.nan
    
    vol = returns.std(ddof=1)
    
    if pd.isna(vol):
        return np.nan
    
    return vol * math.sqrt(periods_per_year)


def calculate_all_metrics(
    portfolio_values: pd.Series,
    initial_investment: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
    periods_per_year: int = DAYS_PER_YEAR
) -> dict:
    """
    Calculate all standard metrics for a strategy.
    
    Args:
        portfolio_values: Series of daily portfolio values
        initial_investment: Initial capital invested
        start_date: Strategy start date
        end_date: Strategy end date
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Dictionary with all calculated metrics
    """
    # Calculate returns
    daily_returns = portfolio_values.pct_change().dropna()
    
    # Time duration
    duration_days = (end_date - start_date).days
    duration_years = duration_days / DAYS_PER_YEAR
    
    # Calculate metrics
    sharpe = calculate_sharpe_ratio(daily_returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(daily_returns, risk_free_rate, periods_per_year)
    max_dd = calculate_max_drawdown(portfolio_values)
    cum_return = calculate_cumulative_return(initial_investment, portfolio_values.iloc[-1])
    annual_return = calculate_annualized_return(initial_investment, portfolio_values.iloc[-1], duration_years)
    volatility = calculate_volatility(daily_returns, periods_per_year)
    
    return {
        "start_date": start_date.date() if hasattr(start_date, 'date') else start_date,
        "end_date": end_date.date() if hasattr(end_date, 'date') else end_date,
        "duration_years": duration_years,
        "initial_value": initial_investment,
        "final_value": float(portfolio_values.iloc[-1]),
        "cumulative_return": cum_return,
        "annualized_return": annual_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "volatility": volatility,
    }
