"""
Global Configuration for Bitcoin Investment Strategies
Centralized constants, paths, and parameters
"""

from pathlib import Path

# =========================
# Directory Paths
# =========================
# PROJECT_ROOT 现在指向项目根目录（src/ 的上一级）
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

# =========================
# Data File Paths
# =========================
TRAIN_DATA_PATH = DATA_DIR / "bitcoin_train_2010_2020 copy.csv"
TEST_DATA_PATH = DATA_DIR / "bitcoin_test_2023_2024 copy.csv"
VALID_DATA_PATH = DATA_DIR / "bitcoin_valid_2021_2022 copy.csv"

# =========================
# Financial Parameters
# =========================
# Initial capital for HODL strategy
INITIAL_CAPITAL = 12_500.0

# DCA parameters
DCA_MONTHLY_AMOUNT = 1_000.0
DCA_NUM_MONTHS = 13

# Risk-free rate (for Sharpe/Sortino ratio calculation)
RISK_FREE_RATE_ANNUAL = 0.03

# Capital cost rate (for HODL opportunity cost calculation)
CAPITAL_COST_RATE_ANNUAL = 0.07

# Trading cost (basis points)
TRADING_COST_BPS = 5  # 5 basis points = 0.05%

# =========================
# Time Constants
# =========================
DAYS_PER_YEAR = 365
TRADING_DAYS_PER_YEAR = 252  # For annualization in some calculations

# =========================
# Quantitative Strategy Parameters
# =========================
# Pre-trained weights for 10-factor model  
# Trained on 2010-2020 data with conservative parameters to reduce overfitting
# 300-step optimization, 100% max position, 15 bps trading cost, L2 penalty 0.012
# Target: $2.5B-$3B on training set (beat DCA), worse on test set (overfitting demo)
TRAINED_WEIGHTS = [
    0.30399229,
    -1.87053642,
    0.3864643,
    0.36818727,
    -0.35161529,
    1.32005279,
    0.41132621,
    2.56072047,
    0.61195289,
    -0.79305953
]

# Factor calculation windows
MOMENTUM_SHORT_WINDOW = 20
MOMENTUM_LONG_WINDOW = 60
MA_SHORT_WINDOW = 50
MA_LONG_WINDOW = 200
VOLATILITY_WINDOW = 20
ATR_WINDOW = 14
PRICE_POSITION_WINDOW = 60

# Rolling standardization window
ROLLING_Z_WINDOW = 90

# Optimization parameters (conservative to reduce overfitting)
OPTIMIZATION_STEPS = 300
OPTIMIZATION_STEP_SIZE = 0.05

# =========================
# Display Formatting
# =========================
DISPLAY_WIDTH = 80
DECIMAL_PLACES = 4
CURRENCY_DECIMAL_PLACES = 2
