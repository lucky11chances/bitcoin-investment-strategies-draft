"""
Bitcoin Investment Strategies Comparison
Main entry point to run and compare all strategies on both training and test data
"""

import sys
from typing import Dict, Any

from config import TRAIN_DATA_PATH, TEST_DATA_PATH, DISPLAY_WIDTH, INITIAL_CAPITAL, TRAINED_WEIGHTS
from utils import print_separator, format_value, print_metrics_table, load_bitcoin_data
from strategies import hodl_compute, dca_compute, build_contributions, quant_run


def run_hodl_strategy() -> Dict[str, Any]:
    """Run HODL strategy on training data"""
    print("\n🏬 Running HODL Strategy...")
    df = load_bitcoin_data(TRAIN_DATA_PATH)
    metrics = hodl_compute(df)
    return metrics


def run_dca_strategy() -> Dict[str, Any]:
    """Run DCA strategy on training data"""
    print("\n💰 Running DCA Strategy...")
    df = load_bitcoin_data(TRAIN_DATA_PATH)
    contribs = build_contributions(df)
    metrics = dca_compute(df, contribs)
    return metrics


def run_quant_strategy() -> Dict[str, Any]:
    """Run Quantitative strategy on training data"""
    print("\n📊 Running Quantitative Strategy...")
    print("(This may take a few minutes due to optimization...)")
    metrics = quant_run(retrain=False)  # Use pre-trained weights
    return metrics


# ========================================
# TEST SET FUNCTIONS
# ========================================

def run_hodl_test() -> Dict[str, Any]:
    """Run HODL strategy on test data"""
    print("\n🏦 Running HODL Strategy on Test Set...")
    df = load_bitcoin_data(TEST_DATA_PATH)
    metrics = hodl_compute(df)
    return metrics


def run_dca_test() -> Dict[str, Any]:
    """Run DCA strategy on test data"""
    print("\n💰 Running DCA Strategy on Test Set...")
    df = load_bitcoin_data(TEST_DATA_PATH)
    contribs = build_contributions(df)
    metrics = dca_compute(df, contribs)
    return metrics


def run_quant_test() -> Dict[str, Any]:
    """Run Quantitative strategy on test data using trained weights"""
    print("\n📊 Running Quantitative Strategy on Test Set...")
    print("(Using pre-trained weights from 2010-2020 data)")
    
    # Import quant_rf functions
    from strategies.quant_rf import load_btc_data, compute_factors, rolling_standardize, weights_to_positions, backtest
    import numpy as np
    
    df = load_btc_data(TEST_DATA_PATH)
    df, factor_cols = compute_factors(df)
    df, z_cols = rolling_standardize(df, factor_cols, window=90)
    df = df.dropna(subset=z_cols + ['ret']).reset_index(drop=True)
    
    # Use trained weights from config
    best_w = np.array(TRAINED_WEIGHTS)
    
    # Backtest on test data
    final_pos = weights_to_positions(df, z_cols, best_w)
    final_res = backtest(df, final_pos, tc_bps=5.0)
    final_value = INITIAL_CAPITAL * (1.0 + final_res['cum_return'])
    
    return {
        'sharpe_ratio': final_res['sharpe'],
        'sortino_ratio': None,
        'max_drawdown': None,
        'cum_return': final_res['cum_return'],
        'final_value': final_value,
        'best_weights': best_w.tolist()
    }


# ========================================
# COMPARISON AND ANALYSIS FUNCTIONS
# ========================================


# ========================================
# COMPARISON AND ANALYSIS FUNCTIONS
# ========================================

def print_comparison_table(hodl_metrics: Dict, dca_metrics: Dict, quant_metrics: Dict, dataset_name: str = "TRAINING SET"):
    """Print side-by-side comparison of all strategies"""
    print_separator(f"📈 {dataset_name} STRATEGIES COMPARISON", 100)
    
    # Header
    print(f"{'Metric':<30} {'HODL':>20} {'DCA':>20} {'Quantitative':>20}")
    print(f"{'-' * 100}")
    
    # Compare common metrics
    comparison_metrics = [
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Sortino Ratio', 'sortino_ratio'),
        ('Max Drawdown', 'max_drawdown'),
        ('Cumulative Return', 'cumulative_return'),
        ('Final Value', 'final_value'),
    ]
    
    for display_name, key in comparison_metrics:
        hodl_val = format_value(key, hodl_metrics.get(key, 'N/A'))
        dca_val = format_value(key, dca_metrics.get(key, 'N/A'))
        quant_val = format_value(key, quant_metrics.get(key, 'N/A'))
        
        print(f"{display_name:<30} {hodl_val:>20} {dca_val:>20} {quant_val:>20}")
    
    print_separator("", 100)


def print_winner_analysis(hodl_metrics: Dict, dca_metrics: Dict, quant_metrics: Dict):
    """Print analysis of which strategy performed best"""
    print_separator("🏆 WINNER ANALYSIS", 80)
    
    # Compare Sharpe ratios
    sharpe_scores = {
        'HODL': hodl_metrics.get('sharpe_ratio', float('-inf')),
        'DCA': dca_metrics.get('sharpe_ratio', float('-inf')),
        'Quantitative': quant_metrics.get('sharpe_ratio', float('-inf'))
    }
    
    best_sharpe = max(sharpe_scores, key=sharpe_scores.get)
    print(f"Best Risk-Adjusted Return (Sharpe): {best_sharpe} ({sharpe_scores[best_sharpe]:.4f})")
    
    # Compare final values
    final_values = {
        'HODL': hodl_metrics.get('final_value', 0),
        'DCA': dca_metrics.get('final_value', 0),
        'Quantitative': quant_metrics.get('final_value', 0)
    }
    
    best_return = max(final_values, key=final_values.get)
    print(f"Best Absolute Return: {best_return} (${final_values[best_return]:,.2f})")
    
    # Compare drawdowns (less negative is better, filter out None values)
    drawdowns = {
        k: v for k, v in {
            'HODL': hodl_metrics.get('max_drawdown'),
            'DCA': dca_metrics.get('max_drawdown'),
            'Quantitative': quant_metrics.get('max_drawdown')
        }.items() if v is not None
    }
    
    if drawdowns:
        best_drawdown = max(drawdowns, key=drawdowns.get)
        print(f"Best Risk Control (Min Drawdown): {best_drawdown} ({drawdowns[best_drawdown]:.2%})")
    else:
        print(f"Best Risk Control (Min Drawdown): N/A (Quantitative strategy missing drawdown data)")
    
    print_separator("", 80)
    
    # Overall recommendation
    print("\n💡 RECOMMENDATION:")
    print("   - For maximum absolute returns: Choose the strategy with highest final value")
    print("   - For risk-adjusted returns: Choose the strategy with highest Sharpe ratio")
    print("   - For stable growth: DCA typically provides the most consistent results")
    print("   - For hands-off investing: HODL requires zero maintenance")
    print()


def print_train_vs_test_comparison(train_sharpes: Dict[str, float], test_sharpes: Dict[str, float]):
    """Compare training vs test Sharpe ratios to detect overfitting"""
    print_separator("📊 TRAINING vs TEST SET PERFORMANCE", 80)
    
    print(f"{'Strategy':<20} {'Training Sharpe':>20} {'Test Sharpe':>20} {'Difference':>20}")
    print(f"{'-' * 80}")
    
    for strategy in ['HODL', 'DCA', 'Quantitative']:
        train_sharpe = train_sharpes.get(strategy, float('nan'))
        test_sharpe = test_sharpes.get(strategy, float('nan'))
        diff = test_sharpe - train_sharpe
        
        diff_str = f"{diff:+.4f}" if not (test_sharpe != test_sharpe) else "N/A"  # NaN check
        
        print(f"{strategy:<20} {train_sharpe:>20.4f} {test_sharpe:>20.4f} {diff_str:>20}")
    
    print_separator("", 80)
    
    # Overfitting warning
    quant_train = train_sharpes.get('Quantitative', 0)
    quant_test = test_sharpes.get('Quantitative', 0)
    
    if quant_test < 0:
        print("\n⚠️  WARNING: Quantitative strategy has NEGATIVE Sharpe on test set!")
        print("   This indicates severe overfitting to training data (2010-2020).")
        print(f"   Training Sharpe: {quant_train:.4f} → Test Sharpe: {quant_test:.4f}")
    elif quant_test < quant_train * 0.5:
        print("\n⚠️  WARNING: Quantitative strategy performance degraded significantly!")
        print("   This suggests overfitting to training data characteristics.")
        print(f"   Training Sharpe: {quant_train:.4f} → Test Sharpe: {quant_test:.4f}")
    
    print()


def main():
    """Main function to run all strategies on both training and test sets"""
    
    print_separator("🚀 BITCOIN INVESTMENT STRATEGIES - COMPLETE ANALYSIS", DISPLAY_WIDTH)
    print("\nThis analysis will run on:")
    print("  📊 Training Set: 2010-2020 (for strategy development)")
    print("  🧪 Test Set: 2023-2024 (for out-of-sample validation)")
    print("\nThree strategies:")
    print("  1. HODL (Buy and Hold)")
    print("  2. DCA (Dollar-Cost Averaging)")
    print("  3. Quantitative (10-Factor Optimization)")
    
    # Check data files
    if not TRAIN_DATA_PATH.exists():
        print(f"\n❌ ERROR: Training data not found at {TRAIN_DATA_PATH}")
        print("Please ensure the CSV file exists in the data/ directory.")
        sys.exit(1)
    
    if not TEST_DATA_PATH.exists():
        print(f"\n❌ ERROR: Test data not found at {TEST_DATA_PATH}")
        print("Please ensure the CSV file exists in the data/ directory.")
        sys.exit(1)
    
    try:
        # =====================================
        # PART 1: TRAINING SET (2010-2020)
        # =====================================
        print_separator("PART 1: TRAINING SET EVALUATION (2010-2020)", DISPLAY_WIDTH)
        
        hodl_train = run_hodl_strategy()
        dca_train = run_dca_strategy()
        quant_train = run_quant_strategy()
        
        print("\n")
        print_metrics_table(hodl_train, "HODL Strategy - Training Results", DISPLAY_WIDTH)
        print_metrics_table(dca_train, "DCA Strategy - Training Results", DISPLAY_WIDTH)
        print_metrics_table(quant_train, "QUANTITATIVE Strategy - Training Results", DISPLAY_WIDTH)
        
        print_comparison_table(hodl_train, dca_train, quant_train, "TRAINING SET (2010-2020)")
        
        # =====================================
        # PART 2: TEST SET (2023-2024)
        # =====================================
        print_separator("PART 2: TEST SET EVALUATION (2023-2024)", DISPLAY_WIDTH)
        
        hodl_test = run_hodl_test()
        dca_test = run_dca_test()
        quant_test = run_quant_test()
        
        print("\n")
        print_metrics_table(hodl_test, "HODL Strategy - Test Results", DISPLAY_WIDTH)
        print_metrics_table(dca_test, "DCA Strategy - Test Results", DISPLAY_WIDTH)
        print_metrics_table(quant_test, "QUANTITATIVE Strategy - Test Results", DISPLAY_WIDTH)
        
        print_comparison_table(hodl_test, dca_test, quant_test, "TEST SET (2023-2024)")
        
        # =====================================
        # PART 3: CROSS-VALIDATION ANALYSIS
        # =====================================
        print_separator("PART 3: CROSS-VALIDATION ANALYSIS", DISPLAY_WIDTH)
        
        # Training vs Test comparison
        train_sharpes = {
            'HODL': hodl_train.get('sharpe_ratio', float('nan')),
            'DCA': dca_train.get('sharpe_ratio', float('nan')),
            'Quantitative': quant_train.get('sharpe_ratio', float('nan'))
        }
        
        test_sharpes = {
            'HODL': hodl_test.get('sharpe_ratio', float('nan')),
            'DCA': dca_test.get('sharpe_ratio', float('nan')),
            'Quantitative': quant_test.get('sharpe_ratio', float('nan'))
        }
        
        print_train_vs_test_comparison(train_sharpes, test_sharpes)
        
        # Final recommendations
        print_separator("💡 FINAL RECOMMENDATIONS", DISPLAY_WIDTH)
        
        print("Based on the complete analysis:\n")
        
        # Check which strategy is most consistent
        hodl_diff = abs(test_sharpes['HODL'] - train_sharpes['HODL'])
        dca_diff = abs(test_sharpes['DCA'] - train_sharpes['DCA'])
        quant_diff = abs(test_sharpes['Quantitative'] - train_sharpes['Quantitative'])
        
        most_consistent = min([('HODL', hodl_diff), ('DCA', dca_diff), ('Quantitative', quant_diff)], 
                             key=lambda x: x[1])[0]
        
        print(f"✅ Most Consistent Strategy: {most_consistent}")
        print(f"   - Smallest difference between training and test Sharpe ratios")
        
        # Best test performance
        best_test = max(test_sharpes, key=test_sharpes.get)
        print(f"\n✅ Best Test Set Performance: {best_test}")
        print(f"   - Highest Sharpe ratio on unseen data (2023-2024)")
        
        # Risk assessment
        print("\n⚠️  Risk Assessment:")
        if quant_diff > 1.0:
            print("   - Quantitative strategy shows signs of overfitting")
            print("   - Consider using DCA or HODL for more reliable results")
        else:
            print("   - All strategies show reasonable generalization")
        
        print("\n" + "="*DISPLAY_WIDTH)
        print("✅ Complete analysis finished! Check docs/ directory for detailed reports.")
        print("="*DISPLAY_WIDTH + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
