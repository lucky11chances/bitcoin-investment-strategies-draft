#!/usr/bin/env python3
"""
Bitcoin Investment Strategies - Main Entry Point

Run complete analysis including:
- Training set evaluation (2010-2020)
- Test set evaluation (2023-2024)
- Cross-validation and overfitting detection
- Final recommendations
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run main analysis
from main import main

if __name__ == "__main__":
    main()
