"""
Bitcoin Investment Strategies Module

Contains implementations of:
- HODL (Buy and Hold)
- DCA (Dollar-Cost Averaging)
- Quantitative (10-Factor Random Forest)
"""

from .hodl import compute_metrics as hodl_compute
from .dca import build_contributions, compute_metrics as dca_compute
from .quant_rf import run as quant_run

__all__ = [
    'hodl_compute',
    'dca_compute',
    'build_contributions',
    'quant_run',
]
