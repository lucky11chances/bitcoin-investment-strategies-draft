"""
Bitcoin Investment Strategies Analysis Package

A comprehensive framework for comparing cryptocurrency investment strategies
including HODL, DCA, and quantitative approaches.
"""

__version__ = "1.0.0"
__author__ = "Haoyu"

from .strategies import hodl_compute, dca_compute, build_contributions, quant_run

__all__ = [
    'hodl_compute',
    'dca_compute',
    'build_contributions',
    'quant_run',
]
