"""
Hierarchical Bayesian Marketing Mix Modeling (MMM) Package

A production-grade MMM system using Meridian and PyMC for digital-first 
marketing attribution and budget optimization.
"""

__version__ = "0.1.0"
__author__ = "devcharlie"

from .config import load_config, Config
from .cli import main

__all__ = ["load_config", "Config", "main"]
