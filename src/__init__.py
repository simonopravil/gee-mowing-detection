"""
Grassland Mowing Detection Package

This package provides utilities for detecting grassland mowing events
using satellite time-series data (Sentinel-1 SAR and Sentinel-2 optical).
"""

from . import utils
from . import functions
from . import management

__version__ = "1.0.0"
__all__ = ["utils", "functions", "management"]
