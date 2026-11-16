"""Mass-spring-damper simulation and constrained parameter identification."""
from .core import FitResult, simulate, identify, bootstrap_intervals

__all__ = ["FitResult", "simulate", "identify", "bootstrap_intervals"]
