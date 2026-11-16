"""Identify positive damping/stiffness from measured motion and forcing."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class FitResult:
    damping_ns_m: float
    stiffness_n_m: float
    rmse_newtons: float
    r_squared: float
    samples: int
    constrained: bool

    def as_dict(self) -> dict:
        return {
            "damping_ns_m": self.damping_ns_m,
            "stiffness_n_m": self.stiffness_n_m,
            "rmse_newtons": self.rmse_newtons,
            "r_squared": self.r_squared,
            "samples": self.samples,
            "constrained": self.constrained,
        }


def simulate(
    mass: float, damping: float, stiffness: float,
    duration: float, dt: float,
    force: Callable[[float], float],
    x0: float = 0.0, v0: float = 0.0,
) -> dict[str, np.ndarray]:
    """RK4 integration of m*x'' + c*x' + k*x = F(t), in SI units."""
    if mass <= 0 or damping < 0 or stiffness < 0 or duration <= 0 or dt <= 0:
        raise ValueError("Require positive mass, duration, dt and nonnegative c,k")
    times = np.linspace(0, duration, round(duration / dt) + 1)
    if len(times) < 2:
        raise ValueError("Step size too large for duration")
    states = np.empty((len(times), 2), dtype=float)
    states[0] = x0, v0

    def derivative(t: float, state: np.ndarray) -> np.ndarray:
        x, v = state
        return np.array([v, (force(t) - damping * v - stiffness * x) / mass])

    for i in range(len(times) - 1):
        t, h = float(times[i]), float(times[i + 1] - times[i])
        y = states[i]
        a = derivative(t, y)
        b = derivative(t + h / 2, y + h * a / 2)
        c = derivative(t + h / 2, y + h * b / 2)
        d = derivative(t + h, y + h * c)
        states[i + 1] = y + (h / 6) * (a + 2 * b + 2 * c + d)

    applied = np.array([force(float(t)) for t in times], dtype=float)
    accel = (applied - damping * states[:, 1] - stiffness * states[:, 0]) / mass
    return {"time": times, "displacement": states[:, 0],
            "velocity": states[:, 1], "acceleration": accel, "force": applied}


def _observations(
    mass: float, displacement: ArrayLike, velocity: ArrayLike,
    acceleration: ArrayLike, force: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Mass must be finite and positive")
    arrays = [np.asarray(x, dtype=float) for x in
              (displacement, velocity, acceleration, force)]
    if any(a.ndim != 1 for a in arrays) or len({a.size for a in arrays}) != 1:
        raise ValueError("Observations must be equally sized 1D arrays")
    if arrays[0].size < 3 or not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Need at least 3 finite observations")
    x, v, a, f = arrays
    matrix = np.column_stack((v, x))
    if np.linalg.matrix_rank(matrix) < 2:
        raise ValueError("Motion cannot independently identify damping and stiffness")
    return matrix, f - mass * a


def _nonnegative_least_squares(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    unconstrained, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    if np.all(unconstrained >= 0):
        return unconstrained
    v, x = matrix[:, 0], matrix[:, 1]
    candidates = [
        np.array([max(0.0, float(v @ target) / float(v @ v)), 0.0]),
        np.array([0.0, max(0.0, float(x @ target) / float(x @ x))]),
        np.array([0.0, 0.0]),
    ]
    return min(candidates, key=lambda p: np.linalg.norm(matrix @ p - target) ** 2)


def identify(
    mass: float, displacement: ArrayLike, velocity: ArrayLike,
    acceleration: ArrayLike, force: ArrayLike,
    nonnegative: bool = True,
) -> FitResult:
    """Fit c,k from F - m*a = c*v + k*x using least squares."""
    A, y = _observations(mass, displacement, velocity, acceleration, force)
    if nonnegative:
        parameters = _nonnegative_least_squares(A, y)
    else:
        parameters, *_ = np.linalg.lstsq(A, y, rcond=None)
    residuals = y - A @ parameters
    ss_res = float(residuals @ residuals)
    ss_total = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_total if ss_total > 0 else (1.0 if ss_res < 1e-20 else 0.0)
    return FitResult(
        damping_ns_m=float(parameters[0]), stiffness_n_m=float(parameters[1]),
        rmse_newtons=float(np.sqrt(np.mean(residuals**2))),
        r_squared=r2, samples=len(y), constrained=nonnegative,
    )


def bootstrap_intervals(
    mass: float, displacement: ArrayLike, velocity: ArrayLike,
    acceleration: ArrayLike, force: ArrayLike,
    resamples: int = 200, seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals from paired i.i.d. bootstrap resampling.

    For correlated time series, these intervals can understate uncertainty.
    """
    if resamples < 20:
        raise ValueError("Need at least 20 resamples")
    A, y = _observations(mass, displacement, velocity, acceleration, force)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        indices = rng.integers(0, len(y), len(y))
        if np.linalg.matrix_rank(A[indices]) < 2:
            continue
        estimates.append(_nonnegative_least_squares(A[indices], y[indices]))
    if len(estimates) < resamples // 2:
        raise ValueError("Too many rank-deficient bootstrap samples")
    intervals = np.percentile(np.asarray(estimates), [2.5, 97.5], axis=0)
    return {
        "damping_ns_m": (float(intervals[0, 0]), float(intervals[1, 0])),
        "stiffness_n_m": (float(intervals[0, 1]), float(intervals[1, 1])),
    }
