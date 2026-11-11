import numpy as np


def uniform_grid(steps: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, steps + 1)


def shifted_grid(steps: int, shift: float) -> np.ndarray:
    sigma: np.ndarray = np.linspace(1.0, 0.0, steps + 1)
    warped: np.ndarray = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    return 1.0 - warped


def optimal_density(
    derivative_norm: np.ndarray,
    order: int,
    floor: float = 1e-12,
) -> np.ndarray:
    return np.maximum(derivative_norm, floor) ** (1.0 / (order + 1))


def cumulative(times: np.ndarray, density: np.ndarray) -> np.ndarray:
    increments: np.ndarray = 0.5 * (density[1:] + density[:-1])
    increments = increments * np.diff(times)
    return np.concatenate(([0.0], np.cumsum(increments)))


def equidistribute(
    times: np.ndarray,
    density: np.ndarray,
    steps: int,
) -> np.ndarray:
    mass: np.ndarray = cumulative(times, density)
    targets: np.ndarray = np.linspace(0.0, mass[-1], steps + 1)
    return np.interp(targets, mass, times)


def mixed_density(
    density: np.ndarray,
    exponent: float,
    floor_share: float,
) -> np.ndarray:
    powered: np.ndarray = density**exponent
    return powered / powered.mean() + floor_share
