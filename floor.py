import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class FloorFit:
    rate_degrees: float
    floor_degrees: float
    r_squared: float


def angle_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    unit_first = first / np.linalg.norm(first, axis=1, keepdims=True)
    unit_second = second / np.linalg.norm(second, axis=1, keepdims=True)
    chord = np.linalg.norm(unit_first - unit_second, axis=1)
    return 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))


def median_lag_angles(
    velocities: np.ndarray,
    lags: np.ndarray,
    first: int,
    last: int,
) -> np.ndarray:
    anchors = velocities[first:last]
    return np.array(
        [
            np.median(
                angle_between(anchors, velocities[first + lag:last + lag])
            )
            for lag in lags
        ]
    )


def fit_floor(
    velocities: np.ndarray,
    lags: np.ndarray,
    first: int,
    last: int,
) -> FloorFit:
    angles = median_lag_angles(velocities, lags, first, last)
    design = np.stack([lags.astype(float) ** 2, np.ones(len(lags))], axis=1)
    target = angles**2
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    total = target - target.mean()
    return FloorFit(
        rate_degrees=float(np.degrees(np.sqrt(max(coefficients[0], 0.0)))),
        floor_degrees=float(np.degrees(np.sqrt(max(coefficients[1], 0.0)))),
        r_squared=float(1.0 - residual @ residual / (total @ total)),
    )
