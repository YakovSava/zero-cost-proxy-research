import dataclasses

import numpy as np

import grids
import monitor
import rig
import units

HALF_WINDOW: int = 4
DEGREE: int = 3


@dataclasses.dataclass(frozen=True)
class Profile:
    times: np.ndarray
    speed: np.ndarray
    derivatives: np.ndarray
    roughness: np.ndarray
    final: np.ndarray


def load_profile(unit: units.Unit) -> Profile:
    stored = np.load(units.reference_path(rig.DUMPS, unit))
    times: np.ndarray = stored["times"][stored["indices"]]
    values: np.ndarray = stored["velocities"].reshape(len(times), -1)
    values = values.astype(np.float64)
    speed: np.ndarray = np.linalg.norm(values, axis=1)
    derivatives, jitter = monitor.local_fit_statistics(
        times, values, HALF_WINDOW, DEGREE
    )
    return Profile(
        times=times,
        speed=speed,
        derivatives=derivatives,
        roughness=jitter / speed,
        final=stored["final"],
    )


def density(
    profile: Profile,
    derivative: int,
    exponent: float,
) -> np.ndarray:
    selected: np.ndarray = profile.derivatives[:, derivative - 1]
    return np.maximum(selected, 1e-12) ** exponent


def static_density(
    others: list[Profile],
    derivative: int,
    exponent: float,
) -> np.ndarray:
    mean: np.ndarray = np.mean(
        [item.derivatives[:, derivative - 1] for item in others], axis=0
    )
    return np.maximum(mean, 1e-12) ** exponent


def build_grid(
    times: np.ndarray,
    weight: np.ndarray,
    steps: int,
) -> np.ndarray:
    return grids.equidistribute(times, weight, steps - 1)


def roughness_density(profile: Profile, exponent: float) -> np.ndarray:
    return np.maximum(profile.roughness, 1e-12) ** exponent


def static_roughness_density(
    others: list[Profile],
    exponent: float,
) -> np.ndarray:
    mean: np.ndarray = np.mean([item.roughness for item in others], axis=0)
    return np.maximum(mean, 1e-12) ** exponent
