import dataclasses
import pathlib

import numpy as np

SIGMA_MIN: float = 0.0089
SEEDS: tuple[int, ...] = (42, 1337, 7)
REFERENCE_POINTS: int = 192
STORED_STRIDE: int = 2


@dataclasses.dataclass(frozen=True)
class Unit:
    prompt: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.prompt}_s{self.seed}"


def all_units(prompts: list[str]) -> list[Unit]:
    return [Unit(prompt, seed) for seed in SEEDS for prompt in prompts]


def time_grid(points: int) -> np.ndarray:
    return np.linspace(0.0, 1.0 - SIGMA_MIN, points)


def sigmas_from_times(times: np.ndarray) -> np.ndarray:
    return np.concatenate([1.0 - times, [0.0]])


def stored_indices(points: int = REFERENCE_POINTS) -> np.ndarray:
    indices = np.arange(0, points, STORED_STRIDE)
    if indices[-1] != points - 1:
        indices = np.append(indices, points - 1)
    return indices


def reference_path(root: pathlib.Path, unit: Unit) -> pathlib.Path:
    return root / "reference" / f"{unit.name}.npz"
