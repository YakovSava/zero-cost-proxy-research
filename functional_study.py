import json
from typing import Any

import numpy as np

import grids
import profiles
import rig
import units

STEPS: tuple[int, ...] = (8, 12, 16, 24)
FINE: int = 4001
NORMAL_SHIFT: float = 3.0


def interpolate(
    times: np.ndarray,
    values: np.ndarray,
    fine: np.ndarray,
) -> np.ndarray:
    return np.interp(fine, times, values)


def smooth_cost(
    grid: np.ndarray,
    fine: np.ndarray,
    monitor: np.ndarray,
) -> float:
    total = 0.0
    for start, stop in zip(grid[:-1], grid[1:]):
        inside = (fine >= start) & (fine <= stop)
        mean = monitor[inside].mean() if inside.any() else 0.0
        total += 0.5 * (stop - start) ** 2 * mean
    return total


def rough_cost(
    grid: np.ndarray,
    fine: np.ndarray,
    jitter: np.ndarray,
) -> float:
    total = 0.0
    for start, stop in zip(grid[:-1], grid[1:]):
        inside = (fine >= start) & (fine <= stop)
        mean = (jitter[inside] ** 2).mean() if inside.any() else 0.0
        total += (stop - start) ** 2 * mean
    return total


def normal_times(steps: int) -> np.ndarray:
    sigma = np.linspace(1.0, 0.0, steps + 1)
    warped = NORMAL_SHIFT * sigma / (1.0 + (NORMAL_SHIFT - 1.0) * sigma)
    inner = 1.0 - warped
    span = 1.0 - units.SIGMA_MIN
    return inner[:steps] / inner[steps - 1] * span


def candidate_grids(
    own: profiles.Profile,
    others: list[profiles.Profile],
    steps: int,
) -> dict[str, np.ndarray]:
    return {
        "normal_like": normal_times(steps),
        "uniform": units.time_grid(steps),
        "static_acc": profiles.build_grid(
            own.times, profiles.static_density(others, 1, 0.5), steps
        ),
        "oracle_acc": profiles.build_grid(
            own.times, profiles.density(own, 1, 0.5), steps
        ),
        "static_rough": profiles.build_grid(
            own.times, profiles.static_roughness_density(others, 1.0), steps
        ),
        "oracle_rough": profiles.build_grid(
            own.times, profiles.roughness_density(own, 1.0), steps
        ),
    }


def main() -> None:
    everything = units.all_units(list(rig.PROMPTS))
    loaded = {unit.name: profiles.load_profile(unit) for unit in everything}
    report: dict[str, Any] = {}
    for steps in STEPS:
        costs: dict[str, dict[str, list[float]]] = {}
        for unit in everything:
            own = loaded[unit.name]
            others = [
                value for key, value in loaded.items() if key != unit.name
            ]
            fine = np.linspace(own.times[0], own.times[-1], FINE)
            monitor = interpolate(own.times, own.derivatives[:, 0], fine)
            jitter = interpolate(
                own.times, own.roughness * own.speed, fine
            )
            for name, grid in candidate_grids(own, others, steps).items():
                cell = costs.setdefault(name, {"smooth": [], "rough": []})
                cell["smooth"].append(smooth_cost(grid, fine, monitor))
                cell["rough"].append(rough_cost(grid, fine, jitter))
        base = costs["static_acc"]
        report[str(steps)] = {
            name: {
                kind: float(
                    np.mean(np.log(np.array(values) / np.array(base[kind])))
                )
                for kind, values in cell.items()
            }
            for name, cell in costs.items()
        }
    path = rig.ROOT / "results" / "functional.json"
    path.write_text(json.dumps(report, indent=2))
    for steps, row in report.items():
        print(f"N={steps}  mean log-cost relative to static_acc")
        for name, cell in row.items():
            print(
                f"   {name:13s} smooth {cell['smooth']:+.3f}"
                f"   rough {cell['rough']:+.3f}"
            )


if __name__ == "__main__":
    main()
