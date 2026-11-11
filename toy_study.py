import argparse
import json
import pathlib
from typing import Any

import numpy as np
import scipy.stats
import torch

import gaussian_mixture
import grids
import integrators
import monitor
import proxies

RESULTS: pathlib.Path = pathlib.Path(__file__).resolve().parent / "results"
FINE_POINTS: int = 513
CONFIGS: dict[str, dict[str, Any]] = {
    "d64_m4": dict(dimension=64, modes=4, separation=1.5, scale=0.5),
    "d256_m8": dict(dimension=256, modes=8, separation=1.2, scale=0.4),
}


def build(name: str, samples: int, seed: int) -> tuple[Any, torch.Tensor]:
    flow = gaussian_mixture.random_mixture(seed=seed, **CONFIGS[name])
    return flow, flow.sample_noise(samples, seed=seed + 1)


def fine_reference(flow: Any, noise: torch.Tensor) -> integrators.Trajectory:
    grid: np.ndarray = grids.uniform_grid(FINE_POINTS - 1)
    return integrators.reference_trajectory(
        flow.velocity, noise, grid, substeps=4
    )


def proxy_table(
    flow: Any,
    reference: integrators.Trajectory,
    steps: int,
) -> dict[str, np.ndarray]:
    grid: np.ndarray = grids.uniform_grid(steps)
    stride: int = (FINE_POINTS - 1) // steps
    index: np.ndarray = np.arange(0, FINE_POINTS, stride)
    states: list[torch.Tensor] = [reference.states[i] for i in index]
    velocities: list[torch.Tensor] = [reference.velocities[i] for i in index]
    step: float = float(grid[1] - grid[0])
    truth: torch.Tensor = integrators.euler_local_errors(
        flow.velocity, states, grid, substeps=16
    )
    rows: dict[str, list[torch.Tensor]] = {
        "rotation": [],
        "turning": [],
        "stretch": [],
        "acceleration": [],
        "curvature": [],
    }
    for newer, older in zip(velocities[1:-1], velocities[:-2]):
        value = proxies.increment(newer, older)
        rows["rotation"].append(proxies.rotation_rate(value, step))
        rows["turning"].append(proxies.turning_rate(value, step))
        rows["stretch"].append(proxies.stretch_rate(value, step))
        rows["acceleration"].append(proxies.acceleration(value, step))
        rows["curvature"].append(proxies.curvature(value, step))
    table: dict[str, np.ndarray] = {
        key: torch.stack(values, dim=1).numpy()
        for key, values in rows.items()
    }
    table["truth"] = (truth[:, 1:] / step**2).numpy()
    return table


def correlation_summary(table: dict[str, np.ndarray]) -> dict[str, Any]:
    truth: np.ndarray = np.log(table["truth"] + 1e-300)
    summary: dict[str, Any] = {}
    for name, values in table.items():
        if name == "truth":
            continue
        log_values: np.ndarray = np.log(values + 1e-300)
        pooled: float = float(
            scipy.stats.pearsonr(log_values.ravel(), truth.ravel())[0]
        )
        profile: float = float(
            scipy.stats.spearmanr(
                log_values.mean(axis=0), truth.mean(axis=0)
            )[0]
        )
        residual_x: np.ndarray = log_values - log_values.mean(axis=0)
        residual_y: np.ndarray = truth - truth.mean(axis=0)
        within: float = float(
            scipy.stats.pearsonr(residual_x.ravel(), residual_y.ravel())[0]
        )
        summary[name] = {
            "pooled_log": pooled,
            "profile_spearman": profile,
            "within_time_log": within,
        }
    return summary


def run_proxy_study(name: str, samples: int, seed: int) -> dict[str, Any]:
    flow, noise = build(name, samples, seed)
    reference = fine_reference(flow, noise)
    result: dict[str, Any] = {}
    for steps in (16, 32):
        table = proxy_table(flow, reference, steps)
        result[str(steps)] = correlation_summary(table)
    return result


def grid_for_density(
    reference: integrators.Trajectory,
    density: np.ndarray,
    steps: int,
) -> np.ndarray:
    return grids.equidistribute(reference.grid, density, steps)


def error_with_grid(
    flow: Any,
    state: torch.Tensor,
    grid: np.ndarray,
    target: torch.Tensor,
    order: int,
) -> float:
    return float(
        integrators.final_error(flow.velocity, state, grid, target, order)
    )


def monitor_density(
    norms: np.ndarray,
    order: int,
    exponent: float | None,
) -> np.ndarray:
    selected: np.ndarray = norms[..., order - 1]
    power: float = 1.0 / (order + 1) if exponent is None else exponent
    return np.maximum(selected, 1e-12) ** power


def run_grid_study(name: str, samples: int, seed: int) -> dict[str, Any]:
    flow, noise = build(name, samples, seed)
    reference = fine_reference(flow, noise)
    norms: np.ndarray = monitor.derivative_norms(reference, 2)
    targets: torch.Tensor = reference.states[-1]
    result: dict[str, Any] = {}
    for order in (1, 2):
        for steps in (8, 12, 16, 24):
            candidates: dict[str, list[float]] = {}
            static: np.ndarray = norms[..., order - 1].mean(axis=1)
            for label, grid in {
                "uniform": grids.uniform_grid(steps),
                "shift1.5": grids.shifted_grid(steps, 1.5),
                "shift3": grids.shifted_grid(steps, 3.0),
                "static_opt": grid_for_density(
                    reference,
                    np.maximum(static, 1e-12) ** (1.0 / (order + 1)),
                    steps,
                ),
                "static_half": grid_for_density(
                    reference,
                    np.maximum(static, 1e-12) ** 0.5,
                    steps,
                ),
                "static_one": grid_for_density(
                    reference,
                    np.maximum(static, 1e-12),
                    steps,
                ),
            }.items():
                candidates[label] = [
                    error_with_grid(
                        flow, noise[[i]], grid, targets[[i]], order
                    )
                    for i in range(samples)
                ]
            for label, exponent in {
                "oracle_opt": None,
                "oracle_half": 0.5,
                "oracle_one": 1.0,
            }.items():
                errors: list[float] = []
                for i in range(samples):
                    density = monitor_density(
                        norms[:, [i]], order, exponent
                    )[:, 0]
                    grid = grid_for_density(reference, density, steps)
                    errors.append(
                        error_with_grid(
                            flow,
                            noise[[i]],
                            grid,
                            targets[[i]],
                            order,
                        )
                    )
                candidates[label] = errors
            result[f"order{order}_n{steps}"] = {
                key: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                }
                for key, values in candidates.items()
            }
            result[f"order{order}_n{steps}"]["paired_oracle_vs_static"] = (
                float(
                    np.mean(
                        np.log(
                            np.array(candidates["oracle_opt"])
                            / np.array(candidates["static_opt"])
                        )
                    )
                )
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="d64_m4")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study", choices=["proxy", "grid"], default="proxy")
    arguments = parser.parse_args()
    runner = run_proxy_study if arguments.study == "proxy" else run_grid_study
    result = runner(arguments.config, arguments.samples, arguments.seed)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"toy_{arguments.study}_{arguments.config}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
