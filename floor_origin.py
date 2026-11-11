import functools
import json
from typing import Any

import numpy as np
import torch

import floor
import rig
import samplers
import units

PERTURBATIONS: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2)
SIGMA_PROBE: float = 0.6
UNIT: units.Unit = units.Unit("face", 42)
LAGS: np.ndarray = np.array([1, 2, 3, 4, 6, 8, 12, 16])
CFG_STEPS: int = 96


def angle_degrees(first: torch.Tensor, second: torch.Tensor) -> float:
    one = first.flatten().cpu().numpy()[None]
    two = second.flatten().cpu().numpy()[None]
    return float(np.degrees(floor.angle_between(one, two)[0]))


@torch.no_grad()
def probe_sampler(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: bool | None = None,
    report: dict[str, Any] | None = None,
) -> torch.Tensor:
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for index in range(len(sigmas) - 1):
        sigma = float(sigmas[index])
        denoised = model(x, sigmas[index] * s_in, **extra_args)
        velocity = samplers.velocity_from(x, denoised, sigma)
        if sigma <= SIGMA_PROBE:
            break
        x = x + (float(sigmas[index + 1]) - sigma) * velocity

    def evaluate(state: torch.Tensor, level: float) -> torch.Tensor:
        sigma_in = torch.tensor([level], device=x.device)
        value = model(state, sigma_in, **extra_args)
        return samplers.velocity_from(state, value, level)

    repeat = evaluate(x, sigma)
    report["sigma"] = sigma
    report["repeat_angle"] = angle_degrees(repeat, velocity)
    generator = torch.Generator(device=x.device).manual_seed(0)
    direction = torch.randn(x.shape, generator=generator, device=x.device)
    direction = direction / direction.norm()
    state_angles = {}
    path_angles = {}
    time_angles = {}
    for delta in PERTURBATIONS:
        shifted = x + delta * x.norm() * direction
        state_angles[delta] = angle_degrees(evaluate(shifted, sigma), velocity)
        along = x - delta * velocity
        later = sigma - delta
        path_angles[delta] = angle_degrees(evaluate(along, later), velocity)
        time_angles[delta] = angle_degrees(evaluate(x, later), velocity)
    report["state_random"] = state_angles
    report["along_path"] = path_angles
    report["time_only"] = time_angles
    return x


def lag_floor(harness: rig.Rig, cfg: float) -> dict[str, float]:
    times = units.time_grid(CFG_STEPS)
    sigmas = torch.tensor(units.sigmas_from_times(times), dtype=torch.float32)
    trace = samplers.Trace()
    sampler = functools.partial(
        samplers.sample_fixed,
        order=2,
        trace=trace,
        keep_velocities=True,
        disable=True,
    )
    harness.sample(UNIT.prompt, UNIT.seed, sigmas, sampler, cfg=cfg)
    values = np.stack(
        [item.numpy().reshape(-1) for item in trace.velocities[:-1]]
    ).astype(np.float64)
    result = {}
    for name, (first, last) in {"early": (4, 16), "middle": (30, 50)}.items():
        fit = floor.fit_floor(values, LAGS, first, last)
        result[name] = {
            "rate_degrees_per_step": fit.rate_degrees,
            "floor_degrees": fit.floor_degrees,
            "r_squared": fit.r_squared,
        }
    return result


def main() -> None:
    rig.limit_gpu_memory()
    harness = rig.Rig()
    report: dict[str, Any] = {}
    sigmas = harness.sigmas(64)
    sampler = functools.partial(probe_sampler, report=report)
    harness.sample(UNIT.prompt, UNIT.seed, sigmas, sampler)
    report["lag_fit"] = {
        f"cfg{cfg:g}": lag_floor(harness, cfg) for cfg in (5.0, 1.0)
    }
    path = rig.ROOT / "results" / "floor_origin.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
