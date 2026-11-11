import argparse
import functools
import json
from typing import Any

import numpy as np
import torch

import rig
import samplers
import units

UNITS: tuple[units.Unit, ...] = (
    units.Unit("flat", 42),
    units.Unit("face", 42),
)
POINTS: int = 96
FORKS: tuple[int, ...] = (3, 8, 16, 28, 44, 60, 76, 88)
FORKS_USED: list[int] = list(FORKS)
SIZES: list[float] = [3e-3]
EXTRA_SIZES: dict[int, tuple[float, ...]] = {28: (1e-3, 3e-2)}


@torch.no_grad()
def sensitivity_sampler(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Any = None,
    disable: bool | None = None,
    report: dict[str, Any] | None = None,
) -> torch.Tensor:
    s_in = x.new_ones([x.shape[0]])
    generator = torch.Generator(device=x.device).manual_seed(0)
    states: dict[int, torch.Tensor] = {}
    previous = None
    schedule = [float(value) for value in sigmas.cpu()]
    for index in range(len(schedule) - 2):
        if index in FORKS_USED:
            states[index] = x.clone()
        sigma = schedule[index]
        denoised = model(x, sigmas[index] * s_in, **(extra_args or {}))
        velocity = samplers.velocity_from(x, denoised, sigma)
        step = schedule[index + 1] - sigma
        if previous is None:
            x = x + step * velocity
        else:
            current, past = samplers.ab2_weights(
                step, sigma - schedule[index - 1]
            )
            x = x + current * velocity + past * previous
        previous = velocity
    continuation = functools.partial(
        samplers.integrate, model, sigmas=sigmas, extra_args=extra_args,
        order=2,
    )
    for fork, state in states.items():
        base = continuation(state, first=fork)
        sizes = list(EXTRA_SIZES.get(fork, ())) + SIZES
        row = {}
        for size in sizes:
            direction = torch.randn(
                state.shape, generator=generator, device=state.device
            )
            direction = direction / direction.norm()
            kick = size * state.norm() * direction
            moved = continuation(state + kick, first=fork)
            row[f"{size:g}"] = float((moved - base).norm() / kick.norm())
        report[str(fork)] = {
            "sigma": schedule[fork],
            "gain": row,
        }
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=float, default=SIZES)
    parser.add_argument("--forks", nargs="+", type=int, default=FORKS)
    parser.add_argument("--output", default="sensitivity.json")
    arguments = parser.parse_args()
    SIZES[:] = arguments.sizes
    FORKS_USED[:] = arguments.forks
    if arguments.output != "sensitivity.json":
        EXTRA_SIZES.clear()
    rig.limit_gpu_memory()
    harness = rig.Rig()
    times = units.time_grid(POINTS)
    sigmas = torch.tensor(units.sigmas_from_times(times), dtype=torch.float32)
    result: dict[str, Any] = {"times": times[FORKS_USED].tolist()}
    for unit in UNITS:
        report: dict[str, Any] = {}
        sampler = functools.partial(sensitivity_sampler, report=report)
        harness.sample(unit.prompt, unit.seed, sigmas, sampler)
        result[unit.name] = report
        print(unit.name, json.dumps(report), flush=True)
    path = rig.ROOT / "results" / arguments.output
    path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
