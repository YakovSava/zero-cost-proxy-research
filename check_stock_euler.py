import functools
import json

import torch

import rig
import samplers
import units

from comfy.k_diffusion import sampling as k_sampling

CASES: tuple[tuple[units.Unit, str, str], ...] = (
    (units.Unit("flat", 42), "8", "propagated"),
    (units.Unit("face", 42), "8", "propagated"),
    (units.Unit("flat", 42), "12", "angle_rate"),
)
EPSILONS: tuple[float, ...] = (1e-7, 1e-6)


def relative(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first - second).norm() / second.norm())


def main() -> None:
    path = rig.ROOT / "results" / "anima_sigmas.json"
    table = json.loads(path.read_text())
    rig.limit_gpu_memory()
    harness = rig.Rig()
    own = functools.partial(samplers.sample_fixed, order=1, disable=True)
    for unit, steps, name in CASES:
        sigmas = torch.tensor(table[steps][name], dtype=torch.float32)
        stock = harness.sample(
            unit.prompt, unit.seed, sigmas, k_sampling.sample_euler
        )
        mine = harness.sample(unit.prompt, unit.seed, sigmas, own)
        shifts = []
        for epsilon in EPSILONS:
            shifted = sigmas.clone()
            shifted[1:-2] = shifted[1:-2] * (1.0 + epsilon)
            result = harness.sample(
                unit.prompt, unit.seed, shifted, k_sampling.sample_euler
            )
            shifts.append(round(relative(result, stock), 3))
        print(
            unit.name,
            steps,
            name,
            "own vs stock:",
            round(relative(mine, stock), 3),
            "stock vs stock with sigmas shifted by",
            EPSILONS,
            shifts,
            flush=True,
        )


if __name__ == "__main__":
    main()
