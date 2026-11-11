import argparse
import functools
import json
import pathlib
import time

import numpy as np
import torch

import profiles
import rig
import samplers
import units

RESULTS: pathlib.Path = rig.ROOT / "results"
LATENTS: pathlib.Path = rig.DUMPS / "latents"
OUTPUT: pathlib.Path = RESULTS / "real_grids.json"
GRID_NAMES: tuple[str, ...] = (
    "normal",
    "uniform",
    "static_acc",
    "static_jerk",
    "oracle_acc",
    "static_rough",
    "oracle_rough",
)
ACCELERATION_EXPONENT: float = 0.5
JERK_EXPONENT: float = 1.0 / 3.0
ROUGHNESS_EXPONENT: float = 1.0


def candidate_times(
    name: str,
    steps: int,
    own: profiles.Profile,
    others: list[profiles.Profile],
) -> np.ndarray | None:
    if name == "uniform":
        return units.time_grid(steps)
    weights = {
        "static_acc": lambda: profiles.static_density(
            others, 1, ACCELERATION_EXPONENT
        ),
        "static_jerk": lambda: profiles.static_density(
            others, 2, JERK_EXPONENT
        ),
        "oracle_acc": lambda: profiles.density(
            own, 1, ACCELERATION_EXPONENT
        ),
        "static_rough": lambda: profiles.static_roughness_density(
            others, ROUGHNESS_EXPONENT
        ),
        "oracle_rough": lambda: profiles.roughness_density(
            own, ROUGHNESS_EXPONENT
        ),
    }
    if name in weights:
        return profiles.build_grid(own.times, weights[name](), steps)
    return None


def run(
    harness: rig.Rig,
    unit: units.Unit,
    name: str,
    steps: int,
    order: int,
    own: profiles.Profile,
    others: list[profiles.Profile],
) -> torch.Tensor:
    times = candidate_times(name, steps, own, others)
    if times is None:
        sigmas = harness.sigmas(steps)
    else:
        sigmas = torch.tensor(
            units.sigmas_from_times(times), dtype=torch.float32
        )
    sampler = functools.partial(
        samplers.sample_fixed, order=order, disable=True
    )
    return harness.sample(unit.prompt, unit.seed, sigmas, sampler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[8, 12, 16, 24]
    )
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--prompts", nargs="+", default=list(rig.PROMPTS))
    arguments = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    LATENTS.mkdir(parents=True, exist_ok=True)
    table: dict[str, float] = (
        json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    )
    everything = units.all_units(arguments.prompts)
    loaded = {unit.name: profiles.load_profile(unit) for unit in everything}
    rig.limit_gpu_memory()
    harness = rig.Rig()
    for unit in everything:
        own = loaded[unit.name]
        others = [
            value for key, value in loaded.items() if key != unit.name
        ]
        started = time.time()
        for steps in arguments.steps:
            for order in arguments.orders:
                for name in GRID_NAMES:
                    key = f"{unit.name}|{order}|{name}|{steps}"
                    if key in table:
                        continue
                    result = run(
                        harness, unit, name, steps, order, own, others
                    )
                    difference = (result.numpy() - own.final).ravel()
                    table[key] = float(
                        np.linalg.norm(difference)
                        / np.linalg.norm(own.final)
                    )
                    np.save(
                        LATENTS / f"{key.replace('|', '_')}.npy",
                        result.numpy().astype(np.float16),
                    )
        OUTPUT.write_text(json.dumps(table, indent=1))
        print(f"{unit.name} {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
