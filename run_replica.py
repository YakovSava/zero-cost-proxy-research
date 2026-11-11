import functools
import json
from typing import Any

import numpy as np
import torch

import profiles
import rig
import samplers
import units

STEPS: tuple[int, ...] = (8, 12, 16, 24)
EXPONENTS: tuple[float, ...] = (0.75, 1.0, 1.5)
BORROWED: tuple[tuple[str, str, str], ...] = (
    ("1", "11", "0.01"),
    ("1", "8", "0.01"),
    ("1", "15", "0.01"),
)
OUTPUT = rig.ROOT / "results" / "real_replica.json"
PROPAGATION_SIZE: str = "0.03"
PROPAGATION_EXPONENTS: tuple[float, ...] = (0.5, 1.0)


def load_propagation(times: np.ndarray) -> np.ndarray:
    report = json.loads(
        (rig.ROOT / "results" / "sensitivity_large.json").read_text()
    )
    fork_times = np.array(report["times"])
    gains = np.mean(
        [
            [
                row["gain"][PROPAGATION_SIZE]
                for row in report[unit].values()
            ]
            for unit in report
            if unit != "times"
        ],
        axis=0,
    )
    return np.interp(times, fork_times, gains)


def turning_density(
    profile: profiles.Profile,
    exponent: float,
) -> np.ndarray:
    rate = profile.derivatives[:, 0] / profile.speed
    return np.maximum(rate, 1e-12) ** exponent


def static_turning_density(
    others: list[profiles.Profile],
    exponent: float,
) -> np.ndarray:
    rate = np.mean(
        [item.derivatives[:, 0] / item.speed for item in others], axis=0
    )
    return np.maximum(rate, 1e-12) ** exponent


def run_times(
    harness: rig.Rig,
    unit: units.Unit,
    times: np.ndarray,
    reference: np.ndarray,
) -> float:
    sigmas = torch.tensor(units.sigmas_from_times(times), dtype=torch.float32)
    sampler = functools.partial(samplers.sample_fixed, order=1, disable=True)
    result = harness.sample(unit.prompt, unit.seed, sigmas, sampler)
    return float(
        np.linalg.norm(result.numpy() - reference) / np.linalg.norm(reference)
    )


def main() -> None:
    table: dict[str, Any] = (
        json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    )
    everything = units.all_units(list(rig.PROMPTS))
    loaded = {unit.name: profiles.load_profile(unit) for unit in everything}
    adaptive = json.loads(
        (rig.ROOT / "results" / "real_adaptive.json").read_text()
    )
    rig.limit_gpu_memory()
    harness = rig.Rig()
    for index, unit in enumerate(everything):
        own = loaded[unit.name]
        others = [
            value for key, value in loaded.items() if key != unit.name
        ]
        reference = own.final
        for exponent in EXPONENTS:
            for steps in STEPS:
                key = f"{unit.name}|static{exponent:g}|{steps}"
                if key not in table:
                    weight = static_turning_density(others, exponent)
                    times = profiles.build_grid(own.times, weight, steps)
                    table[key] = run_times(harness, unit, times, reference)
        for exponent in PROPAGATION_EXPONENTS:
            for steps in STEPS:
                key = f"{unit.name}|propagated{exponent:g}|{steps}"
                if key not in table:
                    rate = static_turning_density(others, 1.0)
                    weight = (load_propagation(own.times) * rate) ** exponent
                    times = profiles.build_grid(own.times, weight, steps)
                    table[key] = run_times(harness, unit, times, reference)
        for steps in STEPS:
            key = f"{unit.name}|oracle1|{steps}"
            if key not in table:
                weight = turning_density(own, 1.0)
                times = profiles.build_grid(own.times, weight, steps)
                table[key] = run_times(harness, unit, times, reference)
        donor = everything[(index + 1) % len(everything)]
        for gamma, degrees, initial in BORROWED:
            setting = f"{gamma}|{degrees}|{initial}"
            key = f"{unit.name}|borrowed|{setting}"
            if key in table:
                continue
            times = np.array(adaptive[f"{donor.name}|{setting}"]["times"])
            table[key] = run_times(harness, unit, times, reference)
            table[f"{key}|donor_nfe"] = len(times)
        OUTPUT.write_text(json.dumps(table))
        print(unit.name, flush=True)


if __name__ == "__main__":
    main()
