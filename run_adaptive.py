import functools
import json
import math

import numpy as np
import torch

import controllers
import rig
import samplers
import units

SETTINGS: tuple[tuple[float, float, float], ...] = tuple(
    (gamma, degrees, initial)
    for initial, gammas in ((0.02, (1.0, 0.5)), (0.01, (1.0,)))
    for gamma in gammas
    for degrees in (5.0, 6.0, 8.0, 11.0, 15.0)
)
OUTPUT = rig.ROOT / "results" / "real_adaptive.json"


def main() -> None:
    table = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    table = {
        (key if key.count("|") == 3 else f"{key}|0.06"): value
        for key, value in table.items()
    }
    everything = units.all_units(list(rig.PROMPTS))
    references = {
        unit.name: np.load(units.reference_path(rig.DUMPS, unit))["final"]
        for unit in everything
    }
    rig.limit_gpu_memory()
    harness = rig.Rig()
    sigmas = torch.tensor(
        [1.0, units.SIGMA_MIN, 0.0], dtype=torch.float32
    )
    for unit in everything:
        for gamma, degrees, initial in SETTINGS:
            key = f"{unit.name}|{gamma:g}|{degrees:g}|{initial:g}"
            if key in table:
                continue
            controller = controllers.AngleController(
                target=math.radians(degrees), gamma=gamma, initial_step=initial
            )
            report = {}
            sampler = functools.partial(
                samplers.sample_adaptive, controller=controller, report=report
            )
            result = harness.sample(unit.prompt, unit.seed, sigmas, sampler)
            reference = references[unit.name]
            error = np.linalg.norm(result.numpy() - reference) / (
                np.linalg.norm(reference)
            )
            table[key] = {
                "nfe": len(report["times"]),
                "error": float(error),
                "times": report["times"],
            }
        OUTPUT.write_text(json.dumps(table))
        print(unit.name, flush=True)


if __name__ == "__main__":
    main()
