import functools
import json

import numpy as np
import torch

import rig
import samplers
import units

STEPS: tuple[int, ...] = (8, 12, 16)
FIRST_STEPS: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02)
OUTPUT = rig.ROOT / "results" / "real_startfix.json"


def startfix_times(
    harness: rig.Rig,
    steps: int,
    first: float,
) -> np.ndarray:
    sigmas = harness.sigmas(steps - 1).numpy().astype(np.float64)
    shape = (1.0 - sigmas[:-1]) / (1.0 - sigmas[-2])
    end = 1.0 - units.SIGMA_MIN
    return np.concatenate([[0.0], first + (end - first) * shape])


def main() -> None:
    table = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    everything = units.all_units(list(rig.PROMPTS))
    references = {
        unit.name: np.load(units.reference_path(rig.DUMPS, unit))["final"]
        for unit in everything
    }
    rig.limit_gpu_memory()
    harness = rig.Rig()
    sampler = functools.partial(samplers.sample_fixed, order=1, disable=True)
    for unit in everything:
        for steps in STEPS:
            for first in FIRST_STEPS:
                key = f"{unit.name}|{steps}|{first:g}"
                if key in table:
                    continue
                times = startfix_times(harness, steps, first)
                sigmas = torch.tensor(
                    units.sigmas_from_times(times), dtype=torch.float32
                )
                result = harness.sample(
                    unit.prompt, unit.seed, sigmas, sampler
                )
                reference = references[unit.name]
                table[key] = float(
                    np.linalg.norm(result.numpy() - reference)
                    / np.linalg.norm(reference)
                )
        OUTPUT.write_text(json.dumps(table))
        print(unit.name, flush=True)


if __name__ == "__main__":
    main()
