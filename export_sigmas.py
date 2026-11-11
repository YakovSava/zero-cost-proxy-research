import json

import numpy as np

import profiles
import rig
import run_replica
import units

STEPS: tuple[int, ...] = (8, 12, 16, 24)
OUTPUT = rig.ROOT / "results" / "anima_sigmas.json"


def sigma_lists(
    everything: list[profiles.Profile],
    steps: int,
) -> dict[str, list[float]]:
    reference = everything[0]
    rate = run_replica.static_turning_density(everything, 1.0)
    weighted = run_replica.load_propagation(reference.times) * rate
    result: dict[str, list[float]] = {}
    for name, weight in (("angle_rate", rate), ("propagated", weighted)):
        times = profiles.build_grid(reference.times, weight, steps)
        sigmas = units.sigmas_from_times(times)
        result[name] = [round(float(value), 5) for value in sigmas]
    return result


def main() -> None:
    loaded = [
        profiles.load_profile(unit)
        for unit in units.all_units(list(rig.PROMPTS))
    ]
    table = {str(steps): sigma_lists(loaded, steps) for steps in STEPS}
    OUTPUT.write_text(json.dumps(table, indent=1))
    for steps, grids in table.items():
        for name, values in grids.items():
            print(steps, name, values)
    print(np.round(np.diff(table["8"]["propagated"]), 3))


if __name__ == "__main__":
    main()
