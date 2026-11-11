import argparse
import functools
import time

import numpy as np
import torch

import rig
import samplers
import units


def dump_unit(harness: rig.Rig, unit: units.Unit) -> None:
    path = units.reference_path(rig.DUMPS, unit)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    times = units.time_grid(units.REFERENCE_POINTS)
    sigmas = torch.tensor(units.sigmas_from_times(times), dtype=torch.float32)
    trace = samplers.Trace()
    sampler = functools.partial(
        samplers.sample_fixed,
        order=2,
        trace=trace,
        keep_velocities=True,
        disable=True,
    )
    started = time.time()
    final = harness.sample(unit.prompt, unit.seed, sigmas, sampler)
    indices = units.stored_indices()
    stored = np.stack(
        [trace.velocities[i].numpy().astype(np.float32) for i in indices]
    )
    np.savez(
        path,
        times=times,
        indices=indices,
        velocities=stored,
        final=final.numpy().astype(np.float32),
    )
    print(f"{unit.name} {time.time() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="+", default=list(rig.PROMPTS))
    arguments = parser.parse_args()
    rig.limit_gpu_memory()
    harness = rig.Rig()
    for unit in units.all_units(arguments.prompts):
        dump_unit(harness, unit)


if __name__ == "__main__":
    main()
