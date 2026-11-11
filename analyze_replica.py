import json
import pathlib
from typing import Any

import numpy as np

import analyze_grids

RESULTS: pathlib.Path = pathlib.Path(__file__).resolve().parent / "results"
STEPS: tuple[int, ...] = (8, 12, 16, 24)
FAMILIES: tuple[str, ...] = (
    "normal",
    "static_acc",
    "static0.75",
    "static1",
    "static1.5",
    "propagated0.5",
    "propagated1",
    "oracle1",
)
SETTINGS: tuple[str, ...] = (
    "1|15|0.01",
    "1|11|0.01",
    "1|8|0.01",
    "1|11|0.02",
)


def family_errors(
    unit: str,
    family: str,
    grids: dict[tuple[str, int, str, int], float],
    replica: dict[str, Any],
) -> dict[int, float]:
    if family in ("normal", "static_acc"):
        return {n: grids[(unit, 1, family, n)] for n in STEPS}
    return {n: replica[f"{unit}|{family}|{n}"] for n in STEPS}


def curve(errors: dict[int, float], evaluations: float) -> float:
    log_steps = np.log(np.array(STEPS, dtype=float))
    log_errors = np.log([errors[n] for n in STEPS])
    position = np.log(evaluations)
    if position <= log_steps[0]:
        slope = (log_errors[1] - log_errors[0]) / (log_steps[1] - log_steps[0])
        return float(np.exp(log_errors[0] + slope * (position - log_steps[0])))
    if position >= log_steps[-1]:
        slope = (log_errors[-1] - log_errors[-2]) / (
            log_steps[-1] - log_steps[-2]
        )
        return float(
            np.exp(log_errors[-1] + slope * (position - log_steps[-1]))
        )
    return float(np.exp(np.interp(position, log_steps, log_errors)))


def format_interval(values: np.ndarray) -> str:
    mean, low, high = analyze_grids.interval(values)
    return f"{mean:+.2f}[{low:+.2f},{high:+.2f}]"


def main() -> None:
    grids = analyze_grids.parse(
        json.loads((RESULTS / "real_grids.json").read_text())
    )
    replica = json.loads((RESULTS / "real_replica.json").read_text())
    adaptive = json.loads((RESULTS / "real_adaptive.json").read_text())
    unit_names = sorted(
        {key.split("|")[0] for key in replica if "|oracle1|" in key}
    )
    print(f"units complete: {len(unit_names)}")
    print("\nstatic families vs normal (log-ratio, same N)")
    for steps in STEPS:
        cells = []
        for family in FAMILIES[1:]:
            ratios = np.array(
                [
                    np.log(
                        family_errors(u, family, grids, replica)[steps]
                        / grids[(u, 1, "normal", steps)]
                    )
                    for u in unit_names
                ]
            )
            mean_error = np.mean(
                [
                    family_errors(u, family, grids, replica)[steps]
                    for u in unit_names
                ]
            )
            cells.append(
                f"{family} {mean_error:.3f} {format_interval(ratios)}"
            )
        print(f"N={steps}: " + " | ".join(cells))
    print("\nborrowed vs own controller grid")
    for setting in SETTINGS[:3]:
        own = np.array(
            [adaptive[f"{u}|{setting}"]["error"] for u in unit_names]
        )
        borrowed = np.array(
            [replica[f"{u}|borrowed|{setting}"] for u in unit_names]
        )
        nfe_own = np.mean(
            [adaptive[f"{u}|{setting}"]["nfe"] for u in unit_names]
        )
        nfe_donor = np.mean(
            [replica[f"{u}|borrowed|{setting}|donor_nfe"] for u in unit_names]
        )
        print(
            f"{setting}: own {own.mean():.3f} borrowed {borrowed.mean():.3f}"
            f" log(borrowed/own) {format_interval(np.log(borrowed / own))}"
            f" NFE own {nfe_own:.1f} donor {nfe_donor:.1f}"
        )
    print()
    print("controller vs static family at the unit NFE")
    print("(negative: controller better)")
    for setting in SETTINGS:
        cells = []
        for family in FAMILIES:
            ratios = np.array(
                [
                    np.log(
                        adaptive[f"{u}|{setting}"]["error"]
                        / curve(
                            family_errors(u, family, grids, replica),
                            adaptive[f"{u}|{setting}"]["nfe"],
                        )
                    )
                    for u in unit_names
                ]
            )
            cells.append(f"{family} {format_interval(ratios)}")
        print(f"{setting}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
