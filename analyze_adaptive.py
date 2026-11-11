import json
import pathlib
from typing import Any

import numpy as np

import analyze_grids

RESULTS: pathlib.Path = pathlib.Path(__file__).resolve().parent / "results"
STEPS: tuple[int, ...] = (8, 12, 16, 24)
REFERENCES: tuple[str, ...] = ("normal", "static_acc", "oracle_acc")


def curve_value(
    grids: dict[tuple[str, int, str, int], float],
    unit: str,
    name: str,
    evaluations: float,
) -> float:
    log_steps = np.log(np.array(STEPS, dtype=float))
    log_errors = np.log([grids[(unit, 1, name, n)] for n in STEPS])
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


def build_report() -> dict[str, Any]:
    grids = analyze_grids.parse(
        json.loads((RESULTS / "real_grids.json").read_text())
    )
    adaptive = json.loads((RESULTS / "real_adaptive.json").read_text())
    adaptive = {
        (key if key.count("|") == 3 else f"{key}|0.06"): value
        for key, value in adaptive.items()
    }
    settings = sorted({tuple(key.split("|")[1:]) for key in adaptive})
    report: dict[str, Any] = {}
    for gamma, degrees, initial in settings:
        rows = {
            key.split("|")[0]: value
            for key, value in adaptive.items()
            if key.split("|")[1:] == [gamma, degrees, initial]
        }
        cell: dict[str, Any] = {
            "units": len(rows),
            "mean_nfe": float(np.mean([r["nfe"] for r in rows.values()])),
            "nfe_range": [
                min(r["nfe"] for r in rows.values()),
                max(r["nfe"] for r in rows.values()),
            ],
            "mean_error": float(np.mean([r["error"] for r in rows.values()])),
        }
        for reference in REFERENCES:
            ratios = np.array(
                [
                    np.log(
                        row["error"]
                        / curve_value(grids, unit, reference, row["nfe"])
                    )
                    for unit, row in rows.items()
                ]
            )
            mean, low, high = analyze_grids.interval(ratios)
            cell[f"vs_{reference}"] = {
                "log_ratio": mean,
                "low": low,
                "high": high,
            }
        report[f"gamma{gamma}_deg{degrees}_h0{initial}"] = cell
    return report


def main() -> None:
    report = build_report()
    (RESULTS / "real_adaptive_report.json").write_text(
        json.dumps(report, indent=2)
    )
    for name, cell in report.items():
        parts = [
            f"{key[3:]} {value['log_ratio']:+.2f}"
            f"[{value['low']:+.2f},{value['high']:+.2f}]"
            for key, value in cell.items()
            if key.startswith("vs_")
        ]
        print(
            f"{name:24s} NFE {cell['mean_nfe']:5.1f} "
            f"{cell['nfe_range']} err {cell['mean_error']:.3f} | "
            + " | ".join(parts)
        )


if __name__ == "__main__":
    main()
