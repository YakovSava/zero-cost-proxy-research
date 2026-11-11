import json
import pathlib
from typing import Any

import numpy as np

import analyze_grids

RESULTS: pathlib.Path = pathlib.Path(__file__).resolve().parent / "results"


def build_report() -> dict[str, Any]:
    startfix = json.loads((RESULTS / "real_startfix.json").read_text())
    grids = analyze_grids.parse(
        json.loads((RESULTS / "real_grids.json").read_text())
    )
    report: dict[str, Any] = {}
    cells = sorted({tuple(key.split("|")[1:]) for key in startfix})
    for steps, first in cells:
        rows = {
            key.split("|")[0]: value
            for key, value in startfix.items()
            if key.split("|")[1:] == [steps, first]
        }
        ratios = np.array(
            [
                np.log(value / grids[(unit, 1, "normal", int(steps))])
                for unit, value in rows.items()
            ]
        )
        static = np.array(
            [
                np.log(value / grids[(unit, 1, "static_acc", int(steps))])
                for unit, value in rows.items()
            ]
        )
        report[f"n{steps}_h{first}"] = {
            "units": len(rows),
            "error": float(np.mean(list(rows.values()))),
            "vs_normal": analyze_grids.interval(ratios),
            "vs_static_acc": analyze_grids.interval(static),
        }
    return report


def main() -> None:
    report = build_report()
    (RESULTS / "real_startfix_report.json").write_text(
        json.dumps(report, indent=2)
    )
    for name, cell in report.items():
        normal = cell["vs_normal"]
        static = cell["vs_static_acc"]
        print(
            f"{name:12s} units {cell['units']} err {cell['error']:.3f} | "
            f"vs normal {normal[0]:+.2f}[{normal[1]:+.2f},{normal[2]:+.2f}]"
            f" | vs static_acc "
            f"{static[0]:+.2f}[{static[1]:+.2f},{static[2]:+.2f}]"
        )


if __name__ == "__main__":
    main()
