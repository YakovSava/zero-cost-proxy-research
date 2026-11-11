import json
import pathlib
from typing import Any

import numpy as np

RESULTS: pathlib.Path = pathlib.Path(__file__).resolve().parent / "results"
SOURCE: pathlib.Path = RESULTS / "real_grids.json"
OUTPUT: pathlib.Path = RESULTS / "real_grids.md"
BOOTSTRAP: int = 4000
SEED: int = 0


def parse(table: dict[str, float]) -> dict[tuple[str, int, str, int], float]:
    parsed: dict[tuple[str, int, str, int], float] = {}
    for key, value in table.items():
        unit, order, name, steps = key.split("|")
        parsed[(unit, int(order), name, int(steps))] = value
    return parsed


def interval(values: np.ndarray) -> tuple[float, float, float]:
    generator = np.random.default_rng(SEED)
    picks = generator.integers(0, len(values), (BOOTSTRAP, len(values)))
    means = values[picks].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def collect(
    parsed: dict[tuple[str, int, str, int], float],
    order: int,
    steps: int,
    name: str,
    unit_names: list[str],
) -> np.ndarray:
    return np.array(
        [parsed[(unit, order, name, steps)] for unit in unit_names]
    )


def build_report(table: dict[str, float]) -> dict[str, Any]:
    parsed = parse(table)
    names = sorted({key[2] for key in parsed})
    report: dict[str, Any] = {}
    for order in sorted({key[1] for key in parsed}):
        for steps in sorted({key[3] for key in parsed}):
            complete = sorted(
                {
                    unit
                    for (unit, o, _, n) in parsed
                    if o == order and n == steps
                    and all((unit, order, g, steps) in parsed for g in names)
                }
            )
            if not complete:
                continue
            cell: dict[str, Any] = {"units": len(complete)}
            base = collect(parsed, order, steps, "normal", complete)
            for name in names:
                errors = collect(parsed, order, steps, name, complete)
                mean, low, high = interval(np.log(errors / base))
                cell[name] = {
                    "error": float(errors.mean()),
                    "log_ratio_vs_normal": mean,
                    "low": low,
                    "high": high,
                    "wins": int(np.sum(errors < base)),
                }
            report[f"order{order}_n{steps}"] = cell
    return report


def render(report: dict[str, Any]) -> str:
    lines = [
        "| cell | units | grid | mean rel. error | log-ratio vs normal "
        "[95% CI] | wins |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell, row in report.items():
        for name, stats in row.items():
            if name == "units":
                continue
            lines.append(
                f"| {cell} | {row['units']} | {name} | "
                f"{stats['error']:.4f} | {stats['log_ratio_vs_normal']:+.3f} "
                f"[{stats['low']:+.3f}, {stats['high']:+.3f}] | "
                f"{stats['wins']}/{row['units']} |"
            )
    return "\n".join(lines)


def main() -> None:
    table = json.loads(SOURCE.read_text())
    report = build_report(table)
    OUTPUT.write_text(render(report))
    (RESULTS / "real_grids_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(render(report))


if __name__ == "__main__":
    main()
