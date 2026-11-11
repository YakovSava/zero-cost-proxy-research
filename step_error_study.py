import json
from typing import Any

import numpy as np
import scipy.stats

import real_proxy_study
import rig
import units

STRIDES: tuple[int, ...] = (2, 4, 6, 8, 12)
SPACING: float = real_proxy_study.SPACING / 2.0


def euler_step_errors(
    velocities: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(velocities)
    starts = np.arange(stride, count - stride)
    errors = np.zeros(len(starts))
    for row, start in enumerate(starts):
        window = velocities[start:start + stride + 1]
        integral = SPACING * (
            window.sum(axis=0) - 0.5 * (window[0] + window[-1])
        )
        errors[row] = np.linalg.norm(integral - stride * SPACING * window[0])
    return starts, errors


def centered(values: np.ndarray) -> np.ndarray:
    logged = np.log(values)
    return (logged - logged.mean(axis=0))[:, 1:-1]


def estimators(
    velocities: np.ndarray,
    starts: np.ndarray,
    stride: int,
) -> dict[str, np.ndarray]:
    step = stride * SPACING
    current = velocities[starts]
    past = velocities[starts - stride]
    angle = real_proxy_study.angle_between(current, past)
    speed = np.linalg.norm(current, axis=1)
    gap = np.linalg.norm(current - past, axis=1)
    return {
        "angle": 0.5 * step * speed * angle,
        "heun": 0.5 * step * gap,
    }


def main() -> None:
    everything = units.all_units(list(rig.PROMPTS))
    series = [real_proxy_study.load_velocities(unit) for unit in everything]
    report: dict[str, Any] = {}
    for stride in STRIDES:
        truths, angles, heuns = [], [], []
        for velocities in series:
            starts, errors = euler_step_errors(velocities, stride)
            found = estimators(velocities, starts, stride)
            truths.append(errors)
            angles.append(found["angle"])
            heuns.append(found["heun"])
        truth, angle, heun = map(np.stack, (truths, angles, heuns))
        report[str(stride)] = {
            "spacing": stride * SPACING,
            "median_ratio_angle": float(np.median(angle / truth)),
            "median_ratio_heun": float(np.median(heun / truth)),
            "pooled_angle": float(
                scipy.stats.pearsonr(
                    np.log(angle).ravel(), np.log(truth).ravel()
                )[0]
            ),
            "pooled_heun": float(
                scipy.stats.pearsonr(
                    np.log(heun).ravel(), np.log(truth).ravel()
                )[0]
            ),
            "within_angle": float(
                scipy.stats.pearsonr(
                    centered(angle).ravel(), centered(truth).ravel()
                )[0]
            ),
            "within_heun": float(
                scipy.stats.pearsonr(
                    centered(heun).ravel(), centered(truth).ravel()
                )[0]
            ),
        }
    path = rig.ROOT / "results" / "step_error.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
