import json
from typing import Any

import numpy as np
import scipy.stats

import profiles
import rig
import units

STRIDES: tuple[int, ...] = (2, 4, 8)
SPACING: float = 2.0 * (1.0 - units.SIGMA_MIN) / (units.REFERENCE_POINTS - 1)


def angle_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    unit_first = first / np.linalg.norm(first, axis=1, keepdims=True)
    unit_second = second / np.linalg.norm(second, axis=1, keepdims=True)
    chord = np.linalg.norm(unit_first - unit_second, axis=1)
    return 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))


def load_velocities(unit: units.Unit) -> np.ndarray:
    stored = np.load(units.reference_path(rig.DUMPS, unit))
    count = len(stored["indices"])
    return stored["velocities"].reshape(count, -1).astype(np.float64)[:-1]


def proxy_series(
    velocities: np.ndarray,
    stride: int,
) -> dict[str, np.ndarray]:
    sampled = velocities[::stride]
    step = stride * SPACING
    one = angle_between(sampled[2:], sampled[1:-1])
    two = angle_between(sampled[2:], sampled[:-2])
    smooth = np.maximum((two**2 - one**2) / 3.0, 0.0)
    gap = np.linalg.norm(sampled[2:] - sampled[1:-1], axis=1)
    return {
        "turning_raw": one / step,
        "acceleration_raw": gap / step,
        "turning_debiased": np.sqrt(smooth) / step,
        "floor": np.sqrt(np.maximum((4.0 * one**2 - two**2) / 3.0, 0.0)),
        "index": np.arange(2, len(sampled)) * stride,
    }


def truth_series(
    profile: profiles.Profile,
    index: np.ndarray,
) -> dict[str, np.ndarray]:
    derivative = profile.derivatives[index, 0]
    speed = profile.speed[index]
    return {"turning": derivative / speed, "acceleration": derivative}


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    return float(scipy.stats.pearsonr(first.ravel(), second.ravel())[0])


def summarize(
    proxy: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:
    log_proxy = np.log(proxy + 1e-9)
    log_truth = np.log(truth + 1e-9)
    within_proxy = log_proxy - log_proxy.mean(axis=0)
    within_truth = log_truth - log_truth.mean(axis=0)
    interior = slice(2, -2)
    return {
        "pooled": pearson(log_proxy, log_truth),
        "profile": float(
            scipy.stats.spearmanr(
                log_proxy.mean(axis=0), log_truth.mean(axis=0)
            )[0]
        ),
        "within": pearson(within_proxy, within_truth),
        "within_interior": pearson(
            within_proxy[:, interior], within_truth[:, interior]
        ),
    }


def main() -> None:
    everything = [
        unit
        for unit in units.all_units(list(rig.PROMPTS))
        if units.reference_path(rig.DUMPS, unit).exists()
    ]
    loaded = [
        (profiles.load_profile(unit), load_velocities(unit))
        for unit in everything
    ]
    report: dict[str, Any] = {}
    for stride in STRIDES:
        rows: dict[str, list[np.ndarray]] = {}
        truths: list[np.ndarray] = []
        floors: list[np.ndarray] = []
        for profile, velocities in loaded:
            series = proxy_series(velocities, stride)
            truths.append(truth_series(profile, series["index"])["turning"])
            floors.append(series["floor"])
            for key in ("turning_raw", "turning_debiased"):
                rows.setdefault(key, []).append(series[key])
        truth = np.stack(truths)
        floor = np.stack(floors)
        report[str(stride)] = {
            key: summarize(np.stack(values), truth)
            for key, values in rows.items()
        }
        report[str(stride)]["floor_degrees_median"] = float(
            np.degrees(np.median(floor))
        )
        report[str(stride)]["spacing"] = stride * SPACING
    path = rig.ROOT / "results" / "real_proxy.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
