import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import profiles
import rig
import units

plt.switch_backend("Agg")
FIGURES: pathlib.Path = rig.ROOT / "figures"
RESULTS: pathlib.Path = rig.ROOT / "results"


def normal_first_step(steps: int) -> float:
    shift, floor = 3.0, 0.003
    u = 1.0 - (1.0 - floor) / (steps - 1)
    return 1.0 - shift * u / (1.0 + (shift - 1.0) * u)


def monitor_figure(loaded: list[profiles.Profile]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    times = loaded[0].times
    for profile in loaded:
        axes[0].plot(times, profile.speed, color="tab:blue", alpha=0.35)
        rate = profile.derivatives[:, 0] / profile.speed
        axes[1].semilogy(times, rate, color="tab:red", alpha=0.35)
        axes[2].semilogy(
            times, profile.roughness, color="tab:green", alpha=0.35
        )
    axes[0].set_title("speed ||v||")
    axes[1].set_title("debiased monitor |v'| / ||v||")
    axes[2].set_title("relative evaluation noise")
    for axis in axes:
        axis.set_xlabel("t = 1 - sigma")
    figure.tight_layout()
    figure.savefig(FIGURES / "monitor_profiles.png", dpi=130)
    plt.close(figure)


def first_step_figure() -> None:
    grids = json.loads((RESULTS / "real_grids.json").read_text())
    adaptive = json.loads((RESULTS / "real_adaptive.json").read_text())
    startfix = RESULTS / "real_startfix.json"
    names = [unit.name for unit in units.all_units(list(rig.PROMPTS))]
    figure, axis = plt.subplots(figsize=(6.2, 4.4))
    h_normal = [normal_first_step(n) for n in (8, 12, 16, 24)]
    e_normal = [
        np.mean([grids[f"{u}|1|normal|{n}"] for u in names])
        for n in (8, 12, 16, 24)
    ]
    axis.loglog(h_normal, e_normal, "o-", label="normal, N=8..24")
    for initial in ("0.06", "0.02", "0.01"):
        errors = [
            value["error"]
            for key, value in adaptive.items()
            if (key.split("|")[3] if key.count("|") == 3 else "0.06")
            == initial
        ]
        axis.plot(
            float(initial),
            np.mean(errors),
            "s",
            color="tab:red",
            label="cos controller" if initial == "0.06" else None,
        )
    if startfix.exists():
        table = json.loads(startfix.read_text())
        palette = ((8, "tab:green"), (12, "tab:olive"), (16, "tab:purple"))
        for steps, color in palette:
            xs = sorted({float(k.split("|")[2]) for k in table})
            ys = [
                np.mean(
                    [
                        v
                        for k, v in table.items()
                        if k.split("|")[1] == str(steps)
                        and float(k.split("|")[2]) == x
                    ]
                )
                for x in xs
            ]
            axis.plot(
                xs,
                ys,
                "^--",
                color=color,
                label=f"normal + fixed first step, N={steps}",
            )
    axis.set_xlabel("first step h0 (in t)")
    axis.set_ylabel("mean relative final-latent error")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURES / "error_vs_first_step.png", dpi=130)
    plt.close(figure)


def estimator_figure() -> None:
    report = json.loads((RESULTS / "step_error.json").read_text())
    strides = sorted(report, key=int)
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    axis.plot(
        [report[s]["spacing"] for s in strides],
        [report[s]["within_angle"] for s in strides],
        "o-",
        label="angle estimator, across-unit",
    )
    axis.plot(
        [report[s]["spacing"] for s in strides],
        [report[s]["median_ratio_angle"] for s in strides],
        "s-",
        label="median estimate / true error",
    )
    axis.set_xlabel("step h")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "step_error_estimator.png", dpi=130)
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    loaded = [
        profiles.load_profile(unit)
        for unit in units.all_units(list(rig.PROMPTS))
    ]
    monitor_figure(loaded)
    first_step_figure()
    estimator_figure()


if __name__ == "__main__":
    main()
