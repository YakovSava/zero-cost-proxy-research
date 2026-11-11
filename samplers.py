import dataclasses
from typing import Any, Callable

import torch

from controllers import AngleController
from integrators import ab2_weights, solve_adaptive

SIGMA_FLOOR: float = 1e-4


@dataclasses.dataclass
class Trace:
    sigmas: list[float] = dataclasses.field(default_factory=list)
    velocities: list[torch.Tensor] = dataclasses.field(default_factory=list)
    steps: list[float] = dataclasses.field(default_factory=list)


def velocity_from(
    x: torch.Tensor,
    denoised: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    return (x - denoised) / max(sigma, SIGMA_FLOOR)


@torch.no_grad()
def integrate(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    first: int = 0,
    extra_args: dict[str, Any] | None = None,
    order: int = 1,
    trace: Trace | None = None,
    keep_velocities: bool = False,
) -> torch.Tensor:
    extra_args = {} if extra_args is None else extra_args
    s_in: torch.Tensor = x.new_ones([x.shape[0]])
    schedule: list[float] = [float(value) for value in sigmas.cpu()]
    previous: torch.Tensor | None = None
    for index in range(first, len(schedule) - 1):
        sigma: float = schedule[index]
        denoised: torch.Tensor = model(x, sigmas[index] * s_in, **extra_args)
        if schedule[index + 1] == 0.0:
            if trace is not None:
                trace.sigmas.append(sigma)
                if keep_velocities:
                    trace.velocities.append(
                        velocity_from(x, denoised, sigma).cpu()
                    )
            return denoised
        velocity: torch.Tensor = velocity_from(x, denoised, sigma)
        step: float = schedule[index + 1] - sigma
        if trace is not None:
            trace.sigmas.append(sigma)
            trace.steps.append(step)
            if keep_velocities:
                trace.velocities.append(velocity.cpu())
        if order == 1 or previous is None:
            x = x + step * velocity
        else:
            previous_step: float = sigma - schedule[index - 1]
            current, past = ab2_weights(step, previous_step)
            x = x + current * velocity + past * previous
        previous = velocity
    return x


def sample_fixed(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Callable[..., Any] | None = None,
    disable: bool | None = None,
    order: int = 1,
    trace: Trace | None = None,
    keep_velocities: bool = False,
) -> torch.Tensor:
    return integrate(
        model, x, sigmas, 0, extra_args, order, trace, keep_velocities
    )


@torch.no_grad()
def sample_adaptive(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any] | None = None,
    callback: Callable[..., Any] | None = None,
    disable: bool | None = None,
    controller: AngleController | None = None,
    report: dict[str, Any] | None = None,
) -> torch.Tensor:
    extra_args = {} if extra_args is None else extra_args
    s_in: torch.Tensor = x.new_ones([x.shape[0]])
    sigma_end: float = float(sigmas[-2])

    def field(state: torch.Tensor, time: float) -> torch.Tensor:
        sigma: float = 1.0 - time
        denoised: torch.Tensor = model(state, sigma * s_in, **extra_args)
        return -velocity_from(state, denoised, sigma)

    result = solve_adaptive(field, x, 1.0 - sigma_end, controller)
    if report is not None:
        report["times"] = result.times
        report["angles"] = result.angles
    return result.x + sigma_end * result.velocity
