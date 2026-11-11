import dataclasses
from typing import Callable

import numpy as np
import torch

import proxies
from controllers import AngleController

Field = Callable[[torch.Tensor, float], torch.Tensor]


@dataclasses.dataclass
class Trajectory:
    grid: np.ndarray
    states: list[torch.Tensor]
    velocities: list[torch.Tensor]


def ab2_weights(step: float, previous_step: float) -> tuple[float, float]:
    ratio: float = step / previous_step
    return step * (1.0 + 0.5 * ratio), -0.5 * step * ratio


def solve(
    field: Field,
    x: torch.Tensor,
    grid: np.ndarray,
    order: int = 1,
) -> Trajectory:
    states: list[torch.Tensor] = [x]
    velocities: list[torch.Tensor] = []
    for index in range(len(grid) - 1):
        step: float = float(grid[index + 1] - grid[index])
        velocity: torch.Tensor = field(x, float(grid[index]))
        velocities.append(velocity)
        if order == 1 or index == 0:
            x = x + step * velocity
        else:
            previous_step: float = float(grid[index] - grid[index - 1])
            current, past = ab2_weights(step, previous_step)
            x = x + current * velocity + past * velocities[index - 1]
        states.append(x)
    return Trajectory(grid=grid, states=states, velocities=velocities)


def rk4_flow(
    field: Field,
    x: torch.Tensor,
    start: float,
    stop: float,
    substeps: int,
) -> torch.Tensor:
    step: float = (stop - start) / substeps
    time: float = start
    for _ in range(substeps):
        k1: torch.Tensor = field(x, time)
        k2: torch.Tensor = field(x + 0.5 * step * k1, time + 0.5 * step)
        k3: torch.Tensor = field(x + 0.5 * step * k2, time + 0.5 * step)
        k4: torch.Tensor = field(x + step * k3, time + step)
        x = x + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        time = time + step
    return x


def reference_trajectory(
    field: Field,
    x: torch.Tensor,
    grid: np.ndarray,
    substeps: int = 64,
) -> Trajectory:
    states: list[torch.Tensor] = [x]
    velocities: list[torch.Tensor] = [field(x, float(grid[0]))]
    for index in range(len(grid) - 1):
        x = rk4_flow(
            field, x, float(grid[index]), float(grid[index + 1]), substeps
        )
        states.append(x)
        velocities.append(field(x, float(grid[index + 1])))
    return Trajectory(grid=grid, states=states, velocities=velocities)


def final_error(
    field: Field,
    x: torch.Tensor,
    grid: np.ndarray,
    reference: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    result: torch.Tensor = solve(field, x, grid, order).states[-1]
    difference: torch.Tensor = (result - reference).reshape(x.shape[0], -1)
    scale: torch.Tensor = reference.reshape(x.shape[0], -1).norm(dim=1)
    return difference.norm(dim=1) / scale


def euler_local_errors(
    field: Field,
    states: list[torch.Tensor],
    grid: np.ndarray,
    substeps: int = 64,
) -> torch.Tensor:
    errors: list[torch.Tensor] = []
    for index in range(len(grid) - 1):
        start: float = float(grid[index])
        stop: float = float(grid[index + 1])
        state: torch.Tensor = states[index]
        exact: torch.Tensor = rk4_flow(field, state, start, stop, substeps)
        euler: torch.Tensor = state + (stop - start) * field(state, start)
        errors.append((exact - euler).reshape(state.shape[0], -1).norm(dim=1))
    return torch.stack(errors, dim=1)


@dataclasses.dataclass
class AdaptiveResult:
    x: torch.Tensor
    velocity: torch.Tensor
    times: list[float]
    angles: list[float]

    @property
    def evaluations(self) -> int:
        return len(self.times)


def solve_adaptive(
    field: Field,
    x: torch.Tensor,
    end: float,
    controller: AngleController,
) -> AdaptiveResult:
    time: float = 0.0
    step: float = controller.initial_step
    previous: torch.Tensor | None = None
    times: list[float] = []
    angles: list[float] = []
    while True:
        velocity: torch.Tensor = field(x, time)
        times.append(time)
        if time >= end - 1e-12:
            return AdaptiveResult(x, velocity, times, angles)
        if previous is not None:
            angle: float = float(proxies.increment(velocity, previous).angle)
            angles.append(angle)
            step = controller.next_step(step, angle)
        step = min(step, end - time)
        if end - time - step < controller.min_step:
            step = end - time
        x = x + step * velocity
        previous = velocity
        time = time + step
