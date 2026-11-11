import numpy as np
import torch

from integrators import Trajectory


def stack_velocities(trajectory: Trajectory) -> np.ndarray:
    stacked: torch.Tensor = torch.stack(trajectory.velocities)
    return stacked.reshape(stacked.shape[0], stacked.shape[1], -1).numpy()


def derivative_norms(trajectory: Trajectory, count: int) -> np.ndarray:
    values: np.ndarray = stack_velocities(trajectory)
    norms: list[np.ndarray] = []
    for _ in range(count):
        values = np.gradient(values, trajectory.grid, axis=0, edge_order=2)
        norms.append(np.linalg.norm(values, axis=2))
    return np.stack(norms, axis=-1)


def speed_profile(trajectory: Trajectory) -> np.ndarray:
    return np.linalg.norm(stack_velocities(trajectory), axis=2)


def local_fit_statistics(
    times: np.ndarray,
    values: np.ndarray,
    half_window: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    count: int = len(times)
    width: int = 2 * half_window + 1
    norms: np.ndarray = np.zeros((count, degree))
    jitter: np.ndarray = np.zeros(count)
    factorials: list[float] = [1.0, 1.0, 2.0, 6.0, 24.0]
    for index in range(count):
        first: int = min(max(index - half_window, 0), count - width)
        window: slice = slice(first, first + width)
        offsets: np.ndarray = times[window] - times[index]
        design: np.ndarray = np.vander(offsets, degree + 1, increasing=True)
        coefficients: np.ndarray = np.linalg.lstsq(
            design, values[window], rcond=None
        )[0]
        residual: np.ndarray = design @ coefficients - values[window]
        noise: float = float(np.sum(residual**2)) / (width - degree - 1)
        jitter[index] = np.sqrt(noise)
        gram_inverse: np.ndarray = np.linalg.inv(design.T @ design)
        for order in range(1, degree + 1):
            power: float = float(np.sum(coefficients[order] ** 2))
            power = power - noise * gram_inverse[order, order]
            norms[index, order - 1] = factorials[order] * np.sqrt(
                max(power, 0.0)
            )
    return norms, jitter


def local_derivative_norms(
    times: np.ndarray,
    values: np.ndarray,
    half_window: int,
    degree: int,
) -> np.ndarray:
    return local_fit_statistics(times, values, half_window, degree)[0]
