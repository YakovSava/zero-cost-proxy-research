import math

import numpy as np
import pytest
import torch

import controllers
import gaussian_mixture
import grids
import integrators
import proxies


def test_single_mode_matches_closed_form_flow() -> None:
    flow = gaussian_mixture.GaussianMixtureFlow(
        means=torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64),
        scale=0.3,
        weights=torch.ones(1, dtype=torch.float64),
    )
    noise = flow.sample_noise(5, seed=0)
    final = integrators.rk4_flow(flow.velocity, noise, 0.0, 1.0, 400)
    expected = flow.means + flow.scale * noise
    assert torch.allclose(final, expected, atol=1e-8)


def isotropic_trajectory(
    start: float,
) -> tuple[gaussian_mixture.GaussianMixtureFlow, integrators.Trajectory]:
    flow = gaussian_mixture.isotropic_gaussian(16, scale=2.0)
    noise = flow.sample_noise(4, seed=1)
    state = math.sqrt(flow.variance(start)) * noise
    grid = np.linspace(start, 1.0, 13)
    return flow, integrators.reference_trajectory(flow.velocity, state, grid)


def test_isotropic_flow_is_straight_but_not_uniform() -> None:
    flow, reference = isotropic_trajectory(start=0.4)
    angles = [
        proxies.increment(newer, older).angle.max()
        for newer, older in zip(
            reference.velocities[1:], reference.velocities[:-1]
        )
    ]
    assert max(angles) < 1e-9
    errors = integrators.euler_local_errors(
        flow.velocity, reference.states, reference.grid
    )
    assert errors.min() > 1e-4


def test_isotropic_gap_is_pure_stretch() -> None:
    _, reference = isotropic_trajectory(start=0.4)
    value = proxies.increment(
        reference.velocities[5], reference.velocities[4]
    )
    assert torch.allclose(value.gap, value.stretch, rtol=1e-8)
    assert value.gap.min() > 1e-3


def test_isotropic_velocity_reverses_at_the_variance_minimum() -> None:
    flow, reference = isotropic_trajectory(start=0.05)
    crossing = 1.0 / (1.0 + flow.scale**2)
    angles = [
        proxies.increment(newer, older).angle.min().item()
        for newer, older in zip(
            reference.velocities[1:], reference.velocities[:-1]
        )
    ]
    flipped = [
        index
        for index, angle in enumerate(angles)
        if angle > 0.99 * math.pi
    ]
    assert len(flipped) == 1
    first = flipped[0]
    interval = reference.grid[first:first + 3]
    assert interval[0] <= crossing <= interval[-1]


def test_mixture_flow_rotates() -> None:
    flow = gaussian_mixture.random_mixture(
        dimension=32, modes=4, separation=1.5, scale=0.3, seed=0
    )
    noise = flow.sample_noise(8, seed=3)
    grid = grids.uniform_grid(24)
    reference = integrators.reference_trajectory(flow.velocity, noise, grid)
    total = torch.zeros(8, dtype=torch.float64)
    for newer, older in zip(
        reference.velocities[1:], reference.velocities[:-1]
    ):
        total = total + proxies.increment(newer, older).angle
    assert total.min() > 0.05


def order_of_convergence(order: int) -> float:
    flow = gaussian_mixture.random_mixture(
        dimension=16, modes=3, separation=1.0, scale=0.4, seed=5
    )
    noise = flow.sample_noise(8, seed=4)
    reference = integrators.rk4_flow(flow.velocity, noise, 0.0, 1.0, 2000)
    coarse = integrators.final_error(
        flow.velocity, noise, grids.uniform_grid(40), reference, order
    ).mean()
    fine = integrators.final_error(
        flow.velocity, noise, grids.uniform_grid(80), reference, order
    ).mean()
    return math.log2((coarse / fine).item())


def test_euler_is_first_order() -> None:
    assert order_of_convergence(1) == pytest.approx(1.0, abs=0.25)


def test_adams_bashforth_is_second_order() -> None:
    assert order_of_convergence(2) == pytest.approx(2.0, abs=0.35)


def test_ab2_weights_reduce_to_classical_for_equal_steps() -> None:
    current, past = integrators.ab2_weights(0.1, 0.1)
    assert current == pytest.approx(0.15)
    assert past == pytest.approx(-0.05)


def test_shifted_grid_endpoints_and_identity() -> None:
    assert np.allclose(grids.shifted_grid(10, 1.0), grids.uniform_grid(10))
    shifted = grids.shifted_grid(10, 3.0)
    assert shifted[0] == pytest.approx(0.0)
    assert shifted[-1] == pytest.approx(1.0)
    assert np.all(np.diff(shifted) > 0.0)
    assert np.diff(shifted)[0] < np.diff(shifted)[-1]


def test_equidistribute_constant_density_is_uniform() -> None:
    times = np.linspace(0.0, 1.0, 201)
    result = grids.equidistribute(times, np.ones_like(times), 10)
    assert np.allclose(result, grids.uniform_grid(10), atol=1e-12)


def test_equidistribute_balances_mass() -> None:
    times = np.linspace(0.0, 1.0, 2001)
    density = 1.0 + 20.0 * np.exp(-((times - 0.8) / 0.05) ** 2)
    result = grids.equidistribute(times, density, 16)
    mass = grids.cumulative(times, density)
    masses = np.diff(np.interp(result, times, mass))
    assert masses.std() / masses.mean() < 1e-3
    assert np.diff(result).min() < 0.5 * np.diff(result).max()


def test_optimal_density_exponent() -> None:
    derivative = np.array([1.0, 8.0, 27.0])
    assert np.allclose(grids.optimal_density(derivative, 2), [1.0, 2.0, 3.0])


def test_adaptive_solver_reaches_the_end_and_counts_evaluations() -> None:
    flow = gaussian_mixture.random_mixture(
        dimension=16, modes=3, separation=1.0, scale=0.4, seed=5
    )
    noise = flow.sample_noise(1, seed=4)
    controller = controllers.AngleController(target=0.05, initial_step=0.1)
    result = integrators.solve_adaptive(flow.velocity, noise, 0.99, controller)
    assert result.times[0] == 0.0
    assert result.times[-1] == pytest.approx(0.99)
    assert np.all(np.diff(result.times) > 0.0)
    assert result.evaluations == len(result.angles) + 2


def test_smaller_target_angle_uses_more_evaluations() -> None:
    flow = gaussian_mixture.random_mixture(
        dimension=16, modes=3, separation=1.0, scale=0.4, seed=5
    )
    noise = flow.sample_noise(1, seed=4)
    counts = []
    for target in (0.2, 0.05, 0.0125):
        controller = controllers.AngleController(target=target)
        result = integrators.solve_adaptive(
            flow.velocity, noise, 0.99, controller
        )
        counts.append(result.evaluations)
    assert counts[0] < counts[1] < counts[2]


def test_controller_shrinks_step_when_angle_exceeds_target() -> None:
    controller = controllers.AngleController(target=0.1)
    assert controller.next_step(0.1, 0.2) == pytest.approx(0.05)
    assert controller.next_step(0.1, 0.05) == pytest.approx(0.2)
    assert controller.next_step(0.1, 1e-9) == pytest.approx(0.2)
    assert controller.next_step(0.2, 1e-9) == pytest.approx(0.25)


def noisy_field(
    flow: gaussian_mixture.GaussianMixtureFlow,
    relative: float,
    seed: int,
) -> integrators.Field:
    generator = torch.Generator().manual_seed(seed)

    def field(x: torch.Tensor, t: float) -> torch.Tensor:
        velocity = flow.velocity(x, t)
        noise = torch.randn(
            velocity.shape, generator=generator, dtype=torch.float64
        )
        scale = relative * velocity.norm() / math.sqrt(velocity.numel())
        return velocity + scale * noise

    return field


def test_angle_controller_collapses_below_the_noise_floor() -> None:
    flow = gaussian_mixture.random_mixture(
        dimension=256, modes=3, separation=1.0, scale=0.4, seed=5
    )
    noise = flow.sample_noise(1, seed=4)
    floor_angle = math.sqrt(2.0) * 0.04
    counts = {}
    targets = {"above": 3.0 * floor_angle, "below": 0.6 * floor_angle}
    for name, target in targets.items():
        field = noisy_field(flow, relative=0.04, seed=9)
        controller = controllers.AngleController(
            target=target, initial_step=0.02, min_step=0.004
        )
        counts[name] = integrators.solve_adaptive(
            field, noise, 0.99, controller
        ).evaluations
    assert counts["below"] > 4 * counts["above"]
    assert counts["below"] > 0.7 * 0.99 / 0.004


def test_euler_error_estimate_tracks_the_true_local_error() -> None:
    flow = gaussian_mixture.random_mixture(
        dimension=64, modes=4, separation=1.5, scale=0.5, seed=0
    )
    noise = flow.sample_noise(16, seed=1)
    grid = grids.uniform_grid(32)
    reference = integrators.reference_trajectory(
        flow.velocity, noise, grid, substeps=8
    )
    truth = integrators.euler_local_errors(
        flow.velocity, reference.states, grid, substeps=16
    )
    step = float(grid[1] - grid[0])
    estimates = []
    for newer, older in zip(
        reference.velocities[1:-1], reference.velocities[:-2]
    ):
        value = proxies.increment(newer, older)
        estimates.append(0.5 * step * value.gap)
    estimate = torch.stack(estimates, dim=1)
    ratio = (estimate / truth[:, 1:]).median()
    assert 0.7 < ratio.item() < 1.4
