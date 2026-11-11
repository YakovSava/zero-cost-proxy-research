import numpy as np

import floor
import monitor


def test_local_derivatives_recover_polynomial_exactly() -> None:
    times = np.linspace(0.0, 1.0, 40)
    direction = np.array([1.0, 2.0, -2.0])
    values = (times**3)[:, None] * direction[None] + 2.0 * times[:, None]
    norms = monitor.local_derivative_norms(times, values, 4, 3)
    expected_first = np.linalg.norm(
        3.0 * times[:, None] ** 2 * direction[None] + 2.0, axis=1
    )
    expected_second = 6.0 * times * np.linalg.norm(direction)
    expected_third = 6.0 * np.linalg.norm(direction)
    assert np.allclose(norms[:, 0], expected_first, atol=1e-8)
    assert np.allclose(norms[:, 1], expected_second, atol=1e-8)
    assert np.allclose(norms[:, 2], expected_third, atol=1e-7)


def test_local_derivatives_average_out_white_noise() -> None:
    generator = np.random.default_rng(0)
    times = np.linspace(0.0, 1.0, 97)
    dimension = 4000
    slope = generator.normal(size=dimension)
    clean = times[:, None] * slope[None]
    noisy = clean + 0.05 * generator.normal(size=clean.shape)
    smooth = monitor.local_derivative_norms(times, noisy, 4, 3)[:, 0]
    raw = np.linalg.norm(np.gradient(noisy, times, axis=0), axis=1)
    truth = np.linalg.norm(slope)
    assert np.abs(smooth[10:-10] / truth - 1.0).max() < 0.15
    assert np.abs(raw[10:-10] / truth - 1.0).max() > 0.2


def test_floor_fit_separates_rate_and_floor() -> None:
    generator = np.random.default_rng(1)
    count, dimension = 80, 3000
    rate, noise = 0.01, 0.03
    base = generator.normal(size=dimension)
    turn = generator.normal(size=dimension)
    turn = turn - base * (turn @ base) / (base @ base)
    turn = turn * np.linalg.norm(base) / np.linalg.norm(turn)
    steps = np.arange(count)[:, None]
    clean = (
        np.cos(rate * steps) * base[None] + np.sin(rate * steps) * turn[None]
    )
    noisy = clean + noise * generator.normal(size=clean.shape)
    fit = floor.fit_floor(noisy, np.array([1, 2, 3, 4, 6, 8, 12]), 5, 40)
    assert abs(fit.rate_degrees - np.degrees(rate)) < 0.1
    expected_floor = np.degrees(np.sqrt(2.0) * noise)
    assert abs(fit.floor_degrees - expected_floor) < 0.15 * expected_floor
    assert fit.r_squared > 0.99
