import math

import pytest
import torch

import proxies


def random_pair(
    count: int,
    dimension: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (count, dimension)
    first = torch.randn(shape, generator=generator, dtype=torch.float64)
    second = torch.randn(shape, generator=generator, dtype=torch.float64)
    return first, second


def test_gap_splits_into_stretch_and_rotation() -> None:
    new, old = random_pair(32, 17, seed=1)
    value = proxies.increment(new, old)
    reconstructed = value.stretch**2 + value.rotation**2
    assert torch.allclose(value.gap**2, reconstructed, rtol=1e-12)


def test_rotation_vanishes_for_parallel_vectors() -> None:
    new, _ = random_pair(8, 5, seed=2)
    value = proxies.increment(new, 3.0 * new)
    assert value.angle.max() < 1e-12
    assert torch.allclose(value.gap, 2.0 * new.norm(dim=1), rtol=1e-12)
    assert torch.allclose(value.gap, value.stretch, rtol=1e-12)


def test_stretch_vanishes_for_equal_norms() -> None:
    new, old = random_pair(8, 5, seed=3)
    old = old * (new.norm(dim=1) / old.norm(dim=1))[:, None]
    value = proxies.increment(new, old)
    assert value.stretch.max() < 1e-12
    assert torch.allclose(value.gap, value.rotation, rtol=1e-10)


@pytest.mark.parametrize("angle", [1e-8, 1e-5, 1e-2, 1.0, 3.0])
def test_angle_is_accurate_for_small_rotations(angle: float) -> None:
    first = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    second = torch.tensor(
        [[math.cos(angle), math.sin(angle)]], dtype=torch.float64
    )
    value = proxies.increment(second, first)
    assert value.angle.item() == pytest.approx(angle, rel=1e-9)


def test_cosine_loses_small_angles_in_single_precision() -> None:
    angle = 1e-3
    first = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    second = torch.tensor(
        [[math.cos(angle), math.sin(angle)]], dtype=torch.float32
    )
    value = proxies.increment(second, first)
    naive = torch.acos(
        (first * second).sum(dim=1).clamp(-1.0, 1.0)
    ).item()
    assert abs(value.angle.item() - angle) < abs(naive - angle) + 1e-12
    assert value.angle.item() == pytest.approx(angle, rel=1e-3)


def test_rates_scale_inversely_with_step() -> None:
    new, old = random_pair(4, 9, seed=4)
    value = proxies.increment(new, old)
    assert torch.allclose(
        proxies.acceleration(value, 0.25),
        4.0 * proxies.acceleration(value, 1.0),
    )
    assert torch.allclose(
        proxies.curvature(value, 0.5), 2.0 * proxies.curvature(value, 1.0)
    )
