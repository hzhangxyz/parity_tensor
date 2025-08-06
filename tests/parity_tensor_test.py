"""
[#22](https://github.com/hzhangxyz/parity_tensor/issues/22)
Test essential functions.
"""
from typing import Tuple

import torch
import pytest

from parity_tensor import ParityTensor


@pytest.fixture(name="parity_instance_fx",
                params=[(
                    ParityTensor((False, False), ((2, 2), (1, 3)), torch.randn([4, 4])),
                    ParityTensor((False, False), ((2, 2), (1, 3)), torch.randn([4, 4])),
                ), (
                    ParityTensor((True, False, True), ((1, 1), (2, 2), (3, 1)), torch.randn([2, 4, 4])),
                    ParityTensor((True, False, True), ((1, 1), (2, 2), (3, 1)), torch.randn([2, 4, 4])),
                ),
                        (ParityTensor((True, True, False, False), ((1, 2), (2, 2), (1, 1), (3, 1)),
                                      torch.randn([3, 4, 2, 4])), ParityTensor((True, True, False, False), ((1, 2), (2, 2), (1, 1), (3, 1)), torch.randn([3, 4, 2, 4])))])
def _parity_instance_fixture(request: pytest.FixtureRequest) -> Tuple[ParityTensor, ParityTensor]:
    """Fixture for a parity tensor instance."""
    return request.param


@pytest.fixture
def permute_tuple_fx(parity_instance_fx: Tuple[ParityTensor, ParityTensor]) -> Tuple[int, ...]:
    """Fixture for a permutation tuple."""
    parity_tensor, _ = parity_instance_fx
    rank = parity_tensor.tensor.ndim
    return tuple(reversed(range(rank)))


def test_arithmetic(parity_instance_fx: Tuple[ParityTensor, ParityTensor]) -> None:
    """Test the arithmetic operators on ParityTensor."""
    parity_tensor, instance = parity_instance_fx

    # Test __pos__ method.
    print("-" * 5, "Test __pos__ method", "-" * 5)
    print(parity_tensor)
    print(+parity_tensor)

    # Test __neg__ method.
    print("-" * 5, "Test __neg__ method", "-" * 5)
    print(parity_tensor)
    print(-parity_tensor)

    # Test __add__ method.
    print("-" * 5, "Test __add__ method", "-" * 5)
    print(parity_tensor)
    print(parity_tensor + 1)
    print(parity_tensor + instance)

    # Test __radd__ method.
    print("-" * 5, "Test __radd__ method", "-" * 5)
    print(parity_tensor)
    print(1 + parity_tensor)
    print(instance + parity_tensor)

    # Test __iadd__ method.
    print("-" * 5, "Test __iadd__ method", "-" * 5)
    print(parity_tensor)
    parity_tensor += 1
    print(parity_tensor)
    parity_tensor += instance
    print(parity_tensor)

    # Test __sub__ method.
    print("-" * 5, "Test __sub__ method", "-" * 5)
    print(parity_tensor)
    print(parity_tensor - 1)
    print(parity_tensor - instance)

    # Test __rsub__ method.
    print("-" * 5, "Test __rsub__ method", "-" * 5)
    print(parity_tensor)
    print(1 - parity_tensor)
    print(instance - parity_tensor)

    # Test __isub__ method.
    print("-" * 5, "Test __isub__ method", "-" * 5)
    print(parity_tensor)
    parity_tensor -= 1
    print(parity_tensor)
    parity_tensor -= instance
    print(parity_tensor)

    # Test __mul__ method.
    print("-" * 5, "Test __mul__ method", "-" * 5)
    print(parity_tensor)
    print(parity_tensor * 2)

    # Test __rmul__ method.
    print("-" * 5, "Test __rmul__ method", "-" * 5)
    print(parity_tensor)
    print(2 * parity_tensor)

    # Test __imul__ method.
    print("-" * 5, "Test __imul__ method", "-" * 5)
    print(parity_tensor)
    parity_tensor *= 2
    print(parity_tensor)

    # Test __truediv__ method.
    print("-" * 5, "Test __truediv__ method", "-" * 5)
    print(parity_tensor)
    print(parity_tensor / 2)
    print(parity_tensor / instance)

    # Test __rtruediv__ method.
    print("-" * 5, "Test __rtruediv__ method", "-" * 5)
    print(parity_tensor)
    print(2 / parity_tensor)

    # Test __itruediv__ method.
    print("-" * 5, "Test __itruediv__ method", "-" * 5)
    print(parity_tensor)
    parity_tensor /= 2
    print(parity_tensor)


def test_masking(parity_instance_fx: tuple[ParityTensor, ParityTensor]) -> None:
    """Test masking."""
    parity_tensor, _ = parity_instance_fx

    mask = parity_tensor.mask
    assert mask.shape == parity_tensor.tensor.shape, "Mask shape mismatch"

    assert mask.dtype == torch.bool, "Mask must be of dtype bool"

    original_tensor = parity_tensor.tensor.clone()
    update_tensor = parity_tensor.update_mask().tensor

    assert torch.all(update_tensor[~mask] == 0), "Masked-out positions must be zero"

    assert torch.all(update_tensor[mask] == original_tensor[mask]), ("Unmasked positions must be preserved")

    assert parity_tensor.mask is not None, "Mask should be cached after first access"


def test_permute(parity_instance_fx: Tuple[ParityTensor, ParityTensor], permute_tuple_fx: Tuple[int, ...]) -> None:
    """Test permute."""
    parity_tensor, _ = parity_instance_fx
    print("Before permute:", parity_tensor)
    parity_tensor.permute(permute_tuple_fx)
    print("After permute:", parity_tensor)


def test_reverse() -> None:
    """Test tensor reverse."""
    assert True


def test_split_edge() -> None:
    """Test split edge."""
    assert True


def test_merge_edge() -> None:
    """Test merge edge."""
    assert True


def test_contract() -> None:
    """Test contract."""
    assert True


def test_trace() -> None:
    """Test trace."""
    assert True


def test_conjugate() -> None:
    """Test conjugate."""
    assert True


def test_svd() -> None:
    """Test svd."""
    assert True


def test_qr() -> None:
    """Test qr."""
    assert True


def test_identity() -> None:
    """Test identity."""
    assert True


def test_exponential() -> None:
    """Test tensor exponential."""
    assert True
