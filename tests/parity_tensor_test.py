"""
[#22](https://github.com/hzhangxyz/parity_tensor/issues/22)
Test essential functions.
"""
from typing import Tuple

import torch
import pytest

from parity_tensor import ParityTensor


@pytest.fixture(name="parity_instance_fx", params=[ParityTensor((False, False), ((2, 2), (1, 3)), torch.randn([4, 4])), ParityTensor((False, False), ((2, 2), (1, 3)), torch.randn([4, 4]))])
def _parity_instance_fixture(request: pytest.FixtureRequest) -> ParityTensor:
    """Fixture for a parity tensor instance."""
    return request.param


@pytest.fixture(name="permute_tuple_fx", params=[(0, 1)])
def permute_tuple_fixture(request: pytest.FixtureRequest) -> Tuple[int, int]:
    """Fixture for a permutation tuple."""
    return request.param


def test_arithmetic(parity_instance_fx: ParityTensor) -> None:
    """Test the arithmetic operators on ParityTensor."""
    instance = ParityTensor((False, False), ((2, 2), (1, 3)), torch.randn([4, 4]))
    # Test __pos__ method.
    print("-" * 5, "Test __pos__ method", "-" * 5)
    print(parity_instance_fx)
    print(+parity_instance_fx)

    # Test __neg__ method.
    print("-" * 5, "Test __neg__ method", "-" * 5)
    print(parity_instance_fx)
    print(-parity_instance_fx)

    # Test __add__ method.
    print("-" * 5, "Test __add__ method", "-" * 5)
    print(parity_instance_fx)
    print(parity_instance_fx + 1)
    print(parity_instance_fx + instance)

    # Test __radd__ method.
    print("-" * 5, "Test __radd__ method", "-" * 5)
    print(parity_instance_fx)
    print(1 + parity_instance_fx)
    print(instance + parity_instance_fx)

    # Test __iadd__ method.
    print("-" * 5, "Test __iadd__ method", "-" * 5)
    print(parity_instance_fx)
    parity_instance_fx += 1
    print(parity_instance_fx)
    parity_instance_fx += instance
    print(parity_instance_fx)

    # Test __sub__ method.
    print("-" * 5, "Test __sub__ method", "-" * 5)
    print(parity_instance_fx)
    print(parity_instance_fx - 1)
    print(parity_instance_fx - instance)

    # Test __rsub__ method.
    print("-" * 5, "Test __rsub__ method", "-" * 5)
    print(parity_instance_fx)
    print(1 - parity_instance_fx)
    print(instance - parity_instance_fx)

    # Test __isub__ method.
    print("-" * 5, "Test __isub__ method", "-" * 5)
    print(parity_instance_fx)
    parity_instance_fx -= 1
    print(parity_instance_fx)
    parity_instance_fx -= instance
    print(parity_instance_fx)

    # Test __mul__ method.
    print("-" * 5, "Test __mul__ method", "-" * 5)
    print(parity_instance_fx)
    print(parity_instance_fx * 2)

    # Test __rmul__ method.
    print("-" * 5, "Test __rmul__ method", "-" * 5)
    print(parity_instance_fx)
    print(2 * parity_instance_fx)

    # Test __imul__ method.
    print("-" * 5, "Test __imul__ method", "-" * 5)
    print(parity_instance_fx)
    parity_instance_fx *= 2
    print(parity_instance_fx)

    # Test __truediv__ method.
    print("-" * 5, "Test __truediv__ method", "-" * 5)
    print(parity_instance_fx)
    print(parity_instance_fx / 2)
    print(parity_instance_fx / instance)

    # Test __rtruediv__ method.
    print("-" * 5, "Test __rtruediv__ method", "-" * 5)
    print(parity_instance_fx)
    print(2 / parity_instance_fx)

    # Test __itruediv__ method.
    print("-" * 5, "Test __itruediv__ method", "-" * 5)
    print(parity_instance_fx)
    parity_instance_fx /= 2
    print(parity_instance_fx)


def test_masking() -> None:
    """Test masking."""
    assert True


def test_permute(parity_instance_fx: ParityTensor, permute_tuple_fx: Tuple[int, ...]) -> None:
    """Test permute."""
    print(parity_instance_fx)
    parity_instance_fx.permute(permute_tuple_fx)
    print(parity_instance_fx)


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
