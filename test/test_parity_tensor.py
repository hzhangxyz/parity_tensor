"""
[#22](https://github.com/hzhangxyz/parity_tensor/issues/22)
Test essential functions.
"""
import torch
import random
import pytest

from parity_tensor import ParityTensor


@pytest.fixture(
    params = [ParityTensor((False, False), ((2, 2), (1, 3)), torch.randn([4, 4])),
              ParityTensor((False, False), ((2, 2) ,(1, 3)), torch.randn([4, 4]))]
)

def parity_instance(request):
    return request.param

@pytest.fixture(params=[(0, 1)])
def permute_tuple(request):
    return request.param

def test_arithmetic(parity_instance) -> None:
    """Test the arithmetic."""
    # Test __pos__ method.
    print("-" * 5, "Test __pos__ method", "-" * 5)
    print(parity_instance)
    print(+parity_instance)
    # Test __neg__ method.
    print("-" * 5, "Test __neg__ method", "-" * 5)
    print(parity_instance)
    print(-parity_instance)
    # Test __add__ method.
    print("-" * 5, "Test __add__ method", "-" * 5)
    print(parity_instance)
    print(parity_instance + 1)
    # Test __radd__ method.
    print("-" * 5, "Test __radd__ method", "-" * 5)
    print(parity_instance)
    print(1 + parity_instance)
    # Test __iadd__ method.
    print("-" * 5, "Test __iadd__ method", "-" * 5)
    print(parity_instance)
    parity_instance += 1
    print(parity_instance)
    # Test __sub__ method.
    print("-" * 5, "Test __sub__ method", "-" * 5)
    print(parity_instance)
    print(parity_instance - 1)
    # Test __rsub__ method.
    print("-" * 5, "Test __rsub__ method", "-" * 5)
    print(parity_instance)
    print(1 - parity_instance)
    # Test __isub__ method.
    print("-" * 5, "Test __isub__ method", "-" * 5)
    print(parity_instance)
    parity_instance -= 1
    print(parity_instance)
    # Test __mul__ method.
    print("-" * 5, "Test __mul__ method", "-" * 5)
    print(parity_instance)
    print(parity_instance * 2)
    # Test __rmul__ method.
    print("-" * 5, "Test __rmul__ method", "-" * 5)
    print(parity_instance)
    print(2 * parity_instance)
    # Test __imul__ method.
    print("-" * 5, "Test __imul__ method", "-" * 5)
    print(parity_instance)
    parity_instance *= 2
    print(parity_instance)
    # Test __truediv__ method.
    print("-" * 5, "Test __truediv__ method", "-" * 5)
    print(parity_instance)
    print(parity_instance / 2)
    # Test __rtruediv__ method.
    print("-" * 5, "Test __rtruediv__ method", "-" * 5)
    print(parity_instance)
    print(2 / parity_instance)
    # Test __itruediv__ method.
    print("-" * 5, "Test __itruediv__ method", "-" * 5)
    print(parity_instance)
    parity_instance /= 2
    print(parity_instance)

def test_masking() -> None:
    """Test masking."""
    pass

def test_permute(parity_instance, permute_tuple) -> None:
    """Test permute."""
    parity_instance.permute(permute_tuple)

def test_reverse(parity_instance) -> None:
    """Test tensor reverse."""
    print(parity_instance)
    print(parity_instance)

def test_split_edge() -> None:
    """Test split edge."""
    pass

def test_merge_edge() -> None:
    """Test merge edge."""
    pass

def test_contract() -> None:
    """Test contract."""
    pass

def test_trace() -> None:
    """Test trace."""
    pass

def test_conjugate() -> None:
    """Test conjugate."""
    pass

def test_svd() -> None:
    """Test svd."""
    pass

def test_qr() -> None:
    """Test qr."""
    pass

def test_identity() -> None:
    """Test identity."""
    pass

def test_exponential() -> None:
    """Test tensor exponential."""
    pass