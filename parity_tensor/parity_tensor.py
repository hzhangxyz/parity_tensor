"""
A parity tensor class.
"""

from __future__ import annotations

__all__ = ["ParityTensor"]

import dataclasses
import functools
import typing
import torch


@dataclasses.dataclass
class ParityTensor:
    """
    A parity tensor class, which stores a tensor along with information about its edges.
    Each dimension of the tensor is composed of an even and an odd part, represented as a pair of integers.
    """

    edges: tuple[tuple[int, int], ...]
    tensor: torch.Tensor
    mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        assert len(self.edges) == self.tensor.dim(), f"Edges length ({len(self.edges)}) must match tensor dimensions ({self.tensor.dim()})."
        for dim, (even, odd) in zip(self.tensor.shape, self.edges):
            assert even >= 0 and odd >= 0 and dim == even + odd, f"Dimension {dim} must equal sum of even ({even}) and odd ({odd}) parts, and both must be non-negative."
        if self.mask is None:
            self.mask = self._tensor_mask()

    @classmethod
    def _unqueeze(cls, tensor: torch.Tensor, index: int, dim: int) -> torch.Tensor:
        return tensor.view([-1 if i == index else 1 for i in range(dim)])

    @classmethod
    def _edge_mask(cls, even: int, odd: int) -> torch.Tensor:
        return torch.cat([torch.zeros(even, dtype=torch.bool), torch.ones(odd, dtype=torch.bool)])

    def _tensor_mask(self) -> torch.Tensor:
        return functools.reduce(
            torch.logical_xor,
            (self._unqueeze(self._edge_mask(even, odd), index, self.tensor.dim()) for index, (even, odd) in enumerate(self.edges)),
            torch.ones_like(self.tensor, dtype=torch.bool),
        )

    def _validate_edge_compatibility(self, other: ParityTensor) -> None:
        """
        Validate that the edges of two ParityTensor instances are compatible for arithmetic operations.
        """
        assert self.edges == other.edges, f"Edges must match for arithmetic operations. Got {self.edges} and {other.edges}."

    def __pos__(self) -> ParityTensor:
        return ParityTensor(
            edges=self.edges,
            tensor=+self.tensor,
        )

    def __neg__(self) -> ParityTensor:
        return ParityTensor(
            edges=self.edges,
            tensor=-self.tensor,
        )

    def __add__(self, other: typing.Any) -> ParityTensor:
        if isinstance(other, ParityTensor):
            self._validate_edge_compatibility(other)
            return ParityTensor(
                edges=self.edges,
                tensor=self.tensor + other.tensor,
            )
        try:
            result = self.tensor + other
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return ParityTensor(
                edges=self.edges,
                tensor=result,
            )
        return NotImplemented

    def __radd__(self, other: typing.Any) -> ParityTensor:
        try:
            result = other + self.tensor
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return ParityTensor(
                edges=self.edges,
                tensor=result,
            )
        return NotImplemented

    def __iadd__(self, other: typing.Any) -> ParityTensor:
        if isinstance(other, ParityTensor):
            self._validate_edge_compatibility(other)
            self.tensor += other.tensor
        else:
            self.tensor += other
        return self

    def __sub__(self, other: typing.Any) -> ParityTensor:
        if isinstance(other, ParityTensor):
            self._validate_edge_compatibility(other)
            return ParityTensor(
                edges=self.edges,
                tensor=self.tensor - other.tensor,
            )
        try:
            result = self.tensor - other
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return ParityTensor(
                edges=self.edges,
                tensor=result,
            )
        return NotImplemented

    def __rsub__(self, other: typing.Any) -> ParityTensor:
        try:
            result = other - self.tensor
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return ParityTensor(
                edges=self.edges,
                tensor=result,
            )
        return NotImplemented

    def __isub__(self, other: typing.Any) -> ParityTensor:
        if isinstance(other, ParityTensor):
            self._validate_edge_compatibility(other)
            self.tensor -= other.tensor
        else:
            self.tensor -= other
        return self

    def __mul__(self, other: typing.Any) -> ParityTensor:
        if isinstance(other, ParityTensor):
            self._validate_edge_compatibility(other)
            return ParityTensor(
                edges=self.edges,
                tensor=self.tensor * other.tensor,
            )
        try:
            result = self.tensor * other
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return ParityTensor(
                edges=self.edges,
                tensor=result,
            )
        return NotImplemented

    def __rmul__(self, other: typing.Any) -> ParityTensor:
        try:
            result = other * self.tensor
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return ParityTensor(
                edges=self.edges,
                tensor=result,
            )
        return NotImplemented

    def __imul__(self, other: typing.Any) -> ParityTensor:
        if isinstance(other, ParityTensor):
            self._validate_edge_compatibility(other)
            self.tensor *= other.tensor
        else:
            self.tensor *= other
        return self

    def __truediv__(self, other: typing.Any) -> ParityTensor:
        if isinstance(other, ParityTensor):
            self._validate_edge_compatibility(other)
            return ParityTensor(
                edges=self.edges,
                tensor=self.tensor / other.tensor,
            )
        try:
            result = self.tensor / other
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return ParityTensor(
                edges=self.edges,
                tensor=result,
            )
        return NotImplemented

    def __rtruediv__(self, other: typing.Any) -> ParityTensor:
        try:
            result = other / self.tensor
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return ParityTensor(
                edges=self.edges,
                tensor=result,
            )
        return NotImplemented

    def __itruediv__(self, other: typing.Any) -> ParityTensor:
        if isinstance(other, ParityTensor):
            self._validate_edge_compatibility(other)
            self.tensor /= other.tensor
        else:
            self.tensor /= other
        return self
