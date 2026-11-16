"""A minimal, framework-independent feed-forward ReLU network.

The reduction only produces integer weights and biases. Keeping the network as plain
integer matrices (instead of building it in PyTorch right away) has two advantages:

* It can be evaluated *exactly* on rational inputs (``fractions.Fraction``), so the
  correctness tests check the mathematics, not floating-point luck.
* The project needs no deep-learning framework. Export formats (ONNX for the
  verification tools) are thin adapters that copy these matrices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class AffineLayer:
    """``y = W x + b``, optionally followed by ReLU. ``weight`` has shape (out, in)."""

    name: str
    weight: np.ndarray
    bias: np.ndarray
    relu: bool

    def __post_init__(self) -> None:
        if self.weight.ndim != 2:
            raise ValueError(f"{self.name}: weight must be 2-D, got shape {self.weight.shape}")
        if self.bias.shape != (self.weight.shape[0],):
            raise ValueError(
                f"{self.name}: bias shape {self.bias.shape} does not match "
                f"weight shape {self.weight.shape}"
            )

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]


@dataclass(frozen=True)
class ReluNetwork:
    """A sequence of affine layers; consecutive layer sizes must match."""

    layers: tuple[AffineLayer, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("a network needs at least one layer")
        for prev, nxt in zip(self.layers, self.layers[1:]):
            if prev.out_features != nxt.in_features:
                raise ValueError(
                    f"layer {prev.name!r} outputs {prev.out_features} values, "
                    f"but {nxt.name!r} expects {nxt.in_features}"
                )

    @property
    def input_dim(self) -> int:
        return self.layers[0].in_features

    @property
    def output_dim(self) -> int:
        return self.layers[-1].out_features

    @property
    def num_relus(self) -> int:
        return sum(layer.out_features for layer in self.layers if layer.relu)

    def forward_all(self, x: Sequence | np.ndarray, *, exact: bool = False) -> list[np.ndarray]:
        """Outputs of every layer (after its activation), in order.

        ``x`` has shape ``(input_dim,)`` or ``(batch, input_dim)``. With ``exact=True``
        the computation uses Python integers and ``Fraction``s (no rounding at all);
        otherwise it uses float64.
        """
        if exact:
            h = np.array([[Fraction(v) for v in row] for row in np.atleast_2d(x)], dtype=object)
        else:
            h = np.atleast_2d(np.asarray(x, dtype=np.float64))
        single = np.ndim(x) == 1
        if h.shape[1] != self.input_dim:
            raise ValueError(f"expected input dimension {self.input_dim}, got {h.shape[1]}")

        outputs = []
        for layer in self.layers:
            if exact:
                weight, bias = layer.weight.astype(object), layer.bias.astype(object)
            else:
                weight, bias = layer.weight.astype(np.float64), layer.bias.astype(np.float64)
            h = h @ weight.T + bias
            if layer.relu:
                h = np.where(h > 0, h, 0) if exact else np.maximum(h, 0.0)
            outputs.append(h[0] if single else h)
        return outputs

    def forward(self, x: Sequence | np.ndarray, *, exact: bool = False) -> np.ndarray:
        """Network output; see :meth:`forward_all` for shapes and ``exact``."""
        return self.forward_all(x, exact=exact)[-1]
