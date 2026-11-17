r"""Reduction from SAT to ReLU network verification.

Based on the construction in A. Wurm, *Robustness Verification in Neural Networks*,
CPAIOR 2024 (arXiv:2403.13441), proof of Theorem 2 ii). The complexity framework
(NNReach and its NP-completeness for ReLU networks) is from A. Wurm, *Complexity of
Reachability Problems in Neural Networks*, RP 2023 (arXiv:2306.05818).

Notation: the formula has ``n`` variables and ``m`` clauses (the paper calls the
number of clauses ``n``). ``s_j`` is the sum of the literal nodes of clause ``j``, and
``Psi(t) = ReLU(t) - ReLU(t - 1)`` clamps ``t`` to ``[0, 1]``.

Two encodings compute the same function on the input box ``[0, 1]^n``:

**compact** (default; ``2n + m`` ReLUs, 2 hidden layers)

=====  ===========  =====  ==========================================================
layer  name         width  computes
=====  ===========  =====  ==========================================================
1      literals     2n     ``pos_i = ReLU(2 x_i - 1)``, ``neg_i = ReLU(1 - 2 x_i)``
2      unsatisfied  m      ``u_j = ReLU(1 - s_j)``, how far clause ``j`` is from true
3      output       1      ``N(x) = m - (u_1 + ... + u_m)`` (linear)
=====  ===========  =====  ==========================================================

**paper** (``7n + 3m`` ReLUs, 6 hidden layers; the construction exactly as written in
the paper, for reference and for comparing encodings)

=====  ==============  =====  ==================================================
layer  name            width  computes
=====  ==============  =====  ==================================================
1      clamp_parts     2n     ``ReLU(x_i)`` and ``ReLU(x_i - 1)``
2      clamp           n      ``a_i = ReLU(ReLU(x_i) - ReLU(x_i - 1)) = Psi(x_i)``
3      complement      2n     ``ReLU(a_i) = a_i`` and ``ReLU(1 - a_i) = 1 - a_i``
4      literals        2n     ``pos_i = ReLU(2 a_i - 1)``, ``neg_i = ReLU(2 (1 - a_i) - 1)``
5      clause_parts    2m     ``ReLU(s_j)`` and ``ReLU(s_j - 1)``
6      clauses         m      ``c_j = Psi(s_j)``
7      output          1      ``N(x) = c_1 + ... + c_m`` (linear)
=====  ==============  =====  ==================================================

In the paper the literal node is ``2 ReLU(a_i - 1/2)``; the factor 2 is moved inside
the ReLU (``ReLU(2t) = 2 ReLU(t)``) so that all weights and biases are integers. The
paper's output node may also apply ReLU; the sum is non-negative, so that changes
nothing, and the output is kept linear as is usual for VNN-COMP benchmarks.

Why the encodings agree. All literal nodes are ``>= 0``, hence ``s_j >= 0`` and
``Psi(s_j) = min(s_j, 1) = 1 - ReLU(1 - s_j)``. On ``[0, 1]`` the paper's layers 1-3
compute ``a_i = x_i`` and ``1 - x_i``, so both encodings have the same literal nodes
and ``N_paper = sum_j min(s_j, 1) = N_compact``. Outside the box both equal ``N`` at
the clamped point (a literal that is ``>= 1`` contributes ``min(., 1) = 1`` either way).

Why it works. At most one of ``pos_i``, ``neg_i`` is non-zero, so every input ``x``
induces a partial assignment: ``x_i`` true if ``x_i > 1/2``, false if ``x_i < 1/2``,
unassigned if ``x_i = 1/2`` (after clamping to ``[0, 1]``). A literal node is non-zero
iff its literal is true under this partial assignment, hence the clause value
``c_j = min(s_j, 1)`` lies in ``[0, 1]`` and is ``> 0`` iff the partial assignment
satisfies clause ``j``. Therefore

* ``N(x)`` is at most the number of clauses satisfied by the partial assignment of
  ``x``, so ``max N = MaxSAT(phi)`` over all of ``R^n``, attained at a 0/1 point;
* on a 0/1 point, ``N(x)`` is *exactly* the number of satisfied clauses.

In particular ``max N = m`` if the formula is satisfiable and ``max N <= m - 1`` if not.

The verification instance asks: does ``N(x) < m - 1/2`` hold for all ``x`` in
``[0, 1]^n``? It is *violated* iff the formula is *satisfiable*, and every
counterexample decodes into a satisfying assignment (:func:`decode_assignment`).
The threshold sits in the middle of the gap, ``1/2`` away from both possible values.

**neurocodebench** (``m + 2n`` ReLUs, 1 hidden layer, 2 outputs): the construction of the
VNN-COMP 2025 benchmark ``sat_relu`` (NeuroCodeBench 2.0; Manino et al., arXiv:2510.23389),
rebuilt for the comparison of encodings. With ``P_j`` and ``N_j`` the variables that occur
positively and negatively in clause ``j``:

=====  ======  ======  ===============================================================
layer  name    width   computes
=====  ======  ======  ===============================================================
1      hidden  m + 2n  ``v_j = ReLU(sum_{N_j} x - sum_{P_j} x + 1 - |N_j|)``, then
                       ``ReLU(x_i)``, then ``ReLU(2 x_i - 1)``
2      output  2       ``Y_0 = 1 - (v_1 + ... + v_m)``,
                       ``Y_1 = sum_i (ReLU(x_i) - ReLU(2 x_i - 1))`` (linear)
=====  ======  ======  ===============================================================

At a 0/1 point ``v_j`` is 1 if clause ``j`` is falsified and 0 otherwise. On ``[0, 1]``,
``Y_1 = sum_i min(x_i, 1 - x_i)`` is 0 exactly at the 0/1 points. The unsafe set is
``Y_0 >= 1 and Y_1 <= 0``: it holds exactly at the 0/1 points that satisfy every clause, so
it is reachable iff the formula is satisfiable, but without any margin (a counterexample must
be an exact 0/1 point). The neuron order is the one of the published ONNX files.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from relu_sat_bench.cnf import CNF
from relu_sat_bench.network import AffineLayer, ReluNetwork

INPUT_LOWER = Fraction(0)
INPUT_UPPER = Fraction(1)
ENCODINGS = ("compact", "paper")       # one output N(x) = satisfied clauses, unsafe set N(x) >= m - 1/2
NEUROCODEBENCH = "neurocodebench"      # two outputs, unsafe set Y_0 >= 1 and Y_1 <= 0 (see above)
ALL_ENCODINGS = (*ENCODINGS, NEUROCODEBENCH)
DEFAULT_ENCODING = "compact"


def literal_index(lit: int, num_vars: int) -> int:
    """Position of a literal's node in the literal layer.

    The first ``n`` nodes are the positive literals ``x_1..x_n``, the next ``n`` the
    negative literals ``not x_1 .. not x_n``.
    """
    if lit == 0 or abs(lit) > num_vars:
        raise ValueError(f"literal {lit} is not in ±1..±{num_vars}")
    return lit - 1 if lit > 0 else num_vars + (-lit) - 1


def clause_selector(cnf: CNF) -> np.ndarray:
    """Matrix ``S`` of shape (m, 2n) with ``(S @ literals)[j] = s_j``.

    ``S[j, k]`` counts how often literal node ``k`` occurs in clause ``j`` (normally 0
    or 1; a literal repeated inside a clause simply gets weight 2).
    """
    n, m = cnf.num_vars, cnf.num_clauses
    selector = np.zeros((m, 2 * n), dtype=np.int64)
    for j, clause in enumerate(cnf.clauses):
        for lit in clause:
            selector[j, literal_index(lit, n)] += 1
    return selector


def _build_compact(cnf: CNF) -> tuple[AffineLayer, ...]:
    n, m = cnf.num_vars, cnf.num_clauses
    eye_n = np.eye(n, dtype=np.int64)
    ones_n = np.ones(n, dtype=np.int64)
    return (
        # 1: [pos, neg] = [ReLU(2x - 1), ReLU(1 - 2x)]
        AffineLayer("literals", np.vstack([2 * eye_n, -2 * eye_n]), np.concatenate([-ones_n, ones_n]), relu=True),
        # 2: u = ReLU(1 - s) with s = selector @ [pos, neg]
        AffineLayer("unsatisfied", -clause_selector(cnf), np.ones(m, dtype=np.int64), relu=True),
        # 3: N(x) = m - sum_j u_j
        AffineLayer("output", -np.ones((1, m), dtype=np.int64), np.array([m], dtype=np.int64), relu=False),
    )


def _build_paper(cnf: CNF) -> tuple[AffineLayer, ...]:
    n, m = cnf.num_vars, cnf.num_clauses
    eye_n = np.eye(n, dtype=np.int64)
    eye_m = np.eye(m, dtype=np.int64)
    zeros_n = np.zeros(n, dtype=np.int64)
    zeros_m = np.zeros(m, dtype=np.int64)
    ones_n = np.ones(n, dtype=np.int64)
    ones_m = np.ones(m, dtype=np.int64)
    selector = clause_selector(cnf)
    return (
        # 1: [ReLU(x), ReLU(x - 1)]
        AffineLayer("clamp_parts", np.vstack([eye_n, eye_n]), np.concatenate([zeros_n, -ones_n]), relu=True),
        # 2: a = ReLU(ReLU(x) - ReLU(x - 1)) = Psi(x)
        AffineLayer("clamp", np.hstack([eye_n, -eye_n]), zeros_n, relu=True),
        # 3: [a, 1 - a]
        AffineLayer("complement", np.vstack([eye_n, -eye_n]), np.concatenate([zeros_n, ones_n]), relu=True),
        # 4: [pos, neg] = ReLU(2 * [a, 1 - a] - 1)
        AffineLayer("literals", 2 * np.eye(2 * n, dtype=np.int64), -np.ones(2 * n, dtype=np.int64), relu=True),
        # 5: [ReLU(s), ReLU(s - 1)] with s = selector @ [pos, neg]
        AffineLayer("clause_parts", np.vstack([selector, selector]), np.concatenate([zeros_m, -ones_m]), relu=True),
        # 6: c = ReLU(ReLU(s) - ReLU(s - 1)) = Psi(s)
        AffineLayer("clauses", np.hstack([eye_m, -eye_m]), zeros_m, relu=True),
        # 7: N(x) = sum_j c_j
        AffineLayer("output", np.ones((1, m), dtype=np.int64), np.zeros(1, dtype=np.int64), relu=False),
    )


def _build_neurocodebench(cnf: CNF) -> tuple[AffineLayer, ...]:
    n, m = cnf.num_vars, cnf.num_clauses
    clause_weight = np.zeros((m, n), dtype=np.int64)
    clause_bias = np.ones(m, dtype=np.int64)
    for j, clause in enumerate(cnf.clauses):
        for lit in set(clause):                    # P_j and N_j are sets: a repeated literal counts once
            if lit > 0:
                clause_weight[j, lit - 1] -= 1     # positive literal: - x
            else:
                clause_weight[j, -lit - 1] += 1    # negative literal: + x, and 1 - |N_j| in the bias
                clause_bias[j] -= 1
    eye_n = np.eye(n, dtype=np.int64)
    zeros_n = np.zeros(n, dtype=np.int64)
    output = np.zeros((2, m + 2 * n), dtype=np.int64)
    output[0, :m] = -1          # Y_0 = 1 - sum_j v_j
    output[1, m:m + n] = 1      # Y_1 = sum_i ReLU(x_i) - ReLU(2 x_i - 1)
    output[1, m + n:] = -1
    return (
        AffineLayer("hidden", np.vstack([clause_weight, eye_n, 2 * eye_n]),
                    np.concatenate([clause_bias, zeros_n, -np.ones(n, dtype=np.int64)]), relu=True),
        AffineLayer("output", output, np.array([1, 0], dtype=np.int64), relu=False),
    )


def build_network(cnf: CNF, encoding: str = DEFAULT_ENCODING) -> ReluNetwork:
    """The ReLU network ``N`` of the reduction; see the module docstring for the layers."""
    if encoding not in ALL_ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; choose one of {ALL_ENCODINGS}")
    if cnf.num_vars == 0 or cnf.num_clauses == 0:
        raise ValueError("the formula needs at least one variable and one clause")
    builders = {"compact": _build_compact, "paper": _build_paper, NEUROCODEBENCH: _build_neurocodebench}
    return ReluNetwork(builders[encoding](cnf))


@dataclass(frozen=True)
class OutputConstraint:
    """One constraint of the unsafe set: ``Y_output >= rhs`` or ``Y_output <= rhs``."""

    output: int
    op: str
    rhs: Fraction

    def holds(self, value, tolerance: float = 0.0) -> bool:
        if self.op == ">=":
            return value >= self.rhs - tolerance
        if self.op == "<=":
            return value <= self.rhs + tolerance
        raise ValueError(f"unknown operator {self.op!r}")


@dataclass(frozen=True)
class VerificationInstance:
    """Network, input box ``[0, 1]^n`` and the property.

    In VNN-LIB terms the *unsafe* set is ``{x in [0, 1]^n : N(x) >= threshold}`` for the
    encodings in :data:`ENCODINGS`, and ``Y_0 >= 1 and Y_1 <= 0`` for neurocodebench
    (``threshold`` is then the right-hand side of the ``Y_0`` constraint, 1). A verifier
    should answer "sat" (counterexample exists) iff the formula is satisfiable.
    """

    cnf: CNF
    network: ReluNetwork
    threshold: Fraction
    encoding: str

    @property
    def unsafe(self) -> tuple[OutputConstraint, ...]:
        """The constraints of the unsafe set; all of them must hold for a counterexample."""
        if self.encoding == NEUROCODEBENCH:
            return (OutputConstraint(0, ">=", Fraction(1)), OutputConstraint(1, "<=", Fraction(0)))
        return (OutputConstraint(0, ">=", self.threshold),)

    @property
    def num_inputs(self) -> int:
        return self.network.input_dim

    @property
    def input_lower(self) -> Fraction:
        return INPUT_LOWER

    @property
    def input_upper(self) -> Fraction:
        return INPUT_UPPER

    def is_counterexample(self, x: Sequence, *, exact: bool = True) -> bool:
        """Whether ``x`` lies in the input box and violates the property."""
        values = [Fraction(v) for v in x] if exact else [float(v) for v in x]
        if len(values) != self.num_inputs:
            raise ValueError(f"expected {self.num_inputs} inputs, got {len(values)}")
        if any(v < INPUT_LOWER or v > INPUT_UPPER for v in values):
            return False
        outputs = self.network.forward(values, exact=exact)
        return all(c.holds(outputs[c.output]) for c in self.unsafe)


def reduce_cnf(cnf: CNF, encoding: str = DEFAULT_ENCODING) -> VerificationInstance:
    """Map a CNF formula to a verification instance that is violated iff it is satisfiable."""
    threshold = Fraction(1) if encoding == NEUROCODEBENCH else Fraction(cnf.num_clauses) - Fraction(1, 2)
    return VerificationInstance(cnf=cnf, network=build_network(cnf, encoding), threshold=threshold,
                                encoding=encoding)


def encode_assignment(assignment: Sequence[bool]) -> np.ndarray:
    """The 0/1 input point of a total assignment (true -> 1, false -> 0)."""
    return np.array([1 if value else 0 for value in assignment], dtype=np.int64)


def decode_partial_assignment(x: Sequence) -> list[bool | None]:
    """The partial assignment induced by an input point (see the module docstring).

    ``x_i`` is clamped to ``[0, 1]`` first; then values above 1/2 mean true, below 1/2
    false, and exactly 1/2 unassigned (``None``).
    """
    half = Fraction(1, 2)
    result: list[bool | None] = []
    for value in x:
        clamped = min(max(Fraction(value), Fraction(0)), Fraction(1))
        result.append(None if clamped == half else clamped > half)
    return result


def decode_assignment(x: Sequence) -> list[bool]:
    """A total assignment from an input point; unassigned variables become false.

    If ``x`` is a counterexample (``N(x) >= m - 1/2``), the result satisfies the formula.
    """
    return [bool(value) for value in decode_partial_assignment(x)]
