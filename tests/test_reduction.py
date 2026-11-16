"""Correctness tests for the reduction SAT -> ReLU verification.

Most tests evaluate the network *exactly* (``exact=True``: integers and Fractions), so
they test the mathematics of the construction, not floating-point behaviour. Tests that
take the ``encoding`` fixture run for both the compact and the paper encoding.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

import numpy as np
import pytest

from formulas import SMALL_FORMULAS, max_sat
from relu_sat_bench.cnf import CNF, brute_force_max_satisfied, iter_assignments
from relu_sat_bench.reduction import (
    build_network,
    clause_selector,
    decode_assignment,
    decode_partial_assignment,
    encode_assignment,
    literal_index,
    reduce_cnf,
)

LAYER_NAMES = {
    "compact": ["literals", "unsatisfied", "output"],
    "paper": ["clamp_parts", "clamp", "complement", "literals", "clause_parts", "clauses", "output"],
}


def sample_points(rng: random.Random, num_vars: int, count: int = 40) -> list[list[F]]:
    """Rational points in [0, 1]^n: all 0/1 vertices (if few), the center, and random
    points mixing the critical values 0, 1/2, 1 with arbitrary sixteenths."""
    points: list[list[F]] = []
    if num_vars <= 7:
        points += [[F(int(b)) for b in bits] for bits in iter_assignments(num_vars)]
    points.append([F(1, 2)] * num_vars)
    special = [F(0), F(1, 4), F(1, 2), F(3, 4), F(1)]
    for _ in range(count):
        points.append(
            [rng.choice(special) if rng.random() < 0.5 else F(rng.randint(0, 16), 16)
             for _ in range(num_vars)]
        )
    return points


def clause_values(network, encoding: str, points) -> np.ndarray:
    """The clause values c_j = min(s_j, 1), one row per point, for either encoding."""
    layers = network.forward_all(points, exact=True)
    if encoding == "paper":
        return layers[5]
    return 1 - layers[1]  # compact: c_j = 1 - u_j


# --------------------------------------------------------------------- structure


def test_layer_shapes_and_activations(small_cnf, encoding):
    n, m = small_cnf.num_vars, small_cnf.num_clauses
    net = build_network(small_cnf, encoding)
    assert [layer.name for layer in net.layers] == LAYER_NAMES[encoding]
    widths = [layer.out_features for layer in net.layers]
    if encoding == "compact":
        assert widths == [2 * n, m, 1]
        assert net.num_relus == 2 * n + m
    else:
        assert widths == [2 * n, n, 2 * n, 2 * n, 2 * m, m, 1]
        assert net.num_relus == 7 * n + 3 * m
    assert [layer.relu for layer in net.layers] == [True] * (len(net.layers) - 1) + [False]
    assert net.input_dim == n and net.output_dim == 1


def test_weights_and_biases_are_small_integers(small_cnf, encoding):
    m = small_cnf.num_clauses
    layers = {layer.name: layer for layer in build_network(small_cnf, encoding).layers}
    for layer in layers.values():
        assert np.issubdtype(layer.weight.dtype, np.integer)
        assert np.issubdtype(layer.bias.dtype, np.integer)
    if encoding == "compact":
        assert set(np.unique(layers["literals"].weight)) <= {-2, 0, 2}
        assert set(np.unique(layers["literals"].bias)) == {-1, 1}
        assert (layers["unsatisfied"].bias == 1).all()
        assert (layers["output"].weight == -1).all() and layers["output"].bias.tolist() == [m]
    else:
        for name, layer in layers.items():
            assert set(np.unique(layer.bias)) <= {-1, 0, 1}
            if name != "clause_parts":  # only there can a repeated literal give weight > 2
                assert set(np.unique(layer.weight)) <= {-1, 0, 1, 2}


def test_literal_index():
    assert [literal_index(lit, 3) for lit in (1, 2, 3, -1, -2, -3)] == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        literal_index(4, 3)


def test_clause_selector_counts_literal_occurrences():
    selector = clause_selector(CNF(2, ((1, -2), (2, 2, -1))))
    assert selector.tolist() == [[1, 0, 0, 1], [0, 2, 1, 0]]


def test_threshold_is_m_minus_one_half(small_cnf, encoding):
    instance = reduce_cnf(small_cnf, encoding)
    assert instance.threshold == small_cnf.num_clauses - F(1, 2)
    assert instance.encoding == encoding


def test_default_encoding_is_compact():
    assert reduce_cnf(CNF(1, ((1,),))).encoding == "compact"


@pytest.mark.parametrize("cnf", [CNF(0, ()), CNF(2, ())])
def test_build_network_rejects_degenerate_formulas(cnf):
    with pytest.raises(ValueError):
        build_network(cnf)


def test_build_network_rejects_unknown_encoding():
    with pytest.raises(ValueError):
        build_network(CNF(1, ((1,),)), "tiny")


# ------------------------------------------------------------- layer semantics


def relu(t):
    return np.maximum(t, 0)


def psi(t):
    """The clamp gadget of the paper, Psi(t) = ReLU(t) - ReLU(t - 1)."""
    return relu(t) - relu(t - 1)


def clause_sums(cnf: CNF, literals: np.ndarray) -> np.ndarray:
    return np.array([sum(literals[literal_index(lit, cnf.num_vars)] for lit in clause)
                     for clause in cnf.clauses], dtype=np.float64)


def reference_layers(cnf: CNF, x: np.ndarray, encoding: str) -> list[np.ndarray]:
    """The layer outputs written directly as the formulas of the construction,
    independently of the weight matrices."""
    if encoding == "compact":
        literals = np.concatenate([relu(2 * x - 1), relu(1 - 2 * x)])
        u = relu(1 - clause_sums(cnf, literals))
        return [literals, u, np.array([cnf.num_clauses - u.sum()])]
    a = psi(x)
    pos, neg = 2 * relu(a - 0.5), 2 * relu((1 - a) - 0.5)  # the paper's literal nodes
    literals = np.concatenate([pos, neg])
    s = clause_sums(cnf, literals)
    c = psi(s)
    return [
        np.concatenate([relu(x), relu(x - 1)]),
        a,
        np.concatenate([a, 1 - a]),
        literals,
        np.concatenate([relu(s), relu(s - 1)]),
        c,
        np.array([c.sum()]),
    ]


def test_layers_match_the_formulas_of_the_construction(small_cnf, encoding):
    rng = np.random.default_rng(0)
    net = build_network(small_cnf, encoding)
    # Include points outside the box.
    for x in rng.uniform(-0.5, 1.5, size=(20, small_cnf.num_vars)):
        for got, expected in zip(net.forward_all(x), reference_layers(small_cnf, x, encoding)):
            np.testing.assert_allclose(got, expected, atol=1e-12)


def test_encodings_compute_the_same_function(small_cnf):
    """compact and paper agree exactly, inside the box and (via clamping) outside it."""
    compact, paper = build_network(small_cnf, "compact"), build_network(small_cnf, "paper")
    rng = random.Random(6)
    points = sample_points(rng, small_cnf.num_vars)
    points += [[F(rng.randint(-16, 32), 16) for _ in range(small_cnf.num_vars)] for _ in range(20)]
    assert (compact.forward(points, exact=True) == paper.forward(points, exact=True)).all()


EXAMPLE_CNF = CNF(3, ((1, -2, 3), (-1, 2, -3)))
EXAMPLE_X = [F(9, 10), F(3, 10), F(1, 2)]


def test_worked_example_paper():
    """phi = (x1 or not x2 or x3) and (not x1 or x2 or not x3) at x = (9/10, 3/10, 1/2),
    every intermediate value of the paper encoding (the walkthrough's example)."""
    layers = [list(h) for h in build_network(EXAMPLE_CNF, "paper").forward_all(EXAMPLE_X, exact=True)]
    assert layers[0] == [F(9, 10), F(3, 10), F(1, 2), 0, 0, 0]                # ReLU(x), ReLU(x-1)
    assert layers[1] == [F(9, 10), F(3, 10), F(1, 2)]                         # a = Psi(x)
    assert layers[2] == [F(9, 10), F(3, 10), F(1, 2), F(1, 10), F(7, 10), F(1, 2)]  # a, 1-a
    assert layers[3] == [F(4, 5), 0, 0, 0, F(2, 5), 0]                        # pos, neg
    assert layers[4] == [F(6, 5), 0, F(1, 5), 0]                              # ReLU(s), ReLU(s-1)
    assert layers[5] == [1, 0]                                                # clause values
    assert layers[6] == [1]                                                   # N(x)


def test_worked_example_compact():
    """The same example in the compact encoding: same literals, u = 1 - min(s, 1)."""
    layers = [list(h) for h in build_network(EXAMPLE_CNF, "compact").forward_all(EXAMPLE_X, exact=True)]
    assert layers[0] == [F(4, 5), 0, 0, 0, F(2, 5), 0]  # pos, neg
    assert layers[1] == [0, 1]                          # u = ReLU(1 - s), s = (6/5, 0)
    assert layers[2] == [1]                             # N(x) = 2 - (0 + 1)


def test_worked_example_counterexample(encoding):
    assert decode_partial_assignment(EXAMPLE_X) == [True, False, None]
    instance = reduce_cnf(EXAMPLE_CNF, encoding)
    assert not instance.is_counterexample(EXAMPLE_X)
    # Moving x3 to 0 makes the second clause true: a counterexample, N = m = 2.
    moved = [F(9, 10), F(3, 10), 0]
    assert instance.is_counterexample(moved)
    assert decode_assignment(moved) == [True, False, False]


# ------------------------------------------------------ the correctness argument


def test_boolean_points_count_satisfied_clauses(small_cnf, encoding):
    """On a 0/1 point the output is exactly the number of satisfied clauses."""
    net = build_network(small_cnf, encoding)
    assignments = list(iter_assignments(small_cnf.num_vars))
    outputs = net.forward([encode_assignment(a) for a in assignments], exact=True)
    for assignment, output in zip(assignments, outputs[:, 0]):
        assert output == small_cnf.count_satisfied(assignment)


def test_clause_values_detect_the_induced_partial_assignment(small_cnf, encoding):
    """0 <= c_j <= 1, and c_j > 0 iff the partial assignment of x satisfies clause j."""
    net = build_network(small_cnf, encoding)
    points = sample_points(random.Random(1), small_cnf.num_vars)
    for x, values in zip(points, clause_values(net, encoding, points)):
        satisfied = small_cnf.satisfied_clauses(decode_partial_assignment(x))
        for value, sat in zip(values, satisfied):
            assert 0 <= value <= 1
            assert (value > 0) == sat


def test_output_never_exceeds_maxsat(small_cnf, encoding):
    """max over all inputs of N(x) equals MaxSAT(phi)."""
    net = build_network(small_cnf, encoding)
    best = max_sat(small_cnf)
    outputs = net.forward(sample_points(random.Random(2), small_cnf.num_vars), exact=True)[:, 0]
    assert max(outputs) == best  # attained at a 0/1 vertex (all of them are sampled)
    assert all(out <= best for out in outputs)


def test_reduction_is_correct(small_cnf, encoding):
    """Main theorem: a counterexample exists iff the formula is satisfiable."""
    instance = reduce_cnf(small_cnf, encoding)
    best, optimal = brute_force_max_satisfied(small_cnf)
    if best == small_cnf.num_clauses:
        witness = encode_assignment(optimal)
        assert instance.network.forward(witness, exact=True)[0] == small_cnf.num_clauses
        assert instance.is_counterexample(witness)
    else:
        points = sample_points(random.Random(3), small_cnf.num_vars)
        outputs = instance.network.forward(points, exact=True)[:, 0]
        assert all(out <= small_cnf.num_clauses - 1 for out in outputs)
        assert all(out < instance.threshold for out in outputs)  # no counterexample


def test_counterexamples_decode_to_satisfying_assignments(small_cnf, encoding):
    """Every counterexample, also a non-0/1 one, yields a satisfying assignment."""
    instance = reduce_cnf(small_cnf, encoding)
    rng = random.Random(4)
    points = sample_points(rng, small_cnf.num_vars)
    best, optimal = brute_force_max_satisfied(small_cnf)
    if best == small_cnf.num_clauses:
        # Points near the satisfying vertex: move every coordinate up to 1/4 inwards.
        for _ in range(20):
            points.append([F(rng.randint(0, 4), 16) if not v else 1 - F(rng.randint(0, 4), 16)
                           for v in optimal])
    found = 0
    for x in points:
        if instance.is_counterexample(x):
            found += 1
            assert small_cnf.is_satisfied_by(decode_assignment(x))
    assert (found > 0) == (best == small_cnf.num_clauses)


def test_center_point_gives_zero(small_cnf, encoding):
    """At x = (1/2, ..., 1/2) no literal is true, so N(x) = 0 (the paper's witness)."""
    net = build_network(small_cnf, encoding)
    assert net.forward([F(1, 2)] * small_cnf.num_vars, exact=True)[0] == 0


def test_points_outside_the_box_act_like_clamped_points_but_are_not_counterexamples(encoding):
    instance = reduce_cnf(CNF(2, ((1, 2), (-1, 2))), encoding)
    outside, clamped = [F(-3), F(5, 2)], [F(0), F(1)]
    assert (instance.network.forward(outside, exact=True)
            == instance.network.forward(clamped, exact=True)).all()
    assert instance.is_counterexample(clamped)
    assert not instance.is_counterexample(outside)


def test_encode_decode_round_trip():
    for assignment in iter_assignments(4):
        assert decode_assignment(encode_assignment(assignment)) == list(assignment)


def test_float_evaluation_agrees_with_exact_evaluation(small_cnf, encoding):
    net = build_network(small_cnf, encoding)
    for x in sample_points(random.Random(5), small_cnf.num_vars, count=10):
        exact = net.forward(x, exact=True)[0]
        approx = net.forward([float(v) for v in x])[0]
        assert abs(float(exact) - approx) < 1e-12


def test_collection_contains_satisfiable_and_unsatisfiable_formulas():
    """Guards against a vacuous test suite (e.g. only satisfiable formulas)."""
    results = {max_sat(cnf) == cnf.num_clauses for _, cnf in SMALL_FORMULAS}
    assert results == {True, False}
