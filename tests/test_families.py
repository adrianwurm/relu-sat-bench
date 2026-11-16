"""Tests for the formula families. They need no SAT solver: small instances are checked by
brute force, larger ones through their witnesses and structural counts."""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from relu_sat_bench.cnf import CNF, brute_force_max_satisfied
from relu_sat_bench.families import (
    Q_HIDDEN,
    GeneratedFormula,
    num_clauses_for_ratio,
    ordering_principle,
    pigeonhole,
    planted_ksat,
    random_ksat,
    random_regular_graph,
    tseitin,
)


def is_satisfiable(cnf: CNF) -> bool:
    return brute_force_max_satisfied(cnf)[0] == cnf.num_clauses


# ------------------------------------------------------------------ random k-SAT


def test_random_ksat_shape():
    formula = random_ksat(random.Random(0), num_vars=20, ratio=4.26)
    assert formula.family == "random" and formula.satisfiable is None and formula.witness is None
    assert formula.cnf.num_vars == 20
    assert formula.cnf.num_clauses == 85  # round(4.26 * 20)
    for clause in formula.cnf.clauses:
        assert len(clause) == 3 and len({abs(lit) for lit in clause}) == 3


def test_random_ksat_is_reproducible_from_the_seed():
    a = random_ksat(random.Random("seed-1"), 10, 4.0).cnf
    b = random_ksat(random.Random("seed-1"), 10, 4.0).cnf
    c = random_ksat(random.Random("seed-2"), 10, 4.0).cnf
    assert a == b and a != c


def test_num_clauses_for_ratio():
    assert num_clauses_for_ratio(100, 4.26) == 426
    assert num_clauses_for_ratio(3, 0.1) == 1


def test_random_ksat_shows_the_phase_transition():
    """Few clauses: almost always satisfiable; many: almost never (n = 12, brute force)."""
    rng = random.Random(1)
    low = sum(is_satisfiable(random_ksat(rng, 12, 3.0).cnf) for _ in range(20))
    high = sum(is_satisfiable(random_ksat(rng, 12, 7.0).cnf) for _ in range(20))
    assert low >= 15 and high <= 5


# ----------------------------------------------------------------------- planted


@pytest.mark.parametrize("mode", ["naive", "q-hidden"])
def test_planted_formulas_are_satisfied_by_their_witness(mode):
    for seed in range(20):
        formula = planted_ksat(random.Random(seed), num_vars=15, ratio=6.0, mode=mode)
        assert formula.satisfiable is True
        assert formula.cnf.num_clauses == 90
        assert formula.cnf.is_satisfied_by(formula.witness)


def test_q_hidden_constant_balances_true_literals():
    """With acceptance weights Q^t, the expected share of true literals is exactly 1/2."""
    weights = {t: math.comb(3, t) * Q_HIDDEN**t for t in (1, 2, 3)}
    expected_true = sum(t * w for t, w in weights.items()) / sum(weights.values())
    assert expected_true / 3 == pytest.approx(0.5)


@pytest.mark.parametrize("mode, expected_share", [("naive", 4 / 7), ("q-hidden", 1 / 2)])
def test_planting_literal_statistics(mode, expected_share):
    """Share of literals that agree with the hidden solution: 4/7 leaks it, 1/2 does not."""
    formula = planted_ksat(random.Random(3), num_vars=200, ratio=50.0, mode=mode)  # 10000 clauses
    hidden = formula.witness
    literals = [lit for clause in formula.cnf.clauses for lit in clause]
    share = sum(CNF.literal_is_true(lit, hidden) for lit in literals) / len(literals)
    assert share == pytest.approx(expected_share, abs=0.01)


def test_planting_rejects_unknown_mode():
    with pytest.raises(ValueError):
        planted_ksat(random.Random(0), 10, 4.0, mode="loud")


# -------------------------------------------------------------------- pigeonhole


@pytest.mark.parametrize("holes", [1, 2, 3])
def test_pigeonhole_is_unsatisfiable(holes):
    formula = pigeonhole(holes)
    assert formula.satisfiable is False
    assert not is_satisfiable(formula.cnf)


def test_pigeonhole_counts():
    formula = pigeonhole(5)  # 6 pigeons, 5 holes
    assert formula.cnf.num_vars == 30
    assert formula.cnf.num_clauses == 6 + 5 * math.comb(6, 2)


def test_pigeonhole_without_one_pigeon_is_satisfiable():
    """Sanity check: dropping the clause of one pigeon makes it fit."""
    cnf = pigeonhole(3).cnf
    assert is_satisfiable(CNF(cnf.num_vars, cnf.clauses[1:]))


# ----------------------------------------------------------------------- Tseitin


@pytest.mark.parametrize("num_vertices", [4, 6, 10, 20])
def test_random_regular_graph(num_vertices):
    edges = random_regular_graph(random.Random(num_vertices), num_vertices, 3)
    degrees = Counter(v for edge in edges for v in edge)
    assert all(degrees[v] == 3 for v in range(num_vertices))
    assert len(set(edges)) == len(edges) == 3 * num_vertices // 2
    assert all(a != b for a, b in edges)


def test_random_regular_graph_rejects_impossible_sizes():
    with pytest.raises(ValueError):
        random_regular_graph(random.Random(0), 5, 3)  # odd number of half-edges


def test_tseitin_shape():
    formula = tseitin(random.Random(0), 10)
    assert formula.cnf.num_vars == 15  # edges of a 3-regular graph on 10 vertices
    assert formula.cnf.num_clauses == 40  # 4 clauses per vertex
    assert all(len(clause) == 3 for clause in formula.cnf.clauses)


@pytest.mark.parametrize("num_vertices", [4, 6, 8])
def test_tseitin_odd_is_unsatisfiable(num_vertices):
    for seed in range(3):
        formula = tseitin(random.Random(seed), num_vertices, parity="odd")
        assert formula.satisfiable is False
        assert not is_satisfiable(formula.cnf)


@pytest.mark.parametrize("num_vertices", [4, 10, 30, 100])
def test_tseitin_even_has_a_valid_witness(num_vertices):
    for seed in range(5):
        formula = tseitin(random.Random(seed), num_vertices, parity="even")
        assert formula.satisfiable is True
        assert formula.cnf.is_satisfied_by(formula.witness)


# ------------------------------------------------------------ ordering principle


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_ordering_principle_is_unsatisfiable(size):
    formula = ordering_principle(size)
    assert formula.satisfiable is False
    assert not is_satisfiable(formula.cnf)


def test_ordering_principle_counts():
    formula = ordering_principle(6)
    assert formula.cnf.num_vars == math.comb(6, 2)
    assert formula.cnf.num_clauses == 2 * math.comb(6, 3) + 6


def test_ordering_principle_without_one_minimum_clause_is_satisfiable():
    """Sanity check: allow element 0 to be the minimum and a total order exists."""
    cnf = ordering_principle(5).cnf
    # The last 5 clauses say "element j has a predecessor", for j = 0..4; drop the one for j = 0.
    without = cnf.clauses[:-5] + cnf.clauses[-4:]
    assert is_satisfiable(CNF(cnf.num_vars, without))


# ------------------------------------------------------------------ GeneratedFormula


def test_generated_formula_rejects_a_wrong_witness():
    cnf = CNF(1, ((1,),))
    with pytest.raises(ValueError):
        GeneratedFormula("x", {}, cnf, satisfiable=True, witness=(False,))
    with pytest.raises(ValueError):
        GeneratedFormula("x", {}, cnf, satisfiable=False, witness=(True,))


def test_all_generators_produce_valid_cnfs():
    rng = random.Random(0)
    formulas = [
        random_ksat(rng, 8, 4.0),
        planted_ksat(rng, 8, 4.0),
        pigeonhole(4),
        tseitin(rng, 8),
        tseitin(rng, 8, parity="even"),
        ordering_principle(5),
    ]
    for formula in formulas:
        # round trip through DIMACS (the CNF constructor validates every literal)
        assert CNF.from_dimacs(formula.cnf.to_dimacs()) == formula.cnf
        assert all(len(clause) > 0 for clause in formula.cnf.clauses)
    assert len({f.family for f in formulas}) == 5
