"""Tests for the SAT-solver wrapper. Skipped if python-sat is not installed."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("pysat")

from formulas import SMALL_FORMULAS, max_sat  # noqa: E402
from relu_sat_bench.cnf import CNF  # noqa: E402
from relu_sat_bench.families import pigeonhole  # noqa: E402
from relu_sat_bench.solve import (  # noqa: E402
    SolverDisagreement,
    _to_assignment,
    cross_check_unsat,
    solve,
)

NONEMPTY = [(name, cnf) for name, cnf in SMALL_FORMULAS if all(cnf.clauses)]


@pytest.mark.parametrize("cnf", [cnf for _, cnf in NONEMPTY], ids=[name for name, _ in NONEMPTY])
@pytest.mark.parametrize("solver", ["cadical195", "glucose4"])
def test_solver_agrees_with_brute_force(cnf, solver):
    result = solve(cnf, solver=solver)
    assert result.satisfiable == (max_sat(cnf) == cnf.num_clauses)
    if result.satisfiable:
        assert cnf.is_satisfied_by(result.model)
    else:
        assert result.model is None


@pytest.mark.parametrize("cnf", [cnf for _, cnf in NONEMPTY[:5]], ids=[name for name, _ in NONEMPTY[:5]])
def test_solve_in_a_child_process_gives_the_same_answer(cnf):
    assert solve(cnf, timeout=30).satisfiable == solve(cnf).satisfiable


def test_timeout_stops_a_solver_that_cannot_be_interrupted():
    """CaDiCaL ignores interrupts; the child process is terminated instead."""
    start = time.perf_counter()
    result = solve(pigeonhole(11).cnf, solver="cadical195", timeout=1.0)
    assert result.satisfiable is None and result.status == "timeout"
    assert time.perf_counter() - start < 15  # 1 s + process start-up, generously


def test_cross_check_confirms_unsatisfiable_formulas():
    assert cross_check_unsat(pigeonhole(3).cnf) == "confirmed"


def test_cross_check_raises_on_a_satisfiable_formula():
    with pytest.raises(SolverDisagreement):
        cross_check_unsat(CNF(2, ((1, 2),)))


def test_variables_missing_from_the_model_become_false():
    assert _to_assignment([1, -2], 4) == (True, False, False, False)


def test_unknown_solver_fails_loudly():
    with pytest.raises(Exception):
        solve(CNF(1, ((1,),)), solver="no-such-solver")
    with pytest.raises(RuntimeError):
        solve(CNF(1, ((1,),)), solver="no-such-solver", timeout=30)
