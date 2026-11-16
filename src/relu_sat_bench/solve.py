"""Reference answers from SAT solvers (via the ``python-sat`` package).

``python-sat`` wraps compiled solvers such as CaDiCaL, Glucose and MiniSat. Not all of
them can be interrupted from Python (CaDiCaL cannot), so a solve with a timeout runs in a
separate process that is terminated when the time is up. This works the same for every
solver and on every operating system.

Every answer is checked:

* "satisfiable" comes with a model, which :func:`solve` verifies against the formula;
* "unsatisfiable" can be confirmed by a second, independent solver
  (:func:`cross_check_unsat`).
"""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass

from relu_sat_bench.cnf import CNF

DEFAULT_SOLVER = "cadical195"
CROSS_CHECK_SOLVER = "glucose4"


class SolverDisagreement(RuntimeError):
    """Two sources (solvers, or a solver and the construction) disagree: a bug somewhere."""


@dataclass(frozen=True)
class SolveResult:
    """``satisfiable`` is True/False, or None if the solver timed out."""

    solver: str
    satisfiable: bool | None
    model: tuple[bool, ...] | None
    seconds: float

    @property
    def status(self) -> str:
        return {True: "sat", False: "unsat", None: "timeout"}[self.satisfiable]


def _run_solver(clauses, solver: str) -> tuple[bool, list[int] | None, float]:
    from pysat.solvers import Solver  # imported here so the module loads without python-sat

    start = time.perf_counter()
    with Solver(name=solver, bootstrap_with=clauses) as sat_solver:
        satisfiable = sat_solver.solve()
        model = sat_solver.get_model() if satisfiable else None
    return satisfiable, model, time.perf_counter() - start


def _worker(clauses, solver: str, connection) -> None:
    connection.send(_run_solver(clauses, solver))
    connection.close()


def _to_assignment(model: list[int], num_vars: int) -> tuple[bool, ...]:
    """pysat models list +v / -v; variables in no clause may be missing (-> False)."""
    values = [False] * num_vars
    for lit in model:
        if abs(lit) <= num_vars:
            values[abs(lit) - 1] = lit > 0
    return tuple(values)


def solve(cnf: CNF, solver: str = DEFAULT_SOLVER, timeout: float | None = None) -> SolveResult:
    """Decide the formula. With ``timeout`` (seconds) the solver runs in a child process.

    ``seconds`` is the time the solver itself needed (process start-up not included); on
    a timeout it is the timeout. Raises :class:`SolverDisagreement` if the solver returns
    a model that does not satisfy the formula.
    """
    clauses = [list(clause) for clause in cnf.clauses]
    if timeout is None:
        satisfiable, model, seconds = _run_solver(clauses, solver)
    else:
        # "spawn" behaves identically on Linux and Windows (Windows only has "spawn").
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(clauses, solver, sender), daemon=True)
        process.start()
        sender.close()
        try:
            if not receiver.poll(timeout):
                return SolveResult(solver, None, None, float(timeout))
            try:
                satisfiable, model, seconds = receiver.recv()
            except EOFError:
                raise RuntimeError(f"the solver process for {solver!r} ended without an answer") from None
        finally:
            if process.is_alive():
                process.terminate()
            process.join()
            receiver.close()

    assignment = _to_assignment(model, cnf.num_vars) if satisfiable else None
    if assignment is not None and not cnf.is_satisfied_by(assignment):
        raise SolverDisagreement(f"{solver} returned a model that does not satisfy the formula")
    return SolveResult(solver, bool(satisfiable), assignment, seconds)


def cross_check_unsat(
    cnf: CNF, solver: str = CROSS_CHECK_SOLVER, timeout: float | None = None
) -> str:
    """Ask a second solver about a formula believed to be unsatisfiable.

    Returns "confirmed" or "timeout"; raises :class:`SolverDisagreement` if the second
    solver finds a (verified) model.
    """
    result = solve(cnf, solver=solver, timeout=timeout)
    if result.satisfiable:
        raise SolverDisagreement(f"{solver} found a model for a formula believed unsatisfiable")
    return "confirmed" if result.satisfiable is False else "timeout"
