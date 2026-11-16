"""CNF formulas in DIMACS conventions.

Conventions (the same as in the DIMACS CNF file format used by every SAT solver):

* Variables are numbered ``1..num_vars``.
* A literal is a non-zero integer: ``+i`` stands for ``x_i``, ``-i`` for ``not x_i``.
* A clause is a tuple of literals (their disjunction); a formula is a tuple of
  clauses (their conjunction).

Assignments are sequences indexed from 0, so ``assignment[i - 1]`` is the value of
``x_i``. A *partial* assignment may contain ``None`` for unassigned variables; a
literal over an unassigned variable counts as not true.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

Clause = tuple[int, ...]
Assignment = Sequence[bool]
PartialAssignment = Sequence["bool | None"]


@dataclass(frozen=True)
class CNF:
    """A formula in conjunctive normal form.

    Empty clauses are allowed (they make the formula unsatisfiable), as are
    clauses of any length; the reduction works for general CNF, not only 3-CNF.
    """

    num_vars: int
    clauses: tuple[Clause, ...]

    def __post_init__(self) -> None:
        if self.num_vars < 0:
            raise ValueError(f"num_vars must be non-negative, got {self.num_vars}")
        # Accept lists as input but always store tuples, so a CNF is immutable and hashable.
        clauses = tuple(tuple(int(lit) for lit in clause) for clause in self.clauses)
        object.__setattr__(self, "clauses", clauses)
        for j, clause in enumerate(clauses):
            for lit in clause:
                if lit == 0 or abs(lit) > self.num_vars:
                    raise ValueError(
                        f"clause {j}: literal {lit} is not in ±1..±{self.num_vars}"
                    )

    @classmethod
    def from_clauses(
        cls, clauses: Iterable[Iterable[int]], num_vars: int | None = None
    ) -> CNF:
        """Build a formula from clauses; ``num_vars`` defaults to the largest variable used."""
        clauses = tuple(tuple(clause) for clause in clauses)
        if num_vars is None:
            num_vars = max((abs(lit) for clause in clauses for lit in clause), default=0)
        return cls(num_vars, clauses)

    @property
    def num_clauses(self) -> int:
        return len(self.clauses)

    # ------------------------------------------------------------------ evaluation

    def _check_length(self, assignment: PartialAssignment) -> None:
        if len(assignment) != self.num_vars:
            raise ValueError(
                f"assignment has {len(assignment)} values, formula has {self.num_vars} variables"
            )

    @staticmethod
    def literal_is_true(lit: int, assignment: PartialAssignment) -> bool:
        """True iff the literal is true under the (possibly partial) assignment."""
        value = assignment[abs(lit) - 1]
        if value is None:
            return False
        return bool(value) if lit > 0 else not bool(value)

    def satisfied_clauses(self, assignment: PartialAssignment) -> list[bool]:
        """For each clause, whether it contains a true literal."""
        self._check_length(assignment)
        return [
            any(self.literal_is_true(lit, assignment) for lit in clause)
            for clause in self.clauses
        ]

    def count_satisfied(self, assignment: PartialAssignment) -> int:
        """Number of clauses satisfied by the (possibly partial) assignment."""
        return sum(self.satisfied_clauses(assignment))

    def is_satisfied_by(self, assignment: PartialAssignment) -> bool:
        return all(self.satisfied_clauses(assignment))

    # ---------------------------------------------------------------- DIMACS I/O

    def to_dimacs(self, comments: Iterable[str] = ()) -> str:
        lines = [f"c {comment}" for comment in comments]
        lines.append(f"p cnf {self.num_vars} {self.num_clauses}")
        lines.extend(" ".join(str(lit) for lit in clause) + " 0" for clause in self.clauses)
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dimacs(cls, text: str) -> CNF:
        """Parse DIMACS CNF text.

        Clauses may span several lines. A line starting with ``%`` ends the formula
        (the SATLIB benchmark files use this as a terminator).
        """
        num_vars: int | None = None
        expected_clauses: int | None = None
        clauses: list[Clause] = []
        current: list[int] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("%"):
                break
            if line.startswith("p"):
                parts = line.split()
                if len(parts) != 4 or parts[1] != "cnf":
                    raise ValueError(f"malformed problem line: {raw_line!r}")
                num_vars, expected_clauses = int(parts[2]), int(parts[3])
                continue
            if num_vars is None:
                raise ValueError("clause before the 'p cnf' problem line")
            for token in line.split():
                lit = int(token)
                if lit == 0:
                    clauses.append(tuple(current))
                    current = []
                else:
                    current.append(lit)
        if num_vars is None:
            raise ValueError("missing 'p cnf' problem line")
        if current:
            raise ValueError("last clause is not terminated by 0")
        if expected_clauses is not None and len(clauses) != expected_clauses:
            raise ValueError(
                f"header announces {expected_clauses} clauses, found {len(clauses)}"
            )
        return cls(num_vars, tuple(clauses))


# ---------------------------------------------------------------------- brute force


def iter_assignments(num_vars: int) -> Iterator[tuple[bool, ...]]:
    """All 2**num_vars total assignments, in lexicographic order."""
    return itertools.product((False, True), repeat=num_vars)


def brute_force_max_satisfied(cnf: CNF, max_vars: int = 20) -> tuple[int, tuple[bool, ...]]:
    """MaxSAT value of the formula and one optimal assignment, by trying all assignments.

    Exponential in ``num_vars``; meant for tests and sanity checks on small formulas.
    The formula is satisfiable iff the returned value equals ``cnf.num_clauses``.
    """
    if cnf.num_vars > max_vars:
        raise ValueError(f"{cnf.num_vars} variables is too many for brute force (max {max_vars})")
    best_value, best_assignment = -1, ()
    for assignment in iter_assignments(cnf.num_vars):
        value = cnf.count_satisfied(assignment)
        if value > best_value:
            best_value, best_assignment = value, assignment
            if value == cnf.num_clauses:
                break
    return best_value, best_assignment
