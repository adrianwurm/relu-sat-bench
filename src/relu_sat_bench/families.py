"""Formula families for the benchmark.

Every generator returns a :class:`GeneratedFormula`: the CNF plus what is known about it
*by construction*:

==================  =====================  ==============================================
family              answer by construction witness
==================  =====================  ==============================================
random              unknown (needs solver) none
planted             satisfiable            the planted assignment
pigeonhole          unsatisfiable          none (n+1 pigeons do not fit into n holes)
tseitin (odd)       unsatisfiable          none (sum of all vertex parities is odd)
tseitin (even)      satisfiable            constructed along a spanning tree
ordering            unsatisfiable          none (every finite total order has a minimum)
==================  =====================  ==============================================

Randomised generators take a ``random.Random`` so that every instance is reproducible
from its seed. See ``walkthrough/notes-hard-sat-formulas.md`` for why these families are
interesting.
"""

from __future__ import annotations

import itertools
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from relu_sat_bench.cnf import CNF, Clause

# Balanced planting (Jia, Moore & Strain 2005): accept a clause with t true literals with
# probability proportional to Q^t. Q = (sqrt(5) - 1) / 2 solves q^2 + q - 1 = 0, which makes
# every literal in the formula true under the planted assignment with probability exactly 1/2,
# so literal frequencies no longer reveal the solution.
Q_HIDDEN = (5**0.5 - 1) / 2
PLANTING_MODES = ("naive", "q-hidden")


@dataclass(frozen=True)
class GeneratedFormula:
    """A formula together with its family, parameters and what is known by construction."""

    family: str
    params: dict[str, Any]
    cnf: CNF
    satisfiable: bool | None  # None: unknown, a solver has to decide
    witness: tuple[bool, ...] | None = field(default=None)  # a satisfying assignment, if known

    def __post_init__(self) -> None:
        if self.witness is not None:
            if self.satisfiable is not True:
                raise ValueError("a witness only makes sense for a satisfiable formula")
            if not self.cnf.is_satisfied_by(self.witness):
                raise ValueError("the witness does not satisfy the formula")


# ------------------------------------------------------------------ random k-SAT


def _random_clause(rng: random.Random, num_vars: int, k: int) -> Clause:
    """k distinct variables, each negated with probability 1/2."""
    variables = rng.sample(range(1, num_vars + 1), k)
    return tuple(v if rng.random() < 0.5 else -v for v in variables)


def num_clauses_for_ratio(num_vars: int, ratio: float) -> int:
    """m = round(ratio * n), at least 1."""
    return max(1, round(ratio * num_vars))


def random_ksat(rng: random.Random, num_vars: int, ratio: float, k: int = 3) -> GeneratedFormula:
    """Uniform random k-SAT: ``round(ratio * n)`` independent clauses of k distinct variables."""
    if num_vars < k:
        raise ValueError(f"need at least k={k} variables, got {num_vars}")
    m = num_clauses_for_ratio(num_vars, ratio)
    clauses = tuple(_random_clause(rng, num_vars, k) for _ in range(m))
    return GeneratedFormula(
        family="random",
        params={"num_vars": num_vars, "ratio": ratio, "k": k},
        cnf=CNF(num_vars, clauses),
        satisfiable=None,
    )


def planted_ksat(
    rng: random.Random, num_vars: int, ratio: float, k: int = 3, mode: str = "q-hidden"
) -> GeneratedFormula:
    """Random k-SAT with a hidden solution.

    A random assignment is chosen first. Clauses are drawn uniformly and accepted only if
    the assignment satisfies them:

    * ``naive``: every satisfied clause is accepted. Literals that agree with the hidden
      solution are then over-represented (4/7 instead of 1/2 for k = 3), which leaks it.
    * ``q-hidden``: a clause with t true literals is accepted with probability ``Q^(t-1)``,
      which removes that bias (Jia, Moore & Strain 2005).
    """
    if mode not in PLANTING_MODES:
        raise ValueError(f"unknown planting mode {mode!r}; choose one of {PLANTING_MODES}")
    if k != 3 and mode == "q-hidden":
        raise ValueError("the q-hidden constant is derived for k = 3")
    if num_vars < k:
        raise ValueError(f"need at least k={k} variables, got {num_vars}")
    hidden = tuple(rng.random() < 0.5 for _ in range(num_vars))
    m = num_clauses_for_ratio(num_vars, ratio)
    clauses: list[Clause] = []
    while len(clauses) < m:
        clause = _random_clause(rng, num_vars, k)
        true_literals = sum(CNF.literal_is_true(lit, hidden) for lit in clause)
        if true_literals == 0:
            continue
        if mode == "q-hidden" and rng.random() >= Q_HIDDEN ** (true_literals - 1):
            continue
        clauses.append(clause)
    return GeneratedFormula(
        family="planted",
        params={"num_vars": num_vars, "ratio": ratio, "k": k, "mode": mode},
        cnf=CNF(num_vars, tuple(clauses)),
        satisfiable=True,
        witness=hidden,
    )


# -------------------------------------------------------------------- pigeonhole


def pigeonhole(holes: int) -> GeneratedFormula:
    """PHP: ``holes + 1`` pigeons, ``holes`` holes, no two pigeons share a hole.

    Variable ``p(i, j)`` means "pigeon i sits in hole j". Clauses: every pigeon sits in
    some hole (length ``holes``), and for every hole no two pigeons sit in it (length 2).
    Exponentially hard for resolution, hence for CDCL solvers (Haken 1985).
    """
    if holes < 1:
        raise ValueError("need at least one hole")
    pigeons = holes + 1

    def p(i: int, j: int) -> int:
        return i * holes + j + 1

    clauses: list[Clause] = [tuple(p(i, j) for j in range(holes)) for i in range(pigeons)]
    for j in range(holes):
        for i, k in itertools.combinations(range(pigeons), 2):
            clauses.append((-p(i, j), -p(k, j)))
    return GeneratedFormula(
        family="pigeonhole",
        params={"holes": holes},
        cnf=CNF(pigeons * holes, tuple(clauses)),
        satisfiable=False,
    )


# ----------------------------------------------------------------------- Tseitin


def random_regular_graph(
    rng: random.Random, num_vertices: int, degree: int = 3, max_tries: int = 10_000
) -> list[tuple[int, int]]:
    """A uniformly random simple, connected d-regular graph (pairing model with rejection).

    Returns the edge list; vertices are ``0 .. num_vertices - 1``. Random 3-regular graphs
    are expanders with high probability, which is what makes Tseitin formulas hard.
    """
    if num_vertices <= degree or (num_vertices * degree) % 2:
        raise ValueError(f"no simple {degree}-regular graph on {num_vertices} vertices")
    for _ in range(max_tries):
        points = [v for v in range(num_vertices) for _ in range(degree)]
        rng.shuffle(points)
        edges = []
        seen = set()
        ok = True
        for a, b in zip(points[::2], points[1::2]):
            edge = (min(a, b), max(a, b))
            if a == b or edge in seen:
                ok = False
                break
            seen.add(edge)
            edges.append(edge)
        if ok and _is_connected(num_vertices, edges):
            return sorted(edges)
    raise RuntimeError(f"no simple connected graph found in {max_tries} tries")


def _is_connected(num_vertices: int, edges: list[tuple[int, int]]) -> bool:
    neighbours = [[] for _ in range(num_vertices)]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    seen = {0}
    queue = deque([0])
    while queue:
        for w in neighbours[queue.popleft()]:
            if w not in seen:
                seen.add(w)
                queue.append(w)
    return len(seen) == num_vertices


def _xor_clauses(variables: list[int], parity: int) -> list[Clause]:
    """CNF of ``x_1 xor ... xor x_d = parity``: one clause per forbidden assignment."""
    clauses = []
    for bits in itertools.product((0, 1), repeat=len(variables)):
        if sum(bits) % 2 != parity:
            # forbid exactly this assignment
            clauses.append(tuple(-v if bit else v for v, bit in zip(variables, bits)))
    return clauses


def tseitin(
    rng: random.Random, num_vertices: int, degree: int = 3, parity: str = "odd"
) -> GeneratedFormula:
    """Tseitin formula on a random d-regular graph.

    One variable per edge, random vertex charges ``c(v)``, and for every vertex the
    constraint "the XOR of its incident edges equals c(v)" (``2^(d-1)`` clauses of length d,
    so d = 3 gives a 3-CNF). Summing all constraints counts every edge twice, so the formula
    is unsatisfiable iff the total charge is odd (on a connected graph). Hard for resolution
    on expanders (Urquhart 1987).
    """
    if parity not in ("odd", "even"):
        raise ValueError("parity must be 'odd' or 'even'")
    edges = random_regular_graph(rng, num_vertices, degree)
    edge_var = {edge: index + 1 for index, edge in enumerate(edges)}
    charge = [rng.randint(0, 1) for _ in range(num_vertices)]
    if sum(charge) % 2 != (1 if parity == "odd" else 0):
        charge[0] ^= 1

    incident: list[list[int]] = [[] for _ in range(num_vertices)]
    for (a, b), var in edge_var.items():
        incident[a].append(var)
        incident[b].append(var)
    clauses: list[Clause] = []
    for v in range(num_vertices):
        clauses.extend(_xor_clauses(incident[v], charge[v]))

    witness = None
    if parity == "even":
        witness = _tseitin_solution(rng, num_vertices, edges, edge_var, incident, charge)
    return GeneratedFormula(
        family="tseitin",
        params={"num_vertices": num_vertices, "degree": degree, "parity": parity},
        cnf=CNF(len(edges), tuple(clauses)),
        satisfiable=parity == "even",
        witness=witness,
    )


def _tseitin_solution(rng, num_vertices, edges, edge_var, incident, charge) -> tuple[bool, ...]:
    """Satisfying assignment for an even total charge, built along a BFS spanning tree.

    Non-tree edges get random values. Then every non-root vertex, from the leaves upwards,
    fixes its parent edge so that its own constraint holds. The root's constraint holds
    automatically because the total charge is even.
    """
    neighbours: list[list[tuple[int, int]]] = [[] for _ in range(num_vertices)]
    for (a, b), var in edge_var.items():
        neighbours[a].append((b, var))
        neighbours[b].append((a, var))
    parent_edge = {0: None}
    order = [0]
    queue = deque([0])
    while queue:
        v = queue.popleft()
        for w, var in neighbours[v]:
            if w not in parent_edge:
                parent_edge[w] = var
                order.append(w)
                queue.append(w)
    tree = {var for var in parent_edge.values() if var is not None}
    value = {var: rng.randint(0, 1) for var in edge_var.values() if var not in tree}
    for v in reversed(order[1:]):
        others = sum(value[var] for var in incident[v] if var != parent_edge[v])
        value[parent_edge[v]] = (charge[v] - others) % 2
    return tuple(bool(value[var]) for var in range(1, len(edges) + 1))


# ------------------------------------------------------------- ordering principle


def ordering_principle(size: int) -> GeneratedFormula:
    """GT_n: "a total order on n elements without a minimum" (unsatisfiable).

    Variable ``x(i, j)`` for i < j means "i comes before j"; "j before i" is its negation,
    so the order is total and antisymmetric by construction. Clauses: no cycle of length
    3 (transitivity, two clauses per triple) and every element has a predecessor (length
    n - 1). Easy for general resolution (polynomial proofs) but needs large width, hence
    exponential tree-like resolution, i.e. plain DPLL (Bonet & Galesi 2001).
    """
    if size < 2:
        raise ValueError("need at least 2 elements")
    index = {pair: k + 1 for k, pair in enumerate(itertools.combinations(range(size), 2))}

    def before(i: int, j: int) -> int:
        return index[(i, j)] if i < j else -index[(j, i)]

    clauses: list[Clause] = []
    for a, b, c in itertools.combinations(range(size), 3):
        clauses.append((-before(a, b), -before(b, c), -before(c, a)))  # no cycle a<b<c<a
        clauses.append((-before(a, c), -before(c, b), -before(b, a)))  # no cycle a<c<b<a
    for j in range(size):
        clauses.append(tuple(before(i, j) for i in range(size) if i != j))
    return GeneratedFormula(
        family="ordering",
        params={"size": size},
        cnf=CNF(len(index), tuple(clauses)),
        satisfiable=False,
    )
