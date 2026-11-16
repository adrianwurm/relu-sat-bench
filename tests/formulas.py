"""Test data: a fixed, reproducible collection of small formulas (plain module, no fixtures)."""

from __future__ import annotations

import random

from relu_sat_bench.cnf import CNF, brute_force_max_satisfied


def random_cnf(rng: random.Random, num_vars: int, num_clauses: int, clause_len: int | None = 3) -> CNF:
    """Random CNF: each clause has distinct variables with random signs.

    ``clause_len=None`` draws a length between 1 and 4 for every clause. This is only a
    helper for tests; the real instance generator is a separate module (step 2).
    """
    clauses = []
    for _ in range(num_clauses):
        length = clause_len if clause_len is not None else rng.randint(1, 4)
        variables = rng.sample(range(1, num_vars + 1), min(length, num_vars))
        clauses.append(tuple(v if rng.random() < 0.5 else -v for v in variables))
    return CNF(num_vars, tuple(clauses))


def _small_formulas() -> list[tuple[str, CNF]]:
    formulas = []
    for seed in range(60):
        rng = random.Random(seed)
        num_vars = rng.randint(1, 7)
        # Up to 7 clauses per variable: well above the 3-SAT threshold (about 4.26),
        # so the collection contains satisfiable and unsatisfiable formulas.
        num_clauses = rng.randint(1, 7 * num_vars)
        clause_len = 3 if seed % 2 == 0 else None
        cnf = random_cnf(rng, num_vars, num_clauses, clause_len)
        formulas.append((f"seed{seed}-n{num_vars}-m{num_clauses}", cnf))

    # Hand-picked edge cases.
    all_sign_patterns = [
        (a * 1, b * 2, c * 3) for a in (1, -1) for b in (1, -1) for c in (1, -1)
    ]
    formulas += [
        ("all-8-clauses-on-3-vars-unsat", CNF(3, tuple(all_sign_patterns))),
        ("7-of-8-clauses-sat", CNF(3, tuple(all_sign_patterns[:7]))),
        ("unit-clauses-contradiction", CNF(1, ((1,), (-1,)))),
        ("empty-clause", CNF(2, ((1, 2), ()))),
        ("repeated-literal", CNF(2, ((1, 1, -2), (-1,), (2, 2)))),
        ("tautology", CNF(2, ((1, -1, 2), (-2,)))),
        ("unused-variable", CNF(4, ((1, 2, 3), (-1, -2, -3)))),
        ("long-clause", CNF(5, ((1, 2, 3, 4, 5), (-1,), (-2,), (-3,), (-4,)))),
    ]
    return formulas


SMALL_FORMULAS = _small_formulas()


def max_sat(cnf: CNF) -> int:
    """MaxSAT value by brute force."""
    return brute_force_max_satisfied(cnf)[0]
