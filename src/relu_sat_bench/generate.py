"""Command-line generator for benchmark formulas with reference answers.

Examples (run from the repository folder)::

    python -m relu_sat_bench.generate --out instances/demo random --vars 10 20 --ratio 3.5 4.26 5 --count 20
    python -m relu_sat_bench.generate --out instances/demo planted --vars 20 --ratio 4.26 --count 10
    python -m relu_sat_bench.generate --out instances/demo pigeonhole --holes 2 3 4 5
    python -m relu_sat_bench.generate --out instances/demo tseitin --vertices 6 8 10 --count 5
    python -m relu_sat_bench.generate --out instances/demo ordering --size 3 4 5 6

Output, all inside ``--out``:

* ``cnf/<id>.cnf``: the formula in DIMACS format;
* ``solutions/<id>.sol``: a satisfying assignment (``v 1 -2 3 ... 0``), for satisfiable formulas;
* ``formulas.csv``: one row per formula, see :data:`COLUMNS`. Running the generator again
  with the same ``--out`` adds rows; a row with the same id is replaced. (The name
  ``instances.csv`` is reserved for the VNN-COMP instance list written by ``export``.)

Every instance is reproducible: its random generator is seeded from ``--seed``, the family,
its parameters and its index. Every answer is checked: models against the formula,
"unsatisfiable" from a solver by a second solver, and construction against solver.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from relu_sat_bench.families import (
    PLANTING_MODES,
    GeneratedFormula,
    ordering_principle,
    pigeonhole,
    planted_ksat,
    random_ksat,
    tseitin,
)
from relu_sat_bench.solve import (
    CROSS_CHECK_SOLVER,
    DEFAULT_SOLVER,
    SolverDisagreement,
    cross_check_unsat,
    solve,
)

FORMULAS_CSV = "formulas.csv"

COLUMNS = [
    "id",             # unique name, also the file name of the formula
    "family",         # random | planted | pigeonhole | tseitin | ordering
    "params",         # generator parameters as JSON
    "seed",           # string the instance's random generator was seeded with ("" if not random)
    "num_vars",
    "num_clauses",
    "ratio",          # num_clauses / num_vars
    "answer",         # sat | unsat | unknown  (the reference answer)
    "answer_source",  # construction | solver | construction+solver
    "solver",
    "solver_status",  # sat | unsat | timeout
    "solver_seconds",
    "check",          # how the answer was double-checked
    "cnf_file",
    "solution_file",
]


# ------------------------------------------------------------ instance streams


def _rng(seed: str) -> random.Random:
    return random.Random(seed)  # string seeds are hashed deterministically (not via hash())


def _seed(base: int, family: str, params: dict, index: int) -> str:
    return f"{base}:{family}:{json.dumps(params, sort_keys=True)}:{index}"


def _instances(args: argparse.Namespace) -> Iterator[tuple[str, str, GeneratedFormula]]:
    """Yield (id, seed, formula) for the family and parameters given on the command line."""
    family = args.family
    if family in ("random", "planted"):
        for n in args.vars:
            for ratio in args.ratio:
                params = {"num_vars": n, "ratio": ratio}
                if family == "planted":
                    params["mode"] = args.mode
                for i in range(args.count):
                    seed = _seed(args.seed, family, params, i)
                    if family == "random":
                        formula = random_ksat(_rng(seed), n, ratio)
                        name = f"random-n{n}-r{ratio:g}-s{args.seed}-{i:03d}"
                    else:
                        formula = planted_ksat(_rng(seed), n, ratio, mode=args.mode)
                        name = f"planted-{args.mode}-n{n}-r{ratio:g}-s{args.seed}-{i:03d}"
                    yield name, seed, formula
    elif family == "pigeonhole":
        for holes in args.holes:
            yield f"pigeonhole-h{holes}", "", pigeonhole(holes)
    elif family == "tseitin":
        for v in args.vertices:
            params = {"num_vertices": v, "parity": args.parity}
            for i in range(args.count):
                seed = _seed(args.seed, family, params, i)
                formula = tseitin(_rng(seed), v, parity=args.parity)
                yield f"tseitin-{args.parity}-v{v}-s{args.seed}-{i:03d}", seed, formula
    elif family == "ordering":
        for size in args.size:
            yield f"ordering-n{size}", "", ordering_principle(size)
    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(family)


# ------------------------------------------------------------ answers and files


def _status(satisfiable: bool | None) -> str:
    return {True: "sat", False: "unsat", None: "unknown"}[satisfiable]


def reference_answer(
    formula: GeneratedFormula,
    solver: str = DEFAULT_SOLVER,
    timeout: float | None = 60.0,
    cross_check_solver: str = CROSS_CHECK_SOLVER,
) -> tuple[dict, tuple[bool, ...] | None]:
    """Run the solver, combine its answer with the construction, double-check it.

    Returns the answer-related CSV fields and a satisfying assignment (or None).
    """
    result = solve(formula.cnf, solver=solver, timeout=timeout)
    known = formula.satisfiable
    if known is not None and result.satisfiable is not None and known != result.satisfiable:
        raise SolverDisagreement(
            f"{solver} says {result.status}, but the construction says {_status(known)}"
        )

    if known is not None:
        answer = known
        source = "construction" if result.satisfiable is None else "construction+solver"
    else:
        answer = result.satisfiable
        source = "solver"

    witness = formula.witness or result.model
    if answer is True:
        check = "assignment verified"
    elif answer is False and known is None:
        verdict = cross_check_unsat(formula.cnf, solver=cross_check_solver, timeout=timeout)
        check = f"{verdict} by {cross_check_solver}"
    elif answer is False:
        check = "theorem"
    else:
        check = ""
    fields = {
        "answer": _status(answer),
        "answer_source": source,
        "solver": solver,
        "solver_status": result.status,
        "solver_seconds": f"{result.seconds:.4f}",
        "check": check,
    }
    return fields, witness if answer is True else None


def _solution_text(assignment: Sequence[bool]) -> str:
    literals = " ".join(str(i + 1 if value else -(i + 1)) for i, value in enumerate(assignment))
    return f"s SATISFIABLE\nv {literals} 0\n"


def read_rows(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def write_rows(csv_path: Path, rows: dict[str, dict]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())


# ------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m relu_sat_bench.generate",
        description="Generate CNF formulas with checked reference answers.",
    )
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--seed", type=int, default=0, help="base seed (default 0)")
    parser.add_argument("--solver", default=DEFAULT_SOLVER, help=f"default {DEFAULT_SOLVER}")
    parser.add_argument("--cross-check-solver", default=CROSS_CHECK_SOLVER,
                        help=f"confirms 'unsat' answers of the solver (default {CROSS_CHECK_SOLVER})")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="seconds per solver call (default 60)")
    families = parser.add_subparsers(dest="family", required=True)

    for name, helptext in [("random", "uniform random 3-SAT"),
                           ("planted", "random 3-SAT with a hidden solution (satisfiable)")]:
        sub = families.add_parser(name, help=helptext)
        sub.add_argument("--vars", type=int, nargs="+", required=True, help="numbers of variables")
        sub.add_argument("--ratio", type=float, nargs="+", required=True,
                         help="clauses per variable, e.g. 4.26")
        sub.add_argument("--count", type=int, default=10, help="instances per (vars, ratio)")
        if name == "planted":
            sub.add_argument("--mode", choices=PLANTING_MODES, default="q-hidden")

    sub = families.add_parser("pigeonhole", help="n+1 pigeons in n holes (unsatisfiable)")
    sub.add_argument("--holes", type=int, nargs="+", required=True)

    sub = families.add_parser("tseitin", help="parity constraints on random 3-regular graphs")
    sub.add_argument("--vertices", type=int, nargs="+", required=True, help="even numbers >= 4")
    sub.add_argument("--count", type=int, default=1, help="instances per size")
    sub.add_argument("--parity", choices=("odd", "even"), default="odd",
                     help="odd: unsatisfiable (default), even: satisfiable")

    sub = families.add_parser("ordering", help="ordering principle GT_n (unsatisfiable)")
    sub.add_argument("--size", type=int, nargs="+", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out: Path = args.out
    (out / "cnf").mkdir(parents=True, exist_ok=True)
    (out / "solutions").mkdir(exist_ok=True)
    csv_path = out / FORMULAS_CSV
    rows = read_rows(csv_path)

    for name, seed, formula in _instances(args):
        cnf_file = Path("cnf") / f"{name}.cnf"
        comments = [f"relu-sat-bench instance {name}", f"family {formula.family}",
                    f"params {json.dumps(formula.params, sort_keys=True)}"]
        if seed:
            comments.append(f"seed {seed}")
        (out / cnf_file).write_text(formula.cnf.to_dimacs(comments), encoding="utf-8")

        fields, assignment = reference_answer(
            formula, args.solver, args.timeout, args.cross_check_solver
        )
        solution_file = ""
        if assignment is not None:
            solution_file = str(Path("solutions") / f"{name}.sol")
            (out / solution_file).write_text(_solution_text(assignment), encoding="utf-8")

        cnf = formula.cnf
        rows[name] = {
            "id": name,
            "family": formula.family,
            "params": json.dumps(formula.params, sort_keys=True),
            "seed": seed,
            "num_vars": cnf.num_vars,
            "num_clauses": cnf.num_clauses,
            "ratio": f"{cnf.num_clauses / cnf.num_vars:.3f}",
            **fields,
            "cnf_file": cnf_file.as_posix(),
            "solution_file": Path(solution_file).as_posix() if solution_file else "",
        }
        print(f"{name:40s} {fields['answer']:8s} {fields['solver_status']:8s} "
              f"{float(fields['solver_seconds']):8.3f}s  {fields['check']}")
        write_rows(csv_path, rows)  # after every instance, so an interrupted run keeps its work

    return 0


if __name__ == "__main__":
    sys.exit(main())
