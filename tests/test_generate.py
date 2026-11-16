"""Tests for the generator command line. Skipped if python-sat is not installed."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("pysat")

from relu_sat_bench.cnf import CNF, brute_force_max_satisfied  # noqa: E402
from relu_sat_bench.families import GeneratedFormula  # noqa: E402
from relu_sat_bench.generate import COLUMNS, main, reference_answer  # noqa: E402
from relu_sat_bench.solve import SolverDisagreement  # noqa: E402

COMMANDS = {
    "random": ["random", "--vars", "8", "--ratio", "3", "6", "--count", "3"],
    "planted": ["planted", "--vars", "10", "--ratio", "4.26", "--count", "2"],
    "pigeonhole": ["pigeonhole", "--holes", "2", "3"],
    "tseitin-odd": ["tseitin", "--vertices", "6", "8", "--count", "2"],
    "tseitin-even": ["tseitin", "--vertices", "8", "--parity", "even"],
    "ordering": ["ordering", "--size", "3", "4"],
}


def run(out: Path, command: list[str]) -> list[dict]:
    assert main(["--out", str(out), "--timeout", "30", *command]) == 0
    with (out / "formulas.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_solution(path: Path, num_vars: int) -> tuple[bool, ...]:
    literals = [int(tok) for line in path.read_text().splitlines() if line.startswith("v")
                for tok in line.split()[1:]]
    assert literals[-1] == 0
    values = {abs(lit): lit > 0 for lit in literals[:-1]}
    return tuple(values[v] for v in range(1, num_vars + 1))


@pytest.mark.parametrize("family", COMMANDS)
def test_generated_files_and_answers_are_consistent(tmp_path, family):
    rows = run(tmp_path, COMMANDS[family])
    assert rows and all(list(row) == COLUMNS for row in rows)
    for row in rows:
        cnf = CNF.from_dimacs((tmp_path / row["cnf_file"]).read_text())
        assert (cnf.num_vars, cnf.num_clauses) == (int(row["num_vars"]), int(row["num_clauses"]))
        assert row["answer"] in ("sat", "unsat")
        # every answer is right (all instances here are small enough for brute force)
        assert (row["answer"] == "sat") == (brute_force_max_satisfied(cnf)[0] == cnf.num_clauses)
        if row["answer"] == "sat":
            assert cnf.is_satisfied_by(read_solution(tmp_path / row["solution_file"], cnf.num_vars))
            assert row["check"] == "assignment verified"
        else:
            assert row["solution_file"] == ""
        if row["family"] == "random":
            assert row["answer_source"] == "solver"
            if row["answer"] == "unsat":
                assert row["check"] == "confirmed by glucose4"
        else:
            assert row["answer_source"] == "construction+solver"


def test_random_family_produces_both_answers(tmp_path):
    rows = run(tmp_path, ["random", "--vars", "10", "--ratio", "2", "8", "--count", "5"])
    assert {row["answer"] for row in rows} == {"sat", "unsat"}


def test_generation_is_reproducible(tmp_path):
    first = run(tmp_path / "a", COMMANDS["random"])
    second = run(tmp_path / "b", COMMANDS["random"])
    for a, b in zip(first, second):
        assert a["id"] == b["id"] and a["seed"] == b["seed"] and a["answer"] == b["answer"]
        assert (tmp_path / "a" / a["cnf_file"]).read_text() == (tmp_path / "b" / b["cnf_file"]).read_text()


def test_different_seeds_give_different_formulas(tmp_path):
    for seed in ("0", "1"):
        assert main(["--out", str(tmp_path / seed), "--seed", seed, "--timeout", "30",
                     *COMMANDS["random"]]) == 0
    clauses = [
        CNF.from_dimacs(sorted((tmp_path / seed / "cnf").glob("*.cnf"))[0].read_text()).clauses
        for seed in ("0", "1")
    ]
    assert clauses[0] != clauses[1]


def test_running_again_replaces_rows_and_adds_new_ones(tmp_path):
    run(tmp_path, COMMANDS["pigeonhole"])
    run(tmp_path, COMMANDS["pigeonhole"])
    rows = run(tmp_path, COMMANDS["ordering"])
    assert sorted(row["id"] for row in rows) == ["ordering-n3", "ordering-n4", "pigeonhole-h2", "pigeonhole-h3"]


def test_params_are_json(tmp_path):
    rows = run(tmp_path, COMMANDS["planted"])
    params = json.loads(rows[0]["params"])
    assert params == {"num_vars": 10, "ratio": 4.26, "k": 3, "mode": "q-hidden"}


def test_construction_and_solver_disagreement_is_detected():
    satisfiable_cnf = CNF(2, ((1, 2),))
    wrong = GeneratedFormula("test", {}, satisfiable_cnf, satisfiable=False)
    with pytest.raises(SolverDisagreement):
        reference_answer(wrong, timeout=None)
