"""Tests for the evaluation: tables from small hand-made result files, figures if matplotlib is there."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from relu_sat_bench import evaluate as ev

META_COLUMNS = ["onnx", "vnnlib", "timeout", "formula_id", "family", "encoding", "expected", "num_inputs",
                "num_relus", "num_clauses", "threshold", "vnnlib_rhs", "sat_solver_seconds"]
RESULT_COLUMNS = ["tool", "onnx", "vnnlib", "formula_id", "family", "encoding", "expected", "result",
                  "verdict", "seconds", "wall_seconds", "timeout"]

FORMULAS = [  # formula_id, family, expected, variables, SAT-solver seconds
    ("random-n10-r3.5-s1-000", "random", "sat", 10, 0.0004),
    ("random-n20-r3.5-s1-000", "random", "sat", 20, 0.0005),
    ("random-n10-r5-s1-000", "random", "unsat", 10, 0.0006),
    ("pigeonhole-h4", "pigeonhole", "unsat", 20, 0.0030),
    ("tseitin-odd-v6-s1-000", "tseitin", "unsat", 9, 0.0010),
]


def write_bench(folder: Path, formulas, results: dict[str, list[tuple[str, str, float]]]) -> Path:
    """A benchmark folder: metadata.csv and results/<tool>.csv with (verdict, result, seconds) per formula."""
    (folder / "results").mkdir(parents=True)
    with (folder / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=META_COLUMNS)
        writer.writeheader()
        for i, (fid, family, expected, n, sat_s) in enumerate(formulas, start=1):
            writer.writerow({"onnx": f"onnx/f{i:04d}.onnx", "vnnlib": f"vnnlib/f{i:04d}.vnnlib", "timeout": 60,
                             "formula_id": fid, "family": family, "encoding": "compact", "expected": expected,
                             "num_inputs": n, "num_relus": 3 * n, "num_clauses": 4 * n, "threshold": 4 * n - 0.5,
                             "vnnlib_rhs": 4 * n - 0.5, "sat_solver_seconds": sat_s})
    for tool, rows in results.items():
        with (folder / "results" / f"{tool}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for i, ((fid, family, expected, _, _), (verdict, result, seconds)) in enumerate(zip(formulas, rows), 1):
                writer.writerow({"tool": tool, "onnx": f"onnx/f{i:04d}.onnx", "vnnlib": f"vnnlib/f{i:04d}.vnnlib",
                                 "formula_id": fid, "family": family, "encoding": "compact", "expected": expected,
                                 "result": result, "verdict": verdict, "seconds": seconds, "wall_seconds": seconds,
                                 "timeout": 60})
    return folder


OK_SAT, OK_UNSAT, TIMEOUT, WRONG = ("correct", "sat", 4.0), ("correct", "unsat", 9.0), \
    ("timeout", "timeout", 70.0), ("wrong", "unsat", 3.0)


@pytest.fixture
def bench(tmp_path):
    return write_bench(tmp_path / "final", FORMULAS, {
        "marabou": [OK_SAT, TIMEOUT, OK_UNSAT, TIMEOUT, OK_UNSAT],
        "abcrown-bab": [OK_SAT, WRONG, OK_UNSAT, OK_UNSAT, TIMEOUT],
    })


def test_series_names():
    assert ev.series_of("random-n20-r3.5-s1-001") == "random r3.5"
    assert ev.series_of("planted-q-hidden-n14-r4.26-s1-002") == "planted r4.26"
    assert ev.series_of("pigeonhole-h7") == "pigeonhole"
    assert ev.series_of("tseitin-even-v14-s1-001") == "tseitin even"
    assert ev.series_of("ordering-n9") == "ordering"


def test_summary_counts_and_score(bench):
    runs = ev.load_runs(bench)
    summary = {row["tool"]: row for row in ev.tool_summary(runs)}
    assert list(summary) == ["marabou", "abcrown-bab"]            # fixed order, not alphabetical
    m, b = summary["marabou"], summary["abcrown-bab"]
    assert (m["correct"], m["correct sat"], m["correct unsat"], m["timeout"], m["wrong"]) == (3, 1, 2, 2, 0)
    assert (b["correct"], b["wrong"], b["timeout"]) == (3, 1, 1)
    assert m["score"] == 30 and b["score"] == 30 - 150             # a wrong answer costs 15 correct ones
    assert m["median s (solved)"] == 9.0


def test_cells_and_frontier(bench):
    runs = ev.load_runs(bench)
    cells = {(c["series"], c["expected"], c["vars"]): c for c in ev.cells(runs)}
    assert cells[("random r3.5", "sat", 10)]["marabou"] == "1/1"
    assert cells[("random r3.5", "sat", 20)]["abcrown-bab"] == "0/1"
    assert cells[("pigeonhole", "unsat", 20)]["CaDiCaL max s"] == 0.003
    front = ev.frontier(runs)
    assert front[("random r3.5", "sat")] == {"marabou": 10, "abcrown-bab": 10}
    assert front[("tseitin odd", "unsat")] == {"marabou": 9, "abcrown-bab": None}


def test_dense_parameters_match_the_exported_network():
    """The formula for the parameter count agrees with the network the reduction builds."""
    from relu_sat_bench.cnf import CNF
    from relu_sat_bench.reduction import reduce_cnf
    cnf = CNF(3, ((1, -2, 3), (-1, 2), (2, 3), (-3,)))
    network = reduce_cnf(cnf, "compact").network
    stored = sum(layer.weight.size + layer.bias.size for layer in network.layers)
    nonzero = sum(int((layer.weight != 0).sum()) + int((layer.bias != 0).sum()) for layer in network.layers)
    literals = sum(len(c) for c in cnf.clauses)
    assert ev.dense_parameters("compact", 3, 4) == stored
    assert ev.nonzero_parameters("compact", 3, 4, literals) == nonzero == 4 * 3 + 8 + 2 * 4 + 1
    assert ev.dense_parameters("paper", 3, 4) is None and ev.nonzero_parameters("compact", 3, 4, None) is None


def test_literal_counts_come_from_the_cnf_files(tmp_path, bench):
    (bench / "cnf").mkdir()
    (bench / "cnf" / "pigeonhole-h4.cnf").write_text("p cnf 2 3\n1 2 0\n-1 0\n-2 0\n")
    with (bench / "formulas.csv").open("w", newline="") as handle:
        handle.write("id,cnf_file\npigeonhole-h4,cnf/pigeonhole-h4.cnf\nrandom-n10-r5-s1-000,cnf/missing.cnf\n")
    assert ev.literal_counts(bench) == {"pigeonhole-h4": 4}
    runs = {r.formula_id: r for r in ev.load_runs(bench) if r.tool == "marabou"}
    assert runs["pigeonhole-h4"].num_literals == 4 and runs["random-n10-r5-s1-000"].num_literals is None


def test_limits_take_the_last_size_solved_in_half_of_the_cases():
    def run(n, solved):
        return ev.Run(tool="t", formula_id=f"random-n{n}-r5-s1-000", family="random", series="random r5",
                      expected="unsat", num_vars=n, num_clauses=5 * n, num_relus=7 * n, sat_seconds=0.001,
                      verdict="correct" if solved else "timeout", result="", seconds=1.0, timeout=60.0, bench="b")
    # 10: 2/2, 20: 0/2 (a dip), 30: 1/2, 40: 0/2, 50: 0/1  ->  solved at 30, fails from 40 on
    runs = [run(10, 1), run(10, 1), run(20, 0), run(20, 0), run(30, 1), run(30, 0), run(40, 0), run(40, 0), run(50, 0)]
    (row,) = ev.limits(runs)
    assert (row["solved vars"], row["solved relus"], row["fails vars"], row["fails relus"]) == (30, 210, 40, 280)
    assert row["solved params"] == ev.dense_parameters("compact", 30, 150) and row["fails clauses"] == 200
    assert row["fails nonzero"] is None   # no CNF files in this test
    (row,) = ev.limits([run(10, 0)])
    assert row["solved vars"] is None and row["fails vars"] == 10
    (row,) = ev.limits([run(10, 1)])
    assert row["solved vars"] == 10 and row["fails vars"] is None


def test_first_folder_wins_and_reruns_are_kept_apart(tmp_path, bench):
    shifted = write_bench(tmp_path / "shifted", FORMULAS, {
        "abcrown-mip": [OK_SAT, OK_SAT, OK_UNSAT, OK_UNSAT, TIMEOUT],
        "abcrown-mip-rerun": [OK_SAT, OK_SAT, OK_UNSAT, OK_UNSAT, OK_UNSAT],
        "marabou": [TIMEOUT] * 5,                                    # a duplicate: the first folder wins
    })
    runs = ev.combine([ev.load_runs(bench), ev.load_runs(shifted)])
    assert ev.tools_of(runs) == ["marabou", "abcrown-bab", "abcrown-mip"]
    assert sum(r.solved for r in runs if r.tool == "marabou") == 3
    reruns = ev.load_runs(shifted, reruns=True)
    assert {r.tool for r in reruns} == {"abcrown-mip-rerun"} and ev.label("abcrown-mip-rerun").endswith("re-run")


def test_evaluate_writes_tables_and_figures(tmp_path, bench):
    pytest.importorskip("matplotlib")
    extension = write_bench(tmp_path / "scaling", [("random-n50-r3.5-s1-000", "random", "sat", 50, 0.001)],
                            {"abcrown-mip": [OK_SAT]})
    out = tmp_path / "report"
    written = ev.evaluate([bench], [extension], out)
    assert {p.name for p in written} == {"summary.md", "cells.csv", "solved-by-size.svg", "runtime-by-size.svg",
                                         "cactus.svg"}
    assert b"\r" not in (out / "cells.csv").read_bytes()               # LF only, as in the repository
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Benchmark set" in text and "## Scaling experiment" in text and "Network size at each tool's limit" in text
    assert "| Marabou | 5 | 3 |" in text
    for name in ("solved-by-size.svg", "runtime-by-size.svg", "cactus.svg"):
        svg = (out / name).read_text(encoding="utf-8")
        assert svg.startswith("<?xml") and "Marabou" in svg            # text stays text in the SVG


def test_evaluate_without_figures(tmp_path, bench):
    written = ev.evaluate([bench], [], tmp_path / "report", figures=False)
    assert [p.name for p in written] == ["summary.md", "cells.csv"]
    assert "Scaling experiment" not in (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
