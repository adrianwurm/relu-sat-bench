"""Tests for the selection of instances at a tool's limit (relu_sat_bench.subset)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("pysat")

from relu_sat_bench import subset  # noqa: E402
from relu_sat_bench.benchmark import build  # noqa: E402
from relu_sat_bench.evaluate import load_runs  # noqa: E402

SPEC = [
    ["random", "--vars", "6", "8", "10", "--ratio", "6", "--count", "2"],
    ["pigeonhole", "--holes", "2", "3", "4"],
]


def write_results(bench: Path, tool: str, solved) -> None:
    """Fake results: ``solved(meta_row)`` decides correct or timeout."""
    meta = list(csv.DictReader((bench / "metadata.csv").open(newline="")))
    (bench / "results").mkdir(exist_ok=True)
    with (bench / "results" / f"{tool}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["onnx", "verdict", "result", "seconds", "timeout"])
        writer.writeheader()
        for row in meta:
            ok = solved(row)
            writer.writerow({"onnx": row["onnx"], "verdict": "correct" if ok else "timeout",
                             "result": row["expected"] if ok else "timeout", "seconds": 1.0 if ok else 61.0,
                             "timeout": 60})


@pytest.fixture
def bench(tmp_path):
    folder = tmp_path / "bench"
    build(folder, seed=0, spec=SPEC, timeout=60, solver_timeout=30)
    # the tool solves random formulas with 6 variables and pigeonhole with 2 holes, nothing larger
    write_results(folder, "marabou", lambda r: int(r["num_inputs"]) <= 6 or r["formula_id"] == "pigeonhole-h2")
    return folder


def test_failing_sizes_and_selection(bench):
    runs = load_runs(bench)
    window = subset.failing_sizes(runs, sizes=1)
    assert window[("pigeonhole", "unsat")] == [12]                 # h2 (6 variables) solved, h3 (12) not
    for (series, expected), sizes in window.items():               # random: the smallest size above 6
        if series.startswith("random"):
            present = sorted({r.num_vars for r in runs if (r.series, r.expected) == (series, expected)})
            assert sizes == [min(n for n in present if n > 6)]
    chosen = subset.select(runs, "marabou", sizes=1)
    assert {r.formula_id for r in chosen} == {r.formula_id for r in runs
                                             if r.num_vars in window[(r.series, r.expected)]}
    assert "pigeonhole-h3" in {r.formula_id for r in chosen} and "pigeonhole-h4" not in {r.formula_id for r in chosen}
    two = subset.select(runs, "marabou", sizes=2)
    assert {r.formula_id for r in two} == {r.formula_id for r in runs if r.verdict == "timeout"}
    assert subset.select(runs, "abcrown-bab") == []


def test_write_subset(bench, tmp_path):
    out = tmp_path / "long"
    assert subset.main(["--bench", str(bench), "--tool", "marabou", "--out", str(out), "--sizes", "2"]) == 0
    meta = list(csv.DictReader((out / "metadata.csv").open(newline="")))
    source = {r["onnx"]: r for r in csv.DictReader((bench / "metadata.csv").open(newline=""))}
    assert [r["onnx"] for r in meta] == [f"onnx/f{i:04d}.onnx" for i in range(1, len(meta) + 1)]
    for row in meta:
        original = source[row["source"].split("/", 1)[1]]
        assert row["formula_id"] == original["formula_id"] and row["expected"] == original["expected"]
        assert (out / row["onnx"]).read_bytes() == (bench / original["onnx"]).read_bytes()
        assert (out / row["vnnlib"]).read_bytes() == (bench / original["vnnlib"]).read_bytes()
        assert row["timeout"] == "600"
    lines = (out / "instances.csv").read_text().splitlines()
    assert lines == [f"{r['onnx']},{r['vnnlib']},600" for r in meta]
    formulas = list(csv.DictReader((out / "formulas.csv").open(newline="")))
    assert sorted(f["id"] for f in formulas) == sorted(r["formula_id"] for r in meta)
    assert all((out / f["cnf_file"]).exists() for f in formulas)
    with pytest.raises(SystemExit):                                  # never into an existing benchmark
        subset.main(["--bench", str(bench), "--tool", "marabou", "--out", str(out)])


def test_skip_and_answer(bench):
    runs = load_runs(bench)
    first = {r.formula_id for r in subset.select(runs, "marabou", sizes=1)}
    both = {r.formula_id for r in subset.select(runs, "marabou", sizes=2)}
    second = {r.formula_id for r in subset.select(runs, "marabou", sizes=1, skip=1)}
    assert second == both - first and not (second & first)
    unsat = subset.select(runs, "marabou", sizes=2, answer="unsat")
    assert unsat and all(r.expected == "unsat" for r in unsat)
    assert {r.formula_id for r in unsat} == {r.formula_id for r in subset.select(runs, "marabou", sizes=2)
                                             if r.expected == "unsat"}
