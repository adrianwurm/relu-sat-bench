"""The tables in ``report/`` are exactly what the evaluation computes from the result files in ``results/``.

Every report is regenerated into a temporary folder (without figures, whose bytes depend on the matplotlib
version) and compared byte by byte with the committed file. A table that no longer matches the result files
fails here, so ``report/`` cannot go stale. The long-timeout folders are also checked against the selection
rule of ``subset.py``.
"""

import csv
from pathlib import Path

import pytest

from relu_sat_bench.evaluate import load_runs, main
from relu_sat_bench.subset import select

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORT = ROOT / "report"

SATRELU = ["satrelu-compact", "satrelu-compact-shifted", "satrelu-compact-two", "satrelu-paper",
           "satrelu-paper-shifted", "satrelu-ncb"]
LONG = ["long-marabou", "long-bab", "long-mip", "long2-bab", "long2-mip"]

# (arguments of `python -m relu_sat_bench.evaluate` without --out, subfolder of report/, files written)
REPORTS = [
    (["--bench", "final", "final-shifted", "--extension", "scaling-shifted"], "", ["summary.md", "cells.csv"]),
    (["--bench", "final-ncb"], "ncb", ["summary.md", "cells.csv"]),
    (["--bench", "large-ncb"], "large-ncb", ["summary.md", "cells.csv"]),
    (["--compare", *SATRELU, "--name", "satrelu"], "", ["encodings-satrelu.md"]),
    (["--compare", "final", "final-shifted", "sample-ncb", "--name", "sample"], "", ["encodings-sample.md"]),
    (["--compare", "scaling-shifted", "scaling-two", "scaling-ncb", "--name", "scaling"], "",
     ["encodings-scaling.md"]),
    (["--compare", "final", "final-shifted", "final-ncb", "--name", "overlap"], "", ["encodings-overlap.md"]),
    (["--long", *LONG], "", ["long-timeout.md"]),
]

# (folder, source folders, tool, skip, answer): the `python -m relu_sat_bench.subset` calls of step 5c
SUBSETS = [
    ("long-marabou", ["final"], "marabou", 0, None),
    ("long-bab", ["final"], "abcrown-bab", 0, None),
    ("long-mip", ["final-shifted", "scaling-shifted"], "abcrown-mip", 0, None),
    ("long2-bab", ["final"], "abcrown-bab", 1, "unsat"),
    ("long2-mip", ["final-shifted", "scaling-shifted"], "abcrown-mip", 1, "unsat"),
]


def _with_paths(args: list[str]) -> list[str]:
    """Folder names to paths in results/; options and the --name value stay as they are."""
    out, previous = [], None
    for arg in args:
        out.append(arg if arg.startswith("--") or previous == "--name" else str(RESULTS / arg))
        previous = arg
    return out


@pytest.mark.parametrize("args, subfolder, files", REPORTS, ids=[f"{sub or 'report'}/{files[0]}" for _, sub, files in REPORTS])
def test_report_matches_results(tmp_path, capsys, args, subfolder, files):
    main([*_with_paths(args), "--out", str(tmp_path), "--no-figures"])
    capsys.readouterr()
    for name in files:
        written = (tmp_path / name).read_bytes()
        committed = (REPORT / subfolder / name).read_bytes()
        assert written == committed, f"report/{subfolder}/{name} does not match results/ (rerun the evaluation)"


@pytest.mark.parametrize("folder, sources, tool, skip, answer", SUBSETS, ids=[s[0] for s in SUBSETS])
def test_long_timeout_selection(folder, sources, tool, skip, answer):
    runs, origin = [], {}
    for source in sources:
        for run in load_runs(RESULTS / source):
            runs.append(run)
            origin[id(run)] = f"{source}/{run.onnx}"
    chosen = sorted(origin[id(run)] for run in select(runs, tool, sizes=1, skip=skip, answer=answer))
    with (RESULTS / folder / "metadata.csv").open(newline="", encoding="utf-8") as f:
        recorded = sorted(row["source"] for row in csv.DictReader(f))
    assert chosen == recorded


def test_no_sat_relu_formulas_redistributed():
    """The sat_relu formulas have no license: their folders keep only our metadata and results, no CNF files."""
    for folder in SATRELU:
        assert not (RESULTS / folder / "cnf").exists()
        assert not (RESULTS / folder / "solutions").exists()


def test_no_logs_or_networks():
    """Only CSV files and formulas: no tool logs (they contain licence ids), no networks or properties."""
    suffixes = {path.suffix for path in RESULTS.rglob("*") if path.is_file()}
    assert suffixes <= {".csv", ".cnf", ".md"}


def test_hangs_files():
    """Every folder with MIP runs has hangs.csv (written by --mark-hangs from the logs, which are not included);
    it lists only MIP first runs that timed out."""
    for results in sorted(RESULTS.glob("*/results/abcrown-mip.csv")):
        folder = results.parent.parent
        assert (folder / "hangs.csv").exists(), f"{folder.name}: hangs.csv missing"
        with results.open(newline="", encoding="utf-8") as f:
            verdicts = {row["onnx"]: row["verdict"] for row in csv.DictReader(f)}
        with (folder / "hangs.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assert row["tool"] == "abcrown-mip" and verdicts[row["onnx"]] == "timeout"
