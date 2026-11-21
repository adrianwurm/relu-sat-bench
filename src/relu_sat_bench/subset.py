"""Copy the instances at a tool's limit into a new folder, for a run with a longer timeout (step 5c).

For one tool and one or more benchmark folders with results, the selection is, per family and answer
(the rows of the evaluation's size tables):

* the *failing sizes*: the first size at which the tool solved less than half of the instances, and the
  ``sizes - 1`` sizes after it;
* from these sizes, every instance the tool did not solve within its timeout (verdict ``timeout``).

The rule depends only on the results, never on single instances chosen by hand. The new folder is a normal
benchmark folder: ``instances.csv`` with the new timeout, ``metadata.csv``, ``formulas.csv``, ``onnx/``,
``vnnlib/``, ``cnf/`` and ``solutions/``. Its file names are new neutral numbers (the source folders may use
the same names); the column ``source`` of ``metadata.csv`` names the original instance.

Usage::

    python -m relu_sat_bench.subset --bench instances/final --tool marabou --out instances/long-marabou
    python -m relu_sat_bench.subset --bench instances/final-shifted instances/scaling-shifted \\
        --tool abcrown-mip --out instances/long-mip
    python -m relu_sat_bench.subset --bench instances/final --tool abcrown-bab --skip 1 --answer unsat \\
        --out instances/long2-bab     # the next failing size, unsatisfiable formulas only
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from relu_sat_bench.evaluate import Run, load_runs
from relu_sat_bench.export import INSTANCES_CSV, METADATA_COLUMNS, METADATA_CSV
from relu_sat_bench.generate import FORMULAS_CSV, read_rows, write_rows


def failing_sizes(runs: list[Run], sizes: int = 1, skip: int = 0) -> dict[tuple[str, str], list[int]]:
    """Per (series, answer): the first size with less than half solved and the sizes after it; of these the
    ``sizes`` sizes after skipping ``skip``."""
    by_cell: dict[tuple[str, str], dict[int, list[Run]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        by_cell[(run.series, run.expected)][run.num_vars].append(run)
    result = {}
    for cell, by_size in sorted(by_cell.items()):
        ordered = sorted(by_size)
        first = next((n for n in ordered if sum(r.solved for r in by_size[n]) / len(by_size[n]) < 0.5), None)
        if first is not None:
            result[cell] = [n for n in ordered if n >= first][skip:skip + sizes]
    return result


def select(runs: list[Run], tool: str, sizes: int = 1, skip: int = 0, answer: str | None = None) -> list[Run]:
    """The runs of ``tool`` that timed out at its failing sizes, in a fixed order; ``answer``: only sat or unsat."""
    mine = [r for r in runs if r.tool == tool and (answer is None or r.expected == answer)]
    window = failing_sizes(mine, sizes, skip)
    chosen = [r for r in mine if r.verdict == "timeout" and r.num_vars in window.get((r.series, r.expected), [])]
    return sorted(chosen, key=lambda r: (r.series, r.expected, r.num_vars, r.formula_id))


def _renamed(path: str, name: str) -> str:
    """``onnx/f0007-paper.onnx`` -> ``f0042-paper.onnx`` for ``name = "f0042"``: keep the encoding suffix."""
    file = Path(path).name
    return name + file[len(file.split("-")[0].split(".")[0]):]


def write_subset(chosen: list[Run], out: Path, timeout: float, seed: int = 0) -> list[dict]:
    """Copy the chosen instances into ``out`` under new neutral names; returns the metadata rows."""
    if (out / METADATA_CSV).exists():
        raise SystemExit(f"{out} already contains a benchmark; use a new folder")
    for sub in ("onnx", "vnnlib", "cnf", "solutions"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    order = list(range(len(chosen)))
    random.Random(seed).shuffle(order)
    rows, formulas = [], {}
    for number, index in enumerate(order, start=1):
        run = chosen[index]
        bench = Path(run.bench)
        info = next(r for r in csv.DictReader((bench / METADATA_CSV).open(newline="", encoding="utf-8"))
                    if r["onnx"] == run.onnx)
        name = f"f{number:04d}"
        onnx = Path("onnx") / _renamed(info["onnx"], name)
        vnnlib = Path("vnnlib") / _renamed(info["vnnlib"], name)
        shutil.copyfile(bench / info["onnx"], out / onnx)
        shutil.copyfile(bench / info["vnnlib"], out / vnnlib)
        formula = read_rows(bench / FORMULAS_CSV)[run.formula_id]
        for column in ("cnf_file", "solution_file"):
            if formula.get(column):
                shutil.copyfile(bench / formula[column], out / formula[column])
        formulas[run.formula_id] = formula
        rows.append({**{k: info.get(k, "") for k in METADATA_COLUMNS}, "onnx": onnx.as_posix(),
                     "vnnlib": vnnlib.as_posix(), "timeout": f"{timeout:g}",
                     "output_constraints": info.get("output_constraints") or run.output_constraints,
                     "source": f"{bench.name}/{info['onnx']}"})
    rows.sort(key=lambda r: r["onnx"])
    with (out / INSTANCES_CSV).open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows([r["onnx"], r["vnnlib"], r["timeout"]] for r in rows)
    with (out / METADATA_CSV).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*METADATA_COLUMNS, "source"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_rows(out / FORMULAS_CSV, formulas)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m relu_sat_bench.subset", description=__doc__.split("\n")[0])
    parser.add_argument("--bench", type=Path, nargs="+", required=True, help="folders with results of the tool")
    parser.add_argument("--tool", required=True, help="e.g. marabou, abcrown-bab, abcrown-mip")
    parser.add_argument("--out", type=Path, required=True, help="new folder")
    parser.add_argument("--sizes", type=int, default=1, help="failing sizes per family (default 1)")
    parser.add_argument("--skip", type=int, default=0,
                        help="skip the first failing sizes (e.g. 1: the next size after an earlier run)")
    parser.add_argument("--answer", choices=["sat", "unsat"], help="only formulas with this answer")
    parser.add_argument("--timeout", type=float, default=600.0, help="timeout in instances.csv (default 600 s)")
    args = parser.parse_args(argv)
    runs = [run for bench in args.bench for run in load_runs(bench)]
    chosen = select(runs, args.tool, args.sizes, args.skip, args.answer)
    if not chosen:
        raise SystemExit(f"no timeouts of {args.tool} at its failing sizes")
    write_subset(chosen, args.out, args.timeout)
    hours = len(chosen) * (args.timeout + 10) / 3600
    print(f"{len(chosen)} instances written to {args.out}; at most about {hours:.1f} h if all time out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
