"""Evaluate verifier runs: tables and figures from the results of :mod:`relu_sat_bench.run`.

Usage::

    python -m relu_sat_bench.evaluate --bench instances/final instances/final-shifted \\
        --extension instances/scaling-shifted --out report

* ``--bench``: benchmark folders that hold the *same* formulas (for example the normal files and the
  ``--threshold-in-network`` variant). Every ``results/<tool>.csv`` in them is one tool.
* ``--extension``: folders with further, larger formulas (the scaling experiment). Their runs are
  reported in a separate section and continue the lines in the figures; they are not counted in the
  benchmark tables.
* ``results/<tool>-rerun.csv`` files are diagnostic re-runs. They are listed separately and never
  replace the first run.

Comparison of encodings (step 5b)::

    python -m relu_sat_bench.evaluate --compare instances/satrelu-compact instances/satrelu-compact-shifted \
        instances/satrelu-paper instances/satrelu-paper-shifted instances/satrelu-ncb --name satrelu --out report

* ``--compare``: folders with the same formulas in different encodings (the ``encoding`` column of
  ``metadata.csv``). For each tool only the formulas it ran in every one of its encodings are compared.
  Writes ``encodings-<name>.md``.

Writes to ``--out``:

* ``summary.md``: per tool the verdicts, the VNN-COMP score, solved instances per family and size next
  to the SAT solver, the largest formula each tool solved, the network size at each tool's limit, and the
  diagnostic re-runs;
* ``cells.csv``: the per family/size table in machine-readable form;
* ``solved-by-size.svg``, ``runtime-by-size.svg``, ``cactus.svg`` if matplotlib is installed
  (``pip install -e ".[report]"``).

Times: the run time of the tool process as measured by the runner (start-up included), and for the SAT
solver the time CaDiCaL needed in the generator. Unsolved runs count as "not solved within the time
limit"; for medians they count as the time limit.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SAT_SOLVER = "CaDiCaL"
RERUN_SUFFIX = "-rerun"
# VNN-COMP scoring (2024 rules): 10 points per correct answer, -150 per wrong one, no time bonus here.
SCORE_CORRECT, SCORE_WRONG = 10, -150

TOOL_LABELS = {
    "marabou": "Marabou",
    "abcrown-bab": "α,β-CROWN (BaB)",
    "abcrown-mip": "α,β-CROWN (MIP)",
}
TOOL_ORDER = ["marabou", "abcrown-bab", "abcrown-mip"]

# Panels of the figures: (series, expected answer, title), satisfiable row first.
PANELS = [
    ("random r3.5", "sat", "random 3-SAT, ratio 3.5"),
    ("random r4.26", "sat", "random 3-SAT, ratio 4.26"),
    ("planted r4.26", "sat", "planted (q-hidden), 4.26"),
    ("tseitin even", "sat", "Tseitin, even"),
    ("random r4.26", "unsat", "random 3-SAT, ratio 4.26"),
    ("random r5", "unsat", "random 3-SAT, ratio 5"),
    ("tseitin odd", "unsat", "Tseitin, odd"),
    ("pigeonhole", "unsat", "pigeonhole"),
    ("ordering", "unsat", "ordering principle"),
]


@dataclass(frozen=True)
class Run:
    """One verifier run on one instance, together with what we know about the formula."""

    tool: str
    formula_id: str
    family: str
    series: str
    expected: str
    num_vars: int
    num_clauses: int
    num_relus: int
    sat_seconds: float
    verdict: str
    result: str
    seconds: float
    timeout: float
    bench: str
    encoding: str = "compact"
    num_literals: int | None = None   # literal occurrences in the CNF, if the CNF file is available
    output_constraints: int = 1       # constraints of the unsafe set in the VNN-LIB file
    sat_solved: bool = True           # False: CaDiCaL hit its time limit, the answer comes from the construction

    @property
    def solved(self) -> bool:
        return self.verdict == "correct"

    @property
    def setup(self) -> str:
        """The encoding, and for compact and paper whether the property was written as two constraints."""
        if self.encoding in ("compact", "paper") and self.output_constraints == 2:
            return f"{self.encoding} (2 constraints)"
        return self.encoding


def series_of(formula_id: str) -> str:
    """The ladder a formula belongs to: ``random-n20-r3.5-s1-001`` -> ``random r3.5``.

    >>> [series_of(f) for f in ("pigeonhole-h4", "tseitin-odd-v10-s1-002", "planted-q-hidden-n12-r4.26-s1-0",
    ...                         "satrelu-v25-c40-unsat")]
    ['pigeonhole', 'tseitin odd', 'planted r4.26', 'satrelu']
    """
    name = re.sub(r"-s\d+-\d+$", "", formula_id)        # seed and index
    name = re.sub(r"-(sat|unsat)$", "", name)            # the answer in imported names (sat_relu)
    name = re.sub(r"-[nhvc]\d+(?=-|$)", "", name)        # the size
    name = name.replace("-q-hidden", "").replace("-naive", "")
    return name.replace("-", " ")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def literal_counts(bench: Path) -> dict[str, int]:
    """Literal occurrences per formula, from ``formulas.csv`` and the CNF files (empty if they are missing)."""
    from relu_sat_bench.cnf import CNF
    counts = {}
    if (bench / "formulas.csv").exists():
        for row in _read_csv(bench / "formulas.csv"):
            path = bench / row["cnf_file"]
            if path.exists():
                counts[row["id"]] = sum(len(c) for c in CNF.from_dimacs(path.read_text(encoding="utf-8")).clauses)
    return counts


def sat_solved(bench: Path) -> dict[str, bool]:
    """Per formula, whether CaDiCaL decided it in the generator (``formulas.csv``, column ``solver_status``)."""
    if not (bench / "formulas.csv").exists():
        return {}
    return {row["id"]: row.get("solver_status", "sat") in ("sat", "unsat") for row in _read_csv(bench / "formulas.csv")}


def load_runs(bench: Path, reruns: bool = False) -> list[Run]:
    """All runs in ``bench/results``: the first runs, or (``reruns=True``) only the diagnostic re-runs."""
    meta = {row["onnx"]: row for row in _read_csv(bench / "metadata.csv")}
    literals = literal_counts(bench)
    solved_by_sat = sat_solved(bench)
    runs = []
    for results in sorted((bench / "results").glob("*.csv")):
        if results.stem.endswith(RERUN_SUFFIX) != reruns:
            continue
        tool = results.stem
        for row in _read_csv(results):
            info = meta[row["onnx"]]
            runs.append(Run(
                tool=tool, formula_id=info["formula_id"], family=info["family"],
                series=series_of(info["formula_id"]), expected=info["expected"],
                num_vars=int(info["num_inputs"]), num_clauses=int(info["num_clauses"]),
                num_relus=int(info["num_relus"]), sat_seconds=float(info["sat_solver_seconds"]),
                verdict=row["verdict"], result=row["result"], seconds=float(row["seconds"]),
                timeout=float(row["timeout"]), bench=str(bench), encoding=info.get("encoding", "compact"),
                num_literals=literals.get(info["formula_id"]),
                output_constraints=int(info.get("output_constraints") or
                                       (2 if info.get("encoding") == "neurocodebench" else 1)),
                sat_solved=solved_by_sat.get(info["formula_id"], True),
            ))
    return runs


def combine(groups: list[list[Run]], by_encoding: bool = False) -> list[Run]:
    """Merge runs from several folders; for a (tool, formula) pair the first folder wins.

    ``by_encoding``: keep one run per (tool, setup, formula) instead, for the comparison of encodings
    (setup: the encoding, and whether the property has two constraints; see :attr:`Run.setup`).
    """
    seen, merged = set(), []
    for runs in groups:
        for run in runs:
            key = (run.tool, run.setup, run.formula_id) if by_encoding else (run.tool, run.formula_id)
            if key not in seen:
                seen.add(key)
                merged.append(run)
    return merged


def formulas_of(runs: list[Run]) -> dict[str, Run]:
    """One run per formula (any tool), for the formula's own data such as the SAT-solver time."""
    return {run.formula_id: run for run in reversed(runs)}


def tools_of(runs: list[Run]) -> list[str]:
    present = {run.tool for run in runs}
    return [t for t in TOOL_ORDER if t in present] + sorted(present - set(TOOL_ORDER))


def label(tool: str) -> str:
    base = tool[: -len(RERUN_SUFFIX)] if tool.endswith(RERUN_SUFFIX) else tool
    text = TOOL_LABELS.get(base, base)
    return f"{text}, re-run" if base != tool else text


# ----------------------------------------------------------------------------- tables


def tool_summary(runs: list[Run]) -> list[dict]:
    rows = []
    for tool in tools_of(runs):
        mine = [r for r in runs if r.tool == tool]
        verdicts = Counter(r.verdict for r in mine)
        solved_times = [r.seconds for r in mine if r.solved]
        rows.append({
            "tool": tool,
            "instances": len(mine),
            "correct": verdicts["correct"],
            "correct sat": sum(r.solved and r.expected == "sat" for r in mine),
            "correct unsat": sum(r.solved and r.expected == "unsat" for r in mine),
            "wrong": verdicts["wrong"],
            "timeout": verdicts["timeout"],
            "other": len(mine) - verdicts["correct"] - verdicts["wrong"] - verdicts["timeout"],
            "score": SCORE_CORRECT * verdicts["correct"] + SCORE_WRONG * verdicts["wrong"],
            "median s (solved)": statistics.median(solved_times) if solved_times else None,
        })
    return rows


def cells(runs: list[Run]) -> list[dict]:
    """Per (series, expected answer, number of variables): instances, solved per tool, SAT-solver times."""
    tools = tools_of(runs)
    groups: dict[tuple, list[Run]] = defaultdict(list)
    for run in runs:
        groups[(run.series, run.expected, run.num_vars)].append(run)
    rows = []
    for (series, expected, num_vars), group in sorted(groups.items()):
        formulas = formulas_of(group)
        row = {"series": series, "expected": expected, "vars": num_vars, "instances": len(formulas)}
        for tool in tools:
            mine = [r for r in group if r.tool == tool]
            row[tool] = f"{sum(r.solved for r in mine)}/{len(mine)}" if mine else ""
        sat_times = [f.sat_seconds for f in formulas.values()]
        if all(f.sat_solved for f in formulas.values()):
            row[f"{SAT_SOLVER} max s"] = max(sat_times)
        else:  # CaDiCaL hit its time limit on at least one formula of this size
            row[f"{SAT_SOLVER} max s"] = f"> {_fmt(max(sat_times))}"
        rows.append(row)
    return rows


def frontier(runs: list[Run]) -> dict[tuple[str, str], dict[str, int | None]]:
    """For each (series, expected answer): the largest number of variables each tool solved."""
    result: dict[tuple[str, str], dict[str, int | None]] = defaultdict(dict)
    for tool in tools_of(runs):
        for run in runs:
            if run.tool != tool:
                continue
            key = (run.series, run.expected)
            best = result[key].get(tool)
            if run.solved and (best is None or run.num_vars > best):
                result[key][tool] = run.num_vars
            else:
                result[key].setdefault(tool, None)
    return dict(result)


def dense_parameters(encoding: str, num_vars: int, num_clauses: int) -> int | None:
    """Weights and biases of the network as stored in the ONNX file, zeros included.

    This is how VNN-COMP reports network sizes. A layer with ``a`` inputs and ``b`` outputs stores
    ``b * a + b`` numbers. Compact (2n literal ReLUs, m clause ReLUs, one output): 2n*n + 2n, then
    m*2n + m, then m + 1. Neurocodebench (m + 2n hidden ReLUs, two outputs): (m + 2n)(n + 1), then
    2(m + 2n) + 2. Paper: the seven layers of :mod:`relu_sat_bench.reduction`. Unknown encoding: None.
    """
    n, m = num_vars, num_clauses
    if encoding == "compact":
        return (2 * n * n + 2 * n) + (m * 2 * n + m) + (m + 1)
    if encoding == "neurocodebench":
        return (m + 2 * n) * (n + 1) + 2 * (m + 2 * n) + 2
    if encoding == "paper":
        widths = [n, 2 * n, n, 2 * n, 2 * n, 2 * m, m, 1]
        return sum(b * a + b for a, b in zip(widths, widths[1:]))
    return None


def nonzero_parameters(encoding: str, num_vars: int, num_clauses: int, num_literals: int | None) -> int | None:
    """Weights and biases that are not zero, i.e. what the network really needs.

    Compact encoding: 2n literal weights (+-2) and 2n biases (+-1); one weight -1 per literal occurrence L and
    one bias 1 per clause; m output weights -1 and the output bias m. Together 4n + L + 2m + 1.
    """
    if encoding != "compact" or num_literals is None:
        return None
    return 4 * num_vars + num_literals + 2 * num_clauses + 1


def limits(runs: list[Run]) -> list[dict]:
    """Network size at each tool's limit, per (series, expected answer).

    "Solved" is the largest size at which the tool solved at least half of the instances; "fails" is the next
    size of the ladder, from which on it solves less than half at every size. On the input box every ReLU of
    these networks is unstable (both phases occur), so the ReLU count is also the number of unstable ReLUs.
    """
    rows = []
    for tool in tools_of(runs):
        cells_: dict[tuple, dict[int, list[Run]]] = defaultdict(lambda: defaultdict(list))
        for run in runs:
            if run.tool == tool:
                cells_[(run.series, run.expected)][run.num_vars].append(run)
        for (series, expected), by_size in sorted(cells_.items()):
            sizes = sorted(by_size)
            share = {n: statistics.mean(r.solved for r in by_size[n]) for n in sizes}
            ok = [n for n in sizes if share[n] >= 0.5]
            last_ok = max(ok) if ok else None
            fails = next((n for n in sizes if last_ok is None or n > last_ok), None)

            def size(n):
                if n is None:
                    return {"vars": None, "clauses": None, "relus": None, "nonzero": None, "params": None}
                group = by_size[n]
                run = group[0]
                nonzero = [nonzero_parameters(r.encoding, n, r.num_clauses, r.num_literals) for r in group]
                return {"vars": n, "clauses": max(r.num_clauses for r in group), "relus": max(r.num_relus for r in group),
                        "nonzero": None if None in nonzero else max(nonzero),   # None: unknown for this encoding
                        "params": max(dense_parameters(r.encoding, n, r.num_clauses) or 0 for r in group) or None}

            solved, failed = size(last_ok), size(fails)
            rows.append({"tool": tool, "series": series, "expected": expected,
                         **{f"solved {k}": v for k, v in solved.items()},
                         **{f"fails {k}": v for k, v in failed.items()}})
    return rows


def _fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.4f}" if value < 0.1 else f"{value:.1f}"
    return str(value)


def _md_table(rows: list[dict], columns: list[str], headers: list[str] | None = None) -> str:
    headers = headers or columns
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


def _section(title: str, runs: list[Run]) -> list[str]:
    tools = tools_of(runs)
    out = [f"## {title}", ""]
    summary = tool_summary(runs)
    for row in summary:
        row["tool"] = label(row["tool"])
    out += [_md_table(summary, list(summary[0])), ""]
    formulas = formulas_of(runs)
    sat_times = sorted(f.sat_seconds for f in formulas.values() if f.sat_solved)
    unsolved = sum(not f.sat_solved for f in formulas.values())
    if not unsolved:
        out += [f"{SAT_SOLVER} solved all {len(formulas)} formulas; at most {_fmt(sat_times[-1])} s, "
                f"median {_fmt(statistics.median(sat_times))} s.", ""]
    else:
        solved_text = (f"; at most {_fmt(sat_times[-1])} s, median {_fmt(statistics.median(sat_times))} s"
                       if sat_times else "")
        out += [f"{SAT_SOLVER} solved {len(sat_times)} of {len(formulas)} formulas{solved_text}. On the other "
                f"{unsolved} it hit its time limit in the generator; their answers come from the construction.", ""]
    out += ["### Solved instances per family and size", ""]
    table = cells(runs)
    columns = ["series", "expected", "vars", "instances", *tools, f"{SAT_SOLVER} max s"]
    out += [_md_table(table, columns, ["series", "answer", "variables", "instances",
                                       *map(label, tools), f"{SAT_SOLVER} max s"]), ""]
    out += ["### Largest formula solved (number of variables)", ""]
    front = frontier(runs)
    rows = [{"series": s, "expected": e, **front[(s, e)]} for (s, e) in sorted(front)]
    out += [_md_table(rows, ["series", "expected", *tools], ["series", "answer", *map(label, tools)]), ""]
    return out


def write_summary(out: Path, main: list[Run], extension: list[Run], reruns: list[Run]) -> Path:
    lines = ["# Evaluation", "",
             "Generated by `python -m relu_sat_bench.evaluate`. Times include each tool's start-up; "
             f"the score is VNN-COMP's ({SCORE_CORRECT} per correct, {SCORE_WRONG} per wrong answer, "
             "no time bonus).", ""]
    lines += _section("Benchmark set", main)
    if extension:
        lines += _section("Scaling experiment (beyond the benchmark set)", extension)
    lines += ["## Network size at each tool's limit", "",
              "Benchmark set and scaling experiment together. *Solved*: the largest size at which the tool solved "
              "at least half of the instances. *Fails*: the next size, from which on it solves less than half. "
              "Inputs = variables of the formula. Every ReLU of these networks is unstable on the input box (both "
              "phases occur), so the ReLU count is also the number of unstable ReLUs. *Parameters as stored* counts "
              "the full weight matrices of the ONNX file, zeros included, as VNN-COMP does; almost all of them are "
              "zero, because each literal neuron reads one input and each clause neuron reads only its literals. "
              "*Nonzero parameters* is what the network really needs (– if the CNF files are not available). "
              "Sizes are the largest of the instances at that size.", ""]
    table = limits(combine([main, extension]))
    for row in table:
        row["tool"] = label(row["tool"])
    columns = ["tool", "series", "expected", "solved vars", "fails vars", "fails clauses", "fails relus",
               "fails nonzero", "fails params"]
    lines += [_md_table(table, columns, ["tool", "series", "answer", "solved up to (inputs)", "fails at: inputs",
                                         "clauses", "ReLUs", "nonzero parameters", "parameters as stored"]), ""]
    if reruns:
        lines += ["## Diagnostic re-runs", "",
                  "Runs repeated once to see whether a failure was sporadic. They do not replace the "
                  "first run in any table above.", ""]
        rows = [{"tool": label(r.tool), "formula": r.formula_id, "expected": r.expected,
                 "result": r.result, "verdict": r.verdict, "seconds": r.seconds} for r in reruns]
        lines += [_md_table(rows, list(rows[0])), ""]
    path = out / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_cells(out: Path, runs: list[Run]) -> Path:
    table = cells(runs)
    path = out / "cells.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        # Unix line endings: the file is committed, and the repository stores text with LF (.gitattributes).
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    return path


# ----------------------------------------------------------------------- comparison of encodings

ENCODING_ORDER = ["compact", "compact (2 constraints)", "paper", "paper (2 constraints)", "neurocodebench"]
ENCODING_NOTES = {
    "compact (2 constraints)": "the compact network; unsafe set m - 1/2 <= N(x) <= m, the same set written as "
                               "two constraints",
    "paper (2 constraints)": "the paper network; unsafe set m - 1/2 <= N(x) <= m, the same set written as two "
                             "constraints",
    "compact": "2n + m ReLUs in 2 hidden layers, one output N(x) = satisfied clauses; unsafe set N(x) >= m - 1/2",
    "paper": "7n + 3m ReLUs in 6 hidden layers, the construction as written in the paper; same output and "
             "unsafe set as compact",
    "neurocodebench": "m + 2n ReLUs in 1 hidden layer, two outputs (the VNN-COMP 2025 benchmark sat_relu); "
                      "unsafe set Y_0 >= 1 and Y_1 <= 0, reachable only at exact 0/1 points",
}


def encodings_of(runs: list[Run]) -> list[str]:
    present = {run.setup for run in runs}
    return [e for e in ENCODING_ORDER if e in present] + sorted(present - set(ENCODING_ORDER))


def compared(runs: list[Run]) -> dict[str, list[Run]]:
    """Per tool, its runs on the formulas it ran in every one of its encodings."""
    result = {}
    for tool in tools_of(runs):
        mine = [r for r in runs if r.tool == tool]
        per_encoding = defaultdict(set)
        for run in mine:
            per_encoding[run.setup].add(run.formula_id)
        common = set.intersection(*per_encoding.values())
        result[tool] = [r for r in mine if r.formula_id in common]
    return result


def encoding_summary(runs: list[Run]) -> list[dict]:
    """One row per (tool, encoding): verdicts on the common formulas, and what only this encoding solved."""
    rows = []
    for tool, mine in compared(runs).items():
        solved_in = defaultdict(set)
        for run in mine:
            if run.solved:
                solved_in[run.setup].add(run.formula_id)
        for encoding in encodings_of(mine):
            group = [r for r in mine if r.setup == encoding]
            others = set().union(*(v for e, v in solved_in.items() if e != encoding))
            verdicts = Counter(r.verdict for r in group)
            times = [r.seconds for r in group if r.solved]
            rows.append({
                "tool": tool, "encoding": encoding, "formulas": len(group), "correct": verdicts["correct"],
                "correct sat": sum(r.solved and r.expected == "sat" for r in group),
                "correct unsat": sum(r.solved and r.expected == "unsat" for r in group),
                "wrong": verdicts["wrong"], "timeout": verdicts["timeout"],
                "other": len(group) - verdicts["correct"] - verdicts["wrong"] - verdicts["timeout"],
                "only here": len(solved_in[encoding] - others),
                "median s (solved)": statistics.median(times) if times else None,
            })
    return rows


def size_bin(series: str, num_vars: int) -> str:
    """Sizes for the per-size table: the ladders as they are, the sat_relu formulas in bins of 10 variables."""
    if series != "satrelu":
        return str(num_vars)
    low = 10 * ((num_vars - 1) // 10) + 1
    return f"{low}-{low + 9}"


def encoding_cells(runs: list[Run]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Solved per (series, answer, size) and (tool, encoding), on the common formulas of each tool."""
    by_tool = compared(runs)
    columns = [(tool, e) for tool, mine in by_tool.items() for e in encodings_of(mine)]
    groups: dict[tuple, list[Run]] = defaultdict(list)
    for mine in by_tool.values():
        for run in mine:
            groups[(run.series, run.expected, size_bin(run.series, run.num_vars), run.num_vars)].append(run)
    merged: dict[tuple, list[Run]] = defaultdict(list)
    for (series, expected, bin_, _), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][3])):
        merged[(series, expected, bin_)].extend(group)
    rows = []
    for (series, expected, bin_), group in merged.items():
        row = {"series": series, "expected": expected, "vars": bin_, "formulas": len({r.formula_id for r in group})}
        for tool, encoding in columns:
            mine = [r for r in group if r.tool == tool and r.setup == encoding]
            row[f"{tool} {encoding}"] = f"{sum(r.solved for r in mine)}/{len(mine)}" if mine else ""
        rows.append(row)
    return rows, columns


def encoding_sizes(runs: list[Run]) -> list[dict]:
    """Network sizes per encoding over the compared formulas (median and largest)."""
    rows = []
    for encoding in encodings_of(runs):
        formulas = {r.formula_id: r for r in runs if r.setup == encoding}.values()
        relus = [f.num_relus for f in formulas]
        params = [dense_parameters(f.encoding, f.num_vars, f.num_clauses) for f in formulas]
        rows.append({"encoding": encoding, "formulas": len(relus),
                     "median ReLUs": int(statistics.median(relus)), "max ReLUs": max(relus),
                     "median params": None if None in params else int(statistics.median(params)),
                     "max params": None if None in params else max(params),
                     "note": ENCODING_NOTES.get(encoding, "")})
    return rows


def write_encoding_comparison(out: Path, runs: list[Run], name: str) -> Path:
    by_tool = compared(runs)
    lines = [f"# Comparison of encodings: {name}", "",
             "Generated by `python -m relu_sat_bench.evaluate --compare`. The same formulas, encoded in different "
             "networks and properties; every encoding is reachable iff the formula is satisfiable. For each tool only "
             "the formulas it ran in every one of its encodings are compared. α,β-CROWN's MIP mode ran on the "
             "`--threshold-in-network` variant of compact and paper (same networks with the threshold moved into "
             "the last bias, see the README); *(2 constraints)* marks the variant with the unsafe set written as "
             "`m - 1/2 <= Y_0 <= m` (`--two-constraints`). *Only here*: solved in this encoding and in none of the tool's other "
             "encodings. Times include each tool's start-up.", ""]
    lines += ["## Encodings", ""]
    lines += [_md_table(encoding_sizes([r for mine in by_tool.values() for r in mine]),
                        ["encoding", "formulas", "median ReLUs", "max ReLUs", "median params", "max params", "note"],
                        ["encoding", "formulas", "ReLUs (median)", "ReLUs (max)", "parameters as stored (median)",
                         "parameters as stored (max)", "network and property"]), ""]
    lines += ["## Verdicts per tool and encoding", ""]
    summary = encoding_summary(runs)
    for row in summary:
        row["tool"] = label(row["tool"])
    lines += [_md_table(summary, list(summary[0])), ""]
    lines += ["## Solved per family and size", ""]
    table, columns = encoding_cells(runs)
    keys = [f"{t} {e}" for t, e in columns]
    lines += [_md_table(table, ["series", "expected", "vars", "formulas", *keys],
                        ["series", "answer", "variables", "formulas", *(f"{label(t)}, {e}" for t, e in columns)]), ""]
    wrong = [r for mine in by_tool.values() for r in mine if r.verdict in ("wrong", "reference-wrong")]
    lines += ["## Wrong answers", ""]
    if wrong:
        rows = [{"tool": label(r.tool), "encoding": r.setup, "formula": r.formula_id, "expected": r.expected,
                 "result": r.result, "verdict": r.verdict} for r in wrong]
        lines += [_md_table(rows, list(rows[0])), ""]
    else:
        lines += ["None.", ""]
    path = out / f"encodings-{name}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- figures

# Reference palette of the dataviz method (light mode): categorical slots 1-3, validated all-pairs.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"marabou": "#2a78d6", "abcrown-bab": "#eb6834", "abcrown-mip": "#1baf7a"}
MARKERS = {"marabou": "o", "abcrown-bab": "s", "abcrown-mip": "^"}
REFERENCE = {"color": INK_2, "marker": "D"}   # the SAT solver: a neutral reference, not a categorical slot


def _style(plt) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9, "svg.fonttype": "none",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.75, "axes.labelcolor": INK_2,
        "axes.titlesize": 9, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.75, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "text.color": INK,
        "svg.hashsalt": "relu-sat-bench",   # fixed ids, so the same data gives byte-identical files
    })


def _line(ax, xs, ys, tool: str | None, markers: bool = True) -> None:
    style = REFERENCE if tool is None else {"color": COLORS.get(tool, MUTED), "marker": MARKERS.get(tool, "o")}
    ax.plot(xs, ys, color=style["color"], marker=style["marker"] if markers else "", linewidth=1.5, markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, solid_capstyle="round", solid_joinstyle="round",
            label=SAT_SOLVER if tool is None else label(tool), zorder=3)


def _panel_grid(plt, title: str, note: str):
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.8), sharey=True)
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.925, "Top row: satisfiable formulas (the tool must find a counterexample). Bottom row: "
             "unsatisfiable formulas (it must prove the property). " + note, ha="left", color=INK_2)
    return fig, axes


def _finish_grid(fig, axes, footnote: str = "") -> None:
    axes[0][4].set_visible(False)
    handles, labels = _collect_handles(axes)
    _legend(fig, handles, labels)
    if footnote:
        fig.text(0.01, 0.01, footnote, ha="left", va="bottom", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04 if footnote else 0, 1, 0.92))


def _panels(runs: list[Run]):
    """(axis index, title, runs) for each panel that has data, in the fixed panel order."""
    layout = []
    for series, expected, title in PANELS:
        mine = [r for r in runs if r.series == series and r.expected == expected]
        row = 0 if expected == "sat" else 1
        col = sum(1 for s, e, _ in PANELS[: PANELS.index((series, expected, title))] if e == expected)
        layout.append(((row, col), title, mine))
    return layout


def _legend(fig, handles, labels) -> None:
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.995, 0.995),
               handlelength=2.2, columnspacing=1.6)


HANG_NOTE = ("Where a line dips and recovers, a run hung before the solver started; the diagnostic re-runs "
             "in summary.md solved most of these.")


def plot_solved_by_size(runs: list[Run], out: Path) -> Path:
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = _panel_grid(plt, "Share of instances solved within 60 s, by formula size",
                            "Beyond the benchmark sizes only α,β-CROWN (MIP) was run.")
    for (row, col), title, mine in _panels(runs):
        ax = axes[row][col]
        ax.set_title(title, loc="left")
        for tool in tools_of(mine):
            sizes = sorted({r.num_vars for r in mine if r.tool == tool})
            share = [100 * statistics.mean(r.solved for r in mine if r.tool == tool and r.num_vars == n)
                     for n in sizes]
            _line(ax, sizes, share, tool)
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xscale("log")
        _size_ticks(ax)
        ax.set_xlabel("variables (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("solved (%)")
    _finish_grid(fig, axes, HANG_NOTE)
    path = out / "solved-by-size.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    return path


def plot_runtime_by_size(runs: list[Run], out: Path) -> Path:
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = _panel_grid(plt, "Median time per formula size: verifiers against the SAT solver",
                            "Unsolved runs count as the 60 s limit (gray line).")
    for (row, col), title, mine in _panels(runs):
        ax = axes[row][col]
        ax.set_title(title, loc="left")
        ax.set_yscale("log")
        for tool in tools_of(mine):
            sizes = sorted({r.num_vars for r in mine if r.tool == tool})
            medians = [statistics.median(r.seconds if r.solved else r.timeout
                                         for r in mine if r.tool == tool and r.num_vars == n) for n in sizes]
            _line(ax, sizes, medians, tool)
        formulas = formulas_of(mine)
        sizes = sorted({f.num_vars for f in formulas.values()})
        sat = [max(statistics.median(f.sat_seconds for f in formulas.values() if f.num_vars == n), 1e-4)
               for n in sizes]
        _line(ax, sizes, sat, None)
        ax.axhline(60, color=AXIS, linewidth=0.75, zorder=1)
        ax.set_ylim(5e-5, 200)
        ax.set_xscale("log")
        _size_ticks(ax)
        ax.set_xlabel("variables (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("seconds (log)")
    _finish_grid(fig, axes, HANG_NOTE)
    path = out / "runtime-by-size.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    return path


def plot_cactus(runs: list[Run], out: Path) -> Path:
    """Instances solved within a time limit t, for every t: the usual solver comparison."""
    import matplotlib.pyplot as plt
    _style(plt)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    formulas = formulas_of(runs)
    total = len(formulas)
    for tool in [*tools_of(runs), None]:
        if tool is None:
            times = sorted(max(f.sat_seconds, 1e-4) for f in formulas.values() if f.sat_solved)
        else:
            times = sorted(r.seconds for r in runs if r.tool == tool and r.solved)
        _line(ax, times, range(1, len(times) + 1), tool, markers=False)
        if times:  # direct label at the end of the line, in text ink
            ax.annotate(f"{SAT_SOLVER if tool is None else label(tool)}: {len(times)}", (times[-1], len(times)),
                        xytext=(-2, 6), textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlim(5e-5, 100)
    ax.set_ylim(0, total * 1.05)
    ax.axhline(total, color=AXIS, linewidth=0.75, zorder=1)
    ax.set_xlabel("time limit per instance (seconds, log)")
    ax.set_ylabel("instances solved")
    fig.suptitle(f"Instances solved within a time limit ({total} instances)", x=0.02, y=0.975, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.text(0.02, 0.92, f"{SAT_SOLVER}: time in the generator. Verifiers: time of the whole run, start-up "
             "included\n(about 0.4 s for Marabou, 4 s for α,β-CROWN).", ha="left", va="top", color=INK_2, fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(0.37, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    path = out / "cactus.svg"
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    return path


def _size_ticks(ax) -> None:
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    low, high = ax.get_xlim()
    ticks = [t for t in (5, 10, 20, 50, 100, 200) if low <= t <= high]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:g}")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def _collect_handles(axes):
    handles, labels = [], []
    for ax in axes.flat:
        for handle, text in zip(*ax.get_legend_handles_labels()):
            if text not in labels:
                handles.append(handle)
                labels.append(text)
    order = [label(t) for t in TOOL_ORDER] + [SAT_SOLVER]
    pairs = sorted(zip(handles, labels), key=lambda p: order.index(p[1]) if p[1] in order else len(order))
    return [p[0] for p in pairs], [p[1] for p in pairs]


# ----------------------------------------------------------------------------- CLI


def evaluate(benches: list[Path], extensions: list[Path], out: Path, figures: bool = True) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    main = combine([load_runs(b) for b in benches])
    extension = combine([load_runs(b) for b in extensions])
    reruns = [r for b in [*benches, *extensions] for r in load_runs(b, reruns=True)]
    written = [write_summary(out, main, extension, reruns), write_cells(out, main)]
    if figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            print("matplotlib is not installed: no figures (pip install -e \".[report]\")")
        else:
            both = combine([main, extension])
            written += [plot_solved_by_size(both, out), plot_runtime_by_size(both, out), plot_cactus(main, out)]
    return written


def compare(folders: list[Path], out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    runs = combine([load_runs(f) for f in folders], by_encoding=True)
    if not runs:
        raise SystemExit("no results in " + ", ".join(map(str, folders)))
    return write_encoding_comparison(out, runs, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m relu_sat_bench.evaluate", description=__doc__.split("\n")[0])
    parser.add_argument("--bench", type=Path, nargs="+", help="folders with the same formulas")
    parser.add_argument("--extension", type=Path, nargs="*", default=[], help="folders with further formulas")
    parser.add_argument("--compare", type=Path, nargs="+", help="folders with the same formulas in other encodings")
    parser.add_argument("--name", default="comparison", help="with --compare: writes encodings-<name>.md")
    parser.add_argument("--out", type=Path, default=Path("report"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if (args.bench is None) == (args.compare is None):
        parser.error("give either --bench or --compare")
    if args.compare:
        print("written:", compare(args.compare, args.out, args.name))
        return 0
    for path in evaluate(args.bench, args.extension, args.out, figures=not args.no_figures):
        print("written:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
