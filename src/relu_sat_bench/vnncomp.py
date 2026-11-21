"""Compare with VNN-COMP 2025: how large are the networks on which the competition's verifiers failed?

Two steps::

    python -m relu_sat_bench.vnncomp collect --results <vnncomp2025_results> --benchmarks <vnncomp2025_benchmarks> \\
        --out results/vnncomp2025
    python -m relu_sat_bench.vnncomp report --data results/vnncomp2025 \\
        --bench results/final results/final-shifted --extension results/scaling-shifted --out report

``collect`` reads clones of the public repositories ``VNN-COMP/vnncomp2025_results`` and
``VNN-COMP/vnncomp2025_benchmarks`` and writes two files (no networks, no properties):

* ``instances.csv``: per instance the benchmark, its track (``regular``: scored, ``extended``: unscored), its index,
  network and property, the official result (``sat``, ``unsat``, or ``unknown`` if no tool answered; from the
  per-instance table ``SCORING-SMALL-TOL/{scored,unscored}/latex/longtable.tex``) and α,β-CROWN's own result;
* ``networks.csv``: per network the parameters as stored and the nonzero ones (see :func:`count_parameters`).

``report`` writes ``vnncomp2025.md``: the smallest networks with an instance no tool solved, or α,β-CROWN did not
solve, next to the limits of the tools on relu-sat-bench (computed from ``--bench``/``--extension`` as in
``summary.md``), and a table per benchmark.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from collections import defaultdict
from pathlib import Path

YEAR = "2025"
INSTANCE_COLUMNS = ["benchmark", "track", "id", "onnx", "vnnlib", "result", "alpha_beta_crown"]
NETWORK_COLUMNS = ["onnx", "stored", "nonzero", "note"]
ABC = "alpha_beta_crown"
# The scoring leaves out these warm-up instances (process_results.py of the results repository).
WARM_UP = ("test_nano", "test_tiny")


# ----------------------------------------------------------------------------- collect


def count_parameters(path: Path) -> tuple[int, int]:
    """Parameters of an ONNX network (``.onnx`` or ``.onnx.gz``): (stored, nonzero).

    Stored: every floating-point number in the initializers and Constant nodes (weights, biases, normalisation
    constants), zeros included. Integer tensors (shapes, indices) are not parameters and are not counted.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    model = onnx.load_from_string(data)
    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        if node.op_type == "Constant":
            tensors += [a.t for a in node.attribute if a.type == onnx.AttributeProto.TENSOR]
    stored = nonzero = 0
    for tensor in tensors:
        array = numpy_helper.to_array(tensor)
        if np.issubdtype(array.dtype, np.floating):
            stored += array.size
            nonzero += int(np.count_nonzero(array))
    return stored, nonzero


def benchmark_key(category: str) -> str:
    """``2025 Cora 2024`` (long table) -> ``cora_2024`` (category in the results files)."""
    return "_".join(category.split()[1:]).lower()


def official_results(results_repo: Path) -> dict[str, dict]:
    """Per benchmark: track and the official result of every instance, in order (``unknown``: no tool answered)."""
    out: dict[str, dict] = {}
    for scored, track in (("scored", "regular"), ("unscored", "extended")):
        table = results_repo / "SCORING-SMALL-TOL" / scored / "latex" / "longtable.tex"
        for line in table.read_text(encoding="utf-8").splitlines():
            if not re.match(rf"^{YEAR} .* & \d+ & ", line):
                continue
            category, ident, result = (x.strip() for x in line.split("&")[:3])
            entry = out.setdefault(benchmark_key(category), {"track": track, "results": []})
            assert int(ident) == len(entry["results"]), f"{category}: instance {ident} out of order"
            text = re.sub(r"[~{}]|\\textsc", "", result).strip()
            entry["results"].append(text if text in ("sat", "unsat") else "unknown")
    return out


def _relative(path: str) -> str:
    """``vnncomp2025_benchmarks/benchmarks/cora_2024/./onnx/x.onnx`` -> ``benchmarks/cora_2024/onnx/x.onnx``."""
    rest = path.split("benchmarks/", 1)[1].replace("/./", "/")
    return rest if rest.startswith("benchmarks/") else "benchmarks/" + rest


def tool_results(results_repo: Path) -> dict[str, dict[str, list[tuple[str, str, str]]]]:
    """Per benchmark and tool: (network, property, result) of every instance, warm-up instances left out."""
    out: dict[str, dict[str, list]] = defaultdict(dict)
    for path in sorted(results_repo.glob("*/results.csv")):
        if path.parent.name.startswith("SCORING"):
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if any(w in row[1] for w in WARM_UP):
                    continue
                out[row[0]].setdefault(path.parent.name, []).append((_relative(row[1]), _relative(row[2]),
                                                                    row[4].strip().lower()))
    return out


def collect_instances(results_repo: Path) -> list[dict]:
    official, tools = official_results(results_repo), tool_results(results_repo)
    rows = []
    for benchmark, info in official.items():
        per_tool = tools[benchmark]
        lists = {tuple((n, p) for n, p, _ in runs) for runs in per_tool.values()}
        if len(lists) != 1 or len(next(iter(lists))) != len(info["results"]):
            raise ValueError(f"{benchmark}: the tools' instance lists differ from the official table")
        abc = [r for _, _, r in per_tool[ABC]] if ABC in per_tool else None
        for i, ((onnx, vnnlib), result) in enumerate(zip(next(iter(lists)), info["results"])):
            rows.append({"benchmark": benchmark, "track": info["track"], "id": i, "onnx": onnx, "vnnlib": vnnlib,
                         "result": result, "alpha_beta_crown": abc[i] if abc else ""})
    return rows


def collect_networks(instances: list[dict], benchmarks_repo: Path) -> list[dict]:
    rows = []
    for onnx in sorted({row["onnx"] for row in instances}):
        path = next((p for p in (benchmarks_repo / onnx, benchmarks_repo / (onnx + ".gz")) if p.exists()), None)
        if path is None:
            rows.append({"onnx": onnx, "stored": "", "nonzero": "", "note": "not in the repository"})
            continue
        try:
            stored, nonzero = count_parameters(path)
        except Exception as error:  # noqa: BLE001 - recorded, the network is then left out
            rows.append({"onnx": onnx, "stored": "", "nonzero": "", "note": f"unreadable: {type(error).__name__}"})
        else:
            rows.append({"onnx": onnx, "stored": stored, "nonzero": nonzero, "note": ""})
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def collect(results_repo: Path, benchmarks_repo: Path, out: Path) -> list[Path]:
    instances = collect_instances(results_repo)
    return [_write_csv(out / "instances.csv", INSTANCE_COLUMNS, instances),
            _write_csv(out / "networks.csv", NETWORK_COLUMNS, collect_networks(instances, benchmarks_repo))]


# ----------------------------------------------------------------------------- report


def solved(result: str) -> bool:
    """A tool's own result counts as solved if it is an answer (sat/unsat, holds/violated)."""
    return result.startswith(("sat", "unsat")) or result in ("holds", "violated")


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(x: int | None) -> str:
    """Compact count: 456, 4.2k, 55k, 1.5M."""
    if x is None:
        return "–"
    for unit, factor in (("M", 10 ** 6), ("k", 10 ** 3)):
        if x >= factor:
            value = x / factor
            return (f"{value:.1f}".rstrip("0").rstrip(".") if value < 10 else f"{value:.0f}") + unit
    return str(x)


def _span(values: list[int]) -> str:
    if not values:
        return "–"
    low, high = fmt(min(values)), fmt(max(values))
    return low if low == high else f"{low}–{high}"


def benchmark_rows(data: Path) -> list[dict]:
    """Per benchmark: instances, how many no tool / α,β-CROWN solved, network sizes, the smallest open networks."""
    networks = {row["onnx"]: row for row in _read(data / "networks.csv")}
    nonzero = {n: int(r["nonzero"]) for n, r in networks.items() if r["nonzero"]}
    stored = {n: int(r["stored"]) for n, r in networks.items() if r["stored"]}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in _read(data / "instances.csv"):
        groups[row["benchmark"]].append(row)
    rows = []
    for benchmark, instances in groups.items():
        nets = sorted({r["onnx"] for r in instances})
        nobody = [r["onnx"] for r in instances if r["result"] == "unknown"]
        ran_abc = any(r["alpha_beta_crown"] for r in instances)
        abc_open = [r["onnx"] for r in instances if ran_abc and not solved(r["alpha_beta_crown"])]
        known = [n for n in nets if n in nonzero]
        rows.append({
            "benchmark": benchmark, "track": instances[0]["track"], "instances": len(instances),
            "no tool": len(nobody), "abc open": len(abc_open) if ran_abc else None,
            "stored": _span([stored[n] for n in known]), "nonzero": _span([nonzero[n] for n in known]),
            "counted": f"{len(known)}/{len(nets)}",
            "min nonzero": min((nonzero[n] for n in known), default=None),
            "no tool min": min((nonzero[n] for n in nobody if n in nonzero), default=None),
            "abc min": min((nonzero[n] for n in abc_open if n in nonzero), default=None),
        })
    rows.sort(key=lambda r: (r["min nonzero"] is None, r["min nonzero"] or 0, r["benchmark"]))
    return rows


def _smallest(rows: list[dict], key: str, track: str | None = None) -> tuple[int | None, str]:
    candidates = [r for r in rows if r[key] is not None and (track is None or r["track"] == track)]
    if not candidates:
        return None, "–"
    best = min(candidates, key=lambda r: r[key])
    return best[key], best["benchmark"]


def our_limits(benches: list[Path], extensions: list[Path]) -> list[dict]:
    """The rows of "Smallest networks at each tool's limit" in summary.md (first real failure, fails on half)."""
    from relu_sat_bench import evaluate as ev
    main = ev.combine([ev.load_runs(b) for b in benches])
    extension = ev.combine([ev.load_runs(b) for b in extensions])
    return ev.smallest_limits(ev.limits(ev.combine([main, extension])))


def write_report(data: Path, out: Path, ours: list[dict] | None = None) -> Path:
    from relu_sat_bench.evaluate import label
    rows = benchmark_rows(data)
    networks = _read(data / "networks.csv")
    left_out = [r for r in networks if not r["nonzero"]]
    lines = [f"# VNN-COMP {YEAR}: network size where the verifiers failed", "",
             "Generated by `python -m relu_sat_bench.vnncomp report` from `results/vnncomp2025/`, which `collect` "
             f"builds from the public repositories `VNN-COMP/vnncomp{YEAR}_results` (official per-instance table: "
             "*no tool* = no participating tool answered; α,β-CROWN's own `results.csv`) and "
             f"`VNN-COMP/vnncomp{YEAR}_benchmarks` (the networks).", "",
             "*Parameters*: every floating-point number in the initializers and Constant nodes of the ONNX file "
             "(weights, biases, normalisation constants), zeros included; *nonzero*: those that are not 0. Trained "
             "networks have almost all parameters nonzero. VNN-COMP ran on AWS instances chosen by the teams, with "
             "time limits set per benchmark; relu-sat-bench: 60 s per instance, 4 CPU cores, no GPU.", ""]
    if left_out:
        notes = ", ".join(f"`{Path(r['onnx']).name}` ({r['note']})" for r in left_out)
        lines += [f"Not counted: {notes}.", ""]
    lines += ["## Smallest networks at which the verifiers fail", "", "| | nonzero parameters |", "|---|---|"]
    for row in ours or []:
        lines.append(f"| relu-sat-bench, {label(row['tool'])}: first real failure / fails on half | "
                     f"{fmt(row['first nonzero'])} ({row['first where']}) / {fmt(row['fails nonzero'])} "
                     f"({row['fails where']}) |")
    every, every_name = _smallest(rows, "no tool min")
    regular, regular_name = _smallest(rows, "no tool min", "regular")
    abc, abc_name = _smallest(rows, "abc min")
    lines += [f"| VNN-COMP {YEAR}, smallest network with an instance no tool solved | {fmt(every)} ({every_name}); "
              f"regular track: {fmt(regular)} ({regular_name}) |",
              f"| VNN-COMP {YEAR}, smallest network with an instance α,β-CROWN did not solve | {fmt(abc)} ({abc_name}) |",
              "", "## Per benchmark", "", "Sorted by the smallest network of the benchmark.", "",
              "| benchmark | track | instances | no tool | α,β-CROWN open | parameters | nonzero | networks counted "
              "| smallest network with an instance no tool solved (nonzero) | ... α,β-CROWN did not solve |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['benchmark']} | {r['track']} | {r['instances']} | {r['no tool']} | "
                     f"{'–' if r['abc open'] is None else r['abc open']} | {r['stored']} | {r['nonzero']} | "
                     f"{r['counted']} | {fmt(r['no tool min'])} | {fmt(r['abc min'])} |")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"vnncomp{YEAR}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m relu_sat_bench.vnncomp", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="read the VNN-COMP repositories, write instances.csv and networks.csv")
    c.add_argument("--results", type=Path, required=True, help=f"clone of VNN-COMP/vnncomp{YEAR}_results")
    c.add_argument("--benchmarks", type=Path, required=True, help=f"clone of VNN-COMP/vnncomp{YEAR}_benchmarks")
    c.add_argument("--out", type=Path, default=Path(f"results/vnncomp{YEAR}"))
    r = sub.add_parser("report", help=f"write vnncomp{YEAR}.md")
    r.add_argument("--data", type=Path, default=Path(f"results/vnncomp{YEAR}"))
    r.add_argument("--bench", type=Path, nargs="*", default=[], help="our benchmark folders (as for evaluate)")
    r.add_argument("--extension", type=Path, nargs="*", default=[], help="our extension folders (as for evaluate)")
    r.add_argument("--out", type=Path, default=Path("report"))
    args = parser.parse_args(argv)
    if args.command == "collect":
        for path in collect(args.results, args.benchmarks, args.out):
            print("written:", path)
    else:
        ours = our_limits(args.bench, args.extension) if args.bench else None
        print("written:", write_report(args.data, args.out, ours))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
