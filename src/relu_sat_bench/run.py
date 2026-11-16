"""Run a verification tool on a benchmark folder and check every answer.

Usage (normally inside a tool's Docker image, see ``docker/``)::

    python -m relu_sat_bench.run --bench /bench --tool marabou

For every line of ``instances.csv`` the tool is started as a separate process with the
VNN-COMP calling convention ``<tool command> ONNX VNNLIB RESULT_FILE TIMEOUT``. The runner

* enforces the time limit from outside: the whole process group is killed at
  ``timeout + grace`` seconds, because tools do not always stop on time;
* reads the result file (first line ``sat``/``unsat``/``timeout``/``unknown``/...);
* compares the answer with the expected one from ``metadata.csv``;
* checks every counterexample independently: inside the input box, network output above
  the threshold (exact float64 network), and decodes to a satisfying assignment of the CNF;
* appends one row per instance to ``results/<tool>.csv`` right away, so an interrupted run
  can be resumed (instances already in the file are skipped);
* measures the run time with a monotonic clock and, separately, the wall-clock time. If the
  computer sleeps during a run, the wall clock runs on while the monotonic clock stops; tools
  that check their time limit with the wall clock then give up early. Such runs are flagged
  and can be repeated with ``--retry-suspended``. Crashed runs can be repeated with
  ``--retry-errors``, single instances with ``--rerun --only ...``.

Verdicts: ``correct``, ``wrong`` (the opposite of the expected answer, or a counterexample
that does not hold), ``sat-no-cex`` (sat without a counterexample), ``timeout`` (killed,
answered too late, or the tool itself reported ``timeout``), ``no-answer`` (the tool gave up
in time with ``unknown``), ``error``, and
``reference-wrong`` (a valid counterexample for an instance we expected to be unsat, which
would mean a bug on our side; it has never happened, but it must not go unnoticed).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from relu_sat_bench.cnf import CNF
from relu_sat_bench.export import METADATA_CSV
from relu_sat_bench.generate import FORMULAS_CSV, read_rows
from relu_sat_bench.reduction import decode_assignment, reduce_cnf
from relu_sat_bench.tools import TOOLS

RESULT_COLUMNS = [
    "tool", "onnx", "vnnlib", "formula_id", "family", "encoding",
    "expected",       # sat | unsat
    "result",         # what the tool answered (normalized), or timeout if it was killed
    "verdict",        # correct | wrong | sat-no-cex | timeout | no-answer | error
    "seconds",        # run time of the tool process (monotonic clock: stops while the computer sleeps)
    "wall_seconds",   # the same on the wall clock; much larger than seconds = the computer slept
    "timeout",
    "cex_check",      # for sat answers: valid | invalid: <reason> | missing
    "cex_output",     # network output at the counterexample (exact float64 network)
    "num_inputs", "num_relus", "num_clauses", "sat_solver_seconds",
]
OUTPUT_TOLERANCE = 1e-6  # counterexample: output >= threshold - tolerance
INPUT_TOLERANCE = 1e-6   # counterexample: inputs within the box up to this tolerance
SYNONYMS = {"violated": "sat", "holds": "unsat", "safe": "unsat", "unsafe": "sat"}
SUSPEND_TOLERANCE = 10.0  # seconds the wall clock may run ahead before a run counts as suspended
_wall_clock = time.time    # replaced in tests


def was_suspended(row: dict) -> bool:
    """Did the computer sleep during this run? (Unknown for rows written before 1 Nov 2026.)"""
    try:
        return float(row["wall_seconds"]) - float(row["seconds"]) > SUSPEND_TOLERANCE
    except (KeyError, TypeError, ValueError):
        return False


def describe_exit(code: int | None) -> str:
    """``exit code 139 = signal 11 (SIGSEGV)`` and the like."""
    if code is None:
        return "no exit code"
    sig = -code if code < 0 else code - 128 if code > 128 else None
    if sig is not None:
        try:
            return f"exit code {code} = signal {sig} ({signal.Signals(sig).name})"
        except ValueError:
            pass
    return f"exit code {code}"


# ------------------------------------------------------------------- parsing


def parse_result(text: str) -> tuple[str, dict[str, float]]:
    """(status, values) from a VNN-COMP result file.

    The first non-empty line is the status. The counterexample, if any, is either
    ``((X_0 0.5) (X_1 1.0) ... (Y_0 3.0))`` or blocks ``X real [1,n]`` followed by values.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", {}
    status = lines[0].lower()
    status = SYNONYMS.get(status, status)
    rest = "\n".join(lines[1:])
    values = {name: float(value) for name, value in
              re.findall(r"\(\s*([XY]_\d+)\s+([-+0-9.eEinfa]+)\s*\)", rest)}
    if not values:  # vnnlib2-style blocks: "X real [1,3]" then one value per line
        current, index = None, 0
        for line in lines[1:]:
            header = re.match(r"([XY])\s+real\b", line)
            if header:
                current, index = header.group(1), 0
            elif current is not None:
                try:
                    values[f"{current}_{index}"] = float(line)
                    index += 1
                except ValueError:
                    current = None
    return status, values


@dataclass(frozen=True)
class Check:
    verdict: str
    cex_check: str = ""
    cex_output: str = ""


def check_answer(status: str, values: dict[str, float], expected: str, cnf: CNF, encoding: str) -> Check:
    """Judge a finished tool run (not killed) against the expected answer."""
    if status == "unsat":
        return Check("correct" if expected == "unsat" else "wrong")
    if status != "sat":
        if status == "timeout":
            return Check("timeout")  # the tool stopped itself at its time limit
        return Check("no-answer" if status == "unknown" else "error")

    n = cnf.num_vars
    xs = [values.get(f"X_{i}") for i in range(n)]
    if any(x is None for x in xs):
        # A sat answer without (complete) counterexample cannot be checked.
        return Check("wrong" if expected == "unsat" else "sat-no-cex", "missing")
    instance = reduce_cnf(cnf, encoding)
    output = float(instance.network.forward(xs)[0])  # exact network, float64
    problems = []
    if any(x < -INPUT_TOLERANCE or x > 1 + INPUT_TOLERANCE for x in xs):
        problems.append("input outside the box")
    if output < float(instance.threshold) - OUTPUT_TOLERANCE:
        problems.append(f"output {output:g} below threshold {float(instance.threshold):g}")
    if not cnf.is_satisfied_by(decode_assignment([min(max(x, 0.0), 1.0) for x in xs])):
        problems.append("does not decode to a satisfying assignment")
    if problems:
        return Check("wrong", "invalid: " + "; ".join(problems), f"{output:.9g}")
    if expected == "unsat":
        # A valid counterexample proves the formula satisfiable: then OUR reference is wrong.
        return Check("reference-wrong", "valid", f"{output:.9g}")
    return Check("correct", "valid", f"{output:.9g}")


# ------------------------------------------------------------------- running


def run_tool(command: Sequence[str], onnx: Path, vnnlib: Path, result_file: Path, timeout: float,
             grace: float, log_file: Path) -> tuple[bool, float, int | None, float]:
    """Run one instance. Returns (killed, seconds, exit code, wall-clock seconds)."""
    result_file.unlink(missing_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    args = [*command, str(onnx), str(vnnlib), str(result_file), f"{timeout:g}"]
    # The adapters are modules of this package: make sure the child finds it even when the
    # package is used from a source checkout instead of being installed.
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, env.get("PYTHONPATH", "")) if p)
    with log_file.open("w", encoding="utf-8") as log:
        start, wall_start = time.perf_counter(), _wall_clock()
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=(os.name == "posix"))
        try:
            code = process.wait(timeout=timeout + grace)
            killed = False
        except subprocess.TimeoutExpired:
            _kill(process)
            killed, code = True, None
        return killed, time.perf_counter() - start, code, _wall_clock() - wall_start


def _kill(process: subprocess.Popen) -> None:
    """Terminate the tool and everything it started (the whole process group)."""
    if os.name == "posix":
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                continue
    else:  # pragma: no cover - the tools only run on Linux (Docker)
        # Windows has no process groups: only the tool process itself is killed here, processes it
        # started may survive. Fine for tests with fake tools; real runs happen in Linux containers.
        process.kill()
        process.wait()


def load_instances(bench: Path) -> list[dict]:
    """instances.csv rows joined with metadata.csv and formulas.csv."""
    metadata = {}
    with (bench / METADATA_CSV).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metadata[row["onnx"]] = row
    formulas = read_rows(bench / FORMULAS_CSV)
    instances = []
    with (bench / "instances.csv").open(newline="", encoding="utf-8") as handle:
        for onnx, vnnlib, timeout in csv.reader(handle):
            meta = metadata[onnx]
            instances.append({**meta, "vnnlib": vnnlib, "timeout": timeout,
                              "cnf_file": formulas[meta["formula_id"]]["cnf_file"]})
    return instances


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m relu_sat_bench.run",
                                     description="Run a verifier on a benchmark and check its answers.")
    parser.add_argument("--bench", type=Path, required=True, help="benchmark folder (with instances.csv)")
    parser.add_argument("--tool", required=True,
                        help=f"tool name for the results file; built-in commands: {', '.join(TOOLS)}")
    parser.add_argument("--command", nargs=argparse.REMAINDER,
                        help="custom tool command (instead of the built-in one); must be the last option")
    parser.add_argument("--grace", type=float, default=10.0,
                        help="seconds after the timeout before the tool is killed (default 10)")
    parser.add_argument("--timeout", type=float, help="override the timeouts from instances.csv")
    parser.add_argument("--only", nargs="+", help="only these instances (ONNX file names without .onnx)")
    parser.add_argument("--limit", type=int, help="run at most this many new instances")
    parser.add_argument("--rerun", action="store_true",
                        help="run instances again that already have a result (with --only: just those)")
    parser.add_argument("--retry-errors", action="store_true", help="run crashed instances (verdict error) again")
    parser.add_argument("--retry-suspended", action="store_true",
                        help="run instances again during which the computer slept")
    args = parser.parse_args(argv)

    command = args.command or TOOLS.get(args.tool)
    if not command:
        parser.error(f"unknown tool {args.tool!r}; use one of {list(TOOLS)} or pass --command")
    bench: Path = args.bench
    out_dir = bench / "results"
    out_dir.mkdir(exist_ok=True)
    results_csv = out_dir / f"{args.tool}.csv"
    done, header = {}, RESULT_COLUMNS
    if results_csv.exists():
        with results_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            done = {row["onnx"]: row for row in reader}
            header = reader.fieldnames
    only = set(args.only) if args.only else None
    drop = {onnx for onnx, row in done.items()
            if (args.rerun and (only is None or Path(onnx).stem in only))
            or (args.retry_errors and row["verdict"] == "error")
            or (args.retry_suspended and was_suspended(row))}
    if drop or header != RESULT_COLUMNS:  # rewrite: without the rows to repeat, in the current columns
        done = {onnx: row for onnx, row in done.items() if onnx not in drop}
        with results_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(done.values())
        if drop:
            print(f"{args.tool}: {len(drop)} instances will be run again")

    instances = load_instances(bench)
    if args.tool.startswith("abcrown-mip") and any(inst.get("vnnlib_rhs") != "0.0" for inst in instances):
        print("WARNING: alpha,beta-CROWN's MIP mode compares single-constraint properties with 0 instead of "
              "their right-hand side, so it fails on these instances. Use a benchmark built with "
              "--threshold-in-network (see the README).", flush=True)
    todo = [inst for inst in instances if inst["onnx"] not in done]
    if args.only:
        todo = [inst for inst in todo if Path(inst["onnx"]).stem in set(args.only)]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{args.tool}: {len(todo)} instances to run, {len(done)} already done")

    new_file = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if new_file:
            writer.writeheader()
        for inst in todo:
            timeout = args.timeout if args.timeout is not None else float(inst["timeout"])
            stem = Path(inst["onnx"]).stem
            raw = out_dir / "raw" / args.tool / f"{stem}.txt"
            raw.parent.mkdir(parents=True, exist_ok=True)
            killed, seconds, code, wall = run_tool(command, bench / inst["onnx"], bench / inst["vnnlib"], raw,
                                                   timeout, args.grace, out_dir / "logs" / args.tool / f"{stem}.log")
            if killed:
                status, check = "timeout", Check("timeout")
            else:
                status, values = parse_result(raw.read_text(encoding="utf-8")) if raw.exists() else ("", {})
                cnf = CNF.from_dimacs((bench / inst["cnf_file"]).read_text(encoding="utf-8"))
                check = check_answer(status, values, inst["expected"], cnf, inst["encoding"])
                if not status:
                    status = f"no result file ({describe_exit(code)})"
                elif seconds > timeout and check.verdict in ("correct", "wrong", "sat-no-cex"):
                    check = Check("timeout", check.cex_check, check.cex_output)  # answered too late
            row = {
                "tool": args.tool, "onnx": inst["onnx"], "vnnlib": inst["vnnlib"],
                "formula_id": inst["formula_id"], "family": inst["family"], "encoding": inst["encoding"],
                "expected": inst["expected"], "result": status, "verdict": check.verdict,
                "seconds": f"{seconds:.3f}", "wall_seconds": f"{wall:.3f}", "timeout": f"{timeout:g}",
                "cex_check": check.cex_check, "cex_output": check.cex_output,
                "num_inputs": inst["num_inputs"], "num_relus": inst["num_relus"],
                "num_clauses": inst["num_clauses"], "sat_solver_seconds": inst["sat_solver_seconds"],
            }
            writer.writerow(row)
            handle.flush()
            done[inst["onnx"]] = row
            flag = "  <-- WRONG" if check.verdict == "wrong" else ""
            if was_suspended(row):
                flag += f"  <-- the computer slept ({wall - seconds:.0f} s); repeat with --retry-suspended"
            print(f"{stem:12s} {inst['formula_id']:36s} expected {inst['expected']:5s} "
                  f"{status:8s} {check.verdict:10s} {seconds:8.2f}s{flag}", flush=True)

    counts = Counter(row["verdict"] for row in done.values())
    print(f"{args.tool}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    suspended = sum(was_suspended(row) for row in done.values())
    if suspended:
        print(f"{args.tool}: the computer slept during {suspended} runs; repeat them with --retry-suspended")
    if counts.get("wrong"):
        print(f"WARNING: {counts['wrong']} wrong answers, see {results_csv}")
    if counts.get("reference-wrong"):
        print(f"ERROR: {counts['reference-wrong']} valid counterexamples for instances expected unsat")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
