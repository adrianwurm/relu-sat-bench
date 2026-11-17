"""Tests for the runner, with small fake tools that behave well or badly on purpose.

Skipped if onnx, onnxruntime or python-sat are missing. The Marabou test additionally needs
maraboupy (Linux/macOS only).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("pysat")

from relu_sat_bench.benchmark import build  # noqa: E402
from relu_sat_bench.cnf import CNF  # noqa: E402
from relu_sat_bench import run as run_module  # noqa: E402
from relu_sat_bench.run import check_answer, describe_exit, main, parse_result  # noqa: E402
from relu_sat_bench.tools import abcrown  # noqa: E402

SPEC = [
    ["random", "--vars", "6", "--ratio", "3", "6", "--count", "2"],
    ["pigeonhole", "--holes", "2"],
    ["ordering", "--size", "3"],
    ["tseitin", "--vertices", "4", "--parity", "even"],
]

# --------------------------------------------------------------- fake tools

PRELUDE = textwrap.dedent('''
    import itertools, re, sys, time
    import numpy as np
    onnx_path, vnnlib_path, result_path, timeout = sys.argv[1:5]

    def write(text):
        with open(result_path, "w") as f:
            f.write(text)

    def num_inputs():
        import onnxruntime
        return onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).get_inputs()[0].shape[1]

    def search():
        """Brute force over all 0/1 points (enough: the maximum is attained there)."""
        import onnxruntime
        session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        n = session.get_inputs()[0].shape[1]
        text = open(vnnlib_path).read()
        threshold = float(re.search(r"\\(assert \\(>= Y_0 ([-\\d.]+)\\)\\)", text).group(1))
        for bits in itertools.product((0.0, 1.0), repeat=n):
            y = session.run(None, {"X": np.array([bits], dtype=np.float32)})[0][0, 0]
            if y >= threshold:
                return bits, float(y)
        return None, None

    def cex(bits, y):
        pairs = [f"(X_{i} {b})" for i, b in enumerate(bits)] + [f"(Y_0 {y})"]
        return "(" + "\\n ".join(pairs) + ")"
''')

FAKE_TOOLS = {
    "oracle": '''
        bits, y = search()
        write("unsat\\n" if bits is None else "sat\\n" + cex(bits, y) + "\\n")
    ''',
    # same answers, counterexample in the block format ("X real [1,n]" + values)
    "oracle-blocks": '''
        bits, y = search()
        if bits is None:
            write("unsat\\n")
        else:
            body = "\\n".join(str(b) for b in bits)
            write(f"sat\\nX real [1,{len(bits)}]\\n{body}\\nY real [1,1]\\n{y}\\n")
    ''',
    # like "oracle", but checks every output constraint of the property (neurocodebench: Y_0 >= 1, Y_1 <= 0)
    "oracle-multi": '''
        import onnxruntime
        session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        n = session.get_inputs()[0].shape[1]
        constraints = re.findall(r"\\(assert \\((<=|>=) Y_(\\d+) ([-\\d.]+)\\)\\)", open(vnnlib_path).read())
        found = None
        for bits in itertools.product((0.0, 1.0), repeat=n):
            y = session.run(None, {"X": np.array([bits], dtype=np.float32)})[0][0]
            if all(y[int(k)] >= float(v) if op == ">=" else y[int(k)] <= float(v) for op, k, v in constraints):
                found = bits, y
                break
        if found is None:
            write("unsat\\n")
        else:
            pairs = [f"(X_{i} {b})" for i, b in enumerate(found[0])] + [f"(Y_{k} {v})" for k, v in enumerate(found[1])]
            write("sat\\n(" + "\\n ".join(pairs) + ")\\n")
    ''',
    "liar": '''
        write("unsat\\n")
    ''',
    # claims sat everywhere with the center point, where the network outputs 0
    "bad-cex": '''
        write("sat\\n" + cex([0.5] * num_inputs(), 0.0) + "\\n")
    ''',
    "no-cex": '''
        bits, y = search()
        write("unsat\\n" if bits is None else "sat\\n")
    ''',
    "gives-up": '''
        write("unknown\\n")
    ''',
    "crasher": '''
        sys.exit(3)
    ''',
    # starts a grandchild, then sleeps: both must be killed
    "sleeper": '''
        import os, subprocess
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        with open(result_path + ".pids", "w") as f:
            f.write(f"{os.getpid()} {child.pid}")
        time.sleep(600)
    ''',
    # correct, but only after 1.5 s
    "slow-oracle": '''
        time.sleep(1.5)
        bits, y = search()
        write("unsat\\n" if bits is None else "sat\\n" + cex(bits, y) + "\\n")
    ''',
}


@pytest.fixture(scope="session")
def template(tmp_path_factory):
    folder = tmp_path_factory.mktemp("bench")
    build(folder / "bench", seed=0, spec=SPEC, timeout=20, solver_timeout=30)
    tools = folder / "tools"
    tools.mkdir()
    for name, body in FAKE_TOOLS.items():
        (tools / f"{name}.py").write_text(PRELUDE + textwrap.dedent(body))
    return folder


@pytest.fixture
def bench(template, tmp_path):
    shutil.copytree(template / "bench", tmp_path / "bench")
    return tmp_path / "bench"


def run(bench: Path, template: Path, tool: str, *extra: str) -> list[dict]:
    script = template / "tools" / f"{tool}.py"
    main(["--bench", str(bench), "--tool", tool, *extra, "--command", sys.executable, str(script)])
    with (bench / "results" / f"{tool}.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def num_instances(bench: Path) -> int:
    return len((bench / "instances.csv").read_text().splitlines())


# ------------------------------------------------------------ parsing / check


def test_parse_result_formats():
    assert parse_result("sat\n((X_0 0.5)\n (X_1 1)\n (Y_0 2.5))\n") == (
        "sat", {"X_0": 0.5, "X_1": 1.0, "Y_0": 2.5})
    assert parse_result("sat\nX real [1,2]\n0.0\n1.0\nY real [1,1]\n3.5\n") == (
        "sat", {"X_0": 0.0, "X_1": 1.0, "Y_0": 3.5})
    assert parse_result("violated\n") == ("sat", {})
    assert parse_result("holds") == ("unsat", {})
    assert parse_result("") == ("", {})


def test_check_accepts_a_counterexample_exactly_on_the_threshold():
    """Like Marabou's answers: not a 0/1 point, network output exactly m - 1/2.

    (x1 or x2) and (not x1 or x2) at x = (1/4, 3/4): neg1 = pos2 = 1/2, so the first clause
    has value 1/2 and the second 1, N = 3/2 = m - 1/2. It rounds to x1 = F, x2 = T.
    """
    cnf = CNF(2, ((1, 2), (-1, 2)))
    for encoding in ("compact", "paper"):
        check = check_answer("sat", {"X_0": 0.25, "X_1": 0.75}, "sat", cnf, encoding)
        assert (check.verdict, check.cex_check, float(check.cex_output)) == ("correct", "valid", 1.5)


def test_check_rejects_bad_counterexamples():
    cnf = CNF(2, ((1, 2), (-1, 2)))
    center = check_answer("sat", {"X_0": 0.5, "X_1": 0.5}, "sat", cnf, "compact")
    assert center.verdict == "wrong" and "below threshold" in center.cex_check
    outside = check_answer("sat", {"X_0": 0.0, "X_1": 1.5}, "sat", cnf, "compact")
    assert outside.verdict == "wrong" and "outside the box" in outside.cex_check
    assert check_answer("sat", {"X_0": 0.0}, "sat", cnf, "compact").verdict == "sat-no-cex"
    assert check_answer("sat", {}, "unsat", cnf, "compact").verdict == "wrong"


def test_check_unsat_and_non_answers():
    cnf = CNF(1, ((1,), (-1,)))
    assert check_answer("unsat", {}, "unsat", cnf, "compact").verdict == "correct"
    assert check_answer("unsat", {}, "sat", cnf, "compact").verdict == "wrong"
    assert check_answer("unknown", {}, "sat", cnf, "compact").verdict == "no-answer"
    # A tool that stops itself at its time limit: the same verdict as being killed.
    assert check_answer("timeout", {}, "sat", cnf, "compact").verdict == "timeout"
    assert check_answer("timeout", {}, "unsat", cnf, "compact").verdict == "timeout"
    assert check_answer("banana", {}, "sat", cnf, "compact").verdict == "error"


def test_exit_codes_name_the_signal():
    assert describe_exit(139) == "exit code 139 = signal 11 (SIGSEGV)"   # 128 + signal, as a shell reports it
    assert describe_exit(-11) == "exit code -11 = signal 11 (SIGSEGV)"   # killed directly
    assert describe_exit(3) == "exit code 3"
    assert describe_exit(None) == "no exit code"


def test_check_with_the_neurocodebench_encoding():
    """Unsafe set Y_0 >= 1 and Y_1 <= 0: only exact 0/1 points (up to the tolerance) are counterexamples."""
    cnf = CNF(2, ((1, 2), (-1, 2)))
    good = check_answer("sat", {"X_0": 0.0, "X_1": 1.0}, "sat", cnf, "neurocodebench")
    assert (good.verdict, good.cex_check, good.cex_output) == ("correct", "valid", "1;0")
    almost = check_answer("sat", {"X_0": 1e-7, "X_1": 1 - 1e-7}, "sat", cnf, "neurocodebench")
    assert almost.verdict == "correct"
    # (1/4, 3/4) is a valid counterexample of the compact encoding, but not here: Y_1 = 1/2
    quarter = check_answer("sat", {"X_0": 0.25, "X_1": 0.75}, "sat", cnf, "neurocodebench")
    assert quarter.verdict == "wrong" and "Y_1 = 0.5 violates <= 0" in quarter.cex_check
    falsified = check_answer("sat", {"X_0": 0.0, "X_1": 0.0}, "sat", cnf, "neurocodebench")
    assert falsified.verdict == "wrong" and "Y_0 = 0 violates >= 1" in falsified.cex_check


def test_valid_counterexample_for_an_unsat_instance_is_flagged_as_our_bug():
    cnf = CNF(1, ((1,),))
    assert check_answer("sat", {"X_0": 1.0}, "unsat", cnf, "compact").verdict == "reference-wrong"


# ------------------------------------------------------------- end to end


@pytest.mark.parametrize("tool", ["oracle", "oracle-blocks"])
def test_a_correct_tool_is_correct_everywhere(bench, template, tool):
    rows = run(bench, template, tool)
    assert len(rows) == num_instances(bench)
    assert {row["verdict"] for row in rows} == {"correct"}
    assert {row["expected"] for row in rows} == {"sat", "unsat"}
    assert all(row["cex_check"] == "valid" for row in rows if row["expected"] == "sat")


def test_threshold_in_network_variant_is_judged_the_same(template, tmp_path, capsys):
    """The workaround benchmark for alpha,beta-CROWN's MIP mode: same verdicts, no warning."""
    bench = tmp_path / "shifted"
    build(bench, seed=0, spec=SPEC, timeout=20, solver_timeout=30, threshold_in_network=True)
    rows = run(bench, template, "oracle")
    assert {row["verdict"] for row in rows} == {"correct"}
    assert {row["expected"] for row in rows} == {"sat", "unsat"}
    assert all(row["cex_check"] == "valid" for row in rows if row["expected"] == "sat")
    script = template / "tools" / "oracle.py"
    capsys.readouterr()
    main(["--bench", str(bench), "--tool", "abcrown-mip", "--limit", "0", "--command", sys.executable, str(script)])
    assert "WARNING" not in capsys.readouterr().out


def test_neurocodebench_benchmark_is_judged_correctly(template, tmp_path, capsys):
    bench = tmp_path / "ncb"
    build(bench, seed=0, spec=SPEC, timeout=20, encodings=["neurocodebench"], solver_timeout=30)
    rows = run(bench, template, "oracle-multi")
    assert {row["verdict"] for row in rows} == {"correct"}
    assert {row["expected"] for row in rows} == {"sat", "unsat"}
    assert all(row["cex_check"] == "valid" for row in rows if row["expected"] == "sat")
    rows = run(bench, template, "bad-cex")
    assert {row["verdict"] for row in rows} == {"wrong"}
    assert all("violates" in row["cex_check"] for row in rows)
    # alpha,beta-CROWN's MIP mode reads the two-constraint property correctly: no warning
    script = template / "tools" / "oracle-multi.py"
    capsys.readouterr()
    main(["--bench", str(bench), "--tool", "abcrown-mip", "--limit", "0", "--command", sys.executable, str(script)])
    assert "WARNING" not in capsys.readouterr().out


def test_two_constraint_variant_is_judged_correctly_and_gives_no_mip_warning(template, tmp_path, capsys):
    bench = tmp_path / "two"
    build(bench, seed=0, spec=SPEC, timeout=20, solver_timeout=30, two_constraints=True)
    rows = run(bench, template, "oracle-multi")
    assert {row["verdict"] for row in rows} == {"correct"}
    assert {row["expected"] for row in rows} == {"sat", "unsat"}
    script = template / "tools" / "oracle-multi.py"
    capsys.readouterr()
    main(["--bench", str(bench), "--tool", "abcrown-mip", "--limit", "0", "--command", sys.executable, str(script)])
    assert "WARNING" not in capsys.readouterr().out


def test_abcrown_mip_on_the_normal_benchmark_gives_a_warning(bench, template, capsys):
    script = template / "tools" / "oracle.py"
    main(["--bench", str(bench), "--tool", "abcrown-mip", "--limit", "0", "--command", sys.executable, str(script)])
    out = capsys.readouterr().out
    assert "WARNING" in out and "--threshold-in-network" in out


def test_a_lying_tool_is_caught(bench, template):
    for row in run(bench, template, "liar"):
        assert row["verdict"] == ("correct" if row["expected"] == "unsat" else "wrong")


def test_bad_counterexamples_are_caught(bench, template):
    rows = run(bench, template, "bad-cex")
    assert {row["verdict"] for row in rows} == {"wrong"}
    assert all("below threshold" in row["cex_check"] for row in rows)


def test_sat_without_counterexample(bench, template):
    for row in run(bench, template, "no-cex"):
        assert row["verdict"] == ("correct" if row["expected"] == "unsat" else "sat-no-cex")


def test_giving_up_and_crashing(bench, template):
    assert {row["verdict"] for row in run(bench, template, "gives-up")} == {"no-answer"}
    rows = run(bench, template, "crasher")
    assert {row["verdict"] for row in rows} == {"error"}
    assert rows[0]["result"] == "no result file (exit code 3)"


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only; the tools run in Linux containers")
def test_timeout_kills_the_tool_and_its_children(bench, template):
    rows = run(bench, template, "sleeper", "--only", "f0001", "--timeout", "1", "--grace", "1")
    assert len(rows) == 1 and rows[0]["verdict"] == "timeout" and rows[0]["result"] == "timeout"
    assert float(rows[0]["seconds"]) < 10
    pids = (bench / "results" / "raw" / "sleeper" / "f0001.txt.pids").read_text().split()
    for pid in map(int, pids):
        try:
            os.kill(pid, 0)
            alive = Path(f"/proc/{pid}/status").read_text().find("zombie") == -1
        except ProcessLookupError:
            alive = False
        assert not alive, f"process {pid} survived"


def test_an_answer_after_the_timeout_counts_as_timeout(bench, template):
    rows = run(bench, template, "slow-oracle", "--only", "f0001", "--timeout", "1", "--grace", "10")
    assert rows[0]["verdict"] == "timeout" and rows[0]["result"] in ("sat", "unsat")


def test_runs_can_be_resumed(bench, template):
    first = run(bench, template, "oracle", "--limit", "2")
    assert len(first) == 2
    second = run(bench, template, "oracle")
    assert len(second) == num_instances(bench)
    assert len({row["onnx"] for row in second}) == len(second)  # no duplicates
    again = run(bench, template, "oracle", "--rerun", "--limit", "1")
    assert len(again) == 1


def read_results(bench: Path, tool: str) -> list[dict]:
    with (bench / "results" / f"{tool}.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_as(bench: Path, template: Path, name: str, script: str, *extra: str) -> list[dict]:
    """Run the fake tool ``script`` under the results name ``name``."""
    main(["--bench", str(bench), "--tool", name, *extra,
          "--command", sys.executable, str(template / "tools" / f"{script}.py")])
    return read_results(bench, name)


def test_rerun_with_only_repeats_just_those_instances(bench, template):
    first = run_as(bench, template, "tool", "oracle")
    rows = run_as(bench, template, "tool", "crasher", "--rerun", "--only", "f0001")
    assert len(rows) == len(first) == num_instances(bench)
    by_file = {row["onnx"]: row for row in rows}
    assert by_file["onnx/f0001.onnx"]["verdict"] == "error"
    assert all(row == by_file[row["onnx"]] for row in first if row["onnx"] != "onnx/f0001.onnx")


def test_retry_errors_repeats_only_the_crashed_runs(bench, template):
    good = run_as(bench, template, "tool", "oracle", "--limit", "2")
    mixed = run_as(bench, template, "tool", "crasher")
    assert [row["verdict"] for row in mixed] == ["correct"] * 2 + ["error"] * (num_instances(bench) - 2)
    rows = run_as(bench, template, "tool", "oracle", "--retry-errors")
    assert len(rows) == num_instances(bench) and {row["verdict"] for row in rows} == {"correct"}
    assert rows[:2] == good  # the runs that had worked are kept as they were


def test_runs_during_which_the_computer_slept_are_flagged_and_repeated(bench, template, monkeypatch, capsys):
    """Simulated sleep: the wall clock jumps by 1000 s during the run, the monotonic clock does not."""
    clock = iter([0.0, 1000.0])
    monkeypatch.setattr(run_module, "_wall_clock", lambda: next(clock))
    (row,) = run_as(bench, template, "tool", "oracle", "--only", "f0001")
    assert float(row["wall_seconds"]) == 1000.0 and run_module.was_suspended(row)
    assert "the computer slept" in capsys.readouterr().out
    monkeypatch.undo()
    (row,) = run_as(bench, template, "tool", "oracle", "--only", "f0001", "--retry-suspended")
    assert not run_module.was_suspended(row) and row["verdict"] == "correct"


def test_results_from_before_the_wall_clock_column_are_kept(bench, template):
    rows = run_as(bench, template, "tool", "oracle", "--limit", "2")
    old_columns = [c for c in rows[0] if c != "wall_seconds"]
    with (bench / "results" / "tool.csv").open("w", newline="") as handle:  # a file in the old format
        writer = csv.DictWriter(handle, fieldnames=old_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    rows = run_as(bench, template, "tool", "oracle")
    assert len(rows) == num_instances(bench) and list(rows[0]) == run_module.RESULT_COLUMNS
    assert rows[0]["wall_seconds"] == "" and rows[-1]["wall_seconds"] != ""
    assert not any(run_module.was_suspended(row) for row in rows)


def test_unknown_tool_name_without_command_fails(bench):
    with pytest.raises(SystemExit):
        main(["--bench", str(bench), "--tool", "no-such-tool"])


# ----------------------------------------------------------------- adapters


def test_marabou_gives_only_correct_answers(bench):
    pytest.importorskip("maraboupy")
    main(["--bench", str(bench), "--tool", "marabou"])
    with (bench / "results" / "marabou.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == num_instances(bench)
    assert {row["verdict"] for row in rows} == {"correct"}


def test_abcrown_adapter_builds_the_vnncomp_command(tmp_path):
    """A stand-in for alpha,beta-CROWN records its arguments and answers unsat."""
    home = tmp_path / "abcrown"
    verifier = home / "complete_verifier"
    (verifier / abcrown.CONFIG).parent.mkdir(parents=True)
    (verifier / abcrown.CONFIG).write_text("general:\n  complete_verifier: mip\n")
    (verifier / "abcrown.py").write_text(textwrap.dedent('''
        import json, os, sys
        args = sys.argv[1:]
        json.dump({"argv": args, "cwd": os.getcwd()}, open(os.environ["ARGS_FILE"], "w"))
        open(args[args.index("--results_file") + 1], "w").write("unsat\\n")
    '''))
    env = {**os.environ, "ABCROWN_HOME": str(home), "ARGS_FILE": str(tmp_path / "args.json"),
           "PYTHONPATH": os.pathsep.join(sys.path)}
    for mode in ("mip", "bab"):
        result = tmp_path / f"{mode}.txt"
        # subprocess quotes the arguments (os.spawnve does not: a space in the path breaks it on Windows)
        code = subprocess.run([sys.executable, "-m", "relu_sat_bench.tools.abcrown", "--mode", mode,
                               "net.onnx", "prop.vnnlib", str(result), "60"], env=env).returncode
        assert code == 0 and result.read_text() == "unsat\n"
        recorded = json.loads((tmp_path / "args.json").read_text())
        argv = recorded["argv"]
        assert argv[argv.index("--config") + 1] == str(verifier / abcrown.CONFIG)
        assert argv[argv.index("--timeout") + 1] == "60"
        assert argv[argv.index("--device") + 1] == "cpu"
        assert "--save_adv_example" in argv
        assert Path(argv[argv.index("--onnx_path") + 1]).is_absolute()
        assert ("--complete_verifier" in argv) == (mode == "bab")
        assert Path(recorded["cwd"]) == verifier


@pytest.mark.skipif(os.name != "posix", reason="signals are POSIX-only; the tools run in Linux containers")
def test_abcrown_adapter_reports_a_crash_like_a_shell(tmp_path):
    """A segmentation fault in alpha,beta-CROWN becomes exit code 139 (128 + 11), with a note in the log."""
    verifier = tmp_path / "abcrown" / "complete_verifier"
    verifier.mkdir(parents=True)
    (verifier / "abcrown.py").write_text("import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n")
    env = {**os.environ, "ABCROWN_HOME": str(tmp_path / "abcrown"), "PYTHONPATH": os.pathsep.join(sys.path)}
    done = subprocess.run([sys.executable, "-m", "relu_sat_bench.tools.abcrown", "--mode", "bab",
                           "net.onnx", "prop.vnnlib", str(tmp_path / "r.txt"), "60"],
                          env=env, capture_output=True, text=True)
    assert done.returncode == 139 and "signal 11 (SIGSEGV)" in done.stdout
