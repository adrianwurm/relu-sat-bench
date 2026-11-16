"""Tests for the ONNX / VNN-LIB export and the benchmark build.

Skipped if onnx, onnxruntime or python-sat are not installed.
"""

from __future__ import annotations

import csv
import importlib.util
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("pysat")

from relu_sat_bench import generate  # noqa: E402
from relu_sat_bench.benchmark import VNNCOMP_MAX_TOTAL_SECONDS, build  # noqa: E402
from relu_sat_bench.cnf import CNF, iter_assignments  # noqa: E402
from relu_sat_bench.export import (  # noqa: E402
    IR_VERSION,
    OPSET,
    ExportCheckFailed,
    check_onnx,
    export,
    file_names,
    run_onnx,
    shift_output,
    to_onnx,
    vnnlib_text,
)
from relu_sat_bench.reduction import encode_assignment, reduce_cnf  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def parse_vnnlib(text: str) -> tuple[int, dict[int, tuple[float, float]], float]:
    """Minimal parser for the VNN-LIB files we write: (#inputs, input bounds, output threshold)."""
    inputs = len(re.findall(r"\(declare-const X_\d+ Real\)", text))
    assert len(re.findall(r"\(declare-const Y_\d+ Real\)", text)) == 1
    lower = {int(i): float(v) for i, v in re.findall(r"\(assert \(>= X_(\d+) ([-\d.]+)\)\)", text)}
    upper = {int(i): float(v) for i, v in re.findall(r"\(assert \(<= X_(\d+) ([-\d.]+)\)\)", text)}
    (threshold,) = re.findall(r"\(assert \(>= Y_0 ([-\d.]+)\)\)", text)
    return inputs, {i: (lower[i], upper[i]) for i in lower}, float(threshold)


# ----------------------------------------------------------------- ONNX model


def test_onnx_model_structure(small_cnf, encoding):
    instance = reduce_cnf(small_cnf, encoding)
    model = to_onnx(instance.network)
    ops = [node.op_type for node in model.graph.node]
    hidden = len(instance.network.layers) - 1
    assert ops == ["Gemm", "Relu"] * hidden + ["Gemm"]
    assert model.opset_import[0].version == OPSET and model.ir_version == IR_VERSION
    (inp,), (out,) = model.graph.input, model.graph.output
    assert [d.dim_value for d in inp.type.tensor_type.shape.dim] == [1, small_cnf.num_vars]
    assert [d.dim_value for d in out.type.tensor_type.shape.dim] == [1, 1]
    weights = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    for layer in instance.network.layers:
        assert weights[f"{layer.name}.weight"].dtype == np.float32
        assert (weights[f"{layer.name}.weight"] == layer.weight).all()
        assert (weights[f"{layer.name}.bias"] == layer.bias).all()


def test_onnxruntime_matches_the_exact_network(small_cnf, encoding):
    instance = reduce_cnf(small_cnf, encoding)
    model_bytes = to_onnx(instance.network).SerializeToString()
    assignments = list(iter_assignments(small_cnf.num_vars))
    got = run_onnx(model_bytes, np.array([encode_assignment(a) for a in assignments]))
    assert list(got) == [float(small_cnf.count_satisfied(a)) for a in assignments]  # exact
    box = np.random.default_rng(0).uniform(0, 1, size=(30, small_cnf.num_vars))
    np.testing.assert_allclose(run_onnx(model_bytes, box), instance.network.forward(box)[:, 0],
                               rtol=1e-5, atol=1e-4)
    check_onnx(model_bytes, instance)  # the check used during export passes


def test_check_onnx_detects_a_wrong_network():
    cnf = CNF(3, ((1, 2, 3), (-1, -2), (-3,)))
    wrong = to_onnx(reduce_cnf(CNF(3, cnf.clauses[:2])).network).SerializeToString()
    with pytest.raises(ExportCheckFailed):
        check_onnx(wrong, reduce_cnf(cnf), witness=(True, False, False))


def test_shifted_network_computes_n_minus_the_threshold(small_cnf, encoding):
    instance = reduce_cnf(small_cnf, encoding)
    shifted = shift_output(instance.network, instance.threshold)
    points = np.array([encode_assignment(a) for a in iter_assignments(small_cnf.num_vars)][:16])
    exact = instance.network.forward(points, exact=True)[:, 0] - instance.threshold
    assert list(shifted.forward(points, exact=True)[:, 0]) == list(exact)  # exact, no rounding
    model_bytes = to_onnx(shifted).SerializeToString()
    assert list(run_onnx(model_bytes, points)) == [float(v) for v in exact]  # exact in float32 too
    check_onnx(model_bytes, instance, offset=instance.threshold)
    with pytest.raises(ExportCheckFailed):  # the check notices a missing shift
        check_onnx(to_onnx(instance.network).SerializeToString(), instance, offset=instance.threshold)
    assert instance.network.layers[-1].bias[0] == shifted.layers[-1].bias[0] + instance.threshold


def test_thresholds_are_exact_in_float32():
    for m in (1, 2, 85, 426, 10_000, 2**22):
        assert float(np.float32(m - 0.5)) == m - 0.5


# -------------------------------------------------------------------- VNN-LIB


def test_vnnlib_text(small_cnf):
    instance = reduce_cnf(small_cnf)
    inputs, bounds, threshold = parse_vnnlib(vnnlib_text(instance, "test"))
    assert inputs == small_cnf.num_vars
    assert bounds == {i: (0.0, 1.0) for i in range(small_cnf.num_vars)}
    assert threshold == small_cnf.num_clauses - 0.5
    _, _, rhs = parse_vnnlib(vnnlib_text(instance, "test", threshold_in_network=True))
    assert rhs == 0.0


# ---------------------------------------------------------------- end to end


def make_formulas(folder: Path) -> None:
    for command in (
        ["pigeonhole", "--holes", "2", "3"],
        ["random", "--vars", "8", "--ratio", "3", "6", "--count", "2"],
        ["planted", "--vars", "8", "--ratio", "4.26", "--count", "1"],
        ["tseitin", "--vertices", "6", "--parity", "even"],
        ["ordering", "--size", "4"],
    ):
        assert generate.main(["--out", str(folder), "--timeout", "30", *command]) == 0


def test_export_writes_a_consistent_vnncomp_folder(tmp_path):
    make_formulas(tmp_path)
    rows = export(tmp_path, encodings=["compact", "paper"], timeout=42, verbose=False)
    formulas = generate.read_rows(tmp_path / "formulas.csv")
    assert len(rows) == 2 * len(formulas)

    lines = (tmp_path / "instances.csv").read_text().splitlines()
    assert len(lines) == len(rows)
    for line, row in zip(lines, rows):
        onnx_file, vnnlib_file, timeout = line.split(",")
        assert (onnx_file, vnnlib_file, timeout) == (row["onnx"], row["vnnlib"], "42")
        assert (tmp_path / onnx_file).exists() and (tmp_path / vnnlib_file).exists()
    with (tmp_path / "metadata.csv").open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [{k: str(v) for k, v in row.items()} for row in rows]

    for row in rows:
        formula = formulas[row["formula_id"]]
        assert row["expected"] == formula["answer"]
        cnf = CNF.from_dimacs((tmp_path / formula["cnf_file"]).read_text())
        model_bytes = (tmp_path / row["onnx"]).read_bytes()
        _, _, threshold = parse_vnnlib((tmp_path / row["vnnlib"]).read_text())
        # The answer as a verifier would compute it: is any 0/1 point a counterexample?
        # (0/1 points suffice: max N over the box is attained at a 0/1 point.)
        points = np.array([encode_assignment(a) for a in iter_assignments(cnf.num_vars)])
        reachable = run_onnx(model_bytes, points).max() >= threshold
        assert reachable == (row["expected"] == "sat")


def test_threshold_in_network_variant_is_equivalent(tmp_path):
    """Same formulas and file names; outputs lowered by m - 1/2; unsafe set Y_0 >= 0."""
    make_formulas(tmp_path)
    normal = export(tmp_path, encodings=["compact", "paper"], verbose=False, names="neutral", name_seed=5)
    files = {row["onnx"]: ((tmp_path / row["onnx"]).read_bytes(), (tmp_path / row["vnnlib"]).read_text())
             for row in normal}
    shifted = export(tmp_path, encodings=["compact", "paper"], verbose=False, names="neutral", name_seed=5,
                     threshold_in_network=True)
    assert [(r["onnx"], r["vnnlib"], r["formula_id"]) for r in shifted] == \
           [(r["onnx"], r["vnnlib"], r["formula_id"]) for r in normal]
    formulas = generate.read_rows(tmp_path / "formulas.csv")
    for before, row in zip(normal, shifted):
        m = int(row["num_clauses"])
        assert (before["vnnlib_rhs"], row["vnnlib_rhs"]) == (f"{m - 0.5:.1f}", "0.0")
        assert row["threshold"] == before["threshold"] == f"{m - 0.5:.1f}"
        old_onnx, old_vnnlib = files[row["onnx"]]
        _, bounds, rhs = parse_vnnlib((tmp_path / row["vnnlib"]).read_text())
        assert rhs == 0.0 and bounds == parse_vnnlib(old_vnnlib)[1]
        cnf = CNF.from_dimacs((tmp_path / formulas[row["formula_id"]]["cnf_file"]).read_text())
        points = np.array([encode_assignment(a) for a in iter_assignments(cnf.num_vars)])
        new_out = run_onnx((tmp_path / row["onnx"]).read_bytes(), points)
        assert list(new_out) == list(run_onnx(old_onnx, points) - (m - 0.5))
        assert (new_out.max() >= 0) == (row["expected"] == "sat")


FAMILY_WORDS = ("random", "planted", "pigeonhole", "tseitin", "ordering", "-odd", "-even")


def test_neutral_names_hide_the_family(tmp_path):
    make_formulas(tmp_path)
    export(tmp_path, verbose=False)  # descriptive first: its files must disappear
    rows = export(tmp_path, encodings=["compact", "paper"], verbose=False, names="neutral", name_seed=3)
    formulas = generate.read_rows(tmp_path / "formulas.csv")
    assert {row["formula_id"] for row in rows} == set(formulas)  # metadata maps back
    assert len({row["vnnlib"] for row in rows}) == len(formulas)  # one property per formula
    for path in (tmp_path / "onnx").iterdir():
        assert re.fullmatch(r"f\d{4}(-paper)?\.onnx", path.name), path.name
    for path in (tmp_path / "vnnlib").iterdir():
        assert re.fullmatch(r"f\d{4}\.vnnlib", path.name), path.name
    tool_facing = [(tmp_path / "instances.csv").read_text()]
    tool_facing += [(tmp_path / row["vnnlib"]).read_text() for row in rows]
    tool_facing += [onnx.load(str(tmp_path / row["onnx"])).graph.name for row in rows]
    for text in tool_facing:
        assert not any(word in text for word in FAMILY_WORDS), text[:80]


def test_neutral_numbering_is_seeded_and_shuffled():
    ids = [f"formula-{i:02d}" for i in range(30)]
    a, b, c = (file_names(ids, "neutral", seed) for seed in (1, 1, 2))
    assert a == b and a != c
    assert sorted(a.values()) == [f"f{k:04d}" for k in range(1, 31)]
    assert [a[i] for i in sorted(ids)] != sorted(a.values())  # not simply the sorted order
    assert file_names(ids) == {i: i for i in ids}  # descriptive is the default


def test_formulas_without_answer_are_skipped(tmp_path):
    make_formulas(tmp_path)
    rows = generate.read_rows(tmp_path / "formulas.csv")
    first = next(iter(rows))
    rows[first]["answer"] = "unknown"
    generate.write_rows(tmp_path / "formulas.csv", rows)
    exported = export(tmp_path, verbose=False)
    assert first not in {row["formula_id"] for row in exported}
    assert len(exported) == len(rows) - 1


def test_build_is_reproducible(tmp_path):
    spec = [["random", "--vars", "6", "--ratio", "4.26", "--count", "2"], ["ordering", "--size", "3"]]
    for name in ("a", "b"):
        build(tmp_path / name, seed=7, spec=spec, timeout=10)
    files = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*")
                   if p.is_file() and p.suffix in (".onnx", ".vnnlib", ".cnf"))
    assert files
    for f in files:
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()
    assert (tmp_path / "a" / "instances.csv").read_text() == (tmp_path / "b" / "instances.csv").read_text()


# ------------------------------------------------------- generate_properties.py


def load_script():
    spec = importlib.util.spec_from_file_location("generate_properties", REPO / "generate_properties.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_instances(spec) -> int:
    parser = generate.build_parser()
    return sum(sum(1 for _ in generate._instances(parser.parse_args(["--out", "unused", *command])))
               for command in spec)


def test_generate_properties_spec_is_valid_and_within_the_vnncomp_cap():
    script = load_script()
    for spec in (script.SPEC, script.PILOT_SPEC, script.SCALING_SPEC):
        assert count_instances(spec) * script.TIMEOUT * len(script.ENCODINGS) <= VNNCOMP_MAX_TOTAL_SECONDS


def test_generate_properties_spec_is_frozen():
    """The ladders were frozen on 31 Oct 2026 after the pilot; changing them must be deliberate."""
    script = load_script()
    assert count_instances(script.SPEC) == 213
    assert count_instances(script.PILOT_SPEC) == 77
    assert count_instances(script.SCALING_SPEC) == 65
    assert script.TIMEOUT == 60.0 and script.ENCODINGS == ["compact"]


def test_generate_properties_arguments():
    parse = load_script().parse_args
    assert parse(["1"]) == (1, Path("benchmark"), False, "benchmark")
    assert parse(["1", "out", "--threshold-in-network"]) == (1, Path("out"), True, "benchmark")
    assert parse(["0", "--pilot", "pilot"]) == (0, Path("pilot"), False, "pilot")
    assert parse(["1", "s", "--scaling", "--threshold-in-network"]) == (1, Path("s"), True, "scaling")
    for bad in ([], ["x"], ["1", "a", "b"], ["1", "--unknown"], ["1", "--pilot", "--scaling"]):
        assert parse(bad) is None


def test_generate_properties_without_seed_prints_usage(capsys):
    assert load_script().main([]) == 2
    assert "generate_properties.py <seed>" in capsys.readouterr().out
