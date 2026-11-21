"""Tests for the VNN-COMP comparison: parameter counts, reading the results repository, the report."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from relu_sat_bench import vnncomp as vc


def tiny_network(path: Path) -> Path:
    """Gemm with W = [[1, 0], [2, 0]] (4 stored, 2 nonzero), a Constant 0.5 (1, 1) and an int64 shape (not counted)."""
    w = numpy_helper.from_array(np.array([[1, 0], [2, 0]], dtype=np.float32), "W")
    shape = numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "shape")
    half = helper.make_node("Constant", [], ["half"], value=numpy_helper.from_array(np.array([0.5], dtype=np.float32)))
    gemm = helper.make_node("Gemm", ["X", "W"], ["G"])
    add = helper.make_node("Add", ["G", "half"], ["A"])
    reshape = helper.make_node("Reshape", ["A", "shape"], ["Y"])
    graph = helper.make_graph([half, gemm, add, reshape], "tiny",
                              [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
                              [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])], [w, shape])
    data = helper.make_model(graph).SerializeToString()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data)
    return path


def test_count_parameters(tmp_path):
    assert vc.count_parameters(tiny_network(tmp_path / "a.onnx")) == (5, 3)
    assert vc.count_parameters(tiny_network(tmp_path / "b.onnx.gz")) == (5, 3)


def test_benchmark_names_and_formatting():
    assert vc.benchmark_key("2025 Cora 2024") == "cora_2024"
    assert vc.benchmark_key("2025 Traffic Signs Recognition 2023") == "traffic_signs_recognition_2023"
    assert [vc.fmt(x) for x in (None, 456, 4199, 54902, 575260, 1005585, 3616144)] == \
        ["–", "456", "4.2k", "55k", "575k", "1M", "3.6M"]
    assert vc.solved("sat") and vc.solved("unsat") and vc.solved("holds")
    assert not vc.solved("timeout") and not vc.solved("run_instance_timeout") and not vc.solved("")


def write_results_repo(root: Path, abc_last: str = "timeout") -> Path:
    """Two benchmarks: tiny_a (regular, 3 instances, the last one unanswered), tiny_b (extended, 1 instance)."""
    for track, lines in (("scored", ["2025 Tiny A & 0 & ~\\textsc{sat} & 1.0 & - \\\\",
                                     "2025 Tiny A & 1 & ~\\textsc{unsat} & 2.0 & 3.0 \\\\",
                                     "2025 Tiny A & 2 & ~? & - & - \\\\"]),
                         ("unscored", ["2025 Tiny B & 0 & ~\\textsc{unsat} & - & 4.0 \\\\"])):
        table = root / "SCORING-SMALL-TOL" / track / "latex" / "longtable.tex"
        table.parent.mkdir(parents=True)
        table.write_text("% header\n" + "\n".join(lines) + "\n", encoding="utf-8")
    prefix = "vnncomp2025_benchmarks/benchmarks"
    rows = {"alpha_beta_crown": [("tiny_a", "a/./onnx/small.onnx", "p0", "sat"),
                                 ("tiny_a", "a/onnx/small.onnx", "p1", "unsat"),
                                 ("tiny_a", "a/onnx/big.onnx", "p2", abc_last)],
            "other": [("tiny_a", "a/onnx/test_nano.onnx", "p", "sat"),       # warm-up, left out
                      ("tiny_a", "a/onnx/small.onnx", "p0", "timeout"),
                      ("tiny_a", "a/onnx/small.onnx", "p1", "unsat"),
                      ("tiny_a", "a/onnx/big.onnx", "p2", "timeout"),
                      ("tiny_b", "b/onnx/net.onnx", "q0", "unsat")]}
    for tool, runs in rows.items():
        (root / tool).mkdir()
        with (root / tool / "results.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            for bench, net, prop, result in runs:
                writer.writerow([bench, f"{prefix}/{bench}/{net.split('/', 1)[1]}", f"{prefix}/{bench}/vnnlib/{prop}.vnnlib",
                                 "1.0", result, "2.0"])
    return root


def test_collect_reads_the_official_table_and_the_tools(tmp_path):
    results = write_results_repo(tmp_path / "results")
    benchmarks = tmp_path / "benchmarks"
    tiny_network(benchmarks / "benchmarks/tiny_a/onnx/small.onnx")
    tiny_network(benchmarks / "benchmarks/tiny_a/onnx/big.onnx.gz")
    (benchmarks / "benchmarks/tiny_b/onnx").mkdir(parents=True)
    (benchmarks / "benchmarks/tiny_b/onnx/net.onnx").write_bytes(b"not an onnx file")
    vc.collect(results, benchmarks, tmp_path / "data")
    instances = list(csv.DictReader((tmp_path / "data/instances.csv").open()))
    assert [(r["benchmark"], r["track"], r["id"], r["result"], r["alpha_beta_crown"]) for r in instances] == [
        ("tiny_a", "regular", "0", "sat", "sat"), ("tiny_a", "regular", "1", "unsat", "unsat"),
        ("tiny_a", "regular", "2", "unknown", "timeout"), ("tiny_b", "extended", "0", "unsat", "")]
    assert instances[0]["onnx"] == "benchmarks/tiny_a/onnx/small.onnx"       # "/./" removed
    networks = {r["onnx"]: r for r in csv.DictReader((tmp_path / "data/networks.csv").open())}
    assert (networks["benchmarks/tiny_a/onnx/big.onnx"]["stored"], networks["benchmarks/tiny_a/onnx/big.onnx"]["nonzero"]) == ("5", "3")
    assert networks["benchmarks/tiny_b/onnx/net.onnx"]["note"].startswith("unreadable")


def test_collect_refuses_differing_instance_lists(tmp_path):
    results = write_results_repo(tmp_path / "results")
    with (results / "other" / "results.csv").open("a", newline="") as handle:
        csv.writer(handle).writerow(["tiny_a", "vnncomp2025_benchmarks/benchmarks/tiny_a/onnx/x.onnx",
                                     "vnncomp2025_benchmarks/benchmarks/tiny_a/vnnlib/p.vnnlib", "1", "sat", "1"])
    with pytest.raises(ValueError, match="tiny_a"):
        vc.collect_instances(results)


def write_data(folder: Path) -> Path:
    folder.mkdir()
    instances = [("a", "regular", "n/small.onnx", "sat", "sat"), ("a", "regular", "n/big.onnx", "unknown", "timeout"),
                 ("b", "extended", "n/mid.onnx", "unknown", "timeout"), ("b", "extended", "n/mid.onnx", "sat", "error"),
                 ("c", "regular", "n/huge.onnx", "unsat", "")]
    vc._write_csv(folder / "instances.csv", vc.INSTANCE_COLUMNS,
                  [{"benchmark": b, "track": t, "id": i, "onnx": n, "vnnlib": "p", "result": r, "alpha_beta_crown": a}
                   for i, (b, t, n, r, a) in enumerate(instances)])
    vc._write_csv(folder / "networks.csv", vc.NETWORK_COLUMNS, [
        {"onnx": "n/small.onnx", "stored": 500, "nonzero": 400, "note": ""},
        {"onnx": "n/big.onnx", "stored": 700000, "nonzero": 600000, "note": ""},
        {"onnx": "n/mid.onnx", "stored": 60000, "nonzero": 55000, "note": ""},
        {"onnx": "n/huge.onnx", "stored": "", "nonzero": "", "note": "not in the repository"}])
    return folder


def test_benchmark_rows_and_report(tmp_path):
    data = write_data(tmp_path / "data")
    rows = {r["benchmark"]: r for r in vc.benchmark_rows(data)}
    assert list(rows) == ["a", "b", "c"]                                    # by smallest network, unknown last
    a, b, c = rows["a"], rows["b"], rows["c"]
    assert (a["no tool"], a["abc open"], a["nonzero"], a["no tool min"], a["abc min"]) == (1, 1, "400–600k", 600000, 600000)
    assert (b["no tool"], b["abc open"], b["no tool min"], b["abc min"]) == (1, 2, 55000, 55000)
    assert c["abc open"] is None and c["counted"] == "0/1"                   # α,β-CROWN did not take part
    ours = [{"tool": "marabou", "first nonzero": 209, "first where": "tseitin odd (unsat), 12 inputs",
             "fails nonzero": 256, "fails where": "random r4.26 (unsat), 10 inputs"}]
    text = vc.write_report(data, tmp_path / "out", ours).read_text(encoding="utf-8")
    assert "| relu-sat-bench, Marabou: first real failure / fails on half | 209 (tseitin odd (unsat), 12 inputs) / 256" in text
    assert "no tool solved | 55k (b); regular track: 600k (a) |" in text
    assert "α,β-CROWN did not solve | 55k (b) |" in text
    assert "Not counted: `huge.onnx` (not in the repository)." in text
    assert vc.main(["report", "--data", str(data), "--out", str(tmp_path / "cli")]) == 0
