"""Tests for reading formulas back out of sat_relu networks (relu_sat_bench.satrelu).

The tests build files in the layout of the VNN-COMP 2025 benchmark sat_relu from our own neurocodebench
encoding, because the original files are not redistributed with this project.
"""

from __future__ import annotations

import csv
import gzip
import random

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("pysat")

from formulas import random_cnf  # noqa: E402
from relu_sat_bench.cnf import CNF  # noqa: E402
from relu_sat_bench.export import to_onnx, vnnlib_text  # noqa: E402
from relu_sat_bench.generate import FORMULAS_CSV  # noqa: E402
from relu_sat_bench.network import AffineLayer, ReluNetwork  # noqa: E402
from relu_sat_bench.reduction import NEUROCODEBENCH, build_network, reduce_cnf  # noqa: E402
from relu_sat_bench.satrelu import (  # noqa: E402
    FormatError,
    check_property,
    check_same_function,
    decode_network,
    import_benchmark,
)
from relu_sat_bench.solve import SolverDisagreement  # noqa: E402

SAT = CNF(3, ((1, -2), (2, 3), (-1, -3)))       # x1 = x2 = 1, x3 = 0
UNSAT = CNF(2, ((1, 2), (1, -2), (-1, 2), (-1, -2)))


def published(cnf: CNF):
    return to_onnx(build_network(cnf, NEUROCODEBENCH))


def canonical(cnf: CNF) -> CNF:
    return CNF(cnf.num_vars, tuple(tuple(sorted(set(c), key=abs)) for c in cnf.clauses))


@pytest.mark.parametrize("seed", range(10))
def test_decoding_inverts_the_encoding(seed):
    rng = random.Random(seed)
    cnf = random_cnf(rng, rng.randint(2, 8), rng.randint(1, 20), clause_len=None)
    assert decode_network(published(cnf)) == canonical(cnf)


def test_published_network_computes_the_same_function():
    model = published(SAT)
    check_same_function(model.SerializeToString(), decode_network(model))


def _with_layer(cnf: CNF, k: int, weight=None, bias=None):
    layers = list(build_network(cnf, NEUROCODEBENCH).layers)
    old = layers[k]
    layers[k] = AffineLayer(old.name, old.weight if weight is None else weight,
                            old.bias if bias is None else bias, old.relu)
    return to_onnx(ReluNetwork(tuple(layers)))


def test_unexpected_structure_is_rejected():
    hidden, output = build_network(SAT, NEUROCODEBENCH).layers
    bias = hidden.bias.copy()
    bias[0] += 1                                    # clause bias is no longer 1 - |N|
    weight = hidden.weight.copy()
    weight[0, 0] = 2                                # weight outside {-1, 0, 1}
    out_bias = output.bias.copy()
    out_bias[0] = 0                                 # Y_0 without the constant 1
    binarise = hidden.bias.copy()
    binarise[-1] = 0                                # ReLU(2x - 1) became ReLU(2x)
    for model in (_with_layer(SAT, 0, bias=bias), _with_layer(SAT, 0, weight=weight),
                  _with_layer(SAT, 1, bias=out_bias), _with_layer(SAT, 0, bias=binarise)):
        with pytest.raises(FormatError):
            decode_network(model)
    with pytest.raises(FormatError):                # compact encoding: three Gemm layers
        decode_network(to_onnx(build_network(SAT, "compact")))


def test_property_check():
    text = vnnlib_text(reduce_cnf(SAT, NEUROCODEBENCH), "x")
    check_property(text, 3)
    for bad in (text.replace("(assert (<= Y_1 0.0))", "(assert (<= Y_1 0.5))"),
                text.replace("(assert (<= X_2 1.0))", "(assert (<= X_2 2.0))"),
                text + "(assert (or (>= Y_0 1.0)))\n",
                vnnlib_text(reduce_cnf(SAT, "compact"), "x")):
        with pytest.raises(FormatError):
            check_property(bad, 3)
    with pytest.raises(FormatError):
        check_property(text, 4)


def write_published(folder, entries):
    """sat_relu layout: instances.csv, onnx/*.onnx.gz, vnnlib/*.vnnlib.gz."""
    (folder / "onnx").mkdir(parents=True)
    (folder / "vnnlib").mkdir()
    rows = []
    for stem, cnf in entries:
        (folder / "onnx" / f"{stem}.onnx.gz").write_bytes(gzip.compress(published(cnf).SerializeToString()))
        text = vnnlib_text(reduce_cnf(cnf, NEUROCODEBENCH), stem)
        (folder / "vnnlib" / f"{stem}.vnnlib.gz").write_bytes(gzip.compress(text.encode()))
        rows.append([f"onnx/{stem}.onnx", f"vnnlib/{stem}.vnnlib", "100"])
    with (folder / "instances.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows(rows + rows[:1])     # sat_relu lists two networks twice


def test_import_benchmark(tmp_path):
    write_published(tmp_path / "src", [("sat_v3_c3", SAT), ("unsat_v2_c4", UNSAT)])
    rows = import_benchmark(tmp_path / "src", tmp_path / "out", verbose=False)
    assert list(rows) == ["satrelu-v3-c3-sat", "satrelu-v2-c4-unsat"]
    sat, unsat = rows.values()
    assert (sat["answer"], unsat["answer"]) == ("sat", "unsat")
    assert unsat["check"].startswith("confirmed")
    assert CNF.from_dimacs((tmp_path / "out" / sat["cnf_file"]).read_text()) == canonical(SAT)
    assert sat["solution_file"] and not unsat["solution_file"]
    with (tmp_path / "out" / FORMULAS_CSV).open() as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_import_rejects_a_wrong_label(tmp_path):
    write_published(tmp_path / "src", [("sat_v2_c4", UNSAT)])
    with pytest.raises(SolverDisagreement):
        import_benchmark(tmp_path / "src", tmp_path / "out", verbose=False)


def test_import_rejects_wrong_sizes_in_the_name(tmp_path):
    write_published(tmp_path / "src", [("sat_v4_c3", SAT)])
    with pytest.raises(FormatError):
        import_benchmark(tmp_path / "src", tmp_path / "out", verbose=False)


@pytest.mark.parametrize("encoding", ["compact", NEUROCODEBENCH])
def test_main_writes_a_complete_instance_folder(tmp_path, encoding):
    from relu_sat_bench.satrelu import main
    write_published(tmp_path / "src", [("sat_v3_c3", SAT), ("unsat_v2_c4", UNSAT)])
    assert main(["--source", str(tmp_path / "src"), "--out", str(tmp_path / "out"), "--encoding", encoding]) == 0
    with (tmp_path / "out" / "metadata.csv").open() as handle:
        meta = list(csv.DictReader(handle))
    assert [m["encoding"] for m in meta] == [encoding] * 2
    assert sorted(m["expected"] for m in meta) == ["sat", "unsat"]
    assert all((tmp_path / "out" / m["onnx"]).exists() and (tmp_path / "out" / m["vnnlib"]).exists() for m in meta)
    with pytest.raises(SystemExit):                       # never mixes two imports in one folder
        main(["--source", str(tmp_path / "src"), "--out", str(tmp_path / "out")])


def test_threshold_in_network_is_refused_for_neurocodebench(tmp_path):
    from relu_sat_bench.export import export
    write_published(tmp_path / "src", [("sat_v3_c3", SAT)])
    import_benchmark(tmp_path / "src", tmp_path / "out", verbose=False)
    with pytest.raises(ValueError):
        export(tmp_path / "out", [NEUROCODEBENCH], threshold_in_network=True, verbose=False)
    with pytest.raises(ValueError):
        vnnlib_text(reduce_cnf(SAT, NEUROCODEBENCH), "x", threshold_in_network=True)
    assert np.isclose(float(reduce_cnf(SAT, NEUROCODEBENCH).threshold), 1.0)
