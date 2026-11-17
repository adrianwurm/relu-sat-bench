"""Read the formulas back out of the VNN-COMP 2025 benchmark ``sat_relu`` and re-encode them.

``sat_relu`` (NeuroCodeBench 2.0; Manino et al., arXiv:2510.23389) turns CNF formulas into ReLU networks
with one hidden layer (see the ``neurocodebench`` encoding in :mod:`relu_sat_bench.reduction`). Its files
are in the repository ``VNN-COMP/vnncomp2025_benchmarks`` under ``benchmarks/sat_relu``: ``instances.csv``,
``onnx/*.onnx.gz`` and ``vnnlib/*.vnnlib.gz``. That repository states no license, so this project does not
redistribute the files; download them yourself.

For every network this module

1. decodes the formula from the weights of the hidden layer (:func:`decode_network`); anything that does
   not have the expected structure raises :class:`FormatError`;
2. checks the published property: input box ``[0, 1]^n`` and unsafe set ``Y_0 >= 1 and Y_1 <= 0``;
3. checks that our neurocodebench network for the decoded formula computes the same function as the
   published network (identical outputs on 0/1 points, close ones on random points of the box);
4. decides the formula with the SAT solver (unsatisfiable answers confirmed by a second solver) and checks
   that the answer agrees with the label in the file name (``sat_...`` / ``unsat_...``).

The formulas are written in the generator's format (``formulas.csv``, ``cnf/``, ``solutions/``), so that
:mod:`relu_sat_bench.export` can encode them in every encoding.

Usage::

    python -m relu_sat_bench.satrelu --source instances/satrelu-original --out instances/satrelu-ncb \\
        --encoding neurocodebench
    python -m relu_sat_bench.satrelu --source instances/satrelu-original --out instances/satrelu-compact
    python -m relu_sat_bench.satrelu --source instances/satrelu-original --out instances/satrelu-compact-shifted \\
        --threshold-in-network
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from relu_sat_bench.cnf import CNF
from relu_sat_bench.families import GeneratedFormula
from relu_sat_bench.generate import FORMULAS_CSV, _solution_text, read_rows, reference_answer, write_rows
from relu_sat_bench.reduction import ALL_ENCODINGS, NEUROCODEBENCH, encode_assignment, reduce_cnf
from relu_sat_bench.solve import CROSS_CHECK_SOLVER, DEFAULT_SOLVER, SolverDisagreement

FAMILY = "satrelu"
NAME = re.compile(r"^(sat|unsat)_v(\d+)_c(\d+)$")


class FormatError(ValueError):
    """A sat_relu file does not have the structure this module expects."""


def _read(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def _initializers(model) -> dict[str, np.ndarray]:
    from onnx import numpy_helper
    return {init.name: numpy_helper.to_array(init).astype(np.float64) for init in model.graph.initializer}


def _layers(model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(W1, b1, W2, b2) of a Gemm-Relu-Gemm network with ``transB=1``."""
    from onnx import helper
    nodes = model.graph.node
    if [n.op_type for n in nodes] != ["Gemm", "Relu", "Gemm"]:
        raise FormatError(f"expected Gemm, Relu, Gemm; got {[n.op_type for n in nodes]}")
    weights = _initializers(model)
    result = []
    for node in (nodes[0], nodes[2]):
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attributes.get("transB", 0) != 1 or attributes.get("alpha", 1.0) != 1.0 or attributes.get("beta", 1.0) != 1.0:
            raise FormatError(f"unexpected Gemm attributes {attributes}")
        result += [weights[node.input[1]], weights[node.input[2]]]
    return tuple(result)


def decode_network(model) -> CNF:
    """The CNF formula a sat_relu network encodes.

    Hidden layer: m clause neurons ``ReLU(sum_N x - sum_P x + 1 - |N|)``, then n neurons ``ReLU(x_i)``,
    then n neurons ``ReLU(2 x_i - 1)``. Output: ``Y_0 = 1 - sum_j v_j``, ``Y_1 = sum ReLU(x_i) - sum
    ReLU(2 x_i - 1)``. A weight -1 in a clause row is a positive literal, +1 a negative one.
    """
    w1, b1, w2, b2 = _layers(model)
    width, n = w1.shape
    m = width - 2 * n
    if m < 1 or w2.shape != (2, width) or b1.shape != (width,) or b2.shape != (2,):
        raise FormatError(f"unexpected shapes {w1.shape}, {w2.shape}")
    eye = np.eye(n)
    if not (np.array_equal(w1[m:m + n], eye) and np.array_equal(b1[m:m + n], np.zeros(n))
            and np.array_equal(w1[m + n:], 2 * eye) and np.array_equal(b1[m + n:], -np.ones(n))):
        raise FormatError("the binarisation neurons ReLU(x_i), ReLU(2 x_i - 1) are not where expected")
    expected_w2 = np.zeros((2, width))
    expected_w2[0, :m] = -1
    expected_w2[1, m:m + n] = 1
    expected_w2[1, m + n:] = -1
    if not (np.array_equal(w2, expected_w2) and np.array_equal(b2, np.array([1.0, 0.0]))):
        raise FormatError("the output layer is not Y_0 = 1 - sum v_j, Y_1 = sum ReLU(x) - ReLU(2x - 1)")
    clauses = []
    for j in range(m):
        row = w1[j]
        if not set(np.unique(row)) <= {-1.0, 0.0, 1.0} or not row.any():
            raise FormatError(f"clause neuron {j}: weights must be -1, 0 or 1, not all 0")
        negative = [i for i in range(n) if row[i] == 1]
        positive = [i for i in range(n) if row[i] == -1]
        if b1[j] != 1 - len(negative):
            raise FormatError(f"clause neuron {j}: bias {b1[j]} is not 1 - |N| = {1 - len(negative)}")
        clauses.append(tuple(sorted([i + 1 for i in positive] + [-(i + 1) for i in negative], key=abs)))
    return CNF(n, tuple(clauses))


def check_property(text: str, num_inputs: int) -> None:
    """The published property must be the box [0, 1]^n and the unsafe set Y_0 >= 1 and Y_1 <= 0."""
    asserts = re.findall(r"\(assert\s+\((<=|>=)\s+([XY]_\d+)\s+([-+0-9.eE]+)\)\)", text)
    if len(asserts) != text.count("(assert"):
        raise FormatError("the property has assertions of an unexpected form")
    box = {(op, var, float(value)) for op, var, value in asserts if var.startswith("X_")}
    expected_box = {(op, f"X_{i}", value) for i in range(num_inputs) for op, value in ((">=", 0.0), ("<=", 1.0))}
    outputs = {(op, var, float(value)) for op, var, value in asserts if var.startswith("Y_")}
    if box != expected_box or outputs != {(">=", "Y_0", 1.0), ("<=", "Y_1", 0.0)}:
        raise FormatError("the property is not [0, 1]^n with the unsafe set Y_0 >= 1 and Y_1 <= 0")


def check_same_function(model_bytes: bytes, cnf: CNF, num_random: int = 20, seed: int = 0) -> None:
    """Our neurocodebench network for ``cnf`` must compute what the published network computes."""
    import random

    from relu_sat_bench.export import run_onnx
    network = reduce_cnf(cnf, NEUROCODEBENCH).network
    rng = random.Random(seed)
    n = cnf.num_vars
    binary = np.array([encode_assignment([rng.random() < 0.5 for _ in range(n)]) for _ in range(num_random)])
    if run_onnx(model_bytes, binary, output=None).tolist() != network.forward(binary).tolist():
        raise FormatError("our neurocodebench network differs from the published one on 0/1 points")
    box = np.array([[rng.random() for _ in range(n)] for _ in range(num_random)])
    if not np.allclose(run_onnx(model_bytes, box, output=None), network.forward(box), rtol=1e-5, atol=1e-5):
        raise FormatError("our neurocodebench network differs from the published one on random points")


def published_networks(source: Path) -> list[str]:
    """The distinct networks listed in ``instances.csv``, in their order (file stems such as sat_v10_c26)."""
    stems = []
    with (source / "instances.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            stem = Path(row[0]).name.removesuffix(".gz").removesuffix(".onnx")
            if stem not in stems:
                stems.append(stem)
    return stems


def _find(folder: Path, stem: str, suffix: str) -> Path:
    for candidate in (folder / f"{stem}{suffix}.gz", folder / f"{stem}{suffix}"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(folder / f"{stem}{suffix}(.gz)")


def import_benchmark(source: Path, out: Path, solver: str = DEFAULT_SOLVER,
                     cross_check_solver: str = CROSS_CHECK_SOLVER, timeout: float = 60.0,
                     verbose: bool = True) -> dict[str, dict]:
    """Decode and check every sat_relu network; write the formulas in the generator's format."""
    import onnx

    (out / "cnf").mkdir(parents=True, exist_ok=True)
    (out / "solutions").mkdir(exist_ok=True)
    rows = {}
    for stem in published_networks(source):
        match = NAME.match(stem)
        if not match:
            raise FormatError(f"unexpected file name {stem}")
        label, n, m = match.group(1), int(match.group(2)), int(match.group(3))
        model_bytes = _read(_find(source / "onnx", stem, ".onnx"))
        cnf = decode_network(onnx.load_from_string(model_bytes))
        if (cnf.num_vars, cnf.num_clauses) != (n, m):
            raise FormatError(f"{stem}: decoded {cnf.num_vars} variables and {cnf.num_clauses} clauses")
        check_property(_read(_find(source / "vnnlib", stem, ".vnnlib")).decode("utf-8"), n)
        check_same_function(model_bytes, cnf)

        name = f"{FAMILY}-v{n}-c{m}-{label}"
        formula = GeneratedFormula(FAMILY, {"source": f"sat_relu/{stem}"}, cnf, satisfiable=None)
        fields, assignment = reference_answer(formula, solver, timeout, cross_check_solver)
        if fields["answer"] != label:
            raise SolverDisagreement(f"{stem}: the file name says {label}, {solver} says {fields['answer']}")
        fields["answer_source"] = "solver, agrees with the sat_relu label"
        cnf_file = Path("cnf") / f"{name}.cnf"
        (out / cnf_file).write_text(cnf.to_dimacs([f"relu-sat-bench import of sat_relu/{stem}",
                                                   f"family {FAMILY}"]), encoding="utf-8")
        solution_file = ""
        if assignment is not None:
            solution_file = (Path("solutions") / f"{name}.sol").as_posix()
            (out / solution_file).write_text(_solution_text(assignment), encoding="utf-8")
        rows[name] = {
            "id": name, "family": FAMILY, "params": json.dumps(formula.params, sort_keys=True), "seed": "",
            "num_vars": n, "num_clauses": m, "ratio": f"{m / n:.3f}", **fields,
            "cnf_file": cnf_file.as_posix(), "solution_file": solution_file,
        }
        if verbose:
            print(f"{stem:22s} -> {name:28s} {fields['answer']:6s} {float(fields['solver_seconds']):8.4f}s  "
                  f"{fields['check']}")
    write_rows(out / FORMULAS_CSV, rows)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    from relu_sat_bench.export import export

    parser = argparse.ArgumentParser(prog="python -m relu_sat_bench.satrelu", description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, required=True,
                        help="the downloaded benchmarks/sat_relu folder (instances.csv, onnx/, vnnlib/)")
    parser.add_argument("--out", type=Path, required=True, help="folder for the formulas and the new instances")
    parser.add_argument("--encoding", choices=ALL_ENCODINGS, default="compact")
    parser.add_argument("--threshold-in-network", action="store_true",
                        help="the variant for alpha,beta-CROWN's MIP mode (compact and paper only)")
    parser.add_argument("--two-constraints", action="store_true",
                        help="unsafe set m - 1/2 <= Y_0 <= m: the same set as two constraints (compact and paper)")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-instance timeout in instances.csv")
    args = parser.parse_args(argv)
    if read_rows(args.out / FORMULAS_CSV):
        raise SystemExit(f"{args.out} already contains formulas; use a new folder")
    import_benchmark(args.source, args.out)
    rows = export(args.out, [args.encoding], args.timeout, names="neutral", name_seed=0,
                  threshold_in_network=args.threshold_in_network, two_constraints=args.two_constraints)
    print(f"benchmark written to {args.out} ({len(rows)} instances)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
