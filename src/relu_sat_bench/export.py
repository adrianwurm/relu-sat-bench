"""Export formulas as verification instances in the VNN-COMP format.

For every formula in ``formulas.csv`` (written by ``generate``) this writes, in the same
folder:

* ``onnx/<id>.onnx`` (compact encoding) and/or ``onnx/<id>-paper.onnx``: the network.
  Each layer is a ``Gemm`` node (``y = x W^T + b``), followed by ``Relu`` for hidden
  layers. Input ``X`` has shape [1, n], output ``Y`` has shape [1, 1]; weights are
  float32. All weights and biases are small integers, so float32 represents them exactly.
* ``vnnlib/<id>.vnnlib``: the property, shared by both encodings. It describes the
  *unsafe* set ``{x in [0, 1]^n : Y_0 >= m - 1/2}``; a verifier answers "sat"
  (counterexample found) iff the formula is satisfiable.
* ``instances.csv``: the VNN-COMP instance list, one line ``onnx,vnnlib,timeout`` per
  instance, without a header.
* ``metadata.csv``: for every instance the expected answer and sizes. The expected answer
  is deliberately *not* written into the ONNX or VNN-LIB files that tools read.

``threshold_in_network=True`` writes an equivalent variant: the last bias of every network is
lowered by the threshold ``m - 1/2``, so the network computes ``N(x) - (m - 1/2)``, and the
unsafe set becomes ``Y_0 >= 0``. Same formulas, same file names, same answers. It exists as a
workaround for alpha,beta-CROWN's MIP mode, which compares single-constraint properties with 0
instead of their right-hand side (see the README).

File names: with ``names="descriptive"`` (default) the files are named after the formula
(``pigeonhole-h5.onnx``). The family name reveals the answer of the constructed families, so
``names="neutral"`` numbers the formulas instead (``f0007.onnx``), in a seeded random order
so that the numbering does not follow the families either. ``metadata.csv`` maps each file
back to its formula.

Every ONNX file is checked with onnxruntime right after writing (see :func:`check_onnx`).
Formulas whose answer is unknown (the SAT solver timed out) are skipped, because an
instance without a reference answer cannot be evaluated.

Usage::

    python -m relu_sat_bench.export --dir instances/demo --timeout 60
    python -m relu_sat_bench.export --dir instances/demo --encoding compact paper
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np

from relu_sat_bench.cnf import CNF
from relu_sat_bench.generate import FORMULAS_CSV, read_rows
from relu_sat_bench.network import AffineLayer, ReluNetwork
from relu_sat_bench.reduction import ENCODINGS, VerificationInstance, encode_assignment, reduce_cnf

OPSET = 13       # widely supported; the verification tools often ship older onnx/onnxruntime
IR_VERSION = 8   # the IR version that belongs to opsets 13-17
INSTANCES_CSV = "instances.csv"
METADATA_CSV = "metadata.csv"
METADATA_COLUMNS = [
    "onnx", "vnnlib", "timeout",  # same as in instances.csv
    "formula_id", "family", "encoding",
    "expected",                   # sat (counterexample exists) | unsat (property holds)
    "num_inputs", "num_relus", "num_clauses",
    "threshold",                  # m - 1/2: the threshold on N(x) = number of satisfied clauses
    "vnnlib_rhs",                 # the right-hand side in the VNN-LIB file: threshold, or 0 (see above)
    "sat_solver_seconds",         # from formulas.csv, for comparing with the verifiers
]


class ExportCheckFailed(RuntimeError):
    """The exported ONNX model does not compute what the exact network computes."""


# --------------------------------------------------------------------- ONNX


def shift_output(network: ReluNetwork, offset: Fraction) -> ReluNetwork:
    """The same network with ``offset`` subtracted from its output (last bias minus ``offset``).

    The bias stays exact (``Fraction``); with ``offset = m - 1/2`` it is a half-integer, which
    float32 represents exactly.
    """
    *hidden, last = network.layers
    shifted = AffineLayer(last.name, last.weight, last.bias.astype(object) - Fraction(offset), last.relu)
    return ReluNetwork((*hidden, shifted))


def to_onnx(network: ReluNetwork, name: str = "relu_sat_bench"):
    """The network as an ``onnx.ModelProto`` (Gemm + Relu layers, float32)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, initializers = [], []
    current = "X"
    last = len(network.layers) - 1
    for k, layer in enumerate(network.layers):
        weight = numpy_helper.from_array(layer.weight.astype(np.float32), f"{layer.name}.weight")
        bias = numpy_helper.from_array(layer.bias.astype(np.float32), f"{layer.name}.bias")
        initializers += [weight, bias]
        affine = "Y" if (k == last and not layer.relu) else f"{layer.name}.affine"
        nodes.append(helper.make_node(
            "Gemm", [current, weight.name, bias.name], [affine], name=f"{layer.name}.gemm", transB=1,
        ))
        current = affine
        if layer.relu:
            activated = "Y" if k == last else layer.name
            nodes.append(helper.make_node("Relu", [affine], [activated], name=f"{layer.name}.relu"))
            current = activated

    graph = helper.make_graph(
        nodes,
        name,
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, network.input_dim])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, network.output_dim])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="relu-sat-bench",
        opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def run_onnx(model_bytes: bytes, points: np.ndarray) -> np.ndarray:
    """Evaluate the ONNX model with onnxruntime on each row of ``points``; returns Y_0 per row."""
    import onnxruntime

    session = onnxruntime.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    rows = np.asarray(points, dtype=np.float32).reshape(-1, 1, session.get_inputs()[0].shape[1])
    return np.array([session.run(None, {"X": row})[0][0, 0] for row in rows], dtype=np.float64)


def check_onnx(
    model_bytes: bytes,
    instance: VerificationInstance,
    witness: Sequence[bool] | None = None,
    num_random: int = 20,
    seed: int = 0,
    offset: Fraction = Fraction(0),
) -> None:
    """Compare onnxruntime with the exact network; raise :class:`ExportCheckFailed` on a mismatch.

    The ONNX model is expected to compute ``N(x) - offset`` (``offset`` is 0, or the threshold
    for ``threshold_in_network``):

    * 0/1 points: the output must equal the number of satisfied clauses minus ``offset``
      *exactly* (integers or half-integers, so float32 introduces no error);
    * the center point (1/2, ..., 1/2): output exactly ``0 - offset``;
    * a known satisfying assignment: output at least ``threshold - offset``, i.e. a
      counterexample in float32 too;
    * random points in the box: close to the exact value (float32 rounding only).
    """
    cnf, n = instance.cnf, instance.num_inputs
    rng = random.Random(seed)
    boolean = [tuple(rng.random() < 0.5 for _ in range(n)) for _ in range(num_random)]
    if witness is not None:
        boolean.append(tuple(witness))
    points = [encode_assignment(a) for a in boolean] + [np.full(n, 0.5)]
    got = run_onnx(model_bytes, np.array(points))
    expected = [float(cnf.count_satisfied(a) - offset) for a in boolean] + [float(-offset)]
    if list(got) != expected:
        raise ExportCheckFailed(f"0/1 points: onnxruntime gives {list(got)}, expected {expected}")
    if witness is not None and not got[len(boolean) - 1] >= float(instance.threshold - offset):
        raise ExportCheckFailed("the satisfying assignment is not a counterexample in float32")

    box = np.array([[rng.random() for _ in range(n)] for _ in range(num_random)])
    exact = instance.network.forward(box)[:, 0] - float(offset)
    if not np.allclose(run_onnx(model_bytes, box), exact, rtol=1e-5, atol=1e-4):
        raise ExportCheckFailed("random points: onnxruntime deviates from the exact network")


# ------------------------------------------------------------------- VNN-LIB


def _decimal(value: Fraction) -> str:
    """Exact decimal notation for the thresholds we use (integers and halves)."""
    if value.denominator not in (1, 2):
        raise ValueError(f"unexpected threshold {value}")
    return f"{float(value):.1f}"


def vnnlib_text(instance: VerificationInstance, name: str, threshold_in_network: bool = False) -> str:
    """The property in VNN-LIB: input box and unsafe output set ``Y_0 >= m - 1/2``.

    With ``threshold_in_network`` the unsafe set is ``Y_0 >= 0``, for networks whose output
    is already lowered by ``m - 1/2`` (see :func:`shift_output`).
    """
    n, m = instance.num_inputs, instance.cnf.num_clauses
    if threshold_in_network:
        unsafe = (f"; unsafe set: Y_0 >= 0, where the network outputs (number of satisfied clauses) - ({m} - 1/2).\n"
                  f"; It is reachable iff the CNF formula with {m} clauses is satisfiable;")
        rhs = Fraction(0)
    else:
        unsafe = (f"; unsafe set: Y_0 >= {m} - 1/2. It is reachable iff the CNF formula with {m} clauses\n"
                  "; is satisfiable;")
        rhs = instance.threshold
    lines = [
        f"; relu-sat-bench instance {name}",
        f"; network input: one variable per propositional variable ({n}), output: one value",
        unsafe + " every counterexample rounds (x_i > 1/2 -> true) to a satisfying assignment.",
        "",
    ]
    lines += [f"(declare-const X_{i} Real)" for i in range(n)]
    lines += ["(declare-const Y_0 Real)", "", "; input box [0, 1]^n"]
    lo, hi = _decimal(instance.input_lower), _decimal(instance.input_upper)
    for i in range(n):
        lines += [f"(assert (>= X_{i} {lo}))", f"(assert (<= X_{i} {hi}))"]
    lines += ["", "; unsafe output set", f"(assert (>= Y_0 {_decimal(rhs)}))", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------- export


def _read_solution(path: Path, num_vars: int) -> tuple[bool, ...]:
    literals = [int(token) for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("v") for token in line.split()[1:]]
    values = {abs(lit): lit > 0 for lit in literals if lit != 0}
    return tuple(values.get(v, False) for v in range(1, num_vars + 1))


def onnx_name(name: str, encoding: str) -> str:
    return f"{name}.onnx" if encoding == "compact" else f"{name}-{encoding}.onnx"


NAMING = ("descriptive", "neutral")


def file_names(formula_ids: Sequence[str], names: str = "descriptive", seed: int = 0) -> dict[str, str]:
    """Map each formula id to the base name of its files.

    ``neutral``: ``f0001``, ``f0002``, ... assigned in a random order seeded by ``seed``.
    """
    if names == "descriptive":
        return {formula_id: formula_id for formula_id in formula_ids}
    if names != "neutral":
        raise ValueError(f"names must be one of {NAMING}")
    order = sorted(formula_ids)
    random.Random(f"{seed}:neutral-names").shuffle(order)
    width = max(4, len(str(len(order))))
    return {formula_id: f"f{k + 1:0{width}d}" for k, formula_id in enumerate(order)}


def export(
    folder: Path,
    encodings: Sequence[str] = ("compact",),
    timeout: float = 60.0,
    verbose: bool = True,
    names: str = "descriptive",
    name_seed: int = 0,
    threshold_in_network: bool = False,
) -> list[dict]:
    """Export every formula with a known answer in ``folder/formulas.csv``.

    ``threshold_in_network``: write the variant with the threshold moved into the network
    (output ``N(x) - (m - 1/2)``, unsafe set ``Y_0 >= 0``); see the module docstring.

    Rewrites ``onnx/``, ``vnnlib/``, ``instances.csv`` and ``metadata.csv`` from scratch
    (old ``.onnx`` and ``.vnnlib`` files there are removed first); returns the metadata rows.
    """
    import onnx

    formulas = read_rows(folder / FORMULAS_CSV)
    if not formulas:
        raise FileNotFoundError(f"no formulas in {folder / FORMULAS_CSV}; run generate first")
    for sub, suffix in (("onnx", ".onnx"), ("vnnlib", ".vnnlib")):
        (folder / sub).mkdir(exist_ok=True)
        for old in (folder / sub).glob(f"*{suffix}"):
            old.unlink()

    skipped = [fid for fid, row in formulas.items() if row["answer"] not in ("sat", "unsat")]
    base = file_names([fid for fid in formulas if fid not in skipped], names, name_seed)
    metadata = []
    for formula_id, row in sorted(formulas.items(), key=lambda item: base.get(item[0], "")):
        if formula_id in skipped:
            continue
        name = base[formula_id]
        cnf = CNF.from_dimacs((folder / row["cnf_file"]).read_text(encoding="utf-8"))
        witness = None
        if row["solution_file"]:
            witness = _read_solution(folder / row["solution_file"], cnf.num_vars)
            if not cnf.is_satisfied_by(witness):
                raise ExportCheckFailed(f"{row['solution_file']} does not satisfy {formula_id}")

        vnnlib_file = Path("vnnlib") / f"{name}.vnnlib"
        for encoding in encodings:
            instance = reduce_cnf(cnf, encoding)
            offset = instance.threshold if threshold_in_network else Fraction(0)
            if encoding == encodings[0]:
                (folder / vnnlib_file).write_text(vnnlib_text(instance, name, threshold_in_network),
                                                  encoding="utf-8")
            network = shift_output(instance.network, offset) if threshold_in_network else instance.network
            model_bytes = to_onnx(network, name=f"{name}-{encoding}").SerializeToString()
            check_onnx(model_bytes, instance, witness, offset=offset)
            onnx_file = Path("onnx") / onnx_name(name, encoding)
            (folder / onnx_file).write_bytes(model_bytes)
            onnx.checker.check_model(str(folder / onnx_file))
            metadata.append({
                "onnx": onnx_file.as_posix(),
                "vnnlib": vnnlib_file.as_posix(),
                "timeout": f"{timeout:g}",
                "formula_id": formula_id,
                "family": row["family"],
                "encoding": encoding,
                "expected": row["answer"],
                "num_inputs": instance.num_inputs,
                "num_relus": instance.network.num_relus,
                "num_clauses": cnf.num_clauses,
                "threshold": _decimal(instance.threshold),
                "vnnlib_rhs": _decimal(instance.threshold - offset),
                "sat_solver_seconds": row["solver_seconds"],
            })
            if verbose:
                print(f"{onnx_file.as_posix():40s} {formula_id:40s} "
                      f"{instance.network.num_relus:6d} ReLUs  expected {row['answer']}")

    with (folder / INSTANCES_CSV).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in metadata:
            writer.writerow([row["onnx"], row["vnnlib"], row["timeout"]])
    with (folder / METADATA_CSV).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metadata)

    if verbose:
        total = sum(float(row["timeout"]) for row in metadata)
        print(f"{len(metadata)} instances, sum of timeouts {total / 3600:.2f} h")
        if skipped:
            print(f"skipped {len(skipped)} formulas without a reference answer: {', '.join(skipped)}")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m relu_sat_bench.export",
        description="Write ONNX networks, VNN-LIB properties and instances.csv for generated formulas.",
    )
    parser.add_argument("--dir", type=Path, required=True,
                        help="folder with formulas.csv (output of generate); files are written there")
    parser.add_argument("--encoding", nargs="+", choices=ENCODINGS, default=["compact"],
                        help="network encodings to export (default: compact)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="per-instance timeout in seconds written to instances.csv (default 60)")
    parser.add_argument("--names", choices=NAMING, default="descriptive",
                        help="file names: after the formula (default) or neutral numbers f0001, ...")
    parser.add_argument("--name-seed", type=int, default=0,
                        help="seed for the order of the neutral numbers (default 0)")
    parser.add_argument("--threshold-in-network", action="store_true",
                        help="lower the network output by m - 1/2 and use the unsafe set Y_0 >= 0 "
                             "(workaround for alpha,beta-CROWN's MIP mode)")
    args = parser.parse_args(argv)
    export(args.dir, args.encoding, args.timeout, names=args.names, name_seed=args.name_seed,
           threshold_in_network=args.threshold_in_network)
    return 0


if __name__ == "__main__":
    sys.exit(main())
