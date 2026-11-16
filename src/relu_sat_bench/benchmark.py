"""Build a complete benchmark folder from a specification in one call.

A specification is a list of generator command lines (the part after ``--out``/``--seed``,
exactly as you would type them for ``python -m relu_sat_bench.generate``). Building means:
run every generator line with the given seed, then export all formulas. The result is a
self-contained folder in the VNN-COMP layout (``instances.csv``, ``onnx/``, ``vnnlib/``),
plus our own ``formulas.csv``, ``metadata.csv``, ``cnf/`` and ``solutions/``.

The benchmark's own specification lives in ``generate_properties.py`` in the repository
root, following the VNN-COMP convention ``python generate_properties.py <seed>``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from relu_sat_bench import export as export_module
from relu_sat_bench import generate as generate_module

VNNCOMP_MAX_TOTAL_SECONDS = 6 * 3600  # VNN-COMP: sum of all per-instance timeouts per benchmark


def build(
    out: Path,
    seed: int,
    spec: Sequence[Sequence[str]],
    timeout: float,
    encodings: Sequence[str] = ("compact",),
    solver_timeout: float = 60.0,
    names: str = "neutral",
    threshold_in_network: bool = False,
) -> list[dict]:
    """Generate all formulas of ``spec`` into ``out`` and export them; returns the metadata rows.

    By default the files get neutral names (``f0001.onnx``), numbered in an order seeded by
    ``seed``; ``metadata.csv`` maps them back to the formulas. ``threshold_in_network`` writes
    the variant for alpha,beta-CROWN's MIP mode (see :mod:`relu_sat_bench.export`); with the same
    seed, both variants contain the same formulas under the same file names.
    """
    out = Path(out)
    for command in spec:
        generate_module.main(["--out", str(out), "--seed", str(seed),
                              "--timeout", str(solver_timeout), *command])
    rows = export_module.export(out, encodings, timeout, names=names, name_seed=seed,
                                threshold_in_network=threshold_in_network)
    total = sum(float(row["timeout"]) for row in rows)
    if total > VNNCOMP_MAX_TOTAL_SECONDS:
        print(f"WARNING: the timeouts add up to {total / 3600:.1f} h; VNN-COMP allows at most 6 h")
    return rows
