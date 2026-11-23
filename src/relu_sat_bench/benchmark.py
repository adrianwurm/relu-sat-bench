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
    two_constraints: bool = False,
) -> list[dict]:
    """Generate all formulas of ``spec`` into ``out`` and export them; returns the metadata rows.

    By default the files get neutral names (``f0001.onnx``), numbered in an order seeded by
    ``seed``; ``metadata.csv`` maps them back to the formulas. ``threshold_in_network`` writes
    the variant for alpha,beta-CROWN's MIP mode (see :mod:`relu_sat_bench.export`); with the same
    seed, both variants contain the same formulas under the same file names. ``two_constraints``
    writes the unsafe set as two constraints (same set; see :mod:`relu_sat_bench.export`).
    """
    out = Path(out)
    for command in spec:
        generate_module.main(["--out", str(out), "--seed", str(seed),
                              "--timeout", str(solver_timeout), *command])
    rows = export_module.export(out, encodings, timeout, names=names, name_seed=seed,
                                threshold_in_network=threshold_in_network, two_constraints=two_constraints)
    total = sum(float(row["timeout"]) for row in rows)
    if total > VNNCOMP_MAX_TOTAL_SECONDS:
        print(f"WARNING: the timeouts add up to {total / 3600:.1f} h; VNN-COMP allows at most 6 h")
    return rows


def formula_ids(spec: Sequence[Sequence[str]], seed: int) -> list[str]:
    """The ids of the formulas that ``spec`` generates with ``seed``, without solving them."""
    parser = generate_module.build_parser()
    return [name for command in spec
            for name, _, _ in generate_module._instances(parser.parse_args(["--out", "unused", "--seed", str(seed),
                                                                            *command]))]


def build_parts(
    out: Path,
    seed: int,
    parts: Sequence[tuple[Sequence[Sequence[str]], str]],
    timeout: float,
    solver_timeout: float = 60.0,
    threshold_in_network: bool = False,
) -> list[dict]:
    """Build one benchmark folder from several (specification, encoding) parts.

    Every formula of a part is exported in that part's encoding. A formula that occurs in two parts (the same
    id: same family, parameters, index and seed) is generated once and exported in both encodings.
    ``threshold_in_network`` applies to the compact parts only. Neutral file names as in :func:`build`.
    """
    out = Path(out)
    chosen: dict[str, list[str]] = {}
    for spec, encoding in parts:
        for command in spec:
            generate_module.main(["--out", str(out), "--seed", str(seed), "--timeout", str(solver_timeout), *command])
        for formula_id in formula_ids(spec, seed):
            if encoding not in chosen.setdefault(formula_id, []):
                chosen[formula_id].append(encoding)
    rows = export_module.export(out, timeout=timeout, names="neutral", name_seed=seed,
                                threshold_in_network=threshold_in_network, encodings_by_formula=chosen)
    total = sum(float(row["timeout"]) for row in rows)
    if total > VNNCOMP_MAX_TOTAL_SECONDS:
        print(f"WARNING: the timeouts add up to {total / 3600:.1f} h; VNN-COMP allows at most 6 h")
    return rows

