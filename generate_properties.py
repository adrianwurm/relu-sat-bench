"""Build the relu-sat-bench benchmark set.

VNN-COMP convention: ``python generate_properties.py <seed> [output folder]``.
The same seed always produces the same instances, on every machine.

Options (after the seed and the folder):

  --threshold-in-network   the equivalent variant for alpha,beta-CROWN's MIP mode: network output
                           lowered by m - 1/2, unsafe set Y_0 >= 0 (same formulas, same file names)
  --pilot                  the 77-instance pilot set of step 3 instead of the frozen set
  --scaling                larger formulas for the scaling experiment with alpha,beta-CROWN's MIP mode
                           (an extra experiment, not part of the benchmark set)

The size ladders in SPEC were calibrated once, on the pilot run of 31 Oct 2026 (seed 0, Marabou
and alpha,beta-CROWN, 60 s per instance), so that every family ranges from "every tool solves it"
to "no tool solves it". They are now frozen. Instances are never selected individually by tool
results. The pilot used seed 0; the evaluation uses a different seed.

The files get neutral names (f0001.onnx, f0001.vnnlib, ...) in a seeded random order, so that
neither names nor order reveal the formula family; metadata.csv maps them back.
"""

from __future__ import annotations

import sys
from pathlib import Path

from relu_sat_bench.benchmark import build

# One generator command line per entry (see `python -m relu_sat_bench.generate --help`).
# Frozen on 31 Oct 2026. Reasons for each line: walkthrough step 4b, section 3.
SPEC = [
    # satisfiable side: both tools reached their limit at 10-20 variables in the pilot
    ["random", "--vars", "10", "12", "14", "16", "18", "20", "25", "30", "--ratio", "3.5", "--count", "5"],
    ["planted", "--vars", "10", "12", "14", "16", "18", "20", "25", "30", "--ratio", "4.26", "--count", "5"],
    # phase transition, mixed answers
    ["random", "--vars", "10", "15", "20", "25", "30", "40", "50", "60", "--ratio", "4.26", "--count", "5"],
    # unsatisfiable side: alpha,beta-CROWN refuted every 30-variable formula in the pilot
    ["random", "--vars", "10", "15", "20", "30", "40", "50", "60", "80", "--ratio", "5", "--count", "5"],
    ["pigeonhole", "--holes", "3", "4", "5", "6", "7"],
    ["tseitin", "--vertices", "6", "8", "10", "12", "14", "16", "18", "20", "--count", "3"],
    ["tseitin", "--vertices", "6", "8", "10", "12", "14", "16", "--count", "3", "--parity", "even"],
    ["ordering", "--size", "4", "5", "6", "7", "8", "9"],
]

# The pilot set of step 3 (77 instances), kept so that the pilot run can be reproduced.
PILOT_SPEC = [
    ["random", "--vars", "10", "20", "30", "--ratio", "3.5", "4.26", "5", "--count", "5"],
    ["planted", "--vars", "10", "20", "30", "--ratio", "4.26", "--count", "5"],
    ["pigeonhole", "--holes", "3", "4", "5", "6"],
    ["tseitin", "--vertices", "6", "10", "14", "--count", "3"],
    ["ordering", "--size", "4", "6", "8", "10"],
]

# Scaling experiment (decided 5 Nov 2026, after the final runs): Gurobi, through alpha,beta-CROWN's MIP
# mode, solved 209 of the 213 frozen instances, so the frozen ladders do not reach its limit. These larger
# formulas continue the ladders beyond the benchmark set, for abcrown-mip only (about 65 instances).
SCALING_SPEC = [
    ["random", "--vars", "90", "100", "120", "150", "--ratio", "4.26", "5", "--count", "3"],
    ["random", "--vars", "50", "100", "200", "--ratio", "3.5", "--count", "3"],
    ["planted", "--vars", "50", "100", "200", "--ratio", "4.26", "--count", "3"],
    ["pigeonhole", "--holes", "8", "9", "10"],
    ["tseitin", "--vertices", "24", "28", "32", "36", "40", "--count", "2"],
    ["tseitin", "--vertices", "24", "32", "40", "--count", "2", "--parity", "even"],
    ["ordering", "--size", "9", "10", "11", "12"],
]

TIMEOUT = 60.0          # seconds per verifier run (instances.csv)
ENCODINGS = ["compact"]
SETS = {"--pilot": "pilot", "--scaling": "scaling"}
OPTIONS = ("--threshold-in-network", *SETS)


def parse_args(argv: list[str]) -> tuple[int, Path, bool, str] | None:
    """(seed, output folder, threshold_in_network, set name), or None if the arguments are invalid.

    The set name is "benchmark" (the frozen SPEC), "pilot" or "scaling".
    """
    flags = [arg for arg in argv if arg.startswith("--")]
    positional = [arg for arg in argv if not arg.startswith("--")]
    if not positional or not positional[0].isdigit() or len(positional) > 2:
        return None
    if any(flag not in OPTIONS for flag in flags):
        return None
    sets = [SETS[flag] for flag in flags if flag in SETS]
    if len(sets) > 1:
        return None
    out = Path(positional[1]) if len(positional) > 1 else Path("benchmark")
    return int(positional[0]), out, "--threshold-in-network" in flags, sets[0] if sets else "benchmark"


def main(argv: list[str]) -> int:
    parsed = parse_args(argv)
    if parsed is None:
        print(__doc__)
        return 2
    seed, out, threshold_in_network, name = parsed
    spec = {"benchmark": SPEC, "pilot": PILOT_SPEC, "scaling": SCALING_SPEC}[name]
    rows = build(out, seed, spec, TIMEOUT, ENCODINGS, threshold_in_network=threshold_in_network)
    print(f"benchmark written to {out} ({len(rows)} instances)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
