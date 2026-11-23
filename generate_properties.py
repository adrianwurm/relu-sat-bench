"""Build the relu-sat-bench benchmark set.

VNN-COMP convention: ``python generate_properties.py <seed> [output folder]``.
The same seed always produces the same instances, on every machine.

Options (after the seed and the folder):

  --threshold-in-network   the equivalent variant for alpha,beta-CROWN's MIP mode: network output
                           lowered by m - 1/2, unsafe set Y_0 >= 0 (same formulas, same file names)
  --pilot                  the 77-instance pilot set of step 3 instead of the frozen set
  --scaling                larger formulas for the scaling experiment with alpha,beta-CROWN's MIP mode
                           (an extra experiment, not part of the benchmark set)
  --large                  still larger formulas, beyond --scaling: where alpha,beta-CROWN's MIP mode stops in
                           the neurocodebench encoding (an extra experiment, use with --neurocodebench)
  --sample                 one formula per size of the frozen set (57 instances): the formulas with index
                           000, identical to those of the frozen set built with the same seed
  --neurocodebench         the neurocodebench encoding (the construction of the VNN-COMP 2025 benchmark
                           sat_relu) instead of compact; not together with --threshold-in-network. Without one
                           of the set options it builds the second benchmark set (NCB_SPEC, 209 instances),
                           whose ladders reach the limits of the tools in that encoding
  --two-constraints        unsafe set m - 1/2 <= Y_0 <= m: the same set, written as two constraints, which
                           alpha,beta-CROWN's MIP mode handles differently; compact only, not together with
                           --threshold-in-network
  --competition            the set for VNN-COMP 2027: both benchmark sets in one folder, thinned to the first 3
                           formulas of every size (SPEC in compact, NCB_SPEC in neurocodebench; 296 instances,
                           4.9 h of timeouts); with --threshold-in-network the compact part gets that variant

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
from typing import NamedTuple

from relu_sat_bench.benchmark import build, build_parts

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

# Beyond SCALING_SPEC (45 instances; decided 17 Nov 2026). In the neurocodebench encoding alpha,beta-CROWN's
# MIP mode solves all of SCALING_SPEC, so these continue every ladder. CaDiCaL needs at most a few seconds for
# the random formulas (n = 300, ratio 5: about 1-4 s); pigeonhole h11-h14 are beyond it (answer by theorem).
LARGE_SPEC = [
    ["random", "--vars", "175", "200", "250", "--ratio", "4.26", "--count", "3"],
    ["random", "--vars", "175", "200", "250", "300", "--ratio", "5", "--count", "3"],
    ["planted", "--vars", "300", "400", "600", "--ratio", "4.26", "--count", "2"],
    ["pigeonhole", "--holes", "11", "12", "13", "14"],
    ["tseitin", "--vertices", "60", "80", "100", "140", "200", "--count", "2"],
    ["ordering", "--size", "14", "16", "18", "20"],
]


def sample_spec(spec: list[list[str]]) -> list[list[str]]:
    """``spec`` with one formula per size: every ``--count`` set to 1.

    Each formula's random generator is seeded by (seed, family, parameters, index), not by the count,
    so the sample contains exactly the formulas with index 000 of the full set (decided 16 Nov 2026,
    for the comparison with the neurocodebench encoding).
    """
    return [[("1" if i > 0 and command[i - 1] == "--count" else arg) for i, arg in enumerate(command)]
            for command in spec]


def thin_spec(spec: list[list[str]], count: int = 3) -> list[list[str]]:
    """``spec`` with at most ``count`` formulas per size: every ``--count`` above it lowered to ``count``.

    As for :func:`sample_spec`, each formula is seeded by (seed, family, parameters, index), so the thinned set
    contains exactly the formulas with index 000 to ``count - 1`` of the full set (decided 21 Nov 2026).
    """
    return [[(str(min(int(arg), count)) if i > 0 and command[i - 1] == "--count" else arg)
             for i, arg in enumerate(command)] for command in spec]


# The second benchmark set, in the neurocodebench encoding (frozen 20 Nov 2026, 209 instances, use with seed 1).
# In that encoding Marabou and branch-and-bound fail on very small formulas, while alpha,beta-CROWN's MIP mode
# gets to 250-300 variables (random, planted) and solves pigeonhole, Tseitin and ordering formulas far beyond
# what CaDiCaL manages (step 5b of the walkthrough). So every ladder starts small and ends beyond MIP's limit.
# With seed 1 the formulas of sizes that also occur in SPEC (10, 20, 50 ...) are identical to those of SPEC.
NCB_SPEC = [
    ["random", "--vars", "10", "20", "50", "100", "200", "300", "400", "--ratio", "3.5", "--count", "5"],
    ["planted", "--vars", "10", "20", "50", "100", "150", "200", "250", "300", "--ratio", "4.26", "--count", "5"],
    # phase transition: up to 250 only, because CaDiCaL needs 20-50 s for the unsatisfiable ones at n = 300 (too
    # close to the 60 s the generator gives it; a slower machine would drop formulas)
    ["random", "--vars", "10", "20", "50", "100", "150", "200", "225", "250", "--ratio", "4.26", "--count", "5"],
    ["random", "--vars", "10", "20", "50", "100", "150", "200", "250", "300", "--ratio", "5", "--count", "5"],
    ["pigeonhole", "--holes", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"],
    ["tseitin", "--vertices", "6", "8", "10", "14", "20", "30", "40", "60", "100", "140", "200", "--count", "2"],
    ["tseitin", "--vertices", "6", "10", "20", "40", "60", "--count", "2", "--parity", "even"],
    ["ordering", "--size", "4", "5", "6", "8", "10", "12", "14", "16", "18", "20"],
]

# The set for VNN-COMP 2027 (decided 21 Nov 2026): both benchmark sets, thinned to the first 3 formulas per size,
# each in its own encoding, in one folder. Formulas shared by the two sets appear in both encodings.
def competition_parts() -> list[tuple[list[list[str]], str]]:
    return [(thin_spec(SPEC), "compact"), (thin_spec(NCB_SPEC), "neurocodebench")]


TIMEOUT = 60.0          # seconds per verifier run (instances.csv)
ENCODINGS = ["compact"]
SETS = {"--pilot": "pilot", "--scaling": "scaling", "--large": "large", "--sample": "sample",
        "--competition": "competition"}
OPTIONS = ("--threshold-in-network", "--neurocodebench", "--two-constraints", *SETS)


class Arguments(NamedTuple):
    seed: int
    out: Path
    threshold_in_network: bool
    set_name: str     # "benchmark" (the frozen SPEC, or NCB_SPEC with neurocodebench), "pilot", "scaling", ...
    encoding: str     # "compact" or "neurocodebench"
    two_constraints: bool = False


def parse_args(argv: list[str]) -> Arguments | None:
    """The parsed command line, or None if the arguments are invalid."""
    flags = [arg for arg in argv if arg.startswith("--")]
    positional = [arg for arg in argv if not arg.startswith("--")]
    if not positional or not positional[0].isdigit() or len(positional) > 2:
        return None
    if any(flag not in OPTIONS for flag in flags):
        return None
    sets = [SETS[flag] for flag in flags if flag in SETS]
    threshold_in_network = "--threshold-in-network" in flags
    neurocodebench = "--neurocodebench" in flags
    two_constraints = "--two-constraints" in flags
    if len(sets) > 1 or sum((threshold_in_network, neurocodebench, two_constraints)) > 1:
        return None
    if "competition" in sets and (neurocodebench or two_constraints):
        return None
    out = Path(positional[1]) if len(positional) > 1 else Path("benchmark")
    return Arguments(int(positional[0]), out, threshold_in_network, sets[0] if sets else "benchmark",
                     "neurocodebench" if neurocodebench else ENCODINGS[0], two_constraints)


def main(argv: list[str]) -> int:
    parsed = parse_args(argv)
    if parsed is None:
        print(__doc__)
        return 2
    seed, out, threshold_in_network, name, encoding, two_constraints = parsed
    if name == "competition":
        rows = build_parts(out, seed, competition_parts(), TIMEOUT, threshold_in_network=threshold_in_network)
        print(f"benchmark written to {out} ({len(rows)} instances)")
        return 0
    spec = {"benchmark": NCB_SPEC if encoding == "neurocodebench" else SPEC, "pilot": PILOT_SPEC,
            "scaling": SCALING_SPEC, "large": LARGE_SPEC, "sample": sample_spec(SPEC)}[name]
    rows = build(out, seed, spec, TIMEOUT, [encoding], threshold_in_network=threshold_in_network,
                 two_constraints=two_constraints)
    print(f"benchmark written to {out} ({len(rows)} instances)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
