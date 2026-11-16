"""Adapters that run verification tools with the VNN-COMP calling convention.

Every adapter is a command that is called as ``<command> ONNX VNNLIB RESULT_FILE TIMEOUT``
and writes a VNN-COMP result file: first line ``sat``, ``unsat``, ``timeout`` or
``unknown``; for ``sat`` followed by the counterexample ``((X_0 v) ... (Y_0 v))``.
"""

import sys

TOOLS = {
    # Marabou via maraboupy, with Marabou's own VNN-LIB parser.
    "marabou": [sys.executable, "-m", "relu_sat_bench.tools.marabou"],
    # alpha,beta-CROWN with the configuration its authors chose for the VNN-COMP 2025
    # benchmark sat_relu (MIP with Gurobi) ...
    "abcrown-mip": [sys.executable, "-m", "relu_sat_bench.tools.abcrown", "--mode", "mip"],
    # ... and the same configuration with its default branch-and-bound instead of MIP.
    "abcrown-bab": [sys.executable, "-m", "relu_sat_bench.tools.abcrown", "--mode", "bab"],
}
