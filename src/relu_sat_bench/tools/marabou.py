"""Marabou adapter: ``python -m relu_sat_bench.tools.marabou ONNX VNNLIB RESULT_FILE TIMEOUT``.

Uses maraboupy (``pip install maraboupy``): the network is read with Marabou's ONNX parser and
the property with Marabou's own VNN-LIB parser (``MarabouCore.loadProperty``), so nothing of
our code sits between the files and the tool. Marabou runs with its default options and the
given timeout; the runner additionally kills it if it overruns.
"""

from __future__ import annotations

import math
import sys
import warnings


def main(argv: list[str]) -> int:
    onnx_path, vnnlib_path, result_path, timeout = argv
    warnings.filterwarnings("ignore", message="Tensorflow parser is unavailable")
    from maraboupy import Marabou, MarabouCore

    network = Marabou.read_onnx(onnx_path)
    query = network.getInputQuery()
    MarabouCore.loadProperty(query, vnnlib_path)
    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=max(1, math.ceil(float(timeout))))
    exit_code, values, _ = MarabouCore.solve(query, options, "")

    status = {"sat": "sat", "unsat": "unsat", "TIMEOUT": "timeout"}.get(exit_code, "unknown")
    lines = [status]
    if status == "sat":
        inputs = network.inputVars[0].flatten()
        outputs = network.outputVars[0].flatten()
        pairs = [f"(X_{i} {values[v]!r})" for i, v in enumerate(inputs)]
        pairs += [f"(Y_{j} {values[v]!r})" for j, v in enumerate(outputs)]
        lines.append("(" + "\n ".join(pairs) + ")")
    with open(result_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Marabou exit code: {exit_code}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
