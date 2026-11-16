"""alpha,beta-CROWN adapter.

``python -m relu_sat_bench.tools.abcrown --mode {mip,bab} ONNX VNNLIB RESULT_FILE TIMEOUT``

Builds the same command as alpha,beta-CROWN's own VNN-COMP wrapper
(``complete_verifier/vnncomp_main.py``), with one choice made explicit: the configuration.
Our benchmark is new, so alpha,beta-CROWN has no configuration for it. We use the one its
authors chose for the most similar VNN-COMP benchmark, ``sat_relu`` (VNN-COMP 2025,
``exp_configs/vnncomp25/sat_relu.yaml``):

* ``--mode mip``: exactly that configuration (complete verifier: MIP, solved with Gurobi);
* ``--mode bab``: the same configuration with the default complete verifier, branch and
  bound, which needs no Gurobi.

Environment: ``ABCROWN_HOME`` (the cloned repository, default ``/opt/abcrown``) and
``ABCROWN_DEVICE`` (``cpu`` or ``cuda``, default ``cpu``).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

CONFIG = Path("exp_configs") / "vnncomp25" / "sat_relu.yaml"


def build_command(mode: str, onnx: str, vnnlib: str, result: str, timeout: str,
                  home: Path, device: str) -> list[str]:
    verifier = home / "complete_verifier"
    command = [
        sys.executable, str(verifier / "abcrown.py"),
        "--config", str(verifier / CONFIG),
        "--onnx_path", str(Path(onnx).resolve()),
        "--vnnlib_path", str(Path(vnnlib).resolve()),
        "--results_file", str(Path(result).resolve()),
        "--timeout", str(timeout),
        "--save_adv_example",
        "--device", device,
    ]
    if mode == "bab":
        command += ["--complete_verifier", "bab"]
    return command


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m relu_sat_bench.tools.abcrown")
    parser.add_argument("--mode", choices=("mip", "bab"), required=True)
    parser.add_argument("onnx")
    parser.add_argument("vnnlib")
    parser.add_argument("result")
    parser.add_argument("timeout")
    args = parser.parse_args(argv)
    home = Path(os.environ.get("ABCROWN_HOME", "/opt/abcrown"))
    device = os.environ.get("ABCROWN_DEVICE", "cpu")
    command = build_command(args.mode, args.onnx, args.vnnlib, args.result, args.timeout, home, device)
    print("command:", " ".join(command), flush=True)
    code = subprocess.call(command, cwd=home / "complete_verifier")
    if code < 0:  # killed by a signal, e.g. a segmentation fault: report it the way a shell does
        print(f"alpha,beta-CROWN was killed by signal {-code} ({signal.Signals(-code).name})", flush=True)
        return 128 - code
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
