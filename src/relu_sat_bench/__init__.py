"""relu-sat-bench: neural network verification instances with known answers, generated from SAT."""

from relu_sat_bench.cnf import CNF
from relu_sat_bench.network import AffineLayer, ReluNetwork
from relu_sat_bench.reduction import (
    ENCODINGS,
    VerificationInstance,
    build_network,
    decode_assignment,
    decode_partial_assignment,
    encode_assignment,
    reduce_cnf,
)

__all__ = [
    "CNF",
    "AffineLayer",
    "ENCODINGS",
    "ReluNetwork",
    "VerificationInstance",
    "build_network",
    "decode_assignment",
    "decode_partial_assignment",
    "encode_assignment",
    "reduce_cnf",
]

__version__ = "0.1.0"
