"""Pytest fixtures shared by all test files."""

import pytest

from formulas import SMALL_FORMULAS
from relu_sat_bench.reduction import ENCODINGS


@pytest.fixture(params=[cnf for _, cnf in SMALL_FORMULAS], ids=[name for name, _ in SMALL_FORMULAS])
def small_cnf(request):
    """Runs a test once for every formula in ``SMALL_FORMULAS``."""
    return request.param


@pytest.fixture(params=ENCODINGS)
def encoding(request):
    """Runs a test once per network encoding ("compact" and "paper")."""
    return request.param
