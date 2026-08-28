"""Test public API imports and package structure."""

import pymoo_binary_problems
from pymoo_binary_problems import (
    BinaryProblem,
    MKP,
    MOBFS,
    MOFS,
    MOSCP,
    MOTSP,
    MSCP,
    MSTSP,
    MUBQP,
    __version__,
)


def test_package_exports() -> None:
    """Verify all expected symbols are exported in __all__ and importable."""
    assert __version__ == "1.0.0"

    expected_all = [
        "BinaryProblem",
        "MKP",
        "MOBFS",
        "MOFS",
        "MOSCP",
        "MOTSP",
        "MSCP",
        "MSTSP",
        "MUBQP",
        "__version__",
    ]
    for symbol in expected_all:
        assert hasattr(pymoo_binary_problems, symbol), f"Missing symbol: {symbol}"
        assert symbol in pymoo_binary_problems.__all__, f"Symbol not in __all__: {symbol}"


def test_aliases() -> None:
    """Verify alias equivalence."""
    assert MOBFS is MOFS
    assert MSCP is MOSCP
    assert MOTSP is MSTSP
