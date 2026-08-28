"""Benchmark problem suite for binary multiobjective optimization in pymoo."""

from .base import BinaryProblem
from .mkp import MKP
from .mofs import MOBFS, MOFS
from .moscp import MOSCP, MSCP
from .mstsp import MOTSP, MSTSP
from .mubqp import MUBQP

__version__ = "0.1.0"

__all__ = [
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
