"""Unit tests for BinaryProblem base class."""

from typing import Any
import numpy as np
import pytest
from pymoo.core.problem import Problem

from pymoo_binary_problems.base import BinaryProblem


def test_binary_problem_init() -> None:
    """Test BinaryProblem initialization and pymoo attribute conformance."""
    problem = BinaryProblem(n_var=10, n_obj=2, n_ieq_constr=3, n_eq_constr=1)

    assert isinstance(problem, Problem)
    assert problem.n_var == 10
    assert problem.n_obj == 2
    assert problem.n_ieq_constr == 3
    assert problem.n_eq_constr == 1
    assert problem.type_var == np.bool_
    assert np.all(problem.xl == 0)
    assert np.all(problem.xu == 1)


def test_binary_problem_abstract_evaluate() -> None:
    """Test that base BinaryProblem raises NotImplementedError when evaluated."""
    problem = BinaryProblem(n_var=5, n_obj=2)
    x = np.zeros((1, 5), dtype=bool)
    out: dict[str, Any] = {}

    with pytest.raises(NotImplementedError, match="Subclasses of BinaryProblem must implement _evaluate"):
        problem._evaluate(x, out)


def test_binary_problem_subclass_evaluation() -> None:
    """Test custom concrete implementation of BinaryProblem."""

    class SimpleOneMax(BinaryProblem):
        def __init__(self, n_bits: int = 10):
            super().__init__(n_var=n_bits, n_obj=2)

        def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            # f1: minimize sum of bits, f2: minimize zeros
            f1 = np.sum(x, axis=1)
            f2 = self.n_var - f1
            out["F"] = np.column_stack([f1, f2])

    p = SimpleOneMax(n_bits=8)
    x_pop = np.array([
        [True] * 8,
        [False] * 8,
        [True, False] * 4,
    ], dtype=bool)

    out: dict[str, Any] = {}
    p._evaluate(x_pop, out)

    assert out["F"].shape == (3, 2)
    assert np.array_equal(out["F"][0], [8, 0])
    assert np.array_equal(out["F"][1], [0, 8])
    assert np.array_equal(out["F"][2], [4, 4])
