"""Unit and integration tests for the MOSCP (Multiobjective Set Covering Problem) benchmark."""

from typing import Any
import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from pymoo_binary_problems import MOSCP, MSCP, BinaryProblem


def test_moscp_instantiation() -> None:
    """Test 1 — Instantiation with incidence matrix and cost matrix."""
    # 4 elements, 5 candidate subsets, 2 objectives
    incidence = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1],
    ], dtype=int)

    costs = np.array([
        [10.0, 15.0, 20.0, 12.0, 18.0],
        [ 5.0,  8.0, 12.0, 25.0,  7.0],
    ], dtype=float)

    problem = MOSCP(incidence_matrix=incidence, costs=costs)

    assert isinstance(problem, BinaryProblem)
    assert problem.n_elements == 4
    assert problem.n_subsets == 5
    assert problem.n_var == 5
    assert problem.n_obj == 2
    assert problem.n_ieq_constr == 4
    assert problem.n_eq_constr == 0
    assert problem.type_var == np.bool_
    assert np.all(problem.xl == 0)
    assert np.all(problem.xu == 1)
    assert MSCP is MOSCP


def test_moscp_from_random_and_from_subsets() -> None:
    """Test 2 — Random generator and subset collection initialization."""
    # 1. from_random
    prob_rand = MOSCP.from_random(n_elements=10, n_subsets=15, n_obj=3, density=0.3, seed=42)
    assert prob_rand.n_elements == 10
    assert prob_rand.n_subsets == 15
    assert prob_rand.n_var == 15
    assert prob_rand.n_obj == 3
    assert prob_rand.n_ieq_constr == 10
    # Every element must be covered by at least 1 subset (guaranteed feasibility)
    assert np.all(np.sum(prob_rand.incidence_matrix, axis=1) >= 1)

    # 2. from_subsets
    # Subset 0 covers {0, 1}, Subset 1 covers {1, 2}, Subset 2 covers {2, 3}
    subsets = [[0, 1], [1, 2], [2, 3]]
    costs = [[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]]
    prob_subsets = MOSCP.from_subsets(universe_size=4, subsets=subsets, costs=costs)

    assert prob_subsets.n_elements == 4
    assert prob_subsets.n_subsets == 3
    assert prob_subsets.n_var == 3
    assert prob_subsets.incidence_matrix[0, 0] == 1
    assert prob_subsets.incidence_matrix[1, 1] == 1
    assert prob_subsets.incidence_matrix[3, 2] == 1


def test_moscp_analytical_evaluation_and_decoding() -> None:
    """Test 3 — Analytical evaluation and coverage decoding."""
    # 3 elements, 3 subsets
    # S0 covers {0}, S1 covers {0, 1}, S2 covers {1, 2}
    incidence = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=int)

    costs = np.array([
        [10.0, 25.0, 15.0],
        [20.0, 10.0, 30.0],
    ], dtype=float)

    problem = MOSCP(incidence_matrix=incidence, costs=costs)

    # Candidate 1: Select S1 and S2 (x = [0, 1, 1])
    # Covers: S1 covers {0, 1}, S2 covers {1, 2} -> Elements covered: {0: 1, 1: 2, 2: 1} -> 100% Feasible
    # Costs: Obj 1 = 25 + 15 = 40.0; Obj 2 = 10 + 30 = 40.0
    x1 = np.array([0, 1, 1], dtype=bool)
    out1: dict[str, Any] = {}
    problem._evaluate(x1, out1)

    assert out1["F"].shape == (1, 2)
    assert out1["F"][0, 0] == pytest.approx(40.0)
    assert out1["F"][0, 1] == pytest.approx(40.0)

    # Constraints in G (1 - coverage <= 0):
    # Element 0: 1 - 1 = 0 <= 0 (OK)
    # Element 1: 1 - 2 = -1 <= 0 (OK)
    # Element 2: 1 - 1 = 0 <= 0 (OK)
    assert np.all(out1["G"] <= 0.0)

    info1 = problem.decode_coverage(x1)
    assert info1["is_feasible"] is True
    assert info1["selected_subsets"] == [1, 2]
    assert len(info1["uncovered_elements"]) == 0

    # Candidate 2: Select only S0 (x = [1, 0, 0])
    # Covers only element 0. Elements 1 and 2 uncovered -> Infeasible
    x2 = np.array([1, 0, 0], dtype=bool)
    out2: dict[str, Any] = {}
    problem._evaluate(x2, out2)

    # G: Element 0 -> 1 - 1 = 0; Element 1 -> 1 - 0 = 1; Element 2 -> 1 - 0 = 1
    assert out2["G"][0, 0] == 0.0
    assert out2["G"][0, 1] == 1.0  # violation
    assert out2["G"][0, 2] == 1.0  # violation

    info2 = problem.decode_coverage(x2)
    assert info2["is_feasible"] is False
    assert info2["uncovered_elements"] == [1, 2]


def test_moscp_validation_errors() -> None:
    """Test 4 — Input validation error handling."""
    # 1D incidence
    with pytest.raises(ValueError, match="Incidence matrix must be 2D"):
        MOSCP(incidence_matrix=np.array([1, 0, 1]), costs=np.ones((2, 3)))

    # Subset count mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        MOSCP(incidence_matrix=np.ones((3, 4), dtype=int), costs=np.ones((2, 5)))

    # n_obj < 2
    with pytest.raises(ValueError, match="Multiobjective optimization requires n_obj >= 2"):
        MOSCP(incidence_matrix=np.ones((3, 4), dtype=int), costs=np.ones((1, 4)))

    # Non-binary incidence
    with pytest.raises(ValueError, match="Incidence matrix entries must be binary"):
        MOSCP(incidence_matrix=np.array([[2, 0], [0, 1]]), costs=np.ones((2, 2)))

    # Unfeasible problem instance (uncoverable element)
    with pytest.raises(ValueError, match="cannot be covered by any subset"):
        MOSCP(incidence_matrix=np.array([[0, 0], [1, 1]]), costs=np.ones((2, 2)))


def test_moscp_pymoo_integration() -> None:
    """Test 5 — Integration with pymoo.optimize.minimize and NSGA2."""
    problem = MOSCP.from_random(n_elements=12, n_subsets=20, n_obj=2, density=0.3, seed=42)
    algorithm = NSGA2(
        pop_size=20,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(prob=0.1),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", 5),
        seed=42,
        verbose=False,
    )

    # 1. Execution success
    assert res is not None

    # 2. Binary solution vectors
    assert res.X is not None
    assert isinstance(res.X, np.ndarray)
    assert len(res.X) > 0
    assert res.X.shape[1] == 20
    assert np.all(np.isin(res.X, [0, 1, True, False]))

    # 3. Objectives
    assert res.F is not None
    assert isinstance(res.F, np.ndarray)
    assert res.F.ndim == 2
    assert res.F.shape[1] == 2
    assert not np.isnan(res.F).any()
