"""Unit and integration tests for the MKP (Multiple Knapsack Problem) benchmark."""

from typing import Any
import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from pymoo_binary_problems import MKP, BinaryProblem


def test_mkp_instantiation() -> None:
    """Test 1 — Instantiation: Create classic MKP instance with 10 items and 3 knapsacks."""
    n_items = 10
    profits = np.array([10, 20, 30, 40, 50, 15, 25, 35, 45, 55], dtype=float)
    weights = np.array([5, 10, 15, 20, 25, 7, 12, 18, 22, 28], dtype=float)
    capacities = np.array([40.0, 50.0, 30.0], dtype=float)

    mkp = MKP(profits=profits, weights=weights, capacities=capacities, n_obj=2)

    assert isinstance(mkp, BinaryProblem)
    assert mkp.n_items == n_items
    assert mkp.n_knapsacks == 3
    assert mkp.n_var == 30  # 10 items * 3 knapsacks
    assert mkp.n_obj == 2
    assert mkp.n_ieq_constr == 13  # 3 capacities + 10 single-assignments
    assert mkp.type_var == np.bool_
    assert np.all(mkp.xl == 0)
    assert np.all(mkp.xu == 1)
    assert np.array_equal(mkp.capacities, capacities)
    assert not mkp.normalize_profit
    assert not mkp.normalize_weight


def test_mkp_from_random() -> None:
    """Test MKP.from_random factory method."""
    mkp = MKP.from_random(n_items=20, n_knapsacks=4, seed=42)
    assert mkp.n_items == 20
    assert mkp.n_knapsacks == 4
    assert mkp.n_var == 80
    assert mkp.n_obj == 2
    assert mkp.n_ieq_constr == 24


def test_mkp_evaluation_3d() -> None:
    """Test 2 — Evaluation: Validate objective and constraint calculation with 3D array input."""
    profits = np.array([10.0, 20.0, 30.0, 40.0])
    weights = np.array([5.0, 10.0, 15.0, 20.0])
    capacities = np.array([20.0, 25.0])  # 2 knapsacks, 4 items -> 6 constraints

    mkp = MKP(profits=profits, weights=weights, capacities=capacities, n_obj=2)

    # Create feasible 3D solution:
    # Item 0 (p=10, w=5) in knapsack 0 -> knapsack 0 weight = 5 <= 20
    # Item 1 (p=20, w=10) and Item 2 (p=30, w=15) in knapsack 1 -> knapsack 1 weight = 25 <= 25
    # Item 3 not allocated
    x_3d = np.zeros((1, 4, 2), dtype=bool)
    x_3d[0, 0, 0] = True
    x_3d[0, 1, 1] = True
    x_3d[0, 2, 1] = True

    out: dict[str, Any] = {}
    mkp._evaluate(x_3d, out)

    # 1. Objectives: Total profit = 10 + 20 + 30 = 60 -> f1 = -60. Total weight = 5 + 25 = 30 -> f2 = 30
    assert out["F"].shape == (1, 2)
    assert out["F"][0, 0] == -60.0
    assert out["F"][0, 1] == 30.0

    # 2. Constraints in out["G"] (M=2 capacities + N=4 assignments):
    # G_capacity[0] = 5 - 20 = -15
    # G_capacity[1] = 25 - 25 = 0
    # G_assignment for all items: item 0=0, item 1=0, item 2=0, item 3=-1
    assert out["G"].shape == (1, 6)
    assert out["G"][0, 0] == -15.0  # knapsack 0 capacity constraint
    assert out["G"][0, 1] == 0.0    # knapsack 1 capacity constraint
    assert np.all(out["G"][0, 2:] <= 0.0)  # item assignment constraints


def test_mkp_evaluation_flattened_2d() -> None:
    """Test 3 — Shape Compatibility: Validate evaluation with 2D flattened binary matrix (N, n_var)."""
    profits = np.array([10.0, 20.0, 30.0])
    weights = np.array([5.0, 10.0, 15.0])
    capacities = np.array([20.0, 25.0])

    mkp = MKP(profits=profits, weights=weights, capacities=capacities, n_obj=2)

    # 2 individuals: 3 items * 2 knapsacks = 6 bits per individual
    x_flat = np.array([
        [True, False, True, False, False, False],  # Ind 0: item 0 in k0, item 1 in k0
        [False, True, False, False, False, True],  # Ind 1: item 0 in k1, item 2 in k1
    ], dtype=bool)

    out: dict[str, Any] = {}
    mkp._evaluate(x_flat, out)

    assert out["F"].shape == (2, 2)
    assert out["G"].shape == (2, 5)  # 2 knapsacks + 3 items = 5 constraints


def test_mkp_normalized_objectives() -> None:
    """Test 4 — Normalization: Verify normalization logic for profit and weight objectives."""
    profits = np.array([10.0, 20.0, 30.0, 40.0])
    weights = np.array([10.0, 20.0, 30.0, 40.0])
    capacities = np.array([100.0])

    mkp = MKP(
        profits=profits,
        weights=weights,
        capacities=capacities,
        n_obj=2,
        normalize_profit=True,
        normalize_weight=True,
    )

    # Allocate all items to single knapsack (sum profit = 100, max profit = 100 -> norm = 1.0)
    x_all = np.ones((1, 4, 1), dtype=bool)
    out: dict[str, Any] = {}
    mkp._evaluate(x_all, out)

    assert np.isclose(out["F"][0, 0], -1.0)
    assert np.isclose(out["F"][0, 1], 1.0)


def test_mkp_validation_errors() -> None:
    """Test 5 — Validations: Ensure invalid dimensions and parameters raise expected errors."""
    with pytest.raises(ValueError, match="The 'profits', 'weights', and 'capacities' arrays must be 1-dimensional"):
        MKP(profits=np.ones((2, 2)), weights=np.ones(4), capacities=np.ones(2))

    with pytest.raises(ValueError, match="Size mismatch between profits"):
        MKP(profits=np.ones(5), weights=np.ones(4), capacities=np.ones(2))

    with pytest.raises(ValueError, match="The 'capacities' array must contain at least one knapsack"):
        MKP(profits=np.ones(4), weights=np.ones(4), capacities=np.array([]))

    with pytest.raises(ValueError, match="The number of objectives 'n_obj' must be at least 2"):
        MKP(profits=np.ones(4), weights=np.ones(4), capacities=np.ones(2), n_obj=1)

    with pytest.raises(ValueError, match="the 'extra_objectives' callable must be provided"):
        MKP(profits=np.ones(4), weights=np.ones(4), capacities=np.ones(2), n_obj=3, extra_objectives=None)


def test_mkp_nsga2_integration() -> None:
    """Test 6 — Integration: Execute full optimization cycle using pymoo NSGA2 on MKP."""
    profits = np.array([12, 15, 20, 25, 30, 35, 40, 45, 50, 60], dtype=float)
    weights = np.array([8, 10, 12, 16, 20, 22, 25, 28, 30, 35], dtype=float)
    capacities = np.array([45.0, 55.0], dtype=float)

    problem = MKP(profits=profits, weights=weights, capacities=capacities, n_obj=2)

    algorithm = NSGA2(
        pop_size=20,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(prob=0.05),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 5),
        seed=42,
        verbose=False,
    )

    assert res.X is not None
    assert res.F is not None
    assert len(res.X) > 0
    assert res.F.shape[1] == 2
