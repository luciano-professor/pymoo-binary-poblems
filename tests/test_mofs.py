"""Unit and integration tests for the MOFS (Multiobjective Feature Selection) benchmark."""

from typing import Any
import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from pymoo_binary_problems import MOBFS, MOFS, BinaryProblem


def test_mofs_instantiation() -> None:
    """Test 1 — Instantiation with numpy dataset and default parameters."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 10))
    y = rng.integers(0, 2, size=50)

    problem = MOFS(X=X, y=y, cv=2, min_features=1)

    assert isinstance(problem, BinaryProblem)
    assert problem.n_samples == 50
    assert problem.n_features == 10
    assert problem.n_var == 10
    assert problem.n_obj == 2
    assert problem.n_ieq_constr == 1
    assert problem.n_eq_constr == 0
    assert problem.type_var == np.bool_
    assert np.all(problem.xl == 0)
    assert np.all(problem.xu == 1)
    assert MOBFS is MOFS


def test_mofs_from_synthetic_and_feature_costs() -> None:
    """Test 2 — Synthetic dataset generator and optional feature costs."""
    # 1. from_synthetic
    prob_synth = MOFS.from_synthetic(
        n_samples=60,
        n_features=15,
        n_informative=5,
        n_redundant=2,
        seed=42,
    )
    assert prob_synth.n_samples == 60
    assert prob_synth.n_features == 15
    assert prob_synth.n_var == 15
    assert prob_synth.n_obj == 2

    # 2. with feature_costs (n_obj = 3)
    costs = np.arange(1, 16, dtype=float)
    prob_costs = MOFS.from_synthetic(
        n_samples=40,
        n_features=15,
        feature_costs=costs,
        seed=1,
    )
    assert prob_costs.n_obj == 3
    assert prob_costs.feature_costs is not None


def test_mofs_evaluation_and_decoding() -> None:
    """Test 3 — Objective calculation, penalty handling, and mask decoding."""
    # Create simple separable dataset where feature 0 perfectly predicts y
    X = np.zeros((40, 4))
    y = np.array([0] * 20 + [1] * 20)
    X[:20, 0] = -5.0
    X[20:, 0] = 5.0
    # Add random noise to other features
    rng = np.random.default_rng(42)
    X[:, 1:] = rng.normal(size=(40, 3))

    problem = MOFS(X=X, y=y, cv=2, min_features=1, seed=42)

    # 1. Candidate selecting only feature 0 (perfect predictive power)
    x_opt = np.array([1, 0, 0, 0], dtype=bool)
    out_opt: dict[str, Any] = {}
    problem._evaluate(x_opt, out_opt)

    assert out_opt["F"].shape == (1, 2)
    # Error rate should be near 0.0, feature ratio = 1/4 = 0.25
    assert out_opt["F"][0, 0] == pytest.approx(0.0, abs=1e-2)
    assert out_opt["F"][0, 1] == pytest.approx(0.25)
    assert out_opt["G"][0, 0] == 0.0  # 1 - 1 = 0 <= 0 (feasible)

    info_opt = problem.decode_features(x_opt)
    assert info_opt["selected_indices"] == [0]
    assert info_opt["n_selected"] == 1
    assert info_opt["ratio"] == pytest.approx(0.25)
    assert info_opt["is_valid"] is True

    # 2. Empty candidate (no features selected)
    x_empty = np.array([0, 0, 0, 0], dtype=bool)
    out_empty: dict[str, Any] = {}
    problem._evaluate(x_empty, out_empty)

    # Error rate = 1.0 (penalty), feature ratio = 0.0, G = 1 - 0 = 1.0 (violation)
    assert out_empty["F"][0, 0] == 1.0
    assert out_empty["F"][0, 1] == 0.0
    assert out_empty["G"][0, 0] == 1.0

    info_empty = problem.decode_features(x_empty)
    assert info_empty["is_valid"] is False


def test_mofs_validation_errors() -> None:
    """Test 4 — Input validation error handling."""
    X = np.ones((10, 5))
    y = np.ones(10)

    # 1D X
    with pytest.raises(ValueError, match="2D array"):
        MOFS(X=np.ones(10), y=y)

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        MOFS(X=X, y=np.ones(12))

    # Features < 2
    with pytest.raises(ValueError, match="at least 2 features"):
        MOFS(X=np.ones((10, 1)), y=y)

    # min_features > n_features
    with pytest.raises(ValueError, match="min_features"):
        MOFS(X=X, y=y, min_features=10)

    # Invalid feature_costs length
    with pytest.raises(ValueError, match="feature_costs"):
        MOFS(X=X, y=y, feature_costs=[1.0, 2.0])

    # Invalid cv
    with pytest.raises(ValueError, match="Invalid cv"):
        MOFS(X=X, y=y, cv=-1)


def test_mofs_pymoo_integration() -> None:
    """Test 5 — Integration with pymoo.optimize.minimize and NSGA2."""
    problem = MOFS.from_synthetic(
        n_samples=50,
        n_features=12,
        n_informative=4,
        n_redundant=2,
        seed=42,
    )
    algorithm = NSGA2(
        pop_size=15,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(prob=0.1),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", 3),
        seed=42,
        verbose=False,
    )

    # 1. Execution success
    assert res is not None

    # 2. Binary solution masks
    assert res.X is not None
    assert isinstance(res.X, np.ndarray)
    assert len(res.X) > 0
    assert res.X.shape[1] == 12
    assert np.all(np.isin(res.X, [0, 1, True, False]))

    # 3. Objectives
    assert res.F is not None
    assert isinstance(res.F, np.ndarray)
    assert res.F.ndim == 2
    assert res.F.shape[1] == 2
    assert not np.isnan(res.F).any()
