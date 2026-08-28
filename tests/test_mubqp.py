"""Unit and integration tests for the MUBQP (Multiobjective Unconstrained Binary Quadratic Problem) benchmark."""

from typing import Any
import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from pymoo_binary_problems import MUBQP, BinaryProblem


def test_mubqp_instantiation_3d() -> None:
    """Test 1 — Instantiation with 3D numpy array."""
    # 2 objectives, 4 variables
    q1 = np.array([
        [10.0, -5.0,  2.0, 0.0],
        [-5.0, 20.0, -1.0, 3.0],
        [ 2.0, -1.0, 15.0, 4.0],
        [ 0.0,  3.0,  4.0, 8.0],
    ])
    q2 = np.array([
        [-8.0,  4.0,  0.0, 2.0],
        [ 4.0, 12.0, -3.0, 1.0],
        [ 0.0, -3.0, 25.0, -2.0],
        [ 2.0,  1.0, -2.0, 18.0],
    ])
    q_3d = np.stack([q1, q2], axis=0)

    problem = MUBQP(Q=q_3d, maximize=True)

    assert isinstance(problem, BinaryProblem)
    assert problem.n_var == 4
    assert problem.n_obj == 2
    assert problem.n_ieq_constr == 0
    assert problem.n_eq_constr == 0
    assert problem.type_var == np.bool_
    assert np.all(problem.xl == 0)
    assert np.all(problem.xu == 1)
    assert problem.maximize is True
    assert problem.Q.shape == (2, 4, 4)


def test_mubqp_instantiation_sequence() -> None:
    """Test 2 — Instantiation with list of 2D arrays and maximize=False."""
    q1 = np.eye(5)
    q2 = np.ones((5, 5))
    q3 = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])

    problem = MUBQP(Q=[q1, q2, q3], maximize=False)

    assert problem.n_var == 5
    assert problem.n_obj == 3
    assert problem.maximize is False
    assert problem.Q.shape == (3, 5, 5)


def test_mubqp_from_random() -> None:
    """Test 3 — Random instance generation classmethod."""
    n_var = 15
    n_obj = 3
    problem = MUBQP.from_random(
        n_var=n_var,
        n_obj=n_obj,
        density=0.8,
        val_range=(-50.0, 50.0),
        symmetric=True,
        seed=123,
    )

    assert problem.n_var == n_var
    assert problem.n_obj == n_obj
    assert problem.Q.shape == (n_obj, n_var, n_var)

    # Check symmetry
    for k in range(n_obj):
        assert np.allclose(problem.Q[k], problem.Q[k].T)


def test_mubqp_evaluation_accuracy() -> None:
    """Test 4 — Evaluation accuracy against analytical quadratic calculation."""
    # 2 objectives, 3 variables
    q1 = np.array([
        [10.0, 2.0, 1.0],
        [ 2.0, 5.0, 3.0],
        [ 1.0, 3.0, 4.0],
    ])
    q2 = np.array([
        [ 1.0, -2.0, 0.0],
        [-2.0,  8.0, 1.0],
        [ 0.0,  1.0, 2.0],
    ])
    q_3d = np.stack([q1, q2], axis=0)

    # Maximize = True (pymoo returns negative objectives)
    prob_max = MUBQP(Q=q_3d, maximize=True)

    # Test candidate 1: x = [1, 1, 0]
    # f1 = [1, 1, 0] @ q1 @ [1, 1, 0]^T = 10 + 5 + 2*2 = 19
    # f2 = [1, 1, 0] @ q2 @ [1, 1, 0]^T = 1 + 8 + 2*(-2) = 5
    x1 = np.array([1, 1, 0], dtype=bool)
    out1: dict[str, Any] = {}
    prob_max._evaluate(x1, out1)

    assert out1["F"].shape == (1, 2)
    assert out1["F"][0, 0] == -19.0
    assert out1["F"][0, 1] == -5.0

    # Test candidate 2: x = [1, 0, 1]
    # f1 = 10 + 4 + 2*1 = 16
    # f2 = 1 + 2 + 2*0 = 3
    # Batch evaluation:
    x_batch = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 0, 0],
    ], dtype=bool)
    out_batch: dict[str, Any] = {}
    prob_max._evaluate(x_batch, out_batch)

    assert out_batch["F"].shape == (3, 2)
    assert np.allclose(out_batch["F"][0], [-19.0, -5.0])
    assert np.allclose(out_batch["F"][1], [-16.0, -3.0])
    assert np.allclose(out_batch["F"][2], [0.0, 0.0])

    # Maximize = False (direct positive objectives)
    prob_min = MUBQP(Q=q_3d, maximize=False)
    out_min: dict[str, Any] = {}
    prob_min._evaluate(x_batch, out_min)
    assert np.allclose(out_min["F"][0], [19.0, 5.0])
    assert np.allclose(out_min["F"][1], [16.0, 3.0])
    assert np.allclose(out_min["F"][2], [0.0, 0.0])


def test_mubqp_validation_errors() -> None:
    """Test 5 — Input validation and error handling."""
    # 1D or 2D array passed directly as Q
    with pytest.raises(ValueError, match="3D array"):
        MUBQP(Q=np.eye(4))

    # n_obj < 2
    with pytest.raises(ValueError, match="Multiobjective optimization requires n_obj >= 2"):
        MUBQP(Q=np.ones((1, 4, 4)))

    # Non-square matrix
    with pytest.raises(ValueError, match="must be square"):
        MUBQP(Q=np.ones((2, 4, 5)))

    # _evaluate with invalid shapes
    prob = MUBQP.from_random(n_var=4, n_obj=2, seed=42)
    with pytest.raises(ValueError, match="Expected decision vector of length 4"):
        prob._evaluate(np.array([1, 0, 1]), {})

    with pytest.raises(ValueError, match="Expected decision vector of length 4"):
        prob._evaluate(np.ones((3, 5)), {})


def test_mubqp_pymoo_integration() -> None:
    """Test 6 — Integration with pymoo.optimize.minimize and NSGA2."""
    problem = MUBQP.from_random(
        n_var=20,
        n_obj=2,
        density=0.7,
        val_range=(-50.0, 50.0),
        seed=42,
    )
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

    # 1. Successful execution
    assert res is not None

    # 2. Valid binary solutions
    assert res.X is not None
    assert isinstance(res.X, np.ndarray)
    assert len(res.X) > 0
    assert res.X.shape[1] == 20
    assert np.all(np.isin(res.X, [0, 1, True, False]))

    # 3. Valid objectives
    assert res.F is not None
    assert isinstance(res.F, np.ndarray)
    assert res.F.ndim == 2
    assert res.F.shape[1] == 2
    assert len(res.F) == len(res.X)
    assert not np.isnan(res.F).any()
