"""Unit and integration tests for the MSTSP / MOTSP benchmark in binary representation."""

from typing import Any
import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from pymoo_binary_problems import MOTSP, MSTSP, BinaryProblem


def test_mstsp_instantiation_3d() -> None:
    """Test 1 — Instantiation with 3D numpy array."""
    # 2 objectives, 4 cities -> n_var = 16
    c1 = np.array([
        [0.0, 10.0, 15.0, 20.0],
        [10.0, 0.0, 35.0, 25.0],
        [15.0, 35.0, 0.0, 30.0],
        [20.0, 25.0, 30.0, 0.0],
    ])
    c2 = np.array([
        [0.0, 5.0, 25.0, 12.0],
        [5.0, 0.0, 10.0, 18.0],
        [25.0, 10.0, 0.0, 8.0],
        [12.0, 18.0, 8.0, 0.0],
    ])
    cost_3d = np.stack([c1, c2], axis=0)

    problem = MSTSP(cost_matrices=cost_3d)

    assert isinstance(problem, BinaryProblem)
    assert problem.n_cities == 4
    assert problem.n_var == 16
    assert problem.n_obj == 2
    assert problem.n_ieq_constr == 9  # 4 rows + 4 cols + 1 deficit
    assert problem.n_eq_constr == 0
    assert problem.type_var == np.bool_
    assert np.all(problem.xl == 0)
    assert np.all(problem.xu == 1)
    assert problem.cost_matrices.shape == (2, 4, 4)


def test_mstsp_from_random_and_coordinates() -> None:
    """Test 2 — Random generation and coordinate-based initialization."""
    # 1. from_random
    prob_rand = MSTSP.from_random(n_cities=6, n_obj=3, cost_range=(10.0, 50.0), seed=42)
    assert prob_rand.n_cities == 6
    assert prob_rand.n_var == 36
    assert prob_rand.n_obj == 3
    assert prob_rand.n_ieq_constr == 13
    for k in range(3):
        assert np.allclose(prob_rand.cost_matrices[k], prob_rand.cost_matrices[k].T)
        assert np.all(np.diag(prob_rand.cost_matrices[k]) == 0.0)

    # 2. from_coordinates
    coords1 = np.array([[0, 0], [0, 4], [3, 0]], dtype=float)
    coords2 = np.array([[0, 0], [1, 1], [2, 0]], dtype=float)
    prob_coords = MSTSP.from_coordinates([coords1, coords2])
    assert prob_coords.n_cities == 3
    assert prob_coords.n_var == 9
    assert prob_coords.n_obj == 2
    # Distance between (0,0) and (0,4) = 4, (0,4) and (3,0) = 5, (3,0) and (0,0) = 3
    assert prob_coords.cost_matrices[0, 0, 1] == pytest.approx(4.0)
    assert prob_coords.cost_matrices[0, 1, 2] == pytest.approx(5.0)
    assert prob_coords.cost_matrices[0, 2, 0] == pytest.approx(3.0)


def test_mstsp_analytical_evaluation_and_constraints() -> None:
    """Test 3 — Analytical cycle cost evaluation and constraint satisfaction."""
    # 3 cities, 2 objectives
    c1 = np.array([
        [0.0, 10.0, 20.0],
        [10.0, 0.0, 15.0],
        [20.0, 15.0, 0.0],
    ])
    c2 = np.array([
        [0.0, 8.0, 12.0],
        [8.0, 0.0, 6.0],
        [12.0, 6.0, 0.0],
    ])
    problem = MSTSP(cost_matrices=[c1, c2])

    # Valid Tour: 0 -> 1 -> 2 -> 0
    # Cost obj 1 = c1[0,1] + c1[1,2] + c1[2,0] = 10 + 15 + 20 = 45.0
    # Cost obj 2 = c2[0,1] + c2[1,2] + c2[2,0] = 8 + 6 + 12 = 26.0
    x_valid = np.zeros((3, 3), dtype=bool)
    x_valid[0, 0] = True  # step 0: city 0
    x_valid[1, 1] = True  # step 1: city 1
    x_valid[2, 2] = True  # step 2: city 2

    out_valid: dict[str, Any] = {}
    problem._evaluate(x_valid.reshape(1, -1), out_valid)

    assert out_valid["F"].shape == (1, 2)
    assert out_valid["F"][0, 0] == pytest.approx(45.0)
    assert out_valid["F"][0, 1] == pytest.approx(26.0)

    # All constraints in G must be <= 0 for feasible tour
    g_valid = out_valid["G"]
    assert g_valid.shape == (1, 7)  # 3 rows + 3 cols + 1 deficit
    assert np.all(g_valid <= 0.0)

    # Decoding valid tour
    tour, is_valid = problem.decode_tour(x_valid)
    assert is_valid is True
    assert tour == [0, 1, 2]


def test_mstsp_constraint_violations() -> None:
    """Test 4 — Infeasible solutions violation flags."""
    problem = MSTSP.from_random(n_cities=4, n_obj=2, seed=1)

    # Infeasible: step 0 has 2 cities, step 1 has 0 cities, city 3 is never visited
    x_invalid = np.zeros((4, 4), dtype=bool)
    x_invalid[0, 0] = True
    x_invalid[0, 1] = True  # Row 0 sum = 2 -> row violation: 2 - 1 = 1 > 0
    x_invalid[2, 2] = True  # Total sum = 3 -> deficit violation: 4 - 3 = 1 > 0

    out_invalid: dict[str, Any] = {}
    problem._evaluate(x_invalid.reshape(1, -1), out_invalid)

    g = out_invalid["G"]
    assert g[0, 0] == 1.0  # row 0 violation
    assert g[0, -1] == 1.0  # deficit violation

    tour, is_valid = problem.decode_tour(x_invalid)
    assert is_valid is False


def test_mstsp_validation_errors() -> None:
    """Test 5 — Input validation error handling."""
    # n_cities < 3
    with pytest.raises(ValueError, match="at least 3 cities"):
        MSTSP(cost_matrices=np.zeros((2, 2, 2)))

    # n_obj < 2
    with pytest.raises(ValueError, match="Multiobjective optimization requires n_obj >= 2"):
        MSTSP(cost_matrices=np.zeros((1, 4, 4)))

    # Non-square matrix
    with pytest.raises(ValueError, match="must be square"):
        MSTSP(cost_matrices=np.zeros((2, 4, 5)))

    # from_coordinates invalid args
    with pytest.raises(ValueError, match="at least 2 coordinate sets"):
        MSTSP.from_coordinates([np.zeros((4, 2))])


def test_mstsp_pymoo_integration() -> None:
    """Test 6 — Integration with pymoo.optimize.minimize and NSGA2."""
    problem = MSTSP.from_random(n_cities=5, n_obj=2, seed=42)
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

    # 2. Population evaluation
    pop = res.pop
    assert pop is not None
    assert len(pop) > 0

    pop_x = pop.get("X")
    assert pop_x is not None
    assert pop_x.shape[1] == 25  # 5 * 5 = 25 bits
    assert np.all(np.isin(pop_x, [0, 1, True, False]))

    pop_f = pop.get("F")
    assert pop_f is not None
    assert pop_f.ndim == 2
    assert pop_f.shape[1] == 2
    assert not np.isnan(pop_f).any()

    pop_g = pop.get("G")
    assert pop_g is not None
    assert pop_g.shape[1] == 2 * 5 + 1

    # 3. MOTSP alias check
    assert MOTSP is MSTSP
