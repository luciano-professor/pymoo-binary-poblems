"""Multiobjective Traveling Salesman Problem (MSTSP / MOTSP) benchmark in binary representation."""

from collections.abc import Sequence
from typing import Any, Optional, Union
import numpy as np

from .base import BinaryProblem


class MSTSP(BinaryProblem):
    """Multiobjective Traveling Salesman Problem (MSTSP / MOTSP) in binary representation.

    Formulated using the Position-City Permutation Matrix (Assignment) encoding.
    For N cities, a candidate solution is an (N x N) binary matrix X (n_var = N * N), where:
        X[p, i] = 1 if city i is visited at step/position p (p = 0, ..., N-1)
        X[p, i] = 0 otherwise.

    Objectives (k = 1, ..., M):
    Given M cost/distance/time matrices C^(1), ..., C^(M) in R^(N x N), the total tour
    cost for objective k is:
        f_k(X) = sum_{p=0}^{N-1} sum_{i=0}^{N-1} sum_{j=0}^{N-1} X[p, i] * C_ij^(k) * X[(p+1)%N, j]

    Constraints in out["G"] (g(x) <= 0):
    1. Single city per position (N constraints):
       sum_i X[p, i] - 1 <= 0
    2. Single visit per city (N constraints):
       sum_p X[p, i] - 1 <= 0
    3. Tour completeness deficit (1 constraint):
       N - sum_{p, i} X[p, i] <= 0

    Total inequality constraints: n_ieq_constr = 2 * N + 1.
    """

    def __init__(
        self,
        cost_matrices: Union[np.ndarray, Sequence[np.ndarray]],
        **kwargs: Any,
    ) -> None:
        """Initialize the Multiobjective Traveling Salesman Problem.

        Parameters
        ----------
        cost_matrices : Union[np.ndarray, Sequence[np.ndarray]]
            Cost matrices for each objective criterion. Can be provided as:
            - 3D numpy array of shape (n_obj, n_cities, n_cities).
            - Sequence of 2D numpy arrays [C_1, C_2, ..., C_M], each of shape (n_cities, n_cities).
        **kwargs : Any
            Additional keyword arguments passed to the BinaryProblem / Problem base class.
        """
        c_arr = np.asarray(cost_matrices, dtype=float)

        if c_arr.ndim != 3:
            raise ValueError(
                f"Cost matrices must form a 3D array of shape (n_obj, n_cities, n_cities), got {c_arr.shape}."
            )

        n_obj, n_cities_1, n_cities_2 = c_arr.shape

        if n_cities_1 != n_cities_2:
            raise ValueError(f"Cost matrices must be square (N x N), got ({n_cities_1}, {n_cities_2}).")

        if n_cities_1 < 3:
            raise ValueError(f"Traveling Salesman Problem requires at least 3 cities, got {n_cities_1}.")

        if n_obj < 2:
            raise ValueError(f"Multiobjective optimization requires n_obj >= 2, got {n_obj}.")

        # Enforce zero diagonal (no self-loops)
        for k in range(n_obj):
            np.fill_diagonal(c_arr[k], 0.0)

        self.cost_matrices: np.ndarray = c_arr
        self.n_cities: int = n_cities_1

        super().__init__(
            n_var=self.n_cities * self.n_cities,
            n_obj=n_obj,
            n_ieq_constr=2 * self.n_cities + 1,
            **kwargs,
        )

    @classmethod
    def from_random(
        cls,
        n_cities: int = 15,
        n_obj: int = 2,
        cost_range: tuple[float, float] = (5.0, 100.0),
        symmetric: bool = True,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "MSTSP":
        """Generate a randomized Multiobjective TSP instance.

        Parameters
        ----------
        n_cities : int, default=15
            Number of cities to visit.
        n_obj : int, default=2
            Number of conflicting cost criteria (e.g., Distance, Time, Tolls).
        cost_range : tuple[float, float], default=(5.0, 100.0)
            Range of edge costs between distinct cities.
        symmetric : bool, default=True
            If True, cost(i, j) == cost(j, i) for all city pairs.
        seed : Optional[int], default=None
            Random seed for reproducibility.
        **kwargs : Any
            Additional keyword arguments passed to MSTSP constructor.

        Returns
        -------
        MSTSP
            Instantiated MSTSP benchmark problem.
        """
        rng = np.random.default_rng(seed)
        matrices = np.zeros((n_obj, n_cities, n_cities), dtype=float)

        for k in range(n_obj):
            raw = rng.uniform(cost_range[0], cost_range[1], size=(n_cities, n_cities))
            if symmetric:
                raw = 0.5 * (raw + raw.T)
            np.fill_diagonal(raw, 0.0)
            matrices[k] = raw

        return cls(cost_matrices=matrices, **kwargs)

    @classmethod
    def from_coordinates(
        cls,
        coordinates_list: Sequence[np.ndarray],
        **kwargs: Any,
    ) -> "MSTSP":
        """Construct Euclidean distance matrices from city coordinate datasets.

        Parameters
        ----------
        coordinates_list : Sequence[np.ndarray]
            List of 2D coordinate arrays [coords_1, coords_2, ..., coords_M],
            each of shape (n_cities, 2), corresponding to different metric layouts
            (e.g., ground distance, aerial path, mountain terrain).
        **kwargs : Any
            Additional keyword arguments passed to MSTSP constructor.

        Returns
        -------
        MSTSP
            Constructed Euclidean multiobjective TSP instance.
        """
        n_obj = len(coordinates_list)
        if n_obj < 2:
            raise ValueError(f"Must provide at least 2 coordinate sets for MOO, got {n_obj}.")

        first_shape = np.asarray(coordinates_list[0]).shape
        n_cities = first_shape[0]

        matrices = np.zeros((n_obj, n_cities, n_cities), dtype=float)

        for k, coords in enumerate(coordinates_list):
            c_arr = np.asarray(coords, dtype=float)
            if c_arr.shape != (n_cities, 2):
                raise ValueError(
                    f"Coordinate set {k} has shape {c_arr.shape}, expected ({n_cities}, 2)."
                )

            # Pairwise Euclidean distances
            diff = c_arr[:, np.newaxis, :] - c_arr[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dist, 0.0)
            matrices[k] = dist

        return cls(cost_matrices=matrices, **kwargs)

    def decode_tour(self, x: np.ndarray) -> tuple[list[int], bool]:
        """Decode binary solution matrix/vector into an ordered list of visited city indices.

        Parameters
        ----------
        x : np.ndarray
            Binary vector of shape (n_cities * n_cities,) or matrix of shape (n_cities, n_cities).

        Returns
        -------
        tuple[list[int], bool]
            - tour: list of city indices visited at each position 0..N-1.
            - is_valid: True if exactly one unique city is visited at each step.
        """
        mat = np.asarray(x, dtype=bool).reshape((self.n_cities, self.n_cities))

        tour = []
        is_valid = True

        for p in range(self.n_cities):
            cities_at_pos = np.where(mat[p, :])[0]
            if len(cities_at_pos) == 1:
                tour.append(int(cities_at_pos[0]))
            elif len(cities_at_pos) > 1:
                tour.append(int(cities_at_pos[0]))
                is_valid = False
            else:
                tour.append(-1)
                is_valid = False

        # Tour is only fully valid if no unvisited cities and no duplicates
        if len(set(tour)) != self.n_cities or -1 in tour:
            is_valid = False

        return tour, is_valid

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Evaluate candidate binary assignment matrices against total tour costs and constraints.

        Parameters
        ----------
        x : np.ndarray
            Binary decision matrix of shape (N, n_cities * n_cities) or (N, n_cities, n_cities).
        out : dict[str, Any]
            Dictionary storing:
            - out["F"]: (N, n_obj) total route costs to minimize.
            - out["G"]: (N, 2 * n_cities + 1) inequality constraint violations.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        if x.ndim == 3:
            x_3d = np.asarray(x, dtype=float)
        elif x.ndim == 2 and x.shape[1] == self.n_var:
            x_3d = np.asarray(x, dtype=float).reshape((-1, self.n_cities, self.n_cities))
        elif x.ndim == 2 and x.shape == (self.n_cities, self.n_cities):
            x_3d = np.asarray(x, dtype=float)[np.newaxis, :, :]
        elif x.ndim == 1 and len(x) == self.n_var:
            x_3d = np.asarray(x, dtype=float).reshape((1, self.n_cities, self.n_cities))
        else:
            raise ValueError(f"Unexpected input shape for x: {x.shape}. Expected (N, {self.n_var}).")

        n_pop = x_3d.shape[0]
        n_c = self.n_cities
        n_obj = self.n_obj

        # 1. Evaluate tour cost for each individual and objective
        # Total cost = sum_{p=0}^{N-1} X[p] @ C @ X[(p+1)%N]^T
        f_matrix = np.zeros((n_pop, n_obj), dtype=float)

        for p in range(n_c):
            next_p = (p + 1) % n_c
            xp = x_3d[:, p, :]       # Shape: (N, n_cities)
            x_next = x_3d[:, next_p, :]  # Shape: (N, n_cities)

            for k in range(n_obj):
                # cost_transition for each individual: (xp @ C_k) * x_next summed over cities
                # xp @ C_k has shape (N, n_cities)
                cost_trans = np.sum((xp @ self.cost_matrices[k]) * x_next, axis=1)
                f_matrix[:, k] += cost_trans

        # 2. Evaluate Constraints:
        # Constraint 1: sum_i X[p, i] - 1 <= 0 for each position p (n_cities constraints)
        g_pos = np.sum(x_3d, axis=2) - 1.0  # Shape: (N, n_cities)

        # Constraint 2: sum_p X[p, i] - 1 <= 0 for each city i (n_cities constraints)
        g_city = np.sum(x_3d, axis=1) - 1.0  # Shape: (N, n_cities)

        # Constraint 3: N - sum_{p, i} X[p, i] <= 0 (1 deficit constraint)
        g_total = (float(n_c) - np.sum(x_3d, axis=(1, 2)))[:, np.newaxis]  # Shape: (N, 1)

        out["F"] = f_matrix
        out["G"] = np.column_stack([g_pos, g_city, g_total])


MOTSP = MSTSP
