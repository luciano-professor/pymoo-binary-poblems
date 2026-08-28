"""Multiobjective Set Covering Problem (MOSCP) benchmark."""

from collections.abc import Sequence
from typing import Any, Optional, Union
import numpy as np

from .base import BinaryProblem


class MOSCP(BinaryProblem):
    """Multiobjective Set Covering Problem (MOSCP / MSCP) in binary representation.

    The Set Covering Problem seeks to cover a universe of m elements by selecting
    a subset of n available candidate sets at minimal cost across K conflicting criteria.

    Parameters & Variables:
    - Binary decision vector x in {0, 1}^n (n_var = n subsets), where:
        x_j = 1 if candidate subset j is selected
        x_j = 0 otherwise
    - Binary incidence matrix A in {0, 1}^(m x n), where a_ij = 1 if subset j covers element i.
    - Cost matrix C in R^(K x n), where C_kj is the cost of choosing subset j under criterion k.

    Objectives (k = 1, ..., K):
    Minimize total selection cost under each objective:
        f_k(x) = sum_{j=1}^n C_kj * x_j

    Constraints in out["G"] (g(x) <= 0):
    Every element i in {1, ..., m} must be covered by at least one selected subset:
        g_i(x) = 1 - sum_{j=1}^n a_ij * x_j <= 0

    Total inequality constraints: n_ieq_constr = m (number of elements to be covered).
    """

    def __init__(
        self,
        incidence_matrix: Union[np.ndarray, Sequence[Sequence[int]]],
        costs: Union[np.ndarray, Sequence[Sequence[float]]],
        **kwargs: Any,
    ) -> None:
        """Initialize the Multiobjective Set Covering Problem.

        Parameters
        ----------
        incidence_matrix : Union[np.ndarray, Sequence[Sequence[int]]]
            Binary matrix A of shape (n_elements, n_subsets) where A[i, j] = 1
            indicates that candidate subset j covers ground element i.
        costs : Union[np.ndarray, Sequence[Sequence[float]]]
            Cost matrix C of shape (n_obj, n_subsets) or (n_subsets,) if n_obj == 1.
            Each row k represents the selection costs under objective k.
        **kwargs : Any
            Additional keyword arguments passed to the BinaryProblem / Problem base class.
        """
        a_arr = np.asarray(incidence_matrix, dtype=int)
        c_arr = np.asarray(costs, dtype=float)

        if a_arr.ndim != 2:
            raise ValueError(f"Incidence matrix must be 2D of shape (n_elements, n_subsets), got {a_arr.shape}.")

        n_elements, n_subsets = a_arr.shape

        if not np.all(np.isin(a_arr, [0, 1])):
            raise ValueError("Incidence matrix entries must be binary (0 or 1).")

        # Verify that every element is coverable by at least one subset
        coverage_potential = np.sum(a_arr, axis=1)
        if np.any(coverage_potential == 0):
            uncoverable = np.where(coverage_potential == 0)[0]
            raise ValueError(
                f"Elements at indices {uncoverable.tolist()} cannot be covered by any subset (unfeasible problem instance)."
            )

        if c_arr.ndim == 1:
            c_arr = c_arr[np.newaxis, :]

        if c_arr.ndim != 2:
            raise ValueError(f"Cost matrix must be 2D of shape (n_obj, n_subsets), got {c_arr.shape}.")

        n_obj, n_cost_subsets = c_arr.shape
        if n_cost_subsets != n_subsets:
            raise ValueError(
                f"Dimension mismatch: incidence matrix has {n_subsets} subsets, "
                f"but cost matrix has {n_cost_subsets} columns."
            )

        if n_obj < 2:
            raise ValueError(f"Multiobjective optimization requires n_obj >= 2, got {n_obj}.")

        self.incidence_matrix: np.ndarray = a_arr
        self.costs: np.ndarray = c_arr
        self.n_elements: int = n_elements
        self.n_subsets: int = n_subsets

        super().__init__(
            n_var=n_subsets,
            n_obj=n_obj,
            n_ieq_constr=n_elements,
            **kwargs,
        )

    @classmethod
    def from_random(
        cls,
        n_elements: int = 50,
        n_subsets: int = 100,
        n_obj: int = 2,
        density: float = 0.15,
        cost_range: tuple[float, float] = (1.0, 100.0),
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "MOSCP":
        """Factory method to generate a randomized, feasible MOSCP benchmark instance.

        Parameters
        ----------
        n_elements : int, default=50
            Number of universe elements to be covered (m).
        n_subsets : int, default=100
            Number of candidate subsets available (n_var = n).
        n_obj : int, default=2
            Number of conflicting cost objectives (K >= 2).
        density : float, default=0.15
            Probability that subset j covers element i.
        cost_range : tuple[float, float], default=(1.0, 100.0)
            Tuple of (min_cost, max_cost) for random cost sampling.
        seed : Optional[int], default=None
            Random seed for reproducibility.
        **kwargs : Any
            Additional arguments passed to the MOSCP initializer.

        Returns
        -------
        MOSCP
            A guaranteed feasible MOSCP instance.
        """
        rng = np.random.default_rng(seed)

        A = rng.binomial(n=1, p=density, size=(n_elements, n_subsets))

        # Guarantee feasibility: ensure every element is covered by at least one subset
        row_sums = np.sum(A, axis=1)
        for i in range(n_elements):
            if row_sums[i] == 0:
                random_subset = rng.integers(0, n_subsets)
                A[i, random_subset] = 1

        C = rng.uniform(cost_range[0], cost_range[1], size=(n_obj, n_subsets))

        return cls(incidence_matrix=A, costs=C, **kwargs)

    @classmethod
    def from_subsets(
        cls,
        universe_size: int,
        subsets: Sequence[Sequence[int]],
        costs: Union[np.ndarray, Sequence[Sequence[float]]],
        **kwargs: Any,
    ) -> "MOSCP":
        """Construct a MOSCP instance from explicit subsets of element indices.

        Parameters
        ----------
        universe_size : int
            Total number of elements in universe U = {0, 1, ..., universe_size - 1}.
        subsets : Sequence[Sequence[int]]
            List of subsets, where each subset is a sequence of integer element indices.
        costs : Union[np.ndarray, Sequence[Sequence[float]]]
            Cost matrix of shape (n_obj, len(subsets)).
        **kwargs : Any
            Additional keyword arguments passed to MOSCP constructor.

        Returns
        -------
        MOSCP
            Configured MOSCP problem.
        """
        n_subsets = len(subsets)
        A = np.zeros((universe_size, n_subsets), dtype=int)

        for j, s in enumerate(subsets):
            for elem in s:
                if 0 <= elem < universe_size:
                    A[elem, j] = 1
                else:
                    raise ValueError(f"Element index {elem} out of bounds for universe size {universe_size}.")

        return cls(incidence_matrix=A, costs=costs, **kwargs)

    def decode_coverage(self, x: np.ndarray) -> dict[str, Any]:
        """Decode a binary decision vector to inspect coverage status and chosen subsets.

        Parameters
        ----------
        x : np.ndarray
            Binary mask vector of shape (n_subsets,).

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "selected_subsets": list of selected subset indices.
            - "n_selected": total number of selected subsets.
            - "coverage_counts": 1D array of length n_elements with times each element is covered.
            - "uncovered_elements": list of elements that fail to be covered.
            - "is_feasible": True if all elements are covered (g_i <= 0 for all i).
            - "total_costs": 1D array of costs across all objectives.
        """
        x_bool = np.asarray(x, dtype=bool).ravel()
        selected = np.where(x_bool)[0].tolist()
        cov = np.dot(self.incidence_matrix, x_bool.astype(float))
        uncovered = np.where(cov < 1.0)[0].tolist()
        costs = np.dot(self.costs, x_bool.astype(float))

        return {
            "selected_subsets": selected,
            "n_selected": len(selected),
            "coverage_counts": cov,
            "uncovered_elements": uncovered,
            "is_feasible": len(uncovered) == 0,
            "total_costs": costs,
        }

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Evaluate candidate binary subset selections against costs and coverage constraints.

        Parameters
        ----------
        x : np.ndarray
            Binary decision matrix of shape (N, n_subsets).
        out : dict[str, Any]
            Dictionary storing:
            - out["F"]: (N, n_obj) matrix of costs to minimize.
            - out["G"]: (N, n_elements) inequality constraint matrix (1 - coverage <= 0).
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        x_bool = np.asarray(x, dtype=bool)
        if x_bool.ndim == 1:
            x_bool = x_bool[np.newaxis, :]

        # x_float shape: (N, n_subsets)
        x_float = x_bool.astype(float)

        # 1. Evaluate Objectives: F = x * C^T -> shape (N, n_obj)
        out["F"] = np.dot(x_float, self.costs.T)

        # 2. Evaluate Constraints: coverage = x * A^T -> shape (N, n_elements)
        # Constraint: g_i(x) = 1.0 - coverage_i <= 0
        coverage = np.dot(x_float, self.incidence_matrix.T)
        out["G"] = 1.0 - coverage


MSCP = MOSCP
