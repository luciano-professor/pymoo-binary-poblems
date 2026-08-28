"""Classic Multiple Knapsack Problem (MKP) benchmark implementation."""

from collections.abc import Callable
from typing import Any, Optional
import numpy as np

from .base import BinaryProblem


class MKP(BinaryProblem):
    """Classic Multiple Knapsack Problem (MKP) for binary optimization in pymoo.

    Models the multiobjective multiple knapsack problem where N items with distinct
    profits and weights are allocated across M knapsacks, each with an individual capacity limit.
    The binary decision variable x[i, j, k] indicates whether item j is allocated to knapsack k.

    Formal inequality constraints implemented in out["G"] (g(x) <= 0):
    1. Knapsack capacity limits (M constraints):
       total_weight[k] - capacities[k] <= 0
    2. Single knapsack assignment per item (N constraints):
       sum_k x[j, k] - 1 <= 0

    Total inequality constraints: n_ieq_constr = n_knapsacks + n_items.
    """

    def __init__(
        self,
        profits: np.ndarray,
        weights: np.ndarray,
        capacities: np.ndarray,
        n_obj: int = 2,
        maximize_profit: bool = True,
        minimize_weight: bool = True,
        normalize_profit: bool = False,
        normalize_weight: bool = False,
        extra_objectives: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Multiple Knapsack Problem.

        Parameters
        ----------
        profits : np.ndarray
            1D array of profits associated with each item, shape (n_items,).
        weights : np.ndarray
            1D array of weights associated with each item, shape (n_items,).
        capacities : np.ndarray
            1D array of capacity limits for each knapsack, shape (n_knapsacks,).
        n_obj : int, default=2
            Number of objective functions (minimum 2).
        maximize_profit : bool, default=True
            If True, profit is converted to minimization in pymoo by multiplying by -1.
        minimize_weight : bool, default=True
            If True, weight is minimized directly.
        normalize_profit : bool, default=False
            If True, normalizes total profit by the sum of all item profits (max_profit).
        normalize_weight : bool, default=False
            If True, normalizes total weight by the sum of all item weights (max_weight).
        extra_objectives : Optional[Callable[[np.ndarray], np.ndarray]], default=None
            Optional callable to compute additional objectives when n_obj > 2.
            Receives candidate 3D binary solution matrix x of shape (N, n_items, n_knapsacks)
            and must return a 2D array of shape exactly (N, n_obj - 2).
        **kwargs : Any
            Additional keyword arguments passed to BinaryProblem / Problem base class.
        """
        self.profits: np.ndarray = np.asarray(profits, dtype=float)
        self.weights: np.ndarray = np.asarray(weights, dtype=float)
        self.capacities: np.ndarray = np.asarray(capacities, dtype=float)

        if self.profits.ndim != 1 or self.weights.ndim != 1 or self.capacities.ndim != 1:
            raise ValueError("The 'profits', 'weights', and 'capacities' arrays must be 1-dimensional (1D).")

        if len(self.profits) != len(self.weights):
            raise ValueError(
                f"Size mismatch between profits ({len(self.profits)}) and weights ({len(self.weights)})."
            )

        if len(self.capacities) == 0:
            raise ValueError("The 'capacities' array must contain at least one knapsack.")

        if n_obj < 2:
            raise ValueError("The number of objectives 'n_obj' must be at least 2.")

        if n_obj > 2 and extra_objectives is None:
            raise ValueError(
                f"When n_obj > 2 (n_obj={n_obj}), the 'extra_objectives' callable must be provided."
            )

        self.n_items: int = len(self.weights)
        self.n_knapsacks: int = len(self.capacities)
        self.maximize_profit: bool = maximize_profit
        self.minimize_weight: bool = minimize_weight
        self.normalize_profit: bool = normalize_profit
        self.normalize_weight: bool = normalize_weight
        self.extra_objectives: Optional[Callable[[np.ndarray], np.ndarray]] = extra_objectives

        self.max_profit: float = float(np.sum(self.profits))
        self.max_weight: float = float(np.sum(self.weights))

        super().__init__(
            n_var=self.n_items * self.n_knapsacks,
            n_obj=n_obj,
            n_ieq_constr=self.n_knapsacks + self.n_items,
            **kwargs,
        )

    @classmethod
    def from_random(
        cls,
        n_items: int = 50,
        n_knapsacks: int = 5,
        profit_range: tuple[float, float] = (10.0, 100.0),
        weight_range: tuple[float, float] = (5.0, 50.0),
        capacity_ratio: float = 0.5,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "MKP":
        """Generate a randomized Multiple Knapsack Problem instance.

        Parameters
        ----------
        n_items : int, default=50
            Number of items.
        n_knapsacks : int, default=5
            Number of knapsacks.
        profit_range : tuple[float, float], default=(10.0, 100.0)
            Range (min, max) for item profits.
        weight_range : tuple[float, float], default=(5.0, 50.0)
            Range (min, max) for item weights.
        capacity_ratio : float, default=0.5
            Ratio of total items weight assigned across all knapsack capacities.
        seed : Optional[int], default=None
            Random seed for reproducibility.
        **kwargs : Any
            Additional arguments passed to MKP initializer.

        Returns
        -------
        MKP
            Instantiated MKP problem instance.
        """
        rng = np.random.default_rng(seed)
        profits = rng.uniform(profit_range[0], profit_range[1], size=n_items)
        weights = rng.uniform(weight_range[0], weight_range[1], size=n_items)

        total_weight = np.sum(weights)
        total_capacity = total_weight * capacity_ratio
        # Random distribution of total capacity among knapsacks
        raw_caps = rng.uniform(0.8, 1.2, size=n_knapsacks)
        capacities = (raw_caps / np.sum(raw_caps)) * total_capacity

        return cls(profits=profits, weights=weights, capacities=capacities, **kwargs)

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Evaluate candidate solutions for the Multiple Knapsack Problem.

        Parameters
        ----------
        x : np.ndarray
            Matrix of candidate binary solutions of shape (N, n_items, n_knapsacks)
            or (N, n_var) where n_var = n_items * n_knapsacks.
        out : dict[str, Any]
            Output dictionary where:
            - 'F' is populated with the objective matrix (N, n_obj).
            - 'G' is populated with concatenated constraints (N, n_knapsacks + n_items).
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        if x.ndim == 3:
            x_3d: np.ndarray = x
        elif x.ndim == 2 and x.shape[1] == self.n_var:
            x_3d = x.reshape((-1, self.n_items, self.n_knapsacks))
        elif x.ndim == 2 and x.shape == (self.n_items, self.n_knapsacks):
            x_3d = x[np.newaxis, :, :]
        elif x.ndim == 1 and len(x) == self.n_var:
            x_3d = x.reshape((1, self.n_items, self.n_knapsacks))
        else:
            raise ValueError(
                f"Unexpected shape for x: {x.shape}. "
                f"Expected (N, {self.n_items}, {self.n_knapsacks}) or (N, {self.n_var})."
            )

        n_pop: int = x_3d.shape[0]

        # 1. Total profit: sum of profits of allocated items across any knapsack
        is_allocated: np.ndarray = np.clip(np.sum(x_3d, axis=2), 0, 1)
        total_profit: np.ndarray = np.dot(is_allocated, self.profits)

        # 2. Weight per knapsack and total weight across all knapsacks
        knapsack_weights: np.ndarray = np.zeros((n_pop, self.n_knapsacks), dtype=float)
        for k in range(self.n_knapsacks):
            knapsack_weights[:, k] = np.dot(x_3d[:, :, k], self.weights)
        total_weight: np.ndarray = np.sum(knapsack_weights, axis=1)

        # 3. Optional objective normalization
        profit_eval: np.ndarray = (
            total_profit / self.max_profit
            if self.normalize_profit and self.max_profit > 0
            else total_profit
        )
        weight_eval: np.ndarray = (
            total_weight / self.max_weight
            if self.normalize_weight and self.max_weight > 0
            else total_weight
        )

        # 4. Pure objectives (pymoo minimizes all functions)
        obj1: np.ndarray = -profit_eval if self.maximize_profit else profit_eval
        obj2: np.ndarray = weight_eval if self.minimize_weight else -weight_eval

        if self.n_obj == 2:
            out["F"] = np.column_stack([obj1, obj2])
        else:
            assert self.extra_objectives is not None
            extra: np.ndarray = np.asarray(self.extra_objectives(x_3d))
            expected_shape: tuple[int, int] = (n_pop, self.n_obj - 2)
            if extra.shape != expected_shape:
                raise ValueError(
                    f"extra_objectives returned shape {extra.shape}, expected exactly {expected_shape}."
                )
            out["F"] = np.column_stack([obj1, obj2, extra])

        # 5. Formal inequality constraints in out["G"]:
        # G_capacity = knapsack_weights - capacities <= 0 (M constraints)
        g_capacity: np.ndarray = knapsack_weights - self.capacities
        # G_assignment = sum_k x[j, k] - 1 <= 0 (N constraints)
        g_assignment: np.ndarray = np.sum(x_3d, axis=2) - 1.0

        out["G"] = np.column_stack([g_capacity, g_assignment])
