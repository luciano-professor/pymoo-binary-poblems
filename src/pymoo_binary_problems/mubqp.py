"""Multiobjective Unconstrained Binary Quadratic Problem (MUBQP) benchmark."""

from collections.abc import Sequence
from typing import Any, Optional, Union
import numpy as np

from .base import BinaryProblem


class MUBQP(BinaryProblem):
    """Multiobjective Unconstrained Binary Quadratic Problem (MUBQP).

    The MUBQP is a classic NP-hard combinatorial benchmark in multiobjective
    optimization. Given n binary decision variables x in {0, 1}^n and M quadratic
    interaction matrices Q^(1), ..., Q^(M) in R^(n x n), each objective is defined as:

        f_k(x) = x^T * Q^(k) * x = sum_{i=1}^n sum_{j=1}^n q_{ij}^(k) * x_i * x_j

    Since pymoo minimizes all objective functions by default:
    - If `maximize=True` (default in UBQP literature), the objectives are minimized
       as -f_k(x).
    - If `maximize=False`, the objectives are minimized as +f_k(x).
    """

    def __init__(
        self,
        Q: Union[np.ndarray, Sequence[np.ndarray]],
        maximize: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Multiobjective Unconstrained Binary Quadratic Problem.

        Parameters
        ----------
        Q : Union[np.ndarray, Sequence[np.ndarray]]
            Quadratic interaction matrices. Can be supplied as:
            - A 3D numpy array of shape (n_obj, n_var, n_var).
            - A sequence / list of 2D numpy arrays [Q_1, Q_2, ..., Q_M],
              each of shape (n_var, n_var).
        maximize : bool, default=True
            If True, objectives are converted to minimization by negating profits
            (f_k = -x^T Q_k x), matching standard multiobjective UBQP benchmark literature.
            If False, values are minimized directly (f_k = +x^T Q_k x).
        **kwargs : Any
            Additional keyword arguments passed to BinaryProblem / Problem base class.
        """
        q_arr = np.asarray(Q, dtype=float)

        if q_arr.ndim != 3:
            raise ValueError(
                f"The Q interaction matrices must form a 3D array of shape (n_obj, n_var, n_var), "
                f"got array of shape {q_arr.shape}."
            )

        n_obj, n_var_1, n_var_2 = q_arr.shape

        if n_var_1 != n_var_2:
            raise ValueError(
                f"Each Q matrix must be square (n_var x n_var), but received dimensions ({n_var_1}, {n_var_2})."
            )

        if n_obj < 2:
            raise ValueError(f"Multiobjective optimization requires n_obj >= 2, but received n_obj={n_obj}.")

        self.Q: np.ndarray = q_arr
        self.maximize: bool = maximize

        super().__init__(
            n_var=n_var_1,
            n_obj=n_obj,
            n_ieq_constr=0,
            n_eq_constr=0,
            **kwargs,
        )

    @classmethod
    def from_random(
        cls,
        n_var: int = 100,
        n_obj: int = 2,
        density: float = 1.0,
        val_range: tuple[float, float] = (-100.0, 100.0),
        symmetric: bool = True,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "MUBQP":
        """Factory method to construct a randomized MUBQP benchmark instance.

        Parameters
        ----------
        n_var : int, default=100
            Number of binary decision variables (bits).
        n_obj : int, default=2
            Number of conflicting quadratic objectives.
        density : float, default=1.0
            Sparsity / density of non-zero entries in Q matrices (between 0.0 and 1.0).
            density=1.0 generates dense matrices; density=0.1 generates sparse matrices.
        val_range : tuple[float, float], default=(-100.0, 100.0)
            Range (min_val, max_val) from which quadratic interaction weights are sampled.
        symmetric : bool, default=True
            If True, enforces symmetry in each Q matrix: Q = 0.5 * (Q + Q.T).
        seed : Optional[int], default=None
            Random seed for reproducibility.
        **kwargs : Any
            Additional arguments passed to MUBQP constructor.

        Returns
        -------
        MUBQP
            Instantiated MUBQP problem instance.
        """
        rng = np.random.default_rng(seed)
        Q = np.zeros((n_obj, n_var, n_var), dtype=float)

        for k in range(n_obj):
            mat = rng.uniform(val_range[0], val_range[1], size=(n_var, n_var))

            if density < 1.0:
                mask = rng.random(size=(n_var, n_var)) < density
                mat = mat * mask

            if symmetric:
                mat = 0.5 * (mat + mat.T)

            Q[k] = mat

        return cls(Q=Q, **kwargs)

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Evaluate candidate binary solution vectors on all quadratic objectives.

        Parameters
        ----------
        x : np.ndarray
            Binary decision matrix of shape (N, n_var) or 1D array of shape (n_var,).
        out : dict[str, Any]
            Dictionary storing evaluation outputs, where out["F"] is assigned the
            (N, n_obj) matrix of objective values to be minimized.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        if x.ndim == 1:
            x_2d = x[np.newaxis, :]
        elif x.ndim == 2:
            x_2d = x
        else:
            raise ValueError(f"Decision vector x must be 1D or 2D, got shape {x.shape}.")

        n_pop, n_vars = x_2d.shape
        if n_vars != self.n_var:
            raise ValueError(f"Expected decision vector of length {self.n_var}, but got {n_vars}.")

        # x_float shape: (N, n_var)
        x_float = x_2d.astype(float)

        # Vectorized evaluation across all population members and all objectives
        # f_k(x) = sum_j (sum_i x_i * Q_k_ij) * x_j = sum_j (x @ Q_k)_j * x_j
        f_matrix = np.zeros((n_pop, self.n_obj), dtype=float)

        for k in range(self.n_obj):
            # x_float @ self.Q[k] shape: (N, n_var)
            # Element-wise product with x_float, then sum along variables axis
            q_k = self.Q[k]
            x_q = np.dot(x_float, q_k)  # (N, n_var)
            quad_vals = np.sum(x_q * x_float, axis=1)  # (N,)

            if self.maximize:
                f_matrix[:, k] = -quad_vals
            else:
                f_matrix[:, k] = quad_vals

        out["F"] = f_matrix
