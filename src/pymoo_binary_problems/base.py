"""Base problem definitions for binary optimization in pymoo."""

from typing import Any
import numpy as np
from pymoo.core.problem import Problem


class BinaryProblem(Problem):
    """Abstract base class for all binary multiobjective optimization problems.

    Inherits from pymoo.core.problem.Problem and pre-configures binary search space
    properties:
    - type_var = np.bool_ (binary/boolean variable space)
    - xl = 0 (lower bound)
    - xu = 1 (upper bound)
    - n_var = number of binary decision variables (bits)
    - n_obj = number of conflicting objective functions
    - n_ieq_constr = number of inequality constraints (g(x) <= 0)
    - n_eq_constr = number of equality constraints (h(x) == 0)

    All binary problems must implement:
        _evaluate(self, x, out, *args, **kwargs)
    where:
    - x is a 2D numpy array of shape (N, n_var) with dtype bool or float.
    - out is a dictionary where out["F"] is assigned the objective matrix of shape (N, n_obj),
      and out["G"] (optional) is assigned the inequality constraint matrix of shape (N, n_ieq_constr).
    """

    def __init__(
        self,
        n_var: int,
        n_obj: int = 2,
        n_ieq_constr: int = 0,
        n_eq_constr: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the BinaryProblem.

        Parameters
        ----------
        n_var : int
            Number of binary decision variables.
        n_obj : int, default=2
            Number of objective functions to optimize.
        n_ieq_constr : int, default=0
            Number of inequality constraints (g(x) <= 0).
        n_eq_constr : int, default=0
            Number of equality constraints (h(x) == 0).
        **kwargs : Any
            Additional keyword arguments passed to pymoo.core.problem.Problem.
        """
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            n_eq_constr=n_eq_constr,
            xl=0,
            xu=1,
            vtype=np.bool_,
            type_var=np.bool_,
            **kwargs,
        )
        self.type_var: type = np.bool_

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Evaluate the binary solutions on objectives and constraints.

        Parameters
        ----------
        x : np.ndarray
            Binary decision matrix of shape (N, n_var).
        out : dict[str, Any]
            Dictionary storing evaluation outputs:
            - out["F"]: objective matrix (N, n_obj) to minimize.
            - out["G"]: inequality constraints (N, n_ieq_constr) with g(x) <= 0.
            - out["H"]: equality constraints (N, n_eq_constr) with h(x) == 0.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        raise NotImplementedError("Subclasses of BinaryProblem must implement _evaluate().")
