"""Multiobjective Feature Selection (MOFS) benchmark."""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Union
import numpy as np

from .base import BinaryProblem


class MOFS(BinaryProblem):
    """Multiobjective Feature Selection (MOFS / MOBFS) benchmark for machine learning.

    Formulates feature selection as a combinatorial multiobjective optimization problem.
    Given a dataset with S samples and D features (X_data in R^(S x D), y in Z^S):
    - Decision vector x in {0, 1}^D (n_var = D features):
        x_j = 1 if feature j is selected
        x_j = 0 if feature j is discarded

    Objectives (Minimization):
    1. Objective 1 (f1): Classification Error Rate:
       f1(x) = 1.0 - Accuracy(X_data[:, x == 1], y)
       If no features are selected (x = 0), f1(x) = 1.0 (maximal error penalty).
    2. Objective 2 (f2): Feature Ratio / Model Complexity:
       f2(x) = sum_{j=1}^D x_j / D = ||x||_1 / D
    3. Optional Objective 3 (f3): Feature Measurement / Acquisition Cost (if feature_costs is provided):
       f3(x) = sum_{j=1}^D cost_j * x_j

    Constraints in out["G"] (g(x) <= 0):
    At least `min_features` features must be selected:
       g(x) = min_features - sum_{j=1}^D x_j <= 0
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator: Optional[Any] = None,
        cv: Union[int, float] = 3,
        min_features: int = 1,
        feature_costs: Optional[Union[np.ndarray, Sequence[float]]] = None,
        scoring: str = "accuracy",
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Multiobjective Feature Selection problem.

        Parameters
        ----------
        X : np.ndarray
            Dataset feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Ground truth labels of shape (n_samples,).
        estimator : Optional[Any], default=None
            Supervised learning estimator implementing `fit` and `predict` / `score`
            (e.g., sklearn KNeighborsClassifier, DecisionTreeClassifier), or None
            to use standard 5-NN classifier.
        cv : Union[int, float], default=3
            Cross-validation strategy. If int >= 2, uses K-Fold cross validation.
            If float in (0, 1), uses a single train/test split with test_size=cv.
        min_features : int, default=1
            Minimum number of features required to form a valid subset.
        feature_costs : Optional[Union[np.ndarray, Sequence[float]]], default=None
            Optional 1D array of acquisition/measurement costs per feature.
            If provided and n_obj > 2, total feature cost is evaluated as objective 3.
        scoring : str, default="accuracy"
            Metric name ("accuracy" or "balanced_accuracy").
        seed : Optional[int], default=None
            Random seed for CV splitting.
        **kwargs : Any
            Additional keyword arguments passed to the BinaryProblem / Problem base class.
        """
        x_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        if x_arr.ndim != 2:
            raise ValueError(f"Dataset 'X' must be a 2D array of shape (n_samples, n_features), got {x_arr.shape}.")

        if y_arr.ndim != 1:
            raise ValueError(f"Target 'y' must be a 1D array of shape (n_samples,), got {y_arr.shape}.")

        n_samples, n_features = x_arr.shape
        if len(y_arr) != n_samples:
            raise ValueError(f"Length mismatch between X ({n_samples}) and y ({len(y_arr)}).")

        if n_features < 2:
            raise ValueError(f"Dataset must contain at least 2 features, got {n_features}.")

        if min_features < 1 or min_features > n_features:
            raise ValueError(
                f"'min_features' must be between 1 and {n_features}, got {min_features}."
            )

        self.dataset_X: np.ndarray = x_arr
        self.dataset_y: np.ndarray = y_arr
        self.n_samples: int = n_samples
        self.n_features: int = n_features
        self.min_features: int = min_features
        self.cv: Union[int, float] = cv
        self.scoring: str = scoring
        self.seed: Optional[int] = seed

        # Optional feature costs
        if feature_costs is not None:
            c_arr = np.asarray(feature_costs, dtype=float)
            if c_arr.ndim != 1 or len(c_arr) != n_features:
                raise ValueError(
                    f"'feature_costs' must be a 1D array of length {n_features}, got {c_arr.shape}."
                )
            self.feature_costs: Optional[np.ndarray] = c_arr
            n_obj = 3
        else:
            self.feature_costs = None
            n_obj = 2

        # Estimator setup (default k-NN if None)
        if estimator is None:
            from sklearn.neighbors import KNeighborsClassifier
            self.estimator: Any = KNeighborsClassifier(n_neighbors=min(5, max(1, n_samples // 4)))
        else:
            self.estimator = estimator

        # Precompute train / validation splits
        self._splits: list[tuple[np.ndarray, np.ndarray]] = self._create_splits(n_samples, cv, seed)

        super().__init__(
            n_var=self.n_features,
            n_obj=n_obj,
            n_ieq_constr=1,
            n_eq_constr=0,
            **kwargs,
        )

    def _create_splits(
        self,
        n_samples: int,
        cv: Union[int, float],
        seed: Optional[int],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Precompute train/validation index splits for consistent evaluation.

        Parameters
        ----------
        n_samples : int
            Number of dataset samples.
        cv : Union[int, float]
            Fold count (int) or test ratio (float).
        seed : Optional[int]
            Random seed.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (train_indices, val_indices) pairs.
        """
        rng = np.random.default_rng(seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        if isinstance(cv, int) and cv >= 2:
            folds = np.array_split(indices, cv)
            splits = []
            for i in range(cv):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(cv) if j != i])
                splits.append((train_idx, val_idx))
            return splits
        elif isinstance(cv, float) and 0.0 < cv < 1.0:
            n_val = max(1, int(n_samples * cv))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            return [(train_idx, val_idx)]
        else:
            raise ValueError(f"Invalid cv parameter: {cv}. Expected int >= 2 or float in (0.0, 1.0).")

    @classmethod
    def from_synthetic(
        cls,
        n_samples: int = 200,
        n_features: int = 30,
        n_informative: int = 8,
        n_redundant: int = 4,
        n_classes: int = 2,
        seed: Optional[int] = 42,
        **kwargs: Any,
    ) -> "MOFS":
        """Factory method to construct a MOFS problem using synthetic classification data.

        Parameters
        ----------
        n_samples : int, default=200
            Number of observations to generate.
        n_features : int, default=30
            Total number of features (n_var).
        n_informative : int, default=8
            Number of informative features directly correlated with target class.
        n_redundant : int, default=4
            Number of linear combinations of informative features.
        n_classes : int, default=2
            Number of target classes.
        seed : Optional[int], default=42
            Random seed for dataset generation.
        **kwargs : Any
            Additional keyword arguments passed to MOFS constructor.

        Returns
        -------
        MOFS
            Initialized MOFS problem instance.
        """
        from sklearn.datasets import make_classification

        X_synth, y_synth = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            random_state=seed,
            shuffle=False,
        )
        return cls(X=X_synth, y=y_synth, seed=seed, **kwargs)

    def _eval_single_mask(self, active_indices: np.ndarray) -> float:
        """Fit estimator and calculate mean validation accuracy over precomputed splits.

        Parameters
        ----------
        active_indices : np.ndarray
            1D array of active feature column indices.

        Returns
        -------
        float
            Mean accuracy across CV splits.
        """
        from sklearn.base import clone

        accs = []
        for train_idx, val_idx in self._splits:
            X_train = self.dataset_X[train_idx][:, active_indices]
            y_train = self.dataset_y[train_idx]
            X_val = self.dataset_X[val_idx][:, active_indices]
            y_val = self.dataset_y[val_idx]

            est = clone(self.estimator)
            est.fit(X_train, y_train)
            preds = est.predict(X_val)

            if self.scoring == "accuracy":
                acc = np.mean(preds == y_val)
            else:
                from sklearn.metrics import balanced_accuracy_score
                acc = balanced_accuracy_score(y_val, preds)

            accs.append(acc)

        return float(np.mean(accs))

    def decode_features(self, x: np.ndarray) -> dict[str, Any]:
        """Decode binary solution vector into selected feature indices and summary metrics.

        Parameters
        ----------
        x : np.ndarray
            Binary mask vector of shape (n_features,).

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "selected_indices": list of selected feature indices.
            - "n_selected": number of selected features.
            - "ratio": feature selection ratio.
            - "is_valid": True if n_selected >= min_features.
        """
        mask = np.asarray(x, dtype=bool).ravel()
        selected = np.where(mask)[0].tolist()
        n_sel = len(selected)
        return {
            "selected_indices": selected,
            "n_selected": n_sel,
            "ratio": n_sel / self.n_features,
            "is_valid": n_sel >= self.min_features,
        }

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Evaluate candidate binary feature masks.

        Parameters
        ----------
        x : np.ndarray
            Binary decision matrix of shape (N, n_features).
        out : dict[str, Any]
            Evaluation dictionary storing:
            - out["F"]: (N, 2) or (N, 3) matrix of objectives (error rate, ratio, cost).
            - out["G"]: (N, 1) constraint violation matrix (min_features - n_active <= 0).
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        x_bool = np.asarray(x, dtype=bool)
        if x_bool.ndim == 1:
            x_bool = x_bool[np.newaxis, :]

        n_pop = x_bool.shape[0]
        n_features = self.n_features

        f_matrix = np.zeros((n_pop, self.n_obj), dtype=float)
        g_matrix = np.zeros((n_pop, 1), dtype=float)

        for i in range(n_pop):
            active_idx = np.where(x_bool[i])[0]
            n_active = len(active_idx)

            # Constraint: min_features - n_active <= 0
            g_matrix[i, 0] = self.min_features - n_active

            # Objective 1: Classification error rate (1.0 - Accuracy)
            if n_active == 0:
                error_rate = 1.0
            else:
                try:
                    mean_acc = self._eval_single_mask(active_idx)
                    error_rate = 1.0 - mean_acc
                except Exception:
                    error_rate = 1.0

            # Objective 2: Feature selection ratio (Model complexity)
            ratio = n_active / n_features

            f_matrix[i, 0] = error_rate
            f_matrix[i, 1] = ratio

            # Objective 3: Feature cost
            if self.feature_costs is not None and self.n_obj >= 3:
                f_matrix[i, 2] = np.sum(self.feature_costs[active_idx])

        out["F"] = f_matrix
        out["G"] = g_matrix


MOBFS = MOFS
