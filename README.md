# pymoo-binary-problems — Benchmark Problem Suite for Binary Multiobjective Optimization in pymoo

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![pymoo](https://img.shields.io/badge/pymoo-%3E%3D0.6.0-orange.svg)](https://pymoo.org/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**`pymoo-binary-problems`** is a standalone, domain-agnostic Python package providing a comprehensive suite of benchmark problems for **Binary Multi-Objective Optimization (BMOO)** built specifically for the [`pymoo`](https://pymoo.org/) framework.

The package is designed to **decouple problem definitions from specific optimization algorithms**, allowing algorithm libraries (such as **`bmopso`**, **`bmopso_cdr`**, **`bpso`**, or standard `pymoo` algorithms like `NSGA2`, `GA`) to install `pymoo-binary-problems` as a dependency and import problems directly.

---

## 📦 Installation

### 1. Direct Installation via `pip` (GitHub)
Install the latest release directly from GitHub:

```bash
pip install git+https://github.com/luciano-professor/pymoo-binary-poblems
```

#### Install a specific branch (e.g., `main`):
```bash
pip install git+https://github.com/luciano-professor/pymoo-binary-poblems@main
```

#### Upgrade to the latest version:
```bash
pip install --upgrade git+https://github.com/luciano-professor/pymoo-binary-poblems
```

---

### 2. Usage in `requirements.txt`
In your project's `requirements.txt` file:

```text
git+https://github.com/luciano-professor/pymoo-binary-poblems
```

---

### 3. Usage as a Dependency in `pyproject.toml` (PEP 508)
To declare `pymoo-binary-problems` as a dependency in algorithms like `bmopso`, `bmopso_cdr`, etc.:

```toml
[project]
dependencies = [
    "pymoo-binary-problems @ git+https://github.com/luciano-professor/pymoo-binary-poblems",
]
```

---

### 4. Local Editable Installation (Development)
To work on the source code locally and have modifications reflected immediately:

```bash
cd C:\dev\custom_packages\binary_moo_problems
pip install -e .
```

---

### 5. Installation via PyPI (when published)
```bash
pip install pymoo-binary-problems
```

---

## 🚀 Quickstart

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from pymoo_binary_problems import MKP, MOFS, MOSCP, MSTSP, MUBQP

# 1. Instantiate any benchmark problem
problem = MKP.from_random(n_items=30, n_knapsacks=3, seed=42)

# 2. Configure algorithm (e.g., NSGA-II, BMOPSO, etc.)
algorithm = NSGA2(
    pop_size=40,
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(prob=0.05),
    eliminate_duplicates=True,
)

# 3. Execute standard pymoo optimization
res = minimize(
    problem,
    algorithm,
    ("n_gen", 50),
    seed=42,
    verbose=True,
)

print(f"Non-dominated Pareto solutions discovered: {len(res.X)}")
```

---

## 📚 Benchmark Problem Suite

All problems inherit from `BinaryProblem` (`pymoo.core.problem.Problem` pre-configured with `type_var=np.bool_`, `xl=0`, `xu=1`).

| Problem | Class / Alias | Variables (`n_var`) | Objectives (`n_obj`) | Constraints (`n_ieq`) | Factory / Utility Methods |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Multiple Knapsack** | `MKP` | `N * M` bits | >= 2 | `M + N` | `MKP.from_random()` |
| **Feature Selection** | `MOFS` / `MOBFS` | `D` features | 2 or 3 | 1 | `MOFS.from_synthetic()`, `decode_features()` |
| **Set Covering** | `MOSCP` / `MSCP` | `n` subsets | >= 2 | `m` elements | `MOSCP.from_random()`, `from_subsets()`, `decode_coverage()` |
| **Traveling Salesman** | `MSTSP` / `MOTSP` | `N^2` bits | >= 2 | `2*N + 1` | `MSTSP.from_random()`, `from_coordinates()`, `decode_tour()` |
| **Unconstrained Quadratic**| `MUBQP` | `n` bits | >= 2 | 0 (Unconstrained) | `MUBQP.from_random()` |

---

### 1. Multiple Knapsack Problem (`MKP`)
Allocates `N` items across `M` knapsacks, each subject to an individual weight capacity limit.
* **Decision Variable**: Binary matrix `X` of shape `(N, M)` flattened to `N * M` bits (`X[j, k] = 1` if item `j` is assigned to knapsack `k`).
* **Objectives**:
  * `f1(x)`: Maximize Total Profit (minimized in pymoo as `-Profit`).
  * `f2(x)`: Minimize Total Weight Loaded.
  * Additional objectives via `extra_objectives` callable.
* **Constraints (`g(x) <= 0`)**: Knapsack capacity limits (`M`) and single knapsack assignment per item (`N`).

```python
from pymoo_binary_problems import MKP

# Direct instantiation from explicit data
problem = MKP(
    profits=[15.0, 25.0, 30.0, 40.0],
    weights=[5.0, 10.0, 12.0, 18.0],
    capacities=[20.0, 25.0],
    n_obj=2,
)

# Or generate a randomized benchmark instance
problem = MKP.from_random(n_items=50, n_knapsacks=5, seed=42)
```

---

### 2. Multiobjective Feature Selection (`MOFS` / `MOBFS`)
Optimal feature subset selection for machine learning classifiers.
* **Decision Variable**: Binary mask `x` of length `D` where `x[j] = 1` indicates feature `j` is selected.
* **Objectives**:
  * `f1(x)`: Classification error rate (`1.0 - Accuracy`) computed via Cross-Validation.
  * `f2(x)`: Feature selection ratio (model complexity, `||x||_1 / D`).
  * `f3(x)` *(optional)*: Total feature acquisition/measurement cost (when `feature_costs` is provided).
* **Constraints**: Minimum feature requirement (`sum(x) >= min_features`).

```python
from pymoo_binary_problems import MOFS

# Generate from synthetic classification dataset
problem = MOFS.from_synthetic(
    n_samples=200,
    n_features=30,
    n_informative=8,
    min_features=1,
    seed=42,
)

# Decode selected features from solution
info = problem.decode_features(res.X[0])
print(f"Selected features ({info['n_selected']}): {info['selected_indices']}")
```

---

### 3. Multiobjective Set Covering Problem (`MOSCP` / `MSCP`)
Covers a universe of `m` elements at minimal cost by selecting candidate subsets from `n` available groups.
* **Decision Variable**: Binary vector `x` of length `n`.
* **Objectives**: Conflicting cost criteria `f_k(x) = C_k * x`.
* **Constraints**: Every universe element must be covered by at least one selected subset (`1 - A*x <= 0`).

```python
from pymoo_binary_problems import MOSCP

# Generate guaranteed feasible benchmark instance
problem = MOSCP.from_random(
    n_elements=50,
    n_subsets=100,
    n_obj=2,
    density=0.15,
    seed=42,
)

# Decode coverage status
info = problem.decode_coverage(res.X[0])
print("Is feasible:", info["is_feasible"])
print("Selected subsets:", info["selected_subsets"])
```

---

### 4. Multiobjective Traveling Salesman Problem (`MSTSP` / `MOTSP`)
Binary Assignment Matrix formulation (`N * N` bits) for visiting `N` cities in a cyclic tour.
* **Decision Variable**: `X[p, i] = 1` if city `i` is visited at position/step `p` of the tour.
* **Objectives**: Total route cost across conflicting distance, travel time, or toll matrices.
* **Constraints**: One city per position (`N`), one visit per city (`N`), and tour completeness (`1`).

```python
from pymoo_binary_problems import MSTSP

# Instantiate from pairwise city coordinates
problem = MSTSP.from_coordinates([coords_distance, coords_time])

# Decode cyclic tour
tour, is_valid = problem.decode_tour(res.X[0])
print(f"Tour order: {tour} (Valid: {is_valid})")
```

---

### 5. Multiobjective Unconstrained Binary Quadratic (`MUBQP`)
Evaluates quadratic interaction matrices `Q_k` of shape `(n, n)`.
* **Decision Variable**: Binary vector `x` of length `n`.
* **Objectives**: `f_k(x) = x^T * Q_k * x`.
* **Constraints**: Unconstrained (`n_ieq = 0`).

```python
from pymoo_binary_problems import MUBQP

problem = MUBQP.from_random(
    n_var=100,
    n_obj=2,
    density=0.8,
    maximize=True,
    seed=42,
)
```

---

## 🛠️ Creating Custom Binary Problems

To implement your own binary problem, inherit from `BinaryProblem`:

```python
from typing import Any
import numpy as np
from pymoo_binary_problems import BinaryProblem

class MyBinaryProblem(BinaryProblem):
    def __init__(self, n_bits: int = 20):
        super().__init__(
            n_var=n_bits,
            n_obj=2,
            n_ieq_constr=0,
        )

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # x is a 2D array of shape (N, n_var)
        f1 = np.sum(x, axis=1)          # Minimize active bits (count of 1s)
        f2 = self.n_var - f1            # Minimize inactive bits (count of 0s)
        out["F"] = np.column_stack([f1, f2])
```

---

## 🧪 Automated Testing

Run the automated test suite with `pytest`:

```bash
pytest -v
```

---

## 📄 License

This project is distributed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.
