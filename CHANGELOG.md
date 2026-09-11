# Changelog

All notable changes to the `pymoo-binary-problems` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.1] - 2026-09-11

### Changed
- Relicensed project from MIT License to Apache License 2.0 (`Apache-2.0`).
- Updated license identifiers and metadata in `LICENSE`, `pyproject.toml`, and `README.md`.

---

## [1.0.0] - 2026-08-28

### Added
- Initial release of the standalone `pymoo-binary-problems` package for `pymoo`.
- Hosted on GitHub: `https://github.com/luciano-professor/pymoo-binary-poblems`.
- **`BinaryProblem`**: Base problem class inheriting from `pymoo.core.problem.Problem` pre-configured for binary search spaces (`type_var=np.bool_`, `xl=0`, `xu=1`).
- **`MKP`**: Multiple Knapsack Problem benchmark with capacity and single-assignment constraints, objective normalization, and `from_random` generator.
- **`MOFS` / `MOBFS`**: Multiobjective Feature Selection benchmark with classification error rate, feature ratio, acquisition costs, synthetic generator (`from_synthetic`), and solution decoding (`decode_features`).
- **`MOSCP` / `MSCP`**: Multiobjective Set Covering Problem benchmark with incidence matrices, `from_random`, `from_subsets`, and coverage decoding (`decode_coverage`).
- **`MSTSP` / `MOTSP`**: Multiobjective Traveling Salesman Problem in permutation matrix assignment representation ($N \times N$ bits) with `from_random`, `from_coordinates`, and tour decoding (`decode_tour`).
- **`MUBQP`**: Multiobjective Unconstrained Binary Quadratic Problem benchmark supporting 3D arrays, lists of 2D matrices, and `from_random`.
- Comprehensive automated test suite in `tests/` covering instantiation, evaluation, validation errors, and `pymoo` solver integration (34 tests passing).
- Full documentation and usage guide in `README.md`.
