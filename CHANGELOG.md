# Changelog

All notable changes to fibremodes are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - 2026-10-16

### Added

- Radial finite-difference solver `RadialModeSolver` for guided modes of an axisymmetric index \(n(r)\). Each solution is a `RadialMode` (`L`, `M`, propagation constant `beta`, mode group) and can be resampled onto a Cartesian grid with `calculate_2d_modes` and `modes_2D`.
- Axisymmetric index profiles `StepIndex`, `GradedIndex`, and `DoubleClad` in `fibremodes.solvers.fiberprofile`, with `index_1d`, `index_2d`, and `plot`.
- Example notebook `tests/radialmodesolver_examples.ipynb`.

## [1.1.1] - 2026-06-29

### Fixed

- Import `factorial` from `scipy.special` in HG mode generation (`analytical/HG/fibremodes.py`).

## [1.1.0] - 2026-06-29

### Added

- Hermite-Gaussian analytical mode support via `makeHGModes` (`fibremodes.analytical.HG.fibremodes`).
- HG modes documentation in README.

## [1.0.0]

Initial release: scalar mode solver, LG analytical modes (GPU), overlaps utilities.
