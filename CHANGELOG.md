# Changelog

All notable changes to fibremodes are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.5.0] - 2026-10-22

### Added

- `calculate_2d_modes`, `resample_modes`, and `ModeSet.resample` take `w_max` or `pixel_pitch`. `w_max` builds `linspace(-w_max, w_max, N)`. `pixel_pitch` builds `arange(-N // 2, N // 2) * pixel_pitch`, which has a sample at 0.

## [1.4.0] - 2026-10-22

### Added

- `FibreProfile.get_init_properties()` returns the constructor arguments that were passed, so `StepIndex`, `GradedIndex`, and `DoubleClad` can be built again with `Type(**profile.get_init_properties())`.

## [1.3.0] - 2026-10-22

### Fixed

- Numerical profiles keep the index equation on the internal fine curve. `n_samples` only sets the solver grid, so a double-clad core and inner cladding stay on their radii and both solvers return the guided count.
- The catalog no longer takes `L_max`. The radial solver stops on its own when an azimuthal order has no guided eigenvalues.
- `ModeSet.get_fields` and `ModeSet.get_fields_resample` return the stacked solved and resampled fields.

### Added

- `catalog` and `FibreEngine`. A spec from `catalog.specs` selects an analytical basis (`LG`, `HG`) or a numerical fibre (`step_index`, `graded_index`, `double_clad`) and returns one `ModeSet`. Existing profile, solver, and basis classes are unchanged.
- `FibreModesWindow` in `fibremodes.ui` (install the `gui` extra). The form follows the catalog. After a solve, resample keeps a second field stack and the viewport shows it when it exists.
- Example notebooks `tests/fibremodes_engine_interface_test.ipynb` and `tests/fibremodes_engine_Widget_example.ipynb`.

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
