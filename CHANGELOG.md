# Changelog

All notable changes to fibremodes are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.1] - 2026-06-29

### Fixed

- Import `factorial` from `scipy.special` in HG mode generation (`analytical/HG/fibremodes.py`).

## [1.1.0] - 2026-06-29

### Added

- Hermite-Gaussian analytical mode support via `makeHGModes` (`fibremodes.analytical.HG.fibremodes`).
- HG modes documentation in README.

## [1.0.0]

Initial release: scalar mode solver, LG analytical modes (GPU), overlaps utilities.
