from numpy import *


def solver_min_numSamples(wav, na, r_max , min_samples = 10):
    """Estimate the minimum number of samples for the solver.
    The Finest lobe (feature) of the mode is given by wavelenght/na. 
    So lets impose to have at least 10 sample in that lobe
    """

    dx = wav / (na)

    dmax = 2 * r_max
    n = min_samples * dmax  / dx 

    return int(ceil(n))


def resolve_solver_samples(wavelength, na, r_max, n_samples=None):
    """Return ``(n_radial, n_cartesian)`` for a profile grid.

    An explicit ``n_samples`` is used for both grids. When it is omitted,
    ``wavelength`` (metres, the same unit as ``r_max``) and a positive ``na``
    are required. The Cartesian count is ``solver_min_numSamples`` across the
    diameter ``2 * r_max``. The radial count is half of that, because the
    stored curve and the radial solver run from ``0`` to ``r_max``.
    """
    if n_samples is not None:
        count = int(n_samples)
        if count < 2:
            raise ValueError("n_samples must be at least 2.")
        return count, count
    if wavelength is None:
        raise ValueError("Pass wavelength or n_samples.")
    wavelength = float(wavelength)
    na = float(na)
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    if na <= 0:
        raise ValueError("na must be positive.")
    n_cartesian = solver_min_numSamples(wavelength, na, r_max)
    n_radial = (n_cartesian + 1) // 2
    if n_radial < 2:
        n_radial = 2
    return n_radial, n_cartesian
