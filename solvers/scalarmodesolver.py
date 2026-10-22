from numpy import *
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import spdiags
from scipy.sparse.linalg import eigsh,eigs

def scalarmodeEigsSolver(n, dx, dy, k0, xx, K = 10, bc = {'r':'a','l':'a','t':'a','b':'a'}, return_eigenvectors = True):
    
    '''
    INPUT-->
    n : refractive index profile,
    dx: x-coordinate step size,
    dy: y-coordinate step size (solutions is not stable if dx == dy make dy = dx * 1e-9, makes modes to be aligned with xy axes),
    k0: wavnumber (2pi/lambda0)
    xx: grid (XY)
    K = number of solutions to find
    bc(optional) = custum boundary conditions
    return_eigenvectors(optinal) = True or False
    
    OUTPUT:
    
    modes -> Valid mode solutions: 3D array with the solutions of the eigenproblem (modes)
    n_eff -> Valid mode effective refractive indices

    '''
    
    #Default boundary conditions
    bct = {'r':'a','l':'a','t':'a','b':'a'}
    bct.update(bc)
    bc = bct
    
    xx = xx.ravel().real
    tx  = ones_like(xx) 
    txp = tx 
    txm = tx

    M = np.prod(n.shape)
    Ny, Nx = n.shape
    
    kk = (k0*n.ravel())**2
    a = kk - 2 * (tx/dx**2 + tx/dy**2)
    b = txp / dx**2
    c = tx / dy**2
    bl = txm / dx**2
    
    #fix boundary
    b[Nx::Nx]=0
    bl[Nx-1::Nx]=0
    
    if bc['r'] == 'a':
        a[Nx-1::Nx] -= 1./dx**2 
    else:
        a[Nx-1::Nx] += 1./dx**2 
        
    if bc['l'] =='a':
        a[0::Nx] -= 1./dx**2 
    else:
        a[0::Nx] += 1./dx**2 
    
    if bc['t'] =='a':
        a[0:Nx] -= 1./dy**2 
    else:
        a[0:Nx] += 1./dy**2 
    
    if bc['b'] =='a':
        a[-Nx:] -= 1./dy**2 
    else:
        a[-Nx:] += 1./dy**2 

    #Biuld the matrix
    diags = np.array([-Nx,-1,0,1,Nx])
    sm = spdiags([c,bl,a,b,c], diags, M, M, format = 'csr')
    #Solve the eigenproblem
    res = eigsh(sm, K, sigma = kk.max() , return_eigenvectors=return_eigenvectors)

    modes = res[1].T.reshape(K, Ny, Nx)
    beta = res[0]

    modes = flip(modes, axis = 0) #order from lowest eigenvalue to highest
    beta = flip(beta)
    
    n_eff = np.sqrt(beta) / k0
    # The outer corner is cladding. Unguided eigenvalues still clear that index
    # by a small fraction of the core-to-cladding step, so the search margin
    # would otherwise be counted as guided.
    index = np.asarray(n)
    n_outer = float(np.min(index[-10:, -10:]))
    n_peak = float(np.max(index))
    guided = n_eff > n_outer + 0.01 * (n_peak - n_outer)
    modes = modes[guided]
    n_eff = n_eff[guided]
    guided_count = int(np.sum(guided))
    print(f'Total guided modes: {guided_count}')

    return modes, n_eff


class ScalarModeSolver:
    """2D scalar mode solver with the same call pattern as ``RadialModeSolver``.

    ``x`` and ``n`` are the Cartesian axis and square index from
    ``FibreProfile.index_2d``. ``n_modes`` is the estimate from
    ``estimate_num_modes``; ``margin`` extra eigenvalues are requested and the
    unguided ones are dropped by ``scalarmodeEigsSolver``.
    """

    def __init__(self, x, n, wavelength, n_modes, margin=4, asymmetric_step=True, step_epsilon=1e-11):
        axis = np.asarray(x, dtype=float).reshape(-1)
        index = np.asarray(n, dtype=float)
        if index.ndim != 2 or index.shape[0] != index.shape[1]:
            raise ValueError("n must be a square 2D refractive-index array.")
        if axis.size != index.shape[0]:
            raise ValueError("x must have one sample per side of n.")
        if axis.size < 4:
            raise ValueError("x needs at least 4 samples.")
        if not np.all(np.diff(axis) > 0):
            raise ValueError("x must be strictly increasing.")
        if float(wavelength) <= 0:
            raise ValueError("wavelength must be positive.")
        if int(n_modes) < 1:
            raise ValueError("n_modes must be at least 1.")
        if int(margin) < 0:
            raise ValueError("margin must be non-negative.")
        requested = int(n_modes) + int(margin)
        if requested >= index.size:
            raise ValueError("n_modes + margin must be smaller than the number of grid points.")
        self.x = axis.copy()
        self.n = index.copy()
        self.wavelength = float(wavelength)
        self.k0 = 2.0 * np.pi / self.wavelength
        self.n_modes = int(n_modes)
        self.margin = int(margin)
        self.dx = float(self.x[1] - self.x[0])
        self.dy = self.dx + float(step_epsilon) if asymmetric_step else self.dx
        self.modes = None
        self.n_eff = None
        self.modes_resampled = None
        self.x_resampled = None
        self._modes_ready = False

    def calculate_modes(self):
        """Solve the guided modes and store them on ``modes`` and ``n_eff``."""
        xx, yy = np.meshgrid(self.x, self.x, indexing="xy")
        modes, n_eff = scalarmodeEigsSolver(
            self.n, self.dx, self.dy, self.k0, xx, K=self.n_modes + self.margin,
        )
        self.modes = np.asarray(modes)
        self.n_eff = np.asarray(n_eff, dtype=float)
        self.modes_resampled = None
        self.x_resampled = None
        self._modes_ready = True
        return self.modes

    def resample_modes(self, N, window_max=None):
        """Interpolate the stored modes onto a new square window.

        ``N`` is the sample count on each side. ``window_max`` is the
        half-width. When it is omitted the half-width is ``x[-1]``. Samples
        outside the solved window are zero. The result is saved on
        ``modes_resampled`` and does not replace ``modes``.
        """
        if not self._modes_ready:
            raise RuntimeError("Call calculate_modes before resample_modes.")
        n_points = int(N)
        if n_points < 2:
            raise ValueError("N must be at least 2.")
        window = float(self.x[-1] if window_max is None else window_max)
        if window <= 0:
            raise ValueError("window_max must be positive.")
        axis = np.linspace(-window, window, n_points)
        # Mode rows are y and columns are x, matching the finite-difference reshape.
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        points = np.stack([yy.ravel(), xx.ravel()], axis=-1)
        resampled = np.empty((self.modes.shape[0], n_points, n_points), dtype=float)
        for index, field in enumerate(self.modes):
            values = np.real(field) if np.iscomplexobj(field) else np.asarray(field, dtype=float)
            interpolant = RegularGridInterpolator(
                (self.x, self.x), values, bounds_error=False, fill_value=0.0,
            )
            resampled[index] = interpolant(points).reshape(n_points, n_points)
        self.modes_resampled = resampled
        self.x_resampled = axis
        return resampled

    def get_mfd(self):
        """Mode-field diameter of the fundamental mode.

        ``modes[0]`` is fit to the Gaussian with the same intensity second
        moment. The diameter is ``2 w0``, with ``w0`` the 1/e² intensity
        radius, in the same length unit as ``x``.
        """
        if not self._modes_ready or self.modes is None or len(self.modes) == 0:
            raise RuntimeError("Call calculate_modes before get_mfd.")
        field = self.modes[0]
        intensity = np.abs(field) ** 2
        xx, yy = np.meshgrid(self.x, self.x, indexing="xy")
        power = np.sum(intensity)
        if power <= 0:
            raise RuntimeError("The fundamental mode has no intensity.")
        variance = 0.5 * np.sum(intensity * (xx**2 + yy**2)) / power
        return float(4.0 * np.sqrt(variance))
    
