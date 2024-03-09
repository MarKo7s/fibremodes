from numpy import *
import numpy as np
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
    
    modes -> 3D array with the solutions of the eigenproblem (modes)
    beta -> propagation constant of the modes
    sm -> eigenproblem to solve (raw matrix in sparse mode)
    kk.max -> use for latter normalize the propagations constant (equal to the highest solution (eigenvalue) )
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
    
    return beta, modes, sm,  kk.max()
    
