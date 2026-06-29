
import numpy as np
from numpy import *
from scipy.special import eval_hermite, factorial

##HG POLYNOMIAL STUFF
def HG(w,J,x):
    #k=2*pi/1.55
    #zr= pi*w*w/.55
    return ((2/pi)**.5/(2**J*factorial(J)*w))**.5*eval_hermite(J,2**.5/w *x)*exp(-(1/w/w)*x**2)

def makebasis(w0,groups,x):
    Nx = len(x)
    N = groups * (groups + 1) // 2
    outX = np.empty((N,Nx),dtype=np.float32)
    outY = np.empty((N,Nx),dtype=np.float32)
    
    outG = np.empty((groups,Nx),dtype=np.float32)
    gi = 0 
    for i in range(groups):
        outG[i] = HG(w0,i,x)
    
    for i in range(groups):
        for j in range(i+1):
            outX[gi] = outG[j]#HG(w0,j,x)
            outY[gi] = outG[i-j]
            gi+=1
    return outX,outY,outG


def makeHGModes(Nf, w0,G):
    """ Generate Hermite-Gaussian modes

    Args:
        Nf (_type_): Number of pixels (use same dimension for both axis - XY - so Nf x Nf)
        w0 (_type_): Beam waist of the fundamental mode (MFD)
        G (_type_): Number of groups of modes to generate

    Returns:
        _type_: 3D array as ModeIdx, Nx, Ny
    """
    x = arange(-Nf//2,Nf//2)#,endpoint=False)
    targetX,targetY,targetGroup = makebasis(w0,G,x)
    target2d = einsum('...ki,...kj->kij', targetY, targetX)
    return target2d