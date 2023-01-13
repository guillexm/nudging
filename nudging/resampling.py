import numpy as np
from firedrake import *
from nudging import *
from nudging.models.sim_model import SimModel


def residual_resampling(weights, model):
    """
    :arg weights : a numpy array of normalised weights, size N

    returns
    :arg s: an array of integers, size N. X_i will be replaced
    with X_{s_i}.
    """
    
    model = SimModel()
    #model = SimModel().setup(comm='...')
    mesh = model.mesh

    N = weights.size
    # resample Algorithm 3.27
    copies = np.array(np.floor(weights*N), dtype=int) 
    L = N - np.sum(copies)
    residual_weights = N*weights - copies
    residual_weights /= np.sum(residual_weights)
    
    # Need to add parent indexing 
    for i in range(L):
        u = FunctionSpace(mesh, "R", 0) 
        u.assign(np.random.rand()) # make the random number into a function in the space FunctionSpace(model.mesh, "R", 0) 
        cs = np.cumsum(residual_weights)
        istar = -1
        while cs[istar+1] < u:
            istar += 1
        copies[istar] += 1

    count = 0
    s = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(copies[i]):
            s[count] = i
            count += 1     
    return s
