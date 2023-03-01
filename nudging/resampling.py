import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
from functools import cached_property
from firedrake import *
from nudging import *

class residual_resampling(object):
    def __init__(self, seed=34523):
        self.seed = seed
        self.initialised = False

    @cached_property
    def rg(self):
        pcg = PCG64(seed=self.seed)
        return RandomGenerator(pcg)

    def resample(self, weights, model):
        """
        :arg weights : a numpy array of normalised weights, size N
        
        returns
        :arg s: an array of integers, size N. X_i will be replaced
        with X_{s_i}.
        """

        if not self.initialised:
            self.initialised = True
            self.R = FunctionSpace(model.mesh, "R", 0)
        
        N = weights.size
        copies = np.array(np.floor(weights*N), dtype=int) 
        L = N - np.sum(copies)
        residual_weights = N*weights - copies
        residual_weights /= np.sum(residual_weights)
        
        for i in range(L):
            u = self.rg.uniform(self.R, 0., 1.0)
            u0 =  u.dat.data[:]

            cs = np.cumsum(residual_weights)
            istar = -1
            while cs[istar+1] < u0:
                istar += 1
            copies[istar] += 1

        count = 0
        s = np.zeros(N, dtype=int)
        for i in range(N):
            for j in range(copies[i]):
                s[count] = i
                count += 1     
        return s
