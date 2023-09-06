import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
from functools import cached_property
from firedrake import *
from nudging import *
from firedrake.petsc import PETSc

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
        cumsum_weight = np.cumsum(weights)
        ensembl_pos = (self.rg.random() + np.arange(N))/N
        s = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if ensembl_pos[i] < cumsum_weight[j]:
                s[i] = j
                i += 1
            else:
                j +=1
        return s
        
