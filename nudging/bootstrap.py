from firedrake import *
from .pfilter import *
import numpy as np

class bootstrap_filter(base_filter):
    def __init__(self, nsteps, noise_shape):
        self.nsteps = nsteps
        
    def assimilation_step(self, y, log_likelihood):
        N = len(self.ensemble)
        weights = np.zeros(N)
        # forward model step
        for i in range(N):
            W = numpy.random.randn(noise_shape)
            self.model.run(self.nsteps, W,
                           self.ensemble[i], self.ensemble[i])
        
            # particle weights
            Y = self.model.obs(self.ensemble[i])
            weights[i] = log_likelihood(y-Y)

        # renormalise
        weights = exp(-weights)
        weights /= np.sum(weights)

        # resample
        copies = np.array(np.floor(weights*N), dtype=int)
        L = N - np.sum(copies)
        residual_weights = M*weights - copies
        residual_weights /= np.sum(residual_weights)

        for i in range(L):
            u =  np.random.rand()
            cs = np.cumsum(residual_weights)
            istar = np.argmin(cs >= u)
            copies[istar] += 1
        new_ensemble = []

        for i in range(N):
            new_ensemble.append(self.ensemble(copies[i]))
        self.ensemble = new_ensemble
