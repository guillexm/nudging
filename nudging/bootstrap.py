from firedrake import *
from .pfilter import *
import numpy as np

class bootstrap_filter(base_filter):
    def __init__(self, nsteps, noise_shape):
        self.nsteps = nsteps
        self.noise_shape = noise_shape

    def assimilation_step(self, y, log_likelihood):
        N = len(self.ensemble)
        weights = np.zeros(N)
        # forward model step
        for i in range(N):
            W = np.random.randn(*(self.noise_shape))
            self.model.run(self.nsteps, W,
                           self.ensemble[i], self.ensemble[i])   # solving FEM with ensemble as input and final sol ensemble
        
            # particle weights
            Y = self.model.obs(self.ensemble[i])
            weights[i] = log_likelihood(y-Y)

        # renormalise
        weights = np.exp(-weights)
        weights /= np.sum(weights)
        self.ess = 1/np.sum(weights**2)

        # resample Algorithm 3.27
        copies = np.array(np.floor(weights*N), dtype=int)  # x_i = integer fun of len(ensemble)*weight
        L = N - np.sum(copies)
        residual_weights = N*weights - copies
        residual_weights /= np.sum(residual_weights)

        # Need to add parent indexing 
        for i in range(L):
            u =  np.random.rand()
            cs = np.cumsum(residual_weights) # cumulative sum 
            istar = np.argmin(cs >= u)
            copies[istar] += 1

        new_ensemble = []

        for i in range(N):
            new_ensemble.append(self.ensemble[copies[i]])
        self.ensemble = new_ensemble
