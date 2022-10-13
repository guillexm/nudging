from firedrake import *
from .pfilter import *
from .resampling import *
import numpy as np

class bootstrap_filter(base_filter):
    def __init__(self, nsteps, noise_shape):
        self.nsteps = nsteps
        self.noise_shape = noise_shape

    def setup(self, nensemble, model):
        super().setup(nensemble, model)
        # allocate working memory for resampling
        self.new_ensemble = []
        for i in range(self.nensemble):
            self.new_ensemble.append(self.model.allocate())

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

        s = 
        replace this with s_i later
        count = 0
        for i in range(N):
            for j in range(copies[i]):
                self.new_ensemble[count].assign(self.ensemble[i])
                count += 1
        for i in range(N):
            self.ensemble[i].assign(self.new_ensemble[i])
