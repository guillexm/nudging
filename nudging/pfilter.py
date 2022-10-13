from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import *
from .resampling import *
import numpy as np

class base_filter(object, metaclass=ABCMeta):
    ensemble = []

    def __init__(self):
        pass
        
    def setup(self, nensemble, model):
        """
        Construct the ensemble

        nensemble - number of ensemble members
        model - the model to use
        """
        self.model = model
        self.nensemble = nensemble
        for i in range(nensemble):
            self.ensemble.append(model.allocate())

    @abstractmethod
    def assimilation_step(self, y, log_likelihood):
        """
        Advance the ensemble to the next assimilation time
        and apply the filtering algorithm
        y - a k-dimensional numpy array containing the observations
        log_likelihood - a function that computes -log(Pi(y|x))
                         for computing the filter weights
        """
        pass


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

        s = residual_resampling(weights)
        for i in range(N):
            self.new_ensemble[i].assign(self.ensemble[s[i]])
        for i in range(N):
            self.ensemble[i].assign(self.new_ensemble[i])
