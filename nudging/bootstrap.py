from firedrake import *
from .filter import *
import numpy as np

class bootstrap_filter(base_filter):
    def __init__(self, nsteps, noise_shape):
        self.nsteps = nsteps

    def assimilation_step(self, y, log_likelihood):
        # forward model step
        for i in range(len(ensemble)):
            W = numpy.random.randn(noise_shape)
            self.model.run(self.nsteps, W, ensemble[i], ensemble[i])
        
        # particle weights
