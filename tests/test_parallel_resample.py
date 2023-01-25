from firedrake import *
from pyop2.mpi import MPI
from nudging import *
from nudging.models.sim_model import SimModel
import numpy as np


nensemble = [2,1,2,2,1]

model = SimModel()

y_true = model.obs()
y_noise = np.random.normal(0.0, 0.01)

y = y_true + y_noise

def log_likelihood(dY):
    return np.dot(dY, dY)/2



simfilter = sim_filter()
simfilter.setup(nensemble, model)
model.ensemble_rank = simfilter.ensemble_rank

simfilter.assimilation_step(y, log_likelihood)
