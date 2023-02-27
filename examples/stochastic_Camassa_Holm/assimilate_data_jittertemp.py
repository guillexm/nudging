from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc

from nudging.models.stochastic_Camassa_Holm import Camsholm

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""
nsteps = 5
model = Camsholm(100, nsteps)
MALA = True

bfilter = jittertemp_filter(n_temp=4, n_jitt = 10, rho= 0.4, MALA=MALA)


nensemble = [10,10,10,10]
bfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 
u0_exp = (1+0.1)*0.2*2/(exp(x-403./15. + 0.01) + exp(-x+403./15. + 0.02)) \
    + (1+0.2)*0.5*2/(exp(x-203./15. + 0.03)+exp(-x+203./15. + 0.01))

for i in range(nensemble[bfilter.ensemble_rank]):
    _, u = bfilter.ensemble[i][0].split()
    u.interpolate(u0_exp)

def log_likelihood(dY):
    return np.dot(dY, dY)/0.05**2/2

def log_likelihood_symbolic(y,Y):
    for i in range(len(Y)):
        if i == 0:
            ll = (Constant(y[0])-Y[0])**2/0.05**2/2
        else:
            ll += (Constant(y[i])-Y[i])**2/0.05**2/2
    return ll
    
#Load data
y_exact = np.load('y_true.npy')
N_obs = y_exact.shape[0]
y = np.load('y_obs.npy') 

# do assimiliation step
for k in range(N_obs):
    #PETSc.Sys.Print("Step", k)
    #print(bfilter.theta_temper)
    if MALA:
        bfilter.assimilation_step(y[k,:], log_likelihood_symbolic)
    else:
        bfilter.assimilation_step(y[k,:], log_likelihood)
