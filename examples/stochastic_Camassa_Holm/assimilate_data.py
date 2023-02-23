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


bfilter = bootstrap_filter()


nensemble = [3,2,5,4]
bfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 
u0_exp = (1+0.1)*0.2*2/(exp(x-403./15. + 0.01) + exp(-x+403./15. + 0.02)) \
    + (1+0.2)*0.5*2/(exp(x-203./15. + 0.03)+exp(-x+203./15. + 0.01))

for i in range(nensemble[bfilter.ensemble_rank]):
    _, u = bfilter.ensemble[i][0].split()
    u.interpolate(u0_exp)

def log_likelihood(dY):
    return np.dot(dY, dY)/0.05**2/2
    
#Load data
y_exact = np.load('y_true.npy')
N_obs = y_exact.shape[0]

y = np.load('y_obs.npy') 
plt.plot(y[:,10], 'b-', label = 'y_obs')

y_ensemble = np.zeros((N_obs, sum(nensemble), y.shape[1]))
y_ensemble_mean = np.zeros((N_obs, sum(nensemble)))

# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    bfilter.assimilation_step(y[k,:], log_likelihood)

    for e in range(sum(nensemble)):
        y_ensemble[k,e,:] = model.obs(bfilter.ensemble[e][0])

y_ensemble_mean = np.mean(y_ensemble_mean[:,:,10], axis=1)

#print(y_e_mean)
plt.plot(y_ensemble[:,:,10], 'y_ensemble')
plt.plot(y_ensemble_mean, 'g-', label='y_ensemble_mean')
#plt.plot(y_e_var, 'r-', label='y_ensemble_std')
plt.title('Ensemble prediction with N_ensemble = ' +str(nensemble))
plt.show()


