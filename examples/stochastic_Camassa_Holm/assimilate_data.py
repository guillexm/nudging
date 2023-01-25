from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt

from nudging.models.stochastic_Camassa_Holm import Camsholm

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""
model = Camsholm(100)

bfilter = bootstrap_filter(5, (5,4))

# bfilter = jittertemp_filter(5, (5, 4), n_temp=5, n_jitt=5, rho=0.46,
#                             verbose=True)
nensemble = [2,1,2,2,1]
bfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 
u0_exp = (1+0.1)*0.2*2/(exp(x-403./15. + 0.01) + exp(-x+403./15. + 0.02)) \
    + (1+0.2)*0.5*2/(exp(x-203./15. + 0.03)+exp(-x+203./15. + 0.01))

for i in range(nensemble[bfilter.ensemble_rank]):
    _, u = bfilter.ensemble[i].split()
    u.interpolate(u0_exp)

def log_likelihood(dY):
    return np.dot(dY, dY)/0.05**2/2
    
#Load data
y_exact = np.load('y_true.npy')
N_obs = y_exact.shape[0]


station_view = 15
plt.plot(y_exact[:,station_view], 'b-', label='Y_true')

y = np.load('y_obs.npy') 

y_e = np.zeros((N_obs, nensemble, y.shape[1]))
y_e_mean = np.zeros((N_obs, nensemble))

# do assimiliation step
for k in range(N_obs):
    print("Step", k)
    bfilter.assimilation_step(y[k,:], log_likelihood)
    for e in range(nensemble):
        y_e[k,e,:] = model.obs(bfilter.ensemble[e])

y_e_mean = np.mean(y_e[:,:,station_view], axis=1)

plt.plot(y_e[:,:,station_view], 'y-.')
plt.plot(y_e_mean, 'g--', label='Y_ensemble_mean')
plt.legend()
plt.title('Ensemble prediction with N_ensemble = ' +str(nensemble))
plt.show()
