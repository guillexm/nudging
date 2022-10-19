from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""
model = Camsholm(100)
x, = SpatialCoordinate(model.mesh) 

bfilter = jittertemp_filter(5, (5, 4), n_temp=5, n_jitt=5, rho=0.46)
nensemble = 40
bfilter.setup(nensemble, model)

dx0 = Constant(0.)
dx1 = Constant(0.)
a = Constant(0.)
b = Constant(0.)

for i in range(nensemble):
    dx0.assign(np.random.randn()*0.1)
    dx1.assign(np.random.randn()*0.1)
    a.assign(np.random.rand())
    b.assign(np.random.rand())

    u0_exp = a*0.2*2/(exp(x-403./15. + dx0) + exp(-x+403./15. + dx0)) \
           + b*0.5*2/(exp(x-203./15. + dx1)+exp(-x+203./15. + dx1))

    _, u = bfilter.ensemble[i].split()
    u.interpolate(u0_exp)  

def log_likelihood(dY):
    return np.dot(dY, dY)/0.1**2
    
#Load data
N_obs = 25
y_exact = np.load('y_true.npy')
plt.plot(y_exact[:,10], 'r-', label='Y_true')

y = np.load('y_obs.npy') 

y_e = np.zeros((N_obs, nensemble, y.shape[1]))
y_e_mean = np.zeros((N_obs, nensemble))

print(y_e.shape)

# do assimiliation step
for k in range(N_obs):
    bfilter.assimilation_step(y[k,:], log_likelihood)
    for e in range(nensemble):
        y_e[k,e,:] = model.obs(bfilter.ensemble[e])

y_e_mean = np.mean(y_e[:,:,10], axis=1)

plt.plot(y_e[:,:,10], 'y-')
plt.plot(y_e_mean, 'g-', label='Y_ensemble_mean')
plt.legend()
plt.title('Ensemble prediction with N_ensemble = ' +str(nensemble))
plt.show()