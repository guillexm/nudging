from firedrake import *
from nudging import *
import numpy as np


""" read obs from saved file 
    Do assimilation step
"""


np.random.seed(17238)


# For mesh generation
model = Camsholm(10)
x, = SpatialCoordinate(model.mesh) 

# bootstrap filter
bfilter = bootstrap_filter(5, (5, 4))
nensemble = 40
#print(bfilter.nsteps)
#bfilter.setup(nensemble, model)

# initialise ensemble
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


    # used perturbed solution data for ensemble[i]
    _, u = bfilter.ensemble[i].split()
    u.interpolate(u0_exp)  
    


def log_likelihood(dY):
    return np.dot(dY, dY)/0.1**2


#Load data
N_obs = 20
y = np.load('y_truth.npy')
#y_obs_rows = iter(y_obsdata)
#print(y[0,:])




# do assimiliation step
for k in range(N_obs):
    bfilter.assimilation_step(y[k,:], log_likelihood)
    print(bfilter.ess)
