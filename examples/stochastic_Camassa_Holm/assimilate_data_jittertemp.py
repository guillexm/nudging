from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyadjoint import AdjFloat

from nudging.models.stochastic_Camassa_Holm import Camsholm

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""

nsteps = 5
model = Camsholm(100, nsteps)
MALA = True

jtfilter = jittertemp_filter(n_temp=4, n_jitt = 4, rho= 0.4,
                            verbose=False, MALA=MALA)


nensemble = [5,5,
             5,5]
jtfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

#randomness of the initial data
dx0 = Constant(0.)
dx1 = Constant(0.)
a = Constant(0.)
b = Constant(0.)

for i in range(nensemble[jtfilter.ensemble_rank]):
    dx0.assign(np.random.randn())
    dx1.assign(np.random.randn())
    a.assign(np.random.rand())
    b.assign(np.random.rand())
    u0_exp = (1+a)*0.2*2/(exp(x-403./15. + dx0) + exp(-x+403./15. + dx1)) \
        + (1+b)*0.5*2/(exp(x-203./15. + dx0)+exp(-x+203./15. + dx1))

    _, u = jtfilter.ensemble[i][0].split()
    u.interpolate(u0_exp)

def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*dx
    return ll
    
#Load data
y_exact = np.load('y_true.npy')
y = np.load('y_obs.npy') 
N_obs = y.shape[0]

y_e = np.zeros((N_obs, sum(nensemble), y.shape[1]))
y_e_mean = np.zeros((N_obs, sum(nensemble)))

yVOM = Function(model.VVOM)

# prepare shared arrays for data
y_e_list_arr = []
for m in range(y.shape[1]):        
    y_e_shared_arr = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_list_arr.append(y_e_shared_arr)

ys = y.shape
if jtfilter.ensemble_rank == 0:
    y_e = np.zeros(np.sum(nensemble), ys[0], ys[1])

# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[k, :]
    jtfilter.assimilation_step(yVOM, log_likelihood)
        
    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.w0.assign(jtfilter.ensemble[i][0])
        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list_arr[m].dlocal[i] = obsdata[m]

    for m in range(y.shape[1]):
        y_e_shared_arr[m].synchronise()
        if jtfilter.ensemble_rank == 0:
            y_e[:, k, m] = y_e_shared_arr[m].data())

np.save("ensemble_simulated_obs.npy", y_e)
