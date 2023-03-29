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
xpoints = 40
model = Camsholm(100, nsteps, xpoints)
MALA = False
verbose = False
jtfilter = jittertemp_filter(n_temp=4, n_jitt = 4, rho= 0.99,
                            verbose=verbose, MALA=MALA)

# jtfilter = bootstrap_filter()

nensemble = [15,15,15,15,15]
#nensemble = [2,2,2,2,2]
jtfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

#prepare the initial ensemble
for i in range(nensemble[jtfilter.ensemble_rank]):
    dx0 = model.rg.normal(model.R, 0., 0.05)
    dx1 = model.rg.normal(model.R, 0., 0.05)
    a = model.rg.uniform(model.R, 0., 1.0)
    b = model.rg.uniform(model.R, 0., 1.0)
    u0_exp = (1+a)*0.2*2/(exp(x-403./15. + dx0) + exp(-x+403./15. + dx0)) \
        + (1+b)*0.5*2/(exp(x-203./15. + dx1)+exp(-x+203./15. + dx1))

    _, u = jtfilter.ensemble[i][0].split()
    u.interpolate(u0_exp)


def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*dx
    return ll
    
#Load data
y_exact = np.load('y_true.npy')
y = np.load('y_obs.npy') 
#print(np.shape(y))
N_obs = y.shape[0]

yVOM = Function(model.VVOM)

# prepare shared arrays for data
y_e_list = []
y_e_fwd_list = []
y_e_asmfwd_list = []
for m in range(y.shape[1]):        
    y_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_fwd_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_asmfwd_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_list.append(y_e_shared)
    y_e_fwd_list.append(y_e_fwd_shared)
    y_e_asmfwd_list.append(y_e_asmfwd_shared)

ys = y.shape
if COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    y_e_fwd = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    y_e_asmfwd = np.zeros((np.sum(nensemble), ys[0], ys[1]))

# Simply forwad model to get  forecast step
for k in range(N_obs):

    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.randomize(jtfilter.ensemble[i])
        model.run(jtfilter.ensemble[i], jtfilter.ext_ensemble[i])
        fwd_obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_fwd_list[m].dlocal[i] = fwd_obsdata[m]




# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[k, :]

    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.randomize(jtfilter.ensemble[i])
        model.run(jtfilter.ensemble[i], jtfilter.asm_ensemble[i])
        asmfwd_obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_asmfwd_list[m].dlocal[i] = asmfwd_obsdata[m]
    

    jtfilter.assimilation_step(yVOM, log_likelihood)
        
    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.w0.assign(jtfilter.ensemble[i][0])
        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list[m].dlocal[i] = obsdata[m]


    

    for m in range(y.shape[1]):
        y_e_list[m].synchronise()
        y_e_fwd_list[m].synchronise()
        y_e_asmfwd_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()
            y_e_fwd [:, k, m] = y_e_fwd_list[m].data()
            y_e_asmfwd [:, k, m] = y_e_asmfwd_list[m].data()

if COMM_WORLD.rank == 0:
    np.save("ensemble_simulated_obs.npy", y_e)
    np.save("ensemble_forward_obs.npy", y_e_fwd)
    np.save("final_ensemble_forward.npy", y_e_asmfwd)
