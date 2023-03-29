from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyadjoint import AdjFloat

import time
start_time = time.time()



from nudging.models.stochastic_euler import Euler_SD

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""

n = 4
nsteps = 5
model = Euler_SD(n, nsteps=nsteps)

MALA = False
verbose = False
# jtfilter = jittertemp_filter(n_temp=4, n_jitt = 4, rho= 0.99,
#                             verbose=verbose, MALA=MALA)

jtfilter = bootstrap_filter()

# #Load data
u1_exact = np.load('u1_true_data.npy')
u2_exact = np.load('u2_true_data.npy')
u_vel_1 = np.load('u1_obs_data.npy') 
u_vel_2 = np.load('u2_obs_data.npy') 

nensemble = [5,5,5,5,5]


jtfilter.setup(nensemble, model)

x = SpatialCoordinate(model.mesh) 

#prepare the initial ensemble
for i in range(nensemble[jtfilter.ensemble_rank]):
    a = model.rg.uniform(model.R, 0., 1.0) # fixed a and b for local ensemble members
    b = model.rg.uniform(model.R, 0., 1.0)
    q0_in = a*sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*b*cos(6*pi*x[0])*cos(6*pi*x[1])\
                +0.02*a*sin(2*pi*x[0])+0.02*a*sin(2*pi*x[1])+0.3*b*cos(10*pi*x[0])*cos(4*pi*x[1]) 

   
    q = jtfilter.ensemble[i][0]
    q.interpolate(q0_in)
   
# #prepare the initial ensemble
# u1_ensembles = np.zeros((len(nensemble), np.size(u1_exact)))
# u2_ensembles = np.zeros((len(nensemble), np.size(u1_exact)))
# for i in range(nensemble[jtfilter.ensemble_rank]):
#     a = model.rg.uniform(model.R, 0., 1.0)
#     b = model.rg.uniform(model.R, 0., 1.0)
#     q0_in = 0.1*(1+a)*sin(x[0]+a)*(1+b)*sin(x[1]+b)
#     q = jtfilter.ensemble[i][0]
#     q.interpolate(q0_in)
#     model.randomize(jtfilter.ensemble[i]) # poppulating noise term with PV 
#     model.run(jtfilter.ensemble[i], jtfilter.ensemble[i]) # use noise term to solve for PV        
#     u1_e_VOM = model.obs()[0] # use PV to get streamfunction and velocity comp1 
#     u2_e_VOM = model.obs()[1] # use PV to get streamfunction and velocity comp2 
#     u_1 = u1_e_VOM.dat.data[:]
#     u_2 = u2_e_VOM.dat.data[:]
#     u1_ensembles[i,:] = u_1
#     u2_ensembles[i,:] = u_2
# np.save("u1_ensemble_data.npy", u1_ensembles)
# np.save("u2_ensemble_data.npy", u2_ensembles)

def log_likelihood(y, Y):
    ll = 0.5*(1/0.05**2)*((y[0]-Y[0])**2 + (y[1]-Y[1])**2)*dx
    return ll

    


N_obs = u_vel_1.shape[0]

# VOM defintion
u1_VOM = Function(model.VVOM)
u2_VOM = Function(model.VVOM)
u_VOM = [u1_VOM]
u_VOM.append(u2_VOM)

# prepare shared arrays for data
u1_e_list = []
u2_e_list = []
u1_e_fwd_list = []
u2_e_fwd_list = []
for m in range(u_vel_1.shape[1]):        
    u1_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    u2_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    u1_e_fwd_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    u2_e_fwd_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    u1_e_list.append(u1_e_shared)
    u2_e_list.append(u2_e_shared)
    u1_e_fwd_list.append(u1_e_fwd_shared)
    u2_e_fwd_list.append(u2_e_fwd_shared)


ushape = u_vel_1.shape
if COMM_WORLD.rank == 0:
    u1_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))
    u2_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))
    u1_e_fwd = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))
    u2_e_fwd = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))



# Simply forwad model to get  forecast step
for k in range(N_obs):
    
    for i in range(nensemble[jtfilter.ensemble_rank]):
        #model.randomize(jtfilter.ensemble[i])
        model.run(jtfilter.ensemble[i], jtfilter.ext_ensemble[i])
        fwd_obsdata_1 = model.obs()[0].dat.data[:]
        fwd_obsdata_2 = model.obs()[1].dat.data[:]
        for m in range(u_vel_1.shape[1]):
            u1_e_fwd_list[m].dlocal[i] = fwd_obsdata_1[m]
            u2_e_fwd_list[m].dlocal[i] = fwd_obsdata_2[m]

# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    u_VOM[0].dat.data[:] = u_vel_1[k, :]
    u_VOM[1].dat.data[:] = u_vel_2[k, :]

    for i in range(nensemble[jtfilter.ensemble_rank]):
        #model.randomize(jtfilter.ensemble[i])
        model.run(jtfilter.ensemble[i], jtfilter.ext_ensemble[i])
        fwd_obsdata_1 = model.obs()[0].dat.data[:]
        fwd_obsdata_2 = model.obs()[1].dat.data[:]
        for m in range(u_vel_1.shape[1]):
            u1_e_fwd_list[m].dlocal[i] = fwd_obsdata_1[m]
            u2_e_fwd_list[m].dlocal[i] = fwd_obsdata_2[m]

    jtfilter.assimilation_step(u_VOM, log_likelihood)
        
    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.q0.assign(jtfilter.ensemble[i][0])
        obsdata_1 = model.obs()[0].dat.data[:]
        obsdata_2 = model.obs()[1].dat.data[:]
        for m in range(u_vel_1 .shape[1]):
            u1_e_list[m].dlocal[i] = obsdata_1[m]
            u2_e_list[m].dlocal[i] = obsdata_2[m]

    for m in range(u_vel_1.shape[1]):
        u1_e_list[m].synchronise()
        u2_e_list[m].synchronise()
        u1_e_fwd_list[m].synchronise()
        u2_e_fwd_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            u1_e[:, k, m] = u1_e_list[m].data()
            u2_e[:, k, m] = u2_e_list[m].data()
            u1_e_fwd[:, k, m] = u1_e_fwd_list[m].data()
            u2_e_fwd[:, k, m] = u2_e_fwd_list[m].data()

PETSc.Sys.Print("--- %s seconds ---" % (time.time() - start_time))

if COMM_WORLD.rank == 0:
    np.save("Velocity_1_ensemble_simulated_obs.npy", u1_e)
    np.save("Velocity_2_ensemble_simulated_obs.npy", u2_e)
    np.save("Velocity_1_ensemble_forward_obs.npy", u1_e)
    np.save("Velocity_2_ensemble_forward_obs.npy", u2_e)

