from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc

from nudging.models.stochastic_euler import Euler_SD

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""
n=16
nsteps = 10
model = Euler_SD(n, 10)


bfilter = bootstrap_filter()


nensemble = [2,2]
bfilter.setup(nensemble, model)

x = SpatialCoordinate(model.mesh) 
q0_exct = 0.1*(1+0.1)*sin(x[0])*sin(x[1])

# for i in range(nensemble[bfilter.ensemble_rank]):
#     q = bfilter.ensemble[i][0]
#     q.interpolate(q0_exct)
#     model.randomize(bfilter.ensemble[i])
#     model.run(bfilter.ensemble[i],bfilter.ensemble[i])
#     u_vel_en = model.obs()

q = bfilter.ensemble[0][0]
q.interpolate(q0_exct)
model.randomize(bfilter.ensemble[0])
model.run(bfilter.ensemble[0],bfilter.ensemble[0])
v_vel_en = model.obs()
print(type(v_vel_en))
v_vel = v_vel_en.dat.data[:]
print(type(v_vel))
v_vel_data = np.save("v_vel.npy", v_vel)

v_vel_fin = np.load('v_vel.npy') 
print('Vel_shape', np.shape(v_vel))



# def log_likelihood(y, Y):
#     return 0.1*x[0]*dx

    
# #Load data
u_exact = np.load('u_true.npy')
u_vel = np.load('u_obs_true.npy') 

N_obs = u_vel.shape[0]

uVOM = Function(model.VVOM)
uVOM.dat.data[:] = u_vel[1, :]
uVOM_data = uVOM.dat.data[:] 


def log_likelihood(y, Y):
    print(0.5*(1/0.05**2)*np.sum((v_vel[0]-uVOM_data[0])**2))
    #ll = (1/0.05**2)*(inner(y[0]-Y[0], y[0]-Y[0])*dx)
    ll = 0.5*(1/0.05**2)*((y[0]-Y[0])**2)*dx
    return ll

print('Log', assemble(log_likelihood(uVOM, v_vel_en)))

# if bfilter.ensemble_rank == 0:
#     plt.plot(u_exact, label = 'original')
#     plt.plot(v_vel_fin, label = 'perturbed')
#     plt.legend()
#     plt.show()


# # prepare shared arrays for data
# u_vel_e_list = []
# for m in range(u_vel.shape[1]):        
#     u_vel_e_shared = SharedArray(partition=nensemble, 
#                                  comm=bfilter.subcommunicators.ensemble_comm)
#     u_vel_e_list.append(u_vel_e_shared)

# ys = u_vel.shape
# print(ys)
# if COMM_WORLD.rank == 0:
#     u_vel_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))

# # do assimiliation step
# for k in range(N_obs):
#     PETSc.Sys.Print("Step", k)
#     uVOM.dat.data[:] = u_vel[k, :]
#     bfilter.assimilation_step(uVOM, log_likelihood)
        
    # for i in range(nensemble[bfilter.ensemble_rank]):
    #     model.q0.assign(bfilter.ensemble[i][0])
    #     obsdata = model.obs().dat.data[:]
    #     for m in range(u_vel.shape[1]):
    #         u_vel_e_list[m].dlocal[i] = obsdata[m]

    # for m in range(u_vel.shape[1]):
    #     u_vel_e_list[m].synchronise()
    #     if COMM_WORLD.rank == 0:
    #        u_vel_e[:, k, m] = u_vel_e_list[m].data()

# if COMM_WORLD.rank == 0:
#     np.save("ensemble_simulated_obs.npy", u_vel_e)