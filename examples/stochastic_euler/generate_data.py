from ctypes import sizeof
from fileinput import filename
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

from nudging.models.stochastic_euler import Euler_SD


"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get true value and obseravation and use paraview for viewing
add observation noise N(0, sigma^2) 
"""
#np.random.seed(138)
n = 4
nsteps = 5
model = Euler_SD(n, nsteps=nsteps)
model.setup()
X_truth = model.allocate()
q0 = X_truth[0]
x = SpatialCoordinate(model.mesh)
q0.interpolate(sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*cos(6*pi*x[0])*cos(6*pi*x[1])
                        +0.02*sin(2*pi*x[0])+0.02*sin(2*pi*x[1])+0.3*cos(10*pi*x[0])*cos(4*pi*x[1])) # sin 8px 



N_obs = 5

model.randomize(X_truth) # poppulating noise term with PV 
model.run(X_truth, X_truth) # use noise term to solve for PV
u_true_VOM_1 = model.obs() # use PV to get streamfunction and velocity comp1 
#u_true_VOM_2 = model.obs()[1] # use PV to get streamfunction and velocity comp2 
u1_true = u_true_VOM_1.dat.data[:]
#u2_true = u_true_VOM_2.dat.data[:]

u1_true_all = np.zeros((N_obs, np.size(u1_true)))
#u2_true_all = np.zeros((N_obs, np.size(u2_true)))
u1_obs_all = np.zeros((N_obs, np.size(u1_true)))
#u2_obs_all = np.zeros((N_obs, np.size(u2_true)))


# Exact numerical approximation 
for i in range(N_obs):
    model.randomize(X_truth)
    model.run(X_truth, X_truth)
    u_VOM_1 = model.obs()
    #u_VOM_2 = model.obs()[1]
    u_1 = u_VOM_1.dat.data[:]
    #u_2 = u_VOM_2.dat.data[:]
    u1_true_all[i,:] = u_1
    #u2_true_all[i,:] = u_2
    np.save("u1_true_data.npy", u1_true_all)
    #np.save("u2_true_data.npy", u2_true_all)
    u_1_noise = np.random.normal(0.0, 0.05, (n+1)**2 ) # mean = 0, sd = 0.05
    u_2_noise = np.random.normal(0.0, 0.05, (n+1)**2 ) 
    u1_obs = u_1 + u_1_noise
    #u2_obs = u_2 + u_2_noise
    u1_obs_all[i,:] = u1_obs
    #u2_obs_all[i,:] = u2_obs
    np.save("u1_obs_data.npy", u1_obs_all)
    #np.save("u2_obs_data.npy", u2_obs_all)
