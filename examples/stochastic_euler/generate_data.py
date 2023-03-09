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

model = Euler_SD(n, 10)
model.setup()
X_truth = model.allocate()
q0 = X_truth[0]
x = SpatialCoordinate(model.mesh)
q0.interpolate(0.1*sin(x[0])*sin(x[1]))


dt = 0.1


model.randomize(X_truth) # poppulating noise term with PV 
model.run(X_truth, X_truth) # use noise term to solve for PV
u_true_VOM_1 = model.obs()[0] # use PV to get streamfunction and velocity comp1 
u_true_VOM_2 = model.obs()[1] # use PV to get streamfunction and velocity comp2 
u_true_1 = u_true_VOM_1.dat.data[:]
u_true_2 = u_true_VOM_2.dat.data[:]
u_true_data_1 = np.save("u_true_1.npy", u_true_1)
u_true_data_2 = np.save("u_true_2.npy", u_true_2)

N_obs = 5

#print(np.shape(u_obs_list))

# Exact numerical approximation 
u_fin_obs_list_1 = []
u_fin_obs_list_2 = []
for i in range(N_obs):
    u_obs_list_1 =[]
    u_obs_list_2 =[]
    model.run(X_truth, X_truth)
    u_VOM_1 = model.obs()[0]
    u_VOM_2 = model.obs()[1]
    u_1 = u_VOM_1.dat.data[:]
    u_2 = u_VOM_2.dat.data[:]
    u_1_noise = np.random.normal(0.0, 0.05, (n+1)**2 ) 
    u_2_noise = np.random.normal(0.0, 0.05, (n+1)**2 ) 
    #u_noisearray = np.transpose(np.array([u_noise, u_noise]), (1,0))
    u_obs_list_1= u_1 + u_1_noise
    u_obs_list_2= u_2 + u_2_noise
    u_fin_obs_list_1.append(u_obs_list_1)
    u_fin_obs_list_2.append(u_obs_list_2)

# Storing data     

u_fin_obs_arr_1 = np.array(u_fin_obs_list_1)
u_fin_obs_arr_2 = np.array(u_fin_obs_list_2)
u_fin_true_data_1 = np.save("u_obs_true_1.npy", u_fin_obs_arr_1)
u_fin_true_data_2 = np.save("u_obs_true_2.npy", u_fin_obs_arr_2)

u_exact_1 = np.load('u_true_1.npy')
u_exact_2 = np.load('u_true_2.npy')

u_all_obs_1 = np.load('u_obs_true_1.npy')
u_all_obs_2 = np.load('u_obs_true_2.npy')

# Plotting data
plt.plot(u_exact_1, 'r-', label = 'exact_1')
plt.plot(u_exact_2, 'b-', label = 'exact_2')
for i in range(N_obs):
    plt.plot(u_all_obs_1[i,:], '-.',label = 'obs_data_1')
    plt.plot(u_all_obs_2[i,:], '--',label = 'obs_data_2')
#plt.plot(u_exact_obs)
plt.legend()
plt.show()