from ctypes import sizeof
from fileinput import filename
from firedrake import *
#from nudging import *
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
u_true_VOM = model.obs() # use PV to get streamfunction and finally velocity
u_true = u_true_VOM.dat.data[:]
u_true_data = np.save("u_true.npy", u_true)

print(np.shape(u_true))

N_obs = 5

#print(np.shape(u_obs_list))

# Exact numerical approximation 
u_fin_obs_list = []
for i in range(N_obs):
    u_obs_list =[]
    model.run(X_truth, X_truth)
    u_true_VOM = model.obs()
    u_true = u_true_VOM.dat.data[:]
    u_noise = np.random.normal(0.0, 0.05, (n+1)**2 ) 
    u_noisearray = np.transpose(np.array([u_noise, u_noise]), (1,0))
    u_obs_list= u_true + u_noisearray
    u_fin_obs_list.append(u_obs_list)

# Storing data     

u_fin_obs_arr = np.array(u_fin_obs_list)

print(np.shape(u_fin_obs_arr))

u_fin_true_data = np.save("u_obs_true.npy", u_fin_obs_arr)

u_exact_obs = np.load('u_true.npy')

u_all_obs = np.load('u_obs_true.npy')

u_vel = np.load('u_obs_true.npy')
print(np.shape(u_vel))

# Plotting data
plt.plot(u_exact_obs, 'r-', label = 'exact')
for i in range(N_obs):
    plt.plot(u_all_obs[i,:,:], ':')
#plt.plot(u_exact_obs)
plt.legend()
plt.show()