from ctypes import sizeof
from fileinput import filename
from firedrake import *
from nudging import *
import numpy as np
"""
To store observation data at T_1 ---- T_Nobs
use as truth 
"""
# create some synthetic data
model = Camsholm(100)
X_truth = model.allocate()
_, u0 = X_truth.split()
x, = SpatialCoordinate(model.mesh)
u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

dt = 0.01
nsteps = 5 # number of time steps
N_obs = 20  # number of observation for different time t_1----t_Nobs
DT = nsteps*dt # time steps for observation data
nobs_step = nsteps*N_obs # no of obs setp in terms of actual steps

# for writing the output at observation points
#y_true = File('y_truth.pvd')
y = model.obs(X_truth)
y_true_complete = np.zeros((N_obs, np.size(y)))

for i in range(N_obs):
    #i_N = i*nsteps
    W_truth = np.random.randn(nsteps, 4) # 4 is the problem noise  parameter
    model.run(nsteps, W_truth, X_truth, X_truth)
    y = model.obs(X_truth)

    #np.random.seed(100) # for fixing the randomness
    y_noise = np.random.normal(0.0, 0.1, 40)  

    y_obs = y + y_noise
    
    y_true_complete[i,:] = y_obs 

    y_true = np.save("y_truth.npy", y_true_complete)
    



