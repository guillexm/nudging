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


model = Euler_SD(8, 5)
model.setup()
X_truth = model.allocate()
q0 = X_truth
x = SpatialCoordinate(model.mesh)
q0.interpolate(0.1*sin(x[0])*sin(x[1]))

dt = 0.1

# Exact numerical approximation 
u = model.obs()