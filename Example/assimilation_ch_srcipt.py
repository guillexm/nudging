from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt


""" read obs from saved file 
    Do assimilation step
"""





# For mesh generation
model = Camsholm(100)
x, = SpatialCoordinate(model.mesh) 

# bootstrap filter
#bfilter = bootstrap_filter(5, (5, 4))
bfilter = jittertemp_filter(5, (5, 4), n_temp=5, n_jitt=5, rho=0.46)
nensemble = 10
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


    # used perturbed solution as initial data for update  ensemble[i] via run/observation method
    _, u = bfilter.ensemble[i].split()
    u.interpolate(u0_exp)  

def log_likelihood(dY):
    return np.dot(dY, dY)/0.1**2

#print([len(a) for a in bfilter.ensemble])
#Load data
N_obs = 5
# true data  load
y_exact = np.load('y_true.npy')
plt.plot(y_exact[:,10], 'r-', label='y_true')
# synthetic data load
y = np.load('y_obs.npy') 
#plt.plot(y[:,10], 'b-', label='y_obs')

# plt.show()

#y_obs_rows = iter(y_obsdata)
#print(y[0,:])
y_e = np.zeros((N_obs, nensemble, y.shape[1])) # y_ensemble
y_e_mean = np.zeros((N_obs, nensemble))
#y_e_var = np.zeros((N_obs, nensemble))

print(y_e.shape)

# do assimiliation step
for k in range(N_obs):
    bfilter.assimilation_step(y[k,:], log_likelihood)
    #print(bfilter.ess)
    #print(bfilter.e_weight, bfilter.e_oldweight)
    #print(bfilter.e_weight)
    
    #print(bfilter.e_s)
    #print("New copies: ", bfilter.e_copies)
    #print("L_new: ", bfilter.e_L)
    for e in range(nensemble):
        y_e[k,e,:] = model.obs(bfilter.ensemble[e])
        #y_e_mean = np.mean(y_e[k,e,10])

y_e_mean = np.mean(y_e[:,:,10], axis=1)
#y_e_var = np.var(y_e[:,:,10])

#print(y_e_mean)
plt.plot(y_e[:,:,10], 'y-')
plt.plot(y_e_mean, 'g-', label='y_ensemble_mean')
#plt.plot(y_e_var, 'r-', label='y_ensemble_std')
plt.legend()
plt.title('Ensemble prediction with N_ensemble = ' +str(nensemble))
plt.show()



# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(k, e, y_e[:,:,20])
# plt.show()


