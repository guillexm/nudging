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


bfilter = bootstrap_filter()


nensemble = [2,2,2,2]
bfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

dx0 = Constant(0.)
dx1 = Constant(0.)
a = Constant(0.)
b = Constant(0.)

for i in range(nensemble[bfilter.ensemble_rank]):
    dx0.assign(np.random.randn())
    dx1.assign(np.random.randn())
    a.assign(np.random.rand())
    b.assign(np.random.rand())
    u0_exp = (1+a)*0.2*2/(exp(x-403./15. + dx0) + exp(-x+403./15. + dx1)) \
        + (1+b)*0.5*2/(exp(x-203./15. + dx0)+exp(-x+203./15. + dx1))

    _, u = bfilter.ensemble[i][0].split()
    u.interpolate(u0_exp)

def log_likelihood(dY):
    # dY is vertex only mesh data
    return dY**2/0.05**2/2*dx
    
#Load data
y_exact = np.load('y_true.npy')

y = np.load('y_obs.npy') 

N_obs = y.shape[0]

y_e = np.zeros((N_obs, sum(nensemble), y.shape[1]))
y_e_mean = np.zeros((N_obs, sum(nensemble)))

# do assimiliation step

obs_arr = SharedArray(partition=nensemble, 
                                      comm=bfilter.subcommunicators.ensemble_comm)

#obs_nparray = np.zeros((y.shape[1]))
#for i in range(y.shape[1]):
#for i in range((bfilter.ensemble_rank)):
       
    
print("==================Rank================", bfilter.ensemble_rank)

obs_lst = []

yVOM = Function(model.VOM)

for k in range(N_obs):
   
    #PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[k, :]
    bfilter.assimilation_step(yVOM, log_likelihood)
    
    obs_list_arr = []
    for m in range(y.shape[1]):
        
        y_e_shared_arr = SharedArray(partition=nensemble, 
                                      comm=bfilter.subcommunicators.ensemble_comm)
        for i in range(nensemble[bfilter.ensemble_rank]):
            obs_arr.dlocal[i] = model.obs(bfilter.ensemble[i]).dat.data[m]
            obs_arr.synchronise()
        obs_list_arr.append(obs_arr.data())
    obs_lst.append(obs_list_arr)


            
           
obs_fin_arr = np.array(obs_lst)
        
        
#obs_nparr = np.array(obs_lst)
#obs_nparr.synchronise(root=0)
    
    
#print(obs_arr.data())
#print(obs_nparr.shape)
print(obs_fin_arr.shape)
#print(obs_fin_arr)

if bfilter.ensemble_rank == 3:
    y_e = np.transpose(obs_fin_arr, (0,2,1))
    y_e_mean = np.mean(y_e[:,:,20], axis=1)
    print(y_e.shape)

if bfilter.ensemble_rank == 0:
    plt.plot(y_exact[:,20], 'r-', label='y_obs')
    plt.plot(y[:,20], 'b-', label='y_obs_+_noise')
    plt.plot(y_e[:,:,20], 'y-')
    plt.plot(y_e_mean, 'g-', label='y_ensemble_mean')
    plt.title('Ensemble prediction with N_ensemble = ' +str(sum(nensemble)))
    plt.legend()
    plt.show()
