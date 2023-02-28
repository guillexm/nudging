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
MALA = True

jtfilter = jittertemp_filter(n_temp=4, n_jitt = 4, rho= 0.4,
                            verbose=False, MALA=MALA)


nensemble = [5,5,
             5,5]
jtfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

#randomness of the initial data
dx0 = Constant(0.)
dx1 = Constant(0.)
a = Constant(0.)
b = Constant(0.)

for i in range(nensemble[jtfilter.ensemble_rank]):
    dx0.assign(np.random.randn())
    dx1.assign(np.random.randn())
    a.assign(np.random.rand())
    b.assign(np.random.rand())
    u0_exp = (1+a)*0.2*2/(exp(x-403./15. + dx0) + exp(-x+403./15. + dx1)) \
        + (1+b)*0.5*2/(exp(x-203./15. + dx0)+exp(-x+203./15. + dx1))

    _, u = jtfilter.ensemble[i][0].split()
    u.interpolate(u0_exp)

def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*dx
    return ll
    
#Load data
y_exact = np.load('y_true.npy')
y = np.load('y_obs.npy') 
N_obs = y.shape[0]

y_e = np.zeros((N_obs, sum(nensemble), y.shape[1]))
y_e_mean = np.zeros((N_obs, sum(nensemble)))

yVOM = Function(model.VVOM)

# do assimiliation step
y_e_lst = []
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[k, :]
    jtfilter.assimilation_step(yVOM, log_likelihood)

    y_e_list_arr = []
    for m in range(y.shape[1]):
        
        y_e_shared_arr = SharedArray(partition=nensemble, 
                                      comm=jtfilter.subcommunicators.ensemble_comm)
        for i in range(nensemble[jtfilter.ensemble_rank]):
            model.w0.assign(jtfilter.ensemble[i][0])
            y_e_shared_arr.dlocal[i] = model.obs().dat.data[m]
            y_e_shared_arr.synchronise()
        y_e_list_arr.append(y_e_shared_arr.data())
    y_e_lst.append(y_e_list_arr)

y_e_fin_arr = np.array(y_e_lst)
y_e = np.transpose(y_e_fin_arr, (0,2,1))
y_e_mean = np.mean(y_e[:,:,20], axis=1)
print(y_e.shape)

if jtfilter.ensemble_rank == 0:
   
    plt.plot(y_exact[:,20], 'r-', label='y_obs')
    #plt.plot(y[:,20], 'b-', label='y_obs_+_noise')
    plt.plot(y_e[:,:,20], 'y-')
    plt.plot(y_e_mean, 'g-', label='y_ensemble_mean')
    plt.title('Ensemble prediction with N_ensemble = ' +str(sum(nensemble)))
    plt.legend()
    plt.show()
