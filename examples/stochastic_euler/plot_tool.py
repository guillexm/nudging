import numpy as np

import matplotlib.pyplot as plt


# True data
u1_true_data = np.load('u1_true_data.npy')
u2_true_data = np.load('u2_true_data.npy')
#print(np.shape(u1_true_data))
#print(u1_true_data-u2_true_data)


# Noisy data
u1_obs_data = np.load('u1_obs_data.npy')
u2_obs_data = np.load('u2_obs_data.npy')
#print(np.shape(u1_obs_data))
#print(u1_all_obs)

# Ensemble member
u1_e_data = np.load('Velocity_1_ensemble_simulated_obs.npy') 
u2_e_data = np.load('Velocity_2_ensemble_simulated_obs.npy')
n_ensemble = np.shape(u2_e_data)[0] 
print(np.shape(u1_e_data)) 
u1_e_obs_obs = np.transpose(u1_e_data, (1,0,2))# use while plotiing against N_obs
u2_e_obs_obs = np.transpose(u2_e_data, (1,0,2))# use while plotiing against N_obs

print(np.shape(u1_e_obs_obs)) 
u1_e_obs_spatial = np.transpose(u1_e_data, (1,2,0)) # use while plotiing against X
u2_e_obs_spatial = np.transpose(u2_e_data, (1,2,0)) # use while plotiing against X

print(np.shape(u2_e_obs_spatial)) 


# # Plotting data
xi = 20
#calculate 4 quantiles
u1_e_obs_mean = np.mean(u1_e_obs_obs[:,:,xi],  axis=1)
u2_e_obs_mean = np.mean(u2_e_obs_obs[:,:,xi],  axis=1)
# q = 0
u1_e_obs_q0 = np.quantile(u1_e_obs_obs[:,:,xi], 0, axis=1, keepdims=True )
u2_e_obs_q0 = np.quantile(u2_e_obs_obs[:,:,xi], 0, axis=1, keepdims=True )
# # For component 1
# f1,  a1 = plt.subplots() 
# a1.plot(u1_true_data[:,xi], 'r-', label = 'exact_1')
# a1.plot(u1_obs_data[:,xi], 'b-', label = 'exact_1 + noise')
# a1.plot(u1_e_obs_obs[:,:,xi], 'y-')
# f1.suptitle('First velocity component with N_ensemble = ' +str(n_ensemble)+' and  x_i = '+str(xi)) 
# a1.legend(loc='lower left')
# f1.savefig('first_component_spatial.png')



# # For component 2
# f2,  a2 = plt.subplots() 
# a2.plot(u2_true_data[:,xi], 'r-', label = 'exact_2')
# a2.plot(u2_obs_data[:,xi], 'b-', label = 'exact_2 + noise')
# a2.plot(u2_e_obs_obs[:,:,xi], 'y-')
# f2.suptitle('Second velocity component with N_ensemble = ' +str(n_ensemble)+' and  x_i = '+str(xi)) 
# a2.legend(loc='lower left')
# f2.savefig('second_component_spatial.png')


Obs_N = 4
u1_e_spatial_mean = np.mean(u1_e_obs_spatial[Obs_N, :,:], axis=1)
u2_e_spatial_mean = np.mean(u2_e_obs_spatial[Obs_N, :,:], axis=1)

# caculate 4 quantlies
# q =1
u1_e_spatial_q1 = np.quantile(u1_e_obs_spatial[Obs_N, :,:], 0, axis=1, keepdims=True)
u2_e_spatial_q1 = np.quantile(u2_e_obs_spatial[Obs_N, :,:], 0, axis=1, keepdims=True)
# q =2
u1_e_spatial_q2 = np.quantile(u1_e_obs_spatial[Obs_N, :,:], 0.5, axis=1, keepdims=True)
u2_e_spatial_q2 = np.quantile(u2_e_obs_spatial[Obs_N, :,:], 0.5, axis=1, keepdims=True)
# q =3
u1_e_spatial_q3 = np.quantile(u1_e_obs_spatial[Obs_N, :,:], 0.75, axis=1, keepdims=True)
u2_e_spatial_q3 = np.quantile(u2_e_obs_spatial[Obs_N, :,:], 0.75, axis=1, keepdims=True)
# q =4
u1_e_spatial_q4 = np.quantile(u1_e_obs_spatial[Obs_N, :,:], 1.0, axis=1, keepdims=True)
u2_e_spatial_q4 = np.quantile(u2_e_obs_spatial[Obs_N, :,:], 1.0, axis=1, keepdims=True)


# #For component 1
fx1,  ax1 = plt.subplots() 
ax1.plot(u1_true_data[Obs_N,:], 'b-', label = 'exact_1')
ax1.plot(u1_obs_data[Obs_N,:], 'g-', label = 'exact_1 + noise')
#ax1.plot(u1_e_spatial_mean, 'y-',  label = 'ensemble_1_mean')
ax1.plot(u1_e_spatial_q1, '-o',  label = 'ensemble_1_quantile_1')
ax1.plot(u1_e_spatial_q2, '-+',  label = 'ensemble_1_quantile_2')
ax1.plot(u1_e_spatial_q3, '-.',  label = 'ensemble_1_quantile_3')
ax1.plot(u1_e_spatial_q4, '-*',  label = 'ensemble_1_quantile_4')
ax1.plot(u1_e_obs_spatial[Obs_N,:,:], 'y-')
fx1.suptitle('For u_1 with N_ensemble = ' +str(n_ensemble)+' and  assimilation step = '+str(Obs_N))  
ax1.legend(loc='upper right')
fx1.savefig('first_component_obs.png')


#For component 2
fx2,  ax2 = plt.subplots() 
ax2.plot(u2_true_data[Obs_N,:], 'b-', label = 'exact_2')
ax2.plot(u2_obs_data[Obs_N,:], 'g-', label = 'exact_2 + noise')
#ax2.plot(u2_e_spatial_mean, 'g-',  label = 'ensemble_2_mean')
ax2.plot(u2_e_spatial_q1, '-o',  label = 'ensemble_2_quantile_1')
ax2.plot(u2_e_spatial_q2, '-+',  label = 'ensemble_2_quantile_2')
ax2.plot(u2_e_spatial_q3, '-.',  label = 'ensemble_2_quantile_3')
ax2.plot(u2_e_spatial_q4, '-*',  label = 'ensemble_2_quantile_4')
ax2.plot(u2_e_obs_spatial[Obs_N,:,:], 'y-')
fx2.suptitle('For u_2 with N_ensemble = ' +str(n_ensemble)+' and  assimilation step = '+str(Obs_N))  
ax2.legend(loc='upper right')
fx2.savefig('second_component_obs.png')



plt.show()