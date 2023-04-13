import numpy as np
import matplotlib.pyplot as plt


mapped_values = [0]
# Loop through the values 0 to 29
for i in range(1,30):

    # Map each value to the corresponding value
    j =  (i)*5 - 1
    
    # Append the mapped value to the list
    mapped_values.append(j)
#print(len(mapped_values))

y_exct = np.load('y_true.npy')                                          

y = np.load('y_obs.npy')                                                

y_e = np.load('assimilated_ensemble.npy')
y_obs_alltime = np.load('simualated_all_time_obs.npy') 



# y_e_fwd = np.load('ensemble_forward_obs.npy')
# y_e_asmfwd = np.load('final_ensemble_forward.npy')   
#print(np.array(y_e-y_e_fwd))

n_ensemble = np.shape(y_e)[0]  


                     
y_alltime = np.transpose(y_obs_alltime, (1,0,2))
y_e_tr_obs = np.transpose(y_e, (1,0,2))# use while plotiing against N_obs
y_e_trans_spatial = np.transpose(y_e, (1,2,0)) # use while plotiing against X

y_e_reshape = np.zeros((y_e_tr_obs.shape[0]*5, y_e_tr_obs.shape[1], y_e_tr_obs.shape[2]))
for j in range(len(mapped_values)):
    #print(j,mapped_values[j])
    y_e_reshape[mapped_values[j]] = y_e_tr_obs[j]


#print(y_e_reshape)

# y_e_fwdtr_obs = np.transpose(y_e_fwd, (1,0,2))# use while plotiing against N_obs
# y_e_fwdtrans_spatial = np.transpose(y_e_fwd, (1,2,0)) # use while plotiing against X

# y_e_asmfwdtr_obs = np.transpose(y_e_asmfwd, (1,0,2))# use while plotiing against N_obs
# y_e_asmfwdtrans_spatial = np.transpose(y_e_asmfwd, (1,2,0)) # use while plotiing against X


xi =39
y_e_mean_obs = np.mean(y_e_tr_obs[:,:,xi], axis=1)
# against N_obs at Xi =20
# plt.plot(y_exct[:,xi], 'r-', label='True')
# # #plt.plot(y[:,xi], 'b-', label='y_obs_+_noise')
# #plt.plot(y_e_fwdtr_obs[:,:,xi], 'y-')
# # plt.plot(y_e_mean_obs, 'b-', label='ensemble mean')
plt.plot(y_alltime[:,:,xi], 'g-')
#plt.plot(y_e_reshape[mapped_values,:,xi])
# plt.title('Ensemble prediction with N_particles = ' +str(n_ensemble)+' and station view  = '+str(xi)) 
# #plt.legend()
# plt.xlabel("Assimialtion time")
plt.show()



# N_obs= 20
# y_e_mean_spatial = np.mean(y_e_trans_spatial[N_obs,:,:], axis=1) # use while plotiing against X
# #Against X at N_obs=2
# plt.plot(y_exct[N_obs,:], 'r-', label = 'True')                          
# #plt.plot(y[N_obs,:], 'b-', label = 'y_obs')                   
# #plt.plot(y_e_mean_spatial, 'b-', label='ensemble mean')
# #plt.plot(y_e_trans_spatial[N_obs,:], 'y-')
# #plt.plot(y_e_fwdtrans_spatial[N_obs,:], '-.')
# plt.plot(y_e_asmfwdtrans_spatial[N_obs,:], '--')
# plt.title('Ensemble prediction with N_particles = ' +str(n_ensemble)+' and  assimailation time  = '+str(N_obs))  
# plt.legend()
# plt.xlabel("discrete points")
# plt.show()
                      



