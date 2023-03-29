import numpy as np
import matplotlib.pyplot as plt


y_exct = np.load('y_true.npy')                                          

y = np.load('y_obs.npy')                                                

y_e = np.load('ensemble_simulated_obs.npy')  
y_e_fwd = np.load('ensemble_forward_obs.npy')
y_e_asmfwd = np.load('final_ensemble_forward.npy')   
#print(np.array(y_e-y_e_fwd))

n_ensemble = np.shape(y_e)[0]  
                     

y_e_tr_obs = np.transpose(y_e, (1,0,2))# use while plotiing against N_obs
y_e_trans_spatial = np.transpose(y_e, (1,2,0)) # use while plotiing against X

y_e_fwdtr_obs = np.transpose(y_e_fwd, (1,0,2))# use while plotiing against N_obs
y_e_fwdtrans_spatial = np.transpose(y_e_fwd, (1,2,0)) # use while plotiing against X

y_e_asmfwdtr_obs = np.transpose(y_e_asmfwd, (1,0,2))# use while plotiing against N_obs
y_e_asmfwdtrans_spatial = np.transpose(y_e_asmfwd, (1,2,0)) # use while plotiing against X

# print(np.shape(y_e_tr_obs))
# y_combo = np.vstack((y_e_asmfwdtr_obs, y_e_tr_obs))
# y_combo_new = np.zeros_like(y_combo)

# y_combo_new[0]=y_combo[0]
# y_combo_new[1]=y_combo[5]
# y_combo_new[2]=y_combo[1]
# y_combo_new[3]=y_combo[6]
# y_combo_new[4]=y_combo[2]
# y_combo_new[5]=y_combo[7]
# y_combo_new[6]=y_combo[3]
# y_combo_new[7]=y_combo[8]
# y_combo_new[8]=y_combo[4]
# y_combo_new[9]=y_combo[9]

#y_combo[1],y_combo[2]  = y_combo[2], y_combo[1]

#y_combo = np.vstack((y_e_fwdtr_obs, y_e_tr_obs)).reshape((-1, 2), order='F')
# print(np.shape(y_combo_new))
# print("Fwd", np.array(y_e_fwdtr_obs))
# print("Asm", np.array(y_e_tr_obs))
# print("Comb", np.array(y_combo))
# print("New Comb", np.array(y_combo_new))

xi =19
y_e_mean_obs = np.mean(y_e_tr_obs[:,:,xi], axis=1)
# against N_obs at Xi =20
plt.plot(y_exct[:,xi], 'r-', label='True')
# #plt.plot(y[:,xi], 'b-', label='y_obs_+_noise')
#plt.plot(y_e_fwdtr_obs[:,:,xi], 'y-')
plt.plot(y_e_mean_obs, 'b-', label='ensemble mean')
plt.plot(y_e_tr_obs[:,:,xi], '.')
# plt.plot(y_e_asmfwdtr_obs[:,:,xi], 'y-')
####plt.plot(y_combo_new[:,:,xi], 'g-')
plt.title('Ensemble prediction with N_particles = ' +str(n_ensemble)+' and station view  = '+str(xi)) 
#plt.legend()
plt.xlabel("Assimialtion time")
plt.show()



# N_obs= 2
# y_e_mean_spatial = np.mean(y_e_trans_spatial[N_obs,:,:], axis=1) # use while plotiing against X
# #Against X at N_obs=2
# plt.plot(y_exct[N_obs,:], 'r-', label = 'True')                          
# #plt.plot(y[N_obs,:], 'b-', label = 'y_obs')                   
# #plt.plot(y_e_mean_spatial, 'b-', label='ensemble mean')
# plt.plot(y_e_trans_spatial[N_obs,:], 'y-')
# #plt.plot(y_e_fwdtrans_spatial[N_obs,:], '-.')
# #plt.plot(y_e_asmfwdtrans_spatial[N_obs,:], 'g-')
# plt.title('Ensemble prediction with N_particles = ' +str(n_ensemble)+' and  assimailation time  = '+str(N_obs))  
# plt.legend()
# plt.xlabel("discrete points")
# plt.show()
                      



