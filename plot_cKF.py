
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from scipy import stats

n = 10
c = 0.1
obs_vars = 'xyz'
suffix = ''
comp_state = 'a'

with open('./KF_cL63' \
    +"_n"+str(n) \
    +"_c"+str(c) \
    +"_" +obs_vars+suffix \
    +'.pkl','rb') as f:
    MAE_A = pickle.load(f)
    MAE_B = pickle.load(f)
    RMSE_A = pickle.load(f)
    RMSE_B = pickle.load(f)
    info = pickle.load(f)
    meths = info['meths']
    n_meths = len(meths)
print(info)

po_vec = info['period_obs']
vo_vec = info['var_obs']

r_A = np.empty((po_vec.size,vo_vec.size,n_meths))
r_B = np.empty((po_vec.size,vo_vec.size,n_meths))

fls = np.empty(n_meths-1,dtype='object')
for n_c in range(n_meths-1):
    fls[n_c] = open("./values/cKF_"+meths[n_c].strip()+obs_vars+".txt",'w')

for ii in range(po_vec.size):
    for jj in range(vo_vec.size):

        r_A[ii,jj,:] = np.mean(RMSE_A[ii,jj,:,:], axis = 0)
        r_B[ii,jj,:] = np.mean(RMSE_B[ii,jj,:,:], axis = 0)

        for n_c in range(n_meths-1):
            if n_c==0:
                fls[n_c].write(str(vo_vec[jj])+" "+str(po_vec[ii])+" "+str(r_A[ii,jj,n_c])+"\n")
            else:
                fls[n_c].write(str(vo_vec[jj])+" "+str(po_vec[ii])+" "+str(r_A[ii,jj,n_c]/r_A[ii,jj,0])+"\n")
            
    for n_c in range(n_meths-1):
        fls[n_c].write("\n")


if comp_state == 'a':
    r = r_A
elif comp_state == 'b':
    r = r_B

alpha = 1-0.99**(1/po_vec.size/vo_vec.size)
alpha=1e-4
dof = 250*50*30-1
f_min = stats.f.ppf(alpha/2  ,dof,dof)
f_max = stats.f.ppf(1-alpha/2,dof,dof)
print("[f_min, f_max] = "+str(f_min)+","+str(f_max))
print("alpha   = "+str(alpha))
print("alpha^n = "+str((1-alpha)**(po_vec.size*vo_vec.size)))

fig, ax = plt.subplots(1,(n_meths-1),figsize=((n_meths-1)*(vo_vec.size-2),(po_vec.size)*.9))
for n_c in range(0,n_meths-1):
    if n_c == 0:
        print(np.min(r[:,:,n_c]),np.max(r[:,:,n_c]))
        cq = ax[n_c].pcolormesh(vo_vec,po_vec,r[:,:,n_c], cmap=plt.cm.inferno)
        cb = plt.colorbar(cq,ax=ax[n_c], orientation = "horizontal")
    else:
        print(np.min(r[:,:,n_c]/r[:,:,0]),np.max(r[:,:,n_c]/r[:,:,0]))
        vmin, vmax = 0.9, 1.1
        cq = ax[n_c].pcolormesh(vo_vec,po_vec,r[:,:,n_c]/r[:,:,0], cmap=plt.cm.bwr,vmin = vmin,vmax=vmax)
        cb = plt.colorbar(cq,ax=ax[n_c], orientation = "horizontal", \
            label = "RMSE/RMSE$_{\\rm g}$")
        
        fls[n_c].write("\n\n\n")
        fls[n_c].write("v_max,p_max\n")
        for i_p in range(po_vec.size):
            for i_v in range(vo_vec.size):
                if r[i_p,i_v,n_c]/r[i_p,i_v,0]>f_max:
                    ax[n_c].scatter(vo_vec[i_v],po_vec[i_p],s=100,marker='x',facecolors='k')
                    fls[n_c].write(str(vo_vec[i_v])+" "+str(po_vec[i_p])+"\n")
        fls[n_c].write("\n\n\n")
        fls[n_c].write("v_min,p_min\n")
        for i_p in range(po_vec.size):
            for i_v in range(vo_vec.size):
                if r[i_p,i_v,n_c]/r[i_p,i_v,0]<f_min:
                    ax[n_c].scatter(vo_vec[i_v],po_vec[i_p],s=100,marker='o',facecolors='none',edgecolors='k')
                    fls[n_c].write(str(vo_vec[i_v])+" "+str(po_vec[i_p])+"\n")

    ax[n_c].set_title(meths[n_c])
    ax[n_c].set_xlabel("Observational variance")
    ax[n_c].set_ylabel("Observation period")
    ax[n_c].set_yticks(po_vec)

for n_c in range(n_meths-1):
    fls[n_c].close()
fig.savefig("plot_cKF.png")