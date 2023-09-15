
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from scipy import stats

n_meths = 7
meths = [ \
    'Gaussian ', \
    'Lognormal', \
    'Rev logn ', \
    'G-LogNorm', \
    'G-RevLog ', \
    'All mixed', \
    'noDA     ' \
]

obs_vars = 'xy'
comp_state = 'a'

with open('./KF_L63_'+obs_vars+'.pkl','rb') as f:
    MAE_A = pickle.load(f)
    MAE_B = pickle.load(f)
    RMSE_A = pickle.load(f)
    RMSE_B = pickle.load(f)
    info = pickle.load(f)
print(info)

po_vec = info['period_obs']
vo_vec = info['var_obs']

r_A = np.empty((po_vec.size,vo_vec.size,n_meths))
r_B = np.empty((po_vec.size,vo_vec.size,n_meths))

fls = np.empty(6,dtype='object')
for n_c in range(6):
    fls[n_c] = open("./values/KF_"+meths[n_c].strip()+obs_vars+".txt",'w')

for ii in range(po_vec.size):
    for jj in range(vo_vec.size):

        r_A[ii,jj,:] = np.mean(RMSE_A[ii,jj,:,:], axis = 0)
        r_B[ii,jj,:] = np.mean(RMSE_B[ii,jj,:,:], axis = 0)
        
        for n_c in range(6):
            if n_c==0:
                fls[n_c].write(str(vo_vec[jj])+" "+str(po_vec[ii])+" "+str(r_A[ii,jj,n_c])+"\n")
            else:
                fls[n_c].write(str(vo_vec[jj])+" "+str(po_vec[ii])+" "+str(r_A[ii,jj,n_c]/r_A[ii,jj,0])+"\n")
            
    for n_c in range(6):
        fls[n_c].write("\n")

if comp_state == 'a':
    r = r_A
elif comp_state == 'b':
    r = r_B

alpha = 1-0.99**(1/po_vec.size/vo_vec.size)
alpha=1e-4
dof = 250*50*3-1
f_min = stats.f.ppf(alpha/2  ,dof,dof)
f_max = stats.f.ppf(1-alpha/2,dof,dof)
print("[f_min, f_max] = "+str(f_min)+","+str(f_max))
print("alpha   = "+str(alpha))
print("alpha^n = "+str((1-alpha)**(po_vec.size*vo_vec.size)))

fig, ax = plt.subplots(2,3,figsize=(3*(vo_vec.size-2),2*(po_vec.size)))
ii, jj = 0, 0
for n_c in range(0,6):
    if n_c == 0:
        print(np.min(r[:,:,n_c]),np.max(r[:,:,n_c]))
        cq = ax[ii,jj].pcolormesh(vo_vec,po_vec,r[:,:,n_c], cmap=plt.cm.inferno)
        cb = plt.colorbar(cq,ax=ax[ii,jj], orientation = "horizontal")
    else:
        print(np.min(r[:,:,n_c]/r[:,:,0]),np.max(r[:,:,n_c]/r[:,:,0]))
        vmin, vmax = 0.7, 1.3
        cq = ax[ii,jj].pcolormesh(vo_vec,po_vec,r[:,:,n_c]/r[:,:,0], cmap=plt.cm.bwr,vmin = vmin,vmax=vmax)
        cb = plt.colorbar(cq,ax=ax[ii,jj], orientation = "horizontal", \
            label = "RMSE/RMSE$_{\\rm g}$")

        fls[n_c].write("\n\n\n")
        fls[n_c].write("v_max,p_max\n")
        for i_p in range(po_vec.size):
            for i_v in range(vo_vec.size):
                if r[i_p,i_v,n_c]/r[i_p,i_v,0]>f_max:
                    ax[ii,jj].scatter(vo_vec[i_v],po_vec[i_p],s=100,marker='x',facecolors='k')
                    fls[n_c].write(str(vo_vec[i_v])+" "+str(po_vec[i_p])+"\n")
        fls[n_c].write("\n\n\n")
        fls[n_c].write("v_min,p_min\n")
        for i_p in range(po_vec.size):
            for i_v in range(vo_vec.size):
                if r[i_p,i_v,n_c]/r[i_p,i_v,0]<f_min:
                    ax[ii,jj].scatter(vo_vec[i_v],po_vec[i_p],s=100,marker='o',facecolors='none',edgecolors='k')
                    fls[n_c].write(str(vo_vec[i_v])+" "+str(po_vec[i_p])+"\n")

    ax[ii,jj].set_title(meths[n_c])
    ax[ii,jj].set_xlabel("Observational variance")
    ax[ii,jj].set_ylabel("Observation period")
    ax[ii,jj].set_yticks(po_vec)
    jj += 1
    if np.mod(jj,3)==0:
        jj = 0
        ii += 1

for n_c in range(6):
    fls[n_c].close()
fig.savefig("plot_KF_grid.png")


n_p = 5
n_plot = n_meths-1
fig, ax = plt.subplots((n_p),1,figsize=(7, 5*(n_p)))
for ie in range(n_p):
    with open("./values/vals_p"+str(po_vec[ie])+'.dat','w') as f:
        for n_c in range(n_plot):
            ax[ie].plot(vo_vec,r[ie,:,n_c],label=meths[n_c])
            f.write("\n")
            f.write(meths[n_c]+"\n")
            for iv in range(vo_vec.size):
                # f.write(iv,r[ie,iv,n_c],min_A[ie,iv,n_c],max_A[ie,iv,n_c])
                f.write(str(vo_vec[iv])+" "+str(r[ie,iv,n_c]))
                f.write("\n")
    ax[ie].legend()
    ax[ie].set_title("$period = $"+str(po_vec[ie]))
fig.savefig("plot_KF.png")