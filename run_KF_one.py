""" 
Run a nongaussian Kalman Filter twin experiment with Lorenz-63 
"""

# Load modules
import numpy as np 
import pickle
import mod_KalmanDA as da
import LorenzModels as LM
import matplotlib.pyplot as plt
import tikzplotlib as tpl


# Options for DA run
n_wind = 25                                # Number of DA windows for each run
obs_vars = 'xyz'                            # Observed variables, either 'xy', or 'xyz'
ml_method = 'array'                         # Nongaussian decision function, either  
                                            #   'array',    selection based on observations before DA run
                                            #   'function', selection from decision function every DA window
seed = 3405                                 # Seed for random number generator, can be set to None
SV_init = [-5.0,-6.0,22.0]                  # Initial values of the Lorenz-63 run


###########################################################################################
###########################################################################################
###########################################################################################

def one_run():
    global n_wind
    global period_obs
    global obs_vars
    global var_obs
    global ml_method
    global ml_file
    global seed
    global SV_init

    # Load the machine learning model
    with open(ml_file,'rb') as f:
        clf = pickle.load(f)
        scaler = pickle.load(f)
        info = pickle.load(f)
        p = info['p']               # Parameters of L63, typically (10,28,8/3)
        dt = info['dt']             # Time step of L63
    rng = np.random.default_rng(seed)
    SV_init += rng.standard_normal(3)

    # Nature run
    t_max = n_wind*period_obs*dt        # End time of the Lorenz-63 run
    t = np.arange(0.0, t_max, dt)  # Evaluation time
    SV = LM.sol_L63(t, SV_init, p)
    xi_SV = np.nanmax(SV[2,:])+5.0      # Parameter of the reverse lognormal distribution


    ###########################################################################################
    ###########################################################################################
    ###########################################################################################


    # Create observation operator
    n_SV = 3
    if obs_vars == 'xyz':
        n_obs = 3
        obs_h = np.eye(n_SV)
    if obs_vars == 'xy':
        n_obs = 2
        obs_h = np.zeros((n_obs,n_SV))
        obs_h[0,0] = 1.0
        obs_h[1,1] = 1.0

    # Predict distribution from which to sample z-variable
    if obs_vars == 'xyz':
        # Predict distribution from nature run
        X_data = scaler.transform(np.transpose(SV[:2,:]))
        z_pred = clf.predict(X_data)
        # Define observation times when lognormal or reverse lognormal noise should be added
        ln_vars_ML = []
        rl_vars_ML = []
        for ii in range(t.size):
            if ii % period_obs == 0:
                if z_pred[ii] > 0.5: # Lognormal distribution predicted
                    ln_vars_ML.append([2])
                    rl_vars_ML.append([])
                elif z_pred[ii] < -0.5: # Reverse lognormal distribution predicted
                    ln_vars_ML.append([])
                    rl_vars_ML.append([2])
                else: # Gaussian distribution predicted
                    ln_vars_ML.append([])
                    rl_vars_ML.append([])
    if obs_vars == 'xy':
        # If z is not observed, ML model not necessary
        ln_vars_ML = []
        rl_vars_ML = []

    # Generate observations
    t_obs, y, R = da.gen_obs(t, SV, period_obs, obs_h, var_obs, \
        ln_vars = ln_vars_ML, rl_vars = rl_vars_ML, xi_obs = xi_SV, seed = seed, sample='mode')



    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    # DA initial guess
    init_guess = SV_init + rng.standard_normal(3)

    # Model function, of the form x = model(t,x_0)
    def model_l63(t, x0):
        y = LM.sol_L63(t, x0, p)
        return y

    # Decision function for lognormal state variables (for a single prediction)
    def f_ln_vars(SV):
        X = scaler.transform(SV[:2].reshape(1,-1))
        z_pred = clf.predict(X)
        if z_pred > 0.5:
            return [2]
        else:
            return []
    # Decision function for reverse-lognormal state variables (for a single prediction)
    def f_rl_vars(SV):
        X = scaler.transform(SV[:2].reshape(1,-1))
        z_pred = clf.predict(X)
        if z_pred < -0.5:
            return [2]
        else:
            return []

    # Create decision functions/arrays
    if ml_method == 'array':
        ln_vars = []
        rl_vars = []
        for ii in range(t_obs.size):
            ln_vars.append(f_ln_vars(y[:2,ii]))
            rl_vars.append(f_rl_vars(y[:2,ii]))
        ln_B0 = ln_vars[0]
        rl_B0 = rl_vars[0]
    if ml_method == 'function':
        ln_vars = f_ln_vars
        rl_vars = f_rl_vars
        ln_B0 = f_ln_vars(init_guess)
        rl_B0 = f_rl_vars(init_guess)

    # Store different decision functions for different (non)gaussian methods
    ln_vars_SV_OBJ = np.array([ \
        [],      # Gaussian for all t
        [2],     # Lognormal for all t
        [],      # Reverse Lognormal for all t
        ln_vars, # Gaussian - Lognormal ML
        [],      # Gaussian - Reverse Lognormal ML
        ln_vars, # Gaussian - Lognormal - Reverse Lognormal ML
        [],      # No DA (background run)
    ], dtype = object)
    rl_vars_SV_OBJ = np.array([ \
        [],      # Gaussian for all t
        [],      # Lognormal for all t
        [2],     # Reverse Lognormal for all t
        [],      # Gaussian - Lognormal ML
        rl_vars, # Gaussian - Reverse Lognormal ML
        rl_vars, # Gaussian - Lognormal - Reverse Lognormal ML
        [],      # No DA (background run)
    ], dtype = object)
    if obs_vars == 'xyz':
        ln_vars_obs_OBJ = ln_vars_SV_OBJ
        rl_vars_obs_OBJ = rl_vars_SV_OBJ
    elif obs_vars == 'xy':
        ln_vars_obs_OBJ = np.array([[],[],[],[],[],[],[-1.0]], dtype = object)
        rl_vars_obs_OBJ = np.array([[],[],[],[],[],[],[-1.0]], dtype = object)

    # Initialize analysis and background state variables
    n_meths = 7
    n_t_obs = t_obs.size           # number of observations
    n_t = n_t_obs * period_obs + 1 # number of total time steps
    X_A = np.empty((n_meths, n_SV, n_t_obs))
    X_B = np.empty((n_meths, n_SV, n_t))

    # Model error covariance matrix
    Q = np.array([ \
        [0.1491, 0.1505, 0.0007], \
        [0.1505, 0.9048, 0.0014], \
        [0.0007, 0.0014, 0.9180] \
    ])

    ln_vars_B_OBJ = np.array([ [], [2], [], ln_B0, [], ln_B0, []], dtype = object)
    rl_vars_B_OBJ = np.array([ [], [], [2], [], rl_B0, rl_B0, []], dtype = object)

    for iM in range(n_meths-1):
        # Create initial background error covariance matrix
        B_case = 'nmc'
        P_a = da.create_B_init(B_case, \
            SV_init = SV_init, \
            p = p, \
            seed = seed, \
            ln_vars=ln_vars_B_OBJ[iM], rl_vars=rl_vars_B_OBJ[iM])

        X_A[iM,:,:], X_B[iM,:,:], _, _ = da.kalman_filter( \
            init_guess, t_obs, period_obs, y, obs_h, P_a, R, Q, \
            model_l63, \
            ln_vars_SV = ln_vars_SV_OBJ[iM], ln_vars_obs = ln_vars_obs_OBJ[iM], \
            rl_vars_SV = rl_vars_SV_OBJ[iM], rl_vars_obs = rl_vars_obs_OBJ[iM], \
            xi_SV = xi_SV, xi_obs = xi_SV \
        )

    # Background run
    iM += 1
    dt_obs = t_obs[1] - t_obs[0] # Assuming evenly spaced observations
    t_true = np.linspace(t_obs[0],t_obs[-1] + dt_obs, n_t) # Assuming one prediction window
    X_B[iM,:,:] = model_l63(t_true, init_guess)
    X_A[iM,:,:] = X_B[iM,:,:-1:period_obs]

    MAE_A = np.empty(n_meths)
    MAE_B = np.empty(n_meths)
    for iM in range(n_meths):
        MAE_A[iM] = np.nanmean(np.abs((SV[:,::period_obs] - X_A[iM,:,:])))
        MAE_B[iM] = np.nanmean(np.abs((SV - X_B[iM,:,:-1])))

    return MAE_A, MAE_B, t_true, t_obs, X_A, X_B, y, SV, ln_vars_ML, rl_vars_ML, ln_vars, rl_vars

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
meths = [ \
    'Gaussian ', \
    'Lognormal', \
    'Rev logn ', \
    'G-LogNorm', \
    'G-RevLog ', \
    'All mixed', \
    'noDA     ' \
]
n_meths = len(meths)
ml_file = './kNN_l63_w14_s1.0.pkl'    # Location of the trained ML model

period_obs = 40
var_obs = 3.0

MAE_A, MAE_B, t, t_obs, X_A, X_B, y, SV, ln_vars_ML, rl_vars_ML, ln_vars, rl_vars = one_run()

print("Finished p = "+str(period_obs)+", s = "+str(var_obs))
for iM in range(n_meths):
    print(meths[iM]+ \
        ': mean(r_a) = '+format(np.round(MAE_A[iM],3),".3f")+ \
        ', mean(r_b) = '+format(np.round(MAE_B[iM],3),".3f"))

y_gs = np.full(y.shape,np.nan)
y_ln = np.full(y.shape,np.nan)
y_rl = np.full(y.shape,np.nan)
for jj in range(t_obs.size):
    if ln_vars_ML[jj]:
        y_ln[:,jj] = y[:,jj]
    elif rl_vars_ML[jj]:
        y_rl[:,jj] = y[:,jj]
    else:
        y_gs[:,jj] = y[:,jj]
y_gs_DA = np.full(y.shape,np.nan)
y_ln_DA = np.full(y.shape,np.nan)
y_rl_DA = np.full(y.shape,np.nan)
for jj in range(t_obs.size):
    if ln_vars[jj]:
        y_ln_DA[:,jj] = y[:,jj]
    elif rl_vars[jj]:
        y_rl_DA[:,jj] = y[:,jj]
    else:
        y_gs_DA[:,jj] = y[:,jj]

# n_plot = n_meths-1
n_plot = 2
fig, ax = plt.subplots(n_plot,3,figsize=(15, 5))
vars = ['$x$', '$y$', '$z$']
meth_select = [0,5]
for iMeth in range(n_plot):
    for jj in range(3):
        # True state
        ax[iMeth,jj].plot(t[:-1],SV[jj,:],label = 'Nature')
    
        # Background state
        ax[iMeth,jj].plot(t[:-1],X_B[meth_select[iMeth],jj,:-1],'--',label = 'Background')

        # Observations
        if not (jj==2 and obs_vars == 'xy'):
            ax[iMeth,jj].plot(t_obs,y_gs[jj,:],'.',label = 'Gs Observations')
            ax[iMeth,jj].plot(t_obs,y_ln[jj,:],'.',label = 'LN Observations')
            ax[iMeth,jj].plot(t_obs,y_rl[jj,:],'.',label = 'RL Observations')
        if not (jj==2 and obs_vars == 'xy'):
            ax[iMeth,jj].scatter(t_obs,y_gs_DA[jj,:],s=50, facecolors='none',edgecolors='#2ca02c',label = 'Gs ML')
            ax[iMeth,jj].scatter(t_obs,y_ln_DA[jj,:],s=50, facecolors='none',edgecolors='#d62728',label = 'LN ML')
            ax[iMeth,jj].scatter(t_obs,y_rl_DA[jj,:],s=50, facecolors='none',edgecolors='#9467bd',label = 'RL ML')

        # # Analysis
        # ax[iMeth,jj].plot(t_obs,X_A[meth_select[iMeth],jj,:],'.',label = 'Analysis')

        ax[iMeth,jj].set_xlabel('$t$')
        ax[iMeth,jj].set_ylabel(vars[jj])
    ax[iMeth,0].set_title(meths[meth_select[iMeth]])
    ax[iMeth,0].set_ylim(-25,25)
    ax[iMeth,1].set_ylim(-30,30)
    ax[iMeth,2].set_ylim(0,50)
ax[0,0].legend()
# tpl.clean_figure()
# tpl.save("KF_ML_example.tex")
plt.show()



# Write to files
vars = ['x', 'y', 'z']
for var in range(3):
    with open("./values/"+vars[var]+"_true.dat",'w') as f:
        for ii in range(t[:-1].size):
            f.write(str(t[ii])+" "+str(SV[var,ii])+"\n")
    with open("./values/"+vars[var]+"_b_gs.dat",'w') as f:
        for ii in range(t[:-1].size):
            f.write(str(t[ii])+" "+str(X_B[0,var,ii])+"\n")
    with open("./values/"+vars[var]+"_b_ml.dat",'w') as f:
        for ii in range(t[:-1].size):
            f.write(str(t[ii])+" "+str(X_B[5,var,ii])+"\n")


    with open("./values/"+vars[var]+"_obs_gs.dat",'w') as f:
        for ii in range(1,t_obs.size):
            f.write(str(t_obs[ii])+" "+str(y_gs[var,ii])+"\n")
    with open("./values/"+vars[var]+"_obs_ln.dat",'w') as f:
        for ii in range(1,t_obs.size):
            f.write(str(t_obs[ii])+" "+str(y_ln[var,ii])+"\n")
    with open("./values/"+vars[var]+"_obs_rl.dat",'w') as f:
        for ii in range(1,t_obs.size):
            f.write(str(t_obs[ii])+" "+str(y_rl[var,ii])+"\n")

    with open("./values/"+vars[var]+"_mlp_gs.dat",'w') as f:
        for ii in range(1,t_obs.size):
            f.write(str(t_obs[ii])+" "+str(y_gs_DA[var,ii])+"\n")
    with open("./values/"+vars[var]+"_mlp_ln.dat",'w') as f:
        for ii in range(1,t_obs.size):
            f.write(str(t_obs[ii])+" "+str(y_ln_DA[var,ii])+"\n")
    with open("./values/"+vars[var]+"_mlp_rl.dat",'w') as f:
        for ii in range(1,t_obs.size):
            f.write(str(t_obs[ii])+" "+str(y_rl_DA[var,ii])+"\n")