""" 
Run a nongaussian Kalman Filter twin experiment with coupled Lorenz-63 
"""

# Load modules
import numpy as np 
import pickle
import mod_KalmanDA as da
import LorenzModels as LM


# Options for DA run
n_runs = 50                                 # Number of DA runs
n_wind = 250                                # Number of DA windows for each run
obs_vars = 'xyz'                            # Observed variables, either 'xy', or 'xyz'
ml_method = 'array'                         # Nongaussian decision function, either  
                                            #   'array',    selection based on observations before DA run
                                            #   'function', selection from decision function every DA window
seed = None                                 # Seed for random number generator, can be set to None
SV_init_0 = [-5.0,-6.0,22.0]                # Initial values of the Lorenz-63 run


###########################################################################################
###########################################################################################
###########################################################################################

def one_run(jj):
    global n_wind
    global period_obs
    global obs_vars
    global var_obs
    global ml_method
    global ml_file
    global seed
    global SV_init_0

    # Load the machine learning model
    with open(ml_file,'rb') as f:
        clf = pickle.load(f)
        scaler = pickle.load(f)
        info = pickle.load(f)
        p = info['p']               # Parameters of L63, typically (10,28,8/3)
        c = info['c']               # Coupling parameters of L63
        n = info['n']               # Number of coupled L63 systems
        dt = info['dt']             # Time step of L63

    rng = np.random.default_rng(seed)
    SV_init = np.empty(3*n)
    z_indices = []
    for ii in range(n):
        SV_init[3*ii:3*ii+3] = SV_init_0 + rng.normal(loc = 0.0, scale = 3.0, size = (3))
        z_indices.append(3*ii+2)

    # Nature run
    t_max = n_wind*period_obs*dt        # End time of the Lorenz-63 run
    t = np.arange(0.0, t_max, dt)  # Evaluation time
    SV = LM.sol_cL63(t, SV_init, p, c, n)
    xi_SV = np.nanmax(SV[z_indices,:])+5.0      # Parameter of the reverse lognormal distribution
    


    ###########################################################################################
    ###########################################################################################
    ###########################################################################################


    # Create observation operator
    n_SV = 3*n
    if obs_vars == 'xyz':
        n_obs = 3*n
        obs_h = np.eye(n_SV)
    if obs_vars == 'xy':
        n_obs = 2*n
        obs_h = np.zeros((n_obs,n_SV))
        for ii in range(n):
            obs_h[0 + 2*ii,0 + 3*ii] = 1.0
            obs_h[1 + 2*ii,1 + 3*ii] = 1.0


    # Decision function for lognormal state variables (for a single prediction)
    def f_ln_vars(SV):
        ln_vars = []
        for ii in range(n):
            X_data = scaler.transform(SV[0+3*ii:2+3*ii].reshape(1,-1))
            z_pred = clf.predict(X_data)
            if z_pred > 0.5:
                ln_vars.append(2+3*ii)
        return ln_vars
    # Decision function for reverse-lognormal state variables (for a single prediction)
    def f_rl_vars(SV):
        rl_vars = []
        for ii in range(n):
            X_data = scaler.transform(SV[0+3*ii:2+3*ii].reshape(1,-1))
            z_pred = clf.predict(X_data)
            if z_pred < -0.5:
                rl_vars.append(2+3*ii)
        return rl_vars

    # Predict distribution from which to sample z-variable
    if obs_vars == 'xyz':
        ln_vars_ML = []
        rl_vars_ML = []
        for ii in range(t.size):
            if ii % period_obs == 0:
                ln_vars_ML.append(f_ln_vars(SV[:,ii]))
                rl_vars_ML.append(f_rl_vars(SV[:,ii]))
    if obs_vars == 'xy':
        # If z is not observed, ML model not necessary
        ln_vars_ML = []
        rl_vars_ML = []

    # Generate observations
    t_obs, y, R = da.gen_obs(t, SV, period_obs, obs_h, var_obs, \
        ln_vars = ln_vars_ML, rl_vars = rl_vars_ML, xi_obs = xi_SV, seed = seed)



    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    # DA initial guess
    init_guess = SV_init + rng.standard_normal(3*n)

    # Model function, of the form x = model(t,x_0)
    def model_l63(t, x0):
        y = LM.sol_cL63(t, x0, p, c, n)
        return y

    # Create decision functions/arrays
    if ml_method == 'array':
        # Decision function for lognormal observations
        if obs_vars == 'xyz':
            f_ln_vars_obs = f_ln_vars
            f_rl_vars_obs = f_rl_vars
        elif obs_vars == 'xy':
            def f_ln_vars_obs(y):
                ln_vars = []
                for ii in range(n):
                    X_data = scaler.transform(y[0+2*ii:2+2*ii].reshape(1,-1))
                    z_pred = clf.predict(X_data)
                    if z_pred > 0.5:
                        ln_vars.append(2+3*ii)
                return ln_vars
            # Decision function for reverse-lognormal observations
            def f_rl_vars_obs(y):
                rl_vars = []
                for ii in range(n):
                    X_data = scaler.transform(y[0+2*ii:2+2*ii].reshape(1,-1))
                    z_pred = clf.predict(X_data)
                    if z_pred < -0.5:
                        rl_vars.append(2+3*ii)
                return rl_vars
        ln_vars = []
        rl_vars = []
        for ii in range(t_obs.size):
            ln_vars.append(f_ln_vars_obs(y[:,ii]))
            rl_vars.append(f_rl_vars_obs(y[:,ii]))
        ln_B0 = ln_vars[0]
        rl_B0 = rl_vars[0]
    if ml_method == 'function':
        ln_vars = f_ln_vars
        rl_vars = f_rl_vars
        ln_B0 = f_ln_vars(init_guess)
        rl_B0 = f_rl_vars(init_guess)

    # Store different decision functions for different (non)gaussian methods
    ln_vars_SV_OBJ = np.array([ \
        [],                         # Gaussian for all t
        z_indices[::2],             # Alternate
        z_indices[:int(n/2)],       # Split
        ln_vars,                    # ML
        [],                         # No DA (background run)
    ], dtype = object)
    rl_vars_SV_OBJ = np.array([ \
        [],                         # Gaussian for all t
        z_indices[1::2],            # Alternate
        z_indices[int(n/2):],       # Split
        rl_vars,                    # ML
        [],                         # No DA (background run)
    ], dtype = object)
    if obs_vars == 'xyz':
        ln_vars_obs_OBJ = ln_vars_SV_OBJ
        rl_vars_obs_OBJ = rl_vars_SV_OBJ
    elif obs_vars == 'xy':
        ln_vars_obs_OBJ = np.array([[],[],[],[],[-1.0]], dtype = object)
        rl_vars_obs_OBJ = np.array([[],[],[],[],[-1.0]], dtype = object)

    # Initialize analysis and background state variables
    n_meths = 5
    n_t_obs = t_obs.size           # number of observations
    n_t = n_t_obs * period_obs + 1 # number of total time steps
    X_A = np.empty((n_meths, n_SV, n_t_obs))
    X_B = np.empty((n_meths, n_SV, n_t))

    # Model error covariance matrix
    Q = np.kron(np.eye(n), \
        np.array([ \
        [0.1491, 0.1505, 0.0007], \
        [0.1505, 0.9048, 0.0014], \
        [0.0007, 0.0014, 0.9180] \
    ]))

    ln_vars_B_OBJ = np.array([ [], z_indices[ ::2], z_indices[:int(n/2)], ln_B0, []], dtype = object)
    rl_vars_B_OBJ = np.array([ [], z_indices[1::2], z_indices[int(n/2):], rl_B0, []], dtype = object)

    for iM in range(n_meths-1):
        # Create initial background error covariance matrix
        B_case = 'nmc'
        P_a = da.create_B_init_n(B_case, n, \
            SV_init_0 = SV_init_0, \
            p = p, \
            c = c, \
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
    RMSE_A = np.empty(n_meths)
    RMSE_B = np.empty(n_meths)
    for iM in range(n_meths):
        MAE_A[iM] = np.nanmean(np.abs((SV[:,::period_obs] - X_A[iM,:,:])))
        MAE_B[iM] = np.nanmean(np.abs((SV - X_B[iM,:,:-1])))
        RMSE_A[iM] = np.sqrt(np.nanmean(((SV[:,::period_obs] - X_A[iM,:,:]))**2))
        RMSE_B[iM] = np.sqrt(np.nanmean(((SV - X_B[iM,:,:-1]))**2))

    return MAE_A, MAE_B, RMSE_A, RMSE_B

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
meths = [ \
    'Gaussian ', \
    'Alternate', \
    'Split    ', \
    'ML       ', \
    'noDA     ' \
]

from multiprocessing import Pool

if __name__ == '__main__':

    n_meths = len(meths)
    n = 10
    c = 0.1
    ml_file = './kNN_cl63_n'+str(n)+'_c'+str(c)+'.pkl'   # Location of the trained ML model
    po_vec = np.arange(20,220,20)
    vo_vec = np.arange(0.5,5.0,0.5)

    MAE_A = np.empty((po_vec.size,vo_vec.size,n_runs,n_meths))
    MAE_B = np.empty((po_vec.size,vo_vec.size,n_runs,n_meths))
    RMSE_A = np.empty((po_vec.size,vo_vec.size,n_runs,n_meths))
    RMSE_B = np.empty((po_vec.size,vo_vec.size,n_runs,n_meths))

    for ii, period_obs in enumerate(po_vec):
        for jj, var_obs in enumerate(vo_vec):

            with Pool(np.min([n_runs,10])) as p:
                X = p.map(one_run, range(n_runs))

            for kk in range(n_runs):
                MAE_A[ii,jj,kk,:], MAE_B[ii,jj,kk,:], \
                RMSE_A[ii,jj,kk,:], RMSE_B[ii,jj,kk,:] = X[kk]

            print("Finished p = "+str(period_obs)+", s = "+str(var_obs))

            for iM in range(n_meths):
                r_a = np.mean(MAE_A[ii,jj,:,iM])
                r_b = np.mean(MAE_B[ii,jj,:,iM])

                print(meths[iM]+ \
                    ': mean(r_a) = '+format(np.round(r_a,3),".3f")+ \
                    ', mean(r_b) = '+format(np.round(r_b,3),".3f"))

    info = {
        "n_runs": n_runs,
        "n_wind": n_wind,
        "obs_vars": obs_vars,
        "period_obs": po_vec,
        "var_obs": vo_vec,
        "meths": meths,
        "ml_method": ml_method,
        "ml_file": ml_file,
        "seed": seed,
        "SV_init": SV_init_0
    }
    with open('./KF_cL63' \
        +"_n"+str(n) \
        +"_c"+str(c) \
        +"_" +obs_vars \
        +'.pkl','wb') as f:
        pickle.dump(MAE_A, f)
        pickle.dump(MAE_B, f)
        pickle.dump(RMSE_A, f)
        pickle.dump(RMSE_B, f)
        pickle.dump(info, f)
    print("Finished")