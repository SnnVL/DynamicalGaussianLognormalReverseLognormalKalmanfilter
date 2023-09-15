""" 
Functions needed for Kalman filter data assimilation techniques

#### Main methods:
- `kalman_filter`   ->  Kalman filter
- `gen_obs`         ->  Generate observations from true state
- `create_B_init`   ->  Create first guess for the background error covariance matrix
                        in the Lorenz-63 model
- `create_B_init_n` ->  Create first guess for the background error covariance matrix
                        in the coupled Lorenz-63 model


#### Author:
Senne Van Loon

#### References and acknowledgements:
[1] Fletcher, S. J. (2017). Data assimilation for the geosciences: 
From theory to application. Elsevier
[2] Fletcher, S. J., Zupanski, M., Goodliff, M. R., Kliewer, A. J., Jones, A.
S., Forsythe, J. M., Wu, T.-C., Hossen, M. J. and Van Loon, S. (2023) Lognormal
and Mixed Gaussian-Lognormal Kalman Filters. Monthly Weather Review, 151, 761-774
[3] Van Loon, S. and Fletcher, S. J., A dynamical gaussian, lognormal, and reverse 
lognormal Kalman filter. In review at Quarterly Journal of the Royal Meteorological Society


"""

# Load modules
import numpy as np
from scipy.linalg import inv
from scipy.stats import rv_continuous
from scipy.special import erf, erfinv
import copy
import LorenzModels as LM

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)



####################################################################################
####################################################################################
####################################################################################

def kalman_filter(init_guess, t_obs, n_t_mod, y, H, P_a, R, Q, model, \
        ln_vars_SV = [], ln_vars_obs = [], \
        rl_vars_SV = [], rl_vars_obs = [], xi_SV = 0.0, xi_obs = 0.0):
    """
    Kalman filter

    Kalman filter data assimilation technique for use in a general model, 
    which allows for mixed Gaussian, lognormal, and reverse lognormal observations 
    and background state variables.

    #### Input
    - `init_guess`  ->  Initial guess for the model analysis, vector of size n_SV
    - `t_obs`       ->  Time values of observations, vector of size n_t_obs
    - `n_t_mod`     ->  Number of time steps to use in each model run
    - `y`           ->  Observations, array of size n_obs x n_t_obs
    - `H`           ->  Observation operator, obtained from create_H, 
                        array of size n_obs x n_SV
    - `P_a`         ->  Analysis error covariance matrix (initial), array of size n_SV x n_SV
    - `R`           ->  Observation error covariance matrix, object array of size n_t_obs,
                        containing arrays of size n_obs x n_obs
    - `Q`           ->  Model error covariance matrix, array of size n_SV x n_SV
    - `model`       ->  Model to use in the analysis, function of the form
                            x = model(t,x_0),
                        where t contains the time values where the model is evaluated
                        and x_0 is the initial condition;
                        the output x are the state variables for all time values
    - `ln_vars_SV`  ->  State variables that should be treated lognormally, 
                        *   If ln_vars is a callable, it should be of the form
                                    ln_var = ln_vars(SV_input),
                            where SV_input contains the state variables at one time step
                            and ln_var is a list of indices between 0 and n_SV-1
                        *   If ln_vars is a list of indices between 0 and n_SV-1,
                            the state variables of these indices are treated as 
                            lognormally distributed for all time steps
                        *   If ln_vars is a list of lists, the length of the top list
                            should be n_t_obs, and the inner lists are indices between 0 and n_SV-1,
                            the state variables of these indices are treated as 
                            lognormally distributed for each specific time step
                        Default is [] for all Gaussian variables
    - `ln_vars_obs` ->  Observations that should be treated lognormally, 
                        *   If ln_vars is a callable, it should be of the form
                                    ln_var = ln_vars(y_input),
                            where y_input contains the observed variables at one time step
                            and ln_var is a list of indices between 0 and n_obs-1
                        *   If ln_vars is a list of indices between 0 and n_obs-1,
                            the observations of these indices are treated as 
                            lognormally distributed for all time steps
                        *   If ln_vars is a list of lists, the length of the top list
                            should be n_t_obs, and the inner lists are indices between 0 and n_obs-1,
                            the observations of these indices are treated as 
                            lognormally distributed for each specific time step
                        Default is [] for all Gaussian variables
    - `rl_vars_SV`  ->  State variables that should be treated reverse lognormally, same as ln_vars_SV
    - `rl_vars_obs` ->  Observations that should be treated reverse lognormally, same as ln_vars_obs
    - `xi_SV`       ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`      ->  Parameter for the reverse lognormal distribution for the observations

    #### Output
    - `x_b`         ->  Background state, array of size n_SV x n_t
    - `x_a`         ->  Analysis state, array of size n_SV x n_t_obs
    - `t_true`      ->  Total time for background, vector of size n_t
    """

    # Save lengths of arrays
    n_obs, n_SV = H.shape           # number of state variables
    n_t_obs = t_obs.size        # number of observations
    n_t = n_t_obs * n_t_mod + 1 # number of total time steps

    # Initialize background, analysis, and "true" states
    x_b = np.empty((n_SV,n_t))
    x_a = np.empty((n_SV,n_t_obs))
    x_t = np.empty((n_SV,n_t_obs))

    # Initialize errors
    e_a_mix = np.sqrt(np.diagonal(P_a))

    # At t=0, background and analysis states are set to the initial guess
    x_b[:,0] = init_guess
    x_a[:,0] = init_guess

    # "True" state at t = 0
    ln_var_SV,  rl_var_SV  = get_ln_rl_var(ln_vars_SV, rl_vars_SV, n_t_obs,0,x_a[:,0])
    e_a = e_a_mix
    e_a[ln_var_SV] = np.exp(e_a_mix[ln_var_SV])
    e_a[rl_var_SV] = xi_SV - np.exp(e_a_mix[rl_var_SV])
    x_t[:,0] = x_a[:,0] + e_a
    x_t[ln_var_SV,0] = x_a[ln_var_SV,0] * e_a[ln_var_SV] 
    x_t[rl_var_SV,0] = xi_SV - (xi_SV - x_a[rl_var_SV,0]) * (xi_SV - e_a[rl_var_SV]) 

    # Total time evolution for model
    dt_obs = t_obs[1] - t_obs[0] # Assuming evenly spaced observations
    t_true = np.linspace(t_obs[0],t_obs[-1] + dt_obs, n_t) # Assuming one prediction window

    fail=False
    # Loop over all observations
    for ii in range(1,n_t_obs):
        # Calculated states are on model time, and not observation time
        tt = ii * n_t_mod       # Index for model time at observation
        tt_prev = tt - n_t_mod  # Index for model time at previous observation
        sim_time = t_true[tt_prev:tt+1]

        ##################################################################
        ###################       Forecast step       ####################
        ##################################################################
        
        # Model forecast
        x_f = model(sim_time, x_a[:,ii-1])
        b_f = model(sim_time, x_t[:,ii-1])  # "True state" at current time

        if fail:
            x_a[:,ii-1] = np.nan*np.empty(n_SV)
            x_t[:,ii-1] = np.nan*np.empty(n_SV)
            fail = False
        if np.any(np.isnan(x_f)) or np.any(np.isnan(b_f)):
            print("Warning: fail in model run, skipping this assimilation step!")
            print("    x_a_init = "+str(x_a[:,ii-1]))
            print("    x_t_init = "+str(x_t[:,ii-1]))

            # If fail, something went wrong in previous analysis
            x_a[:,ii-1] = np.nan*np.empty(n_SV)
            x_b[:,tt_prev:tt+1] = np.nan*np.empty_like(sim_time)

            # Restart DA from initial guess next step
            x_a[:,ii] = init_guess
            x_t[:,ii] = x_a[:,ii] + e_a
            x_t[ln_var_SV,ii] = x_a[ln_var_SV,ii] * e_a[ln_var_SV] 
            x_t[rl_var_SV,ii] = xi_SV - (xi_SV - x_a[rl_var_SV,ii]) * (xi_SV - e_a[rl_var_SV]) 
            fail = True
            continue

        # Save background forecast
        x_b[:,tt_prev+1:tt+1] = x_f[:,1:]
        
        ##################################################################
        ##################        Transform step       ###################
        ##################################################################

        # Get indices for lognormally and reverse lognormally distributed state variables 
        # and observations for this time step
        ln_var_SV,  rl_var_SV  = get_ln_rl_var(ln_vars_SV, rl_vars_SV, n_t_obs,ii,x_f[:,-1])
        ln_var_obs, rl_var_obs = get_ln_rl_var(ln_vars_obs,rl_vars_obs,n_t_obs,ii,y[:,ii])

        # Transform forecast and observations to mixed variables
        Hx = H @ x_f[:,-1]
        x_f_mix, b_f_mix, ln_var_SV, rl_var_SV\
            = transform_vars(x_f[:,-1], b_f[:,-1], ln_var_SV, rl_var_SV, xi_SV)
        y_mix, Hx_mix, ln_var_obs, rl_var_obs \
            = transform_vars(y[:,ii], Hx, ln_var_obs, rl_var_obs, xi_obs)
        
        # Forecast error and error covariance matrix
        e_f_mix = b_f_mix - x_f_mix
        P_f = np.outer(e_f_mix,e_f_mix) + Q

        # W_b = (dX(x)/dx)^-1
        W_b = np.eye(n_SV)
        W_b[ln_var_SV,ln_var_SV] = x_f[ln_var_SV,-1]
        W_b[rl_var_SV,rl_var_SV] = x_f[rl_var_SV,-1] - xi_SV
        
        # W_o^-1 = dH(h)/dh
        W_o_inv = np.eye(n_obs)
        W_o_inv[ln_var_obs,ln_var_obs] = 1.0/Hx[ln_var_obs]
        W_o_inv[rl_var_obs,rl_var_obs] = 1.0/(Hx[rl_var_obs] - xi_obs)

        # Scaled observation operator
        H_til = W_o_inv @ H @ W_b
        
        ##################################################################
        ####################       Update step       #####################
        ##################################################################

        # Kalman gain matrix (size n_SV x n_obs)
        K = P_f @ H_til.T @ inv(H_til @ P_f @ H_til.T + R[ii])

        # Analysis state
        x_a_mix = x_f_mix + K @ (y_mix - Hx_mix)

        # Analysis error covariance matrix
        e_o = np.sqrt(np.diagonal(R[ii]))
        e_a_mix = (np.eye(n_SV) - K @ H_til) @ e_f_mix + K @ e_o

        # Transform back to normal variables
        x_a[:,ii] = x_a_mix
        x_a[ln_var_SV,ii] = np.exp(x_a[ln_var_SV,ii])
        x_a[rl_var_SV,ii] = xi_SV - np.exp(x_a[rl_var_SV,ii])

        # "True state" for this time step
        x_t[:,ii] = x_a_mix + e_a_mix
        x_t[ln_var_SV,ii] = np.exp(x_t[ln_var_SV,ii])
        x_t[rl_var_SV,ii] = xi_SV - np.exp(x_t[rl_var_SV,ii])

    # Final forecasting step
    ii += 1
    tt = ii * n_t_mod       # Index for model time at observation
    tt_prev = tt - n_t_mod  # Index for model time at previous observation
    sim_time = t_true[tt_prev:tt+1]

    # Model run
    x_f = model(sim_time, x_a[:,ii-1])
    x_b[:,tt_prev+1:tt+1] = x_f[:,1:]
    
    return x_a, x_b, x_t, t_true


####################################################################################
####################################################################################
####################################################################################


def get_ln_rl_var(ln_vars,rl_vars,n_t,ii,SV):
    """
    Get the correct distribution to be used for a certain time step

    #### Input
    - `ln_vars`   ->  State variables that should be treated lognormally, 
                    * If ln_vars is a callable, it should be of the form
                            ln_var = ln_vars(SV_input),
                        where SV_input contains the state variables at one time step
                        and ln_var is a list of indices between 0 and n_SV-1
                    * If ln_vars is a list of indices between 0 and n_SV-1,
                        the state variables of these indices are treated as 
                        lognormally distributed for all time steps
                    * If ln_vars is a list of lists, the length of the top list
                        should be n_t_obs, and the inner lists are indices between 0 and n_SV-1,
                        the state variables of these indices are treated as 
                        lognormally distributed for each specific time step
    - `rl_vars`   ->  State variables that should be treated reverse lognormally, same as ln_vars
    - `n_t`       ->  Length of time vector to be compared to in the case ln_vars and rl_vars
                        are a list of indices
    - `ii`        ->  Index to be used in the case ln_vars and rl_vars are a list of indices
    - `SV`        ->  State variables to be used in the case ln_vars and rl_vars are callable

    #### Output
    - `ln_var`    ->  State variables that should be treated lognormally, 
                        as a list of indices between 0 and n_SV-1
    - `rl_var`    ->  State variables that should be treated reverse lognormally, 
                        as a list of indices between 0 and n_SV-1
    """

    # Get lognormally distributed state variables for this time step
    if callable(ln_vars):
        ln_var = ln_vars(SV)
    elif len(ln_vars) == n_t:
        ln_var = ln_vars[ii]
    else: # Same treatment for all time steps
        ln_var = ln_vars

    # Get reverse lognormally distributed state variables for this time step
    if callable(rl_vars):
        rl_var = rl_vars(SV)
    elif len(rl_vars) == n_t:
        rl_var = rl_vars[ii]
    else: # Same treatment for all time steps
        rl_var = rl_vars

    # Check if there is any overlap in lognormal and reverse lognormal variables
    s_ln_var = set(ln_var)
    if any(x in s_ln_var for x in rl_var):
        raise RuntimeError("State variables can not be lognormal and reverse lognormal at the same time!")

    return ln_var, rl_var


def transform_vars(x_a, x_b, ln_var, rl_var, xi):
    """
    Transform state variables to mixed representation

    Two arrays of state variables are transformed to a mixed representation at the same time. 
    The method tests if a logarithm of a negative value is ever taken,
    and if so removes the index of the state variable for which this is the case from ln_var or rl_var

    #### Input
    - `x_a`       ->  State variables that need to be transformed, array of size n_SV
    - `x_b`       ->  State variables that need to be transformed, array of size n_SV
    - `ln_var`    ->  State variables that should be treated lognormally, 
                        as a list of indices between 0 and n_SV-1
    - `rl_var`    ->  State variables that should be treated reverse lognormally, 
                        as a list of indices between 0 and n_SV-1
    - `xi`        ->  Parameter for the reverse lognormal distribution

    #### Output
    - `x_a_mix`   ->  Mixed representation of x_a, such that 
                        x_a_mix[gs_var] = x_a
                        x_a_mix[ln_var] = log(x_a)
                        x_a_mix[rl_var] = log(xi - x_a)
    - `x_b_mix`   ->  Mixed representation of x_b, such that 
                        x_b_mix[gs_var] = x_b
                        x_b_mix[ln_var] = log(x_b)
                        x_b_mix[rl_var] = log(xi - x_b)
    - `ln_var`    ->  State variables that should be treated lognormally, 
                        as a list of indices between 0 and n_SV-1.
                        If a lognormal state variable had a negative value, this state variable
                        is instead treated as Gaussian, and the index is removed from ln_var
    - `rl_var`    ->  State variables that should be treated reverse lognormally, 
                        as a list of indices between 0 and n_SV-1.
                        If a reverse lognormal state variable had a value larger than xi, 
                        this state variable is instead treated as Gaussian, 
                        and the index is removed from rl_var
    """

    # Initialize mixed variables
    x_a_mix = copy.copy(x_a)
    x_b_mix = copy.copy(x_b)

    ln_var_new = copy.copy(ln_var)
    rl_var_new = copy.copy(rl_var)

    # Lognormal variables
    try:
        x_a_mix[ln_var] = np.log(x_a[ln_var])
        x_b_mix[ln_var] = np.log(x_b[ln_var])
    except RuntimeWarning:
        # Value of negative value taken, find component where this is the case and
        # use Gaussian distribution instead for this time step
        print('Warning: logarithm of negative value taken')
        for iLN in range(len(ln_var)):
            try:
                x_a_mix[ln_var[iLN]] = np.log(x_a[ln_var[iLN]])
                x_b_mix[ln_var[iLN]] = np.log(x_b[ln_var[iLN]])
            except RuntimeWarning:
                x_a_mix[ln_var[iLN]] = x_a[ln_var[iLN]]
                x_b_mix[ln_var[iLN]] = x_b[ln_var[iLN]]
                ln_var_new.remove(ln_var[iLN])

    # Reverse lognormal variables
    try:
        x_a_mix[rl_var] = np.log(xi - x_a[rl_var])
        x_b_mix[rl_var] = np.log(xi - x_b[rl_var])
    except RuntimeWarning:
        # Value of negative value taken, find component where this is the case and
        # use Gaussian distribution instead for this time step
        print('Warning: logarithm of negative value taken (rl)')
        for iRL in range(len(rl_var)):
            try:
                x_a_mix[rl_var[iRL]] = np.log(xi - x_a[rl_var[iRL]])
                x_b_mix[rl_var[iRL]] = np.log(xi - x_b[rl_var[iRL]])
            except RuntimeWarning:
                x_a_mix[rl_var[iRL]] = x_a[rl_var[iRL]]
                x_b_mix[rl_var[iRL]] = x_b[rl_var[iRL]]
                rl_var_new.remove(rl_var[iRL])

    return x_a_mix, x_b_mix, ln_var_new, rl_var_new



####################################################################################
####################################################################################
####################################################################################
# Observations

def gen_obs(t, SV, period_obs, H, var_obs, ln_vars = [], rl_vars = [], xi_obs = 0.0, \
            seed = None, sample = 'mode'):
    """
    Generate noisy observations from true state nature run

    #### Input
    - `t`               ->  Time of truth, vector of length n_t
    - `SV`              ->  State variables of truth, array of size n_SV x n_t,
                            with n_SV the number of variables, and n_t the number of time steps
    - `period_obs`      ->  Observation period, in steps of truth time
    - `H`               ->  Observation operator, either
                                * array of size n_obs x n_SV,  
                                * function of the form y = h(SV)
    - `var_obs`         ->  Observational error variance, diagonal values of 
                            observational error covariance matrix R
    - `ln_vars`         ->  State variables that should be treated lognormally, 
                            * If ln_vars is a callable, it should be of the form
                                    ln_var = ln_vars(SV_input),
                            where SV_input contains the state variables at one time step
                            and ln_var is a list of indices between 0 and n_SV-1
                            * If ln_vars is a list of indices between 0 and n_SV-1,
                            the state variables of these indices are treated as 
                            lognormally distributed for all time steps
                            * If ln_vars is a list of lists, the length of the top list
                            should be n_t_obs, and the inner lists are indices between 0 and n_SV-1,
                            the state variables of these indices are treated as 
                            lognormally distributed for each specific time step
                            Default is [] for all Gaussian variables
    - `rl_vars`         ->  State variables that should be treated reverse lognormally, same as ln_vars
    - `xi_obs`          ->  Parameter for the reverse lognormal distribution
    - `seed`            ->  Seed of the random number generator. Default is random seed
    - `sample`          ->  Descriptor to sample around, can be either 
                            `mode`, `median`, or `mean`. 
                            Default is `mode`.

    #### Output 
    - `t_obs`           ->  Time of observations, vector of length n_t_obs = n_t//period_obs
    - `y`               ->  Observations, array of size n_obs x n_t_obs
    - `R`               ->  Observational error covariance matrix of size n_obs x n_obs
    """

    # Count lognormal and reverse lognormal observations
    n_ln, n_rl = 0, 0

    # Create time array of the observations
    t_obs = t[::period_obs]
    n_t_obs = t_obs.size

    # State variables at observation times
    SV_obs = SV[:,::period_obs]

    # Initialize observations array
    if callable(H):
        y = H(SV_obs)
    else:
        y = H @ SV_obs 
    n_obs, _ = y.shape

    # Create observational error covariance matrix 
    r0 = var_obs
    R = np.empty(n_t_obs,dtype='object')

    # Create random number generator
    rng = np.random.default_rng(seed)
    rln = reverse_lognormal(shapes = 'mu, sigma, T', seed = seed)

    # Loop over observations
    for ii in range(1,n_t_obs):

        # Get indices for lognormally and reverse lognormally distributed state variables 
        # and observations for this time step
        ln_var, rl_var = get_ln_rl_var(ln_vars, rl_vars, n_t_obs, ii, SV_obs[:,ii])

        R[ii] = np.eye(n_obs)

        # Add noise
        for jj in range(n_obs):
            y0 = y[jj,ii]
            if jj in ln_var: # Lognormal noise

                # Calculate mu and sigma for given descriptor
                if sample == 'mean':
                    ln_mu = np.log(y0**2/np.sqrt(r0+y0**2))
                    ln_sd = np.sqrt(np.log(1.0 + r0/y0**2))
                if sample == 'median':
                    ln_mu = np.log(y0)
                    ln_sd = np.sqrt(np.log(0.5 + 0.5*np.sqrt(4.0*r0/y0**2 + 1)))
                if sample == 'mode':
                    p = np.polynomial.polynomial.Polynomial((-r0/y0**2,0,0,-1.0,1.0))
                    rts = p.roots()
                    realRoot = np.real(rts[np.logical_and(np.imag(rts) < 1e-5, np.real(rts) > 1.0)])
                    if not realRoot.size == 1 :
                        raise RuntimeError("Multiple roots found while solving for the mode variance!")
                    ln_mu = np.log(y0*realRoot)
                    ln_sd = np.sqrt(np.log(realRoot))
                
                # Sample from lognormal distribution
                y[jj,ii] = rng.lognormal(ln_mu, ln_sd)
                R[ii][jj,jj] = ln_sd**2
                n_ln += 1
            elif jj in rl_var:  # Reverse lognormal noise

                # Calculate mu and sigma for given descriptor
                if sample == 'mean':
                    rl_mu = np.log((xi_obs-y0)**2/np.sqrt(r0+(xi_obs-y0)**2))
                    rl_sd = np.sqrt(np.log(1.0 + r0/(xi_obs-y0)**2))
                if sample == 'median':
                    rl_mu = np.log(xi_obs-y0)
                    rl_sd = np.sqrt(np.log(0.5 + 0.5*np.sqrt(4.0*r0/(xi_obs-y0)**2 + 1)))
                if sample == 'mode':
                    p = np.polynomial.polynomial.Polynomial((-r0/(xi_obs-y0)**2,0,0,-1.0,1.0))
                    rts = p.roots()
                    realRoot = np.real(rts[np.logical_and(np.imag(rts) < 1e-5, np.real(rts) > 1.0)])
                    if not realRoot.size == 1 :
                        raise RuntimeError("Multiple roots found while solving for the mode variance!")
                    rl_mu = np.log((xi_obs-y0)*realRoot)
                    rl_sd = np.sqrt(np.log(realRoot))

                # Sample from reverse lognormal distribution
                y[jj,ii] = rln.rvs(mu = rl_mu, sigma = rl_sd, T = xi_obs)
                R[ii][jj,jj] = rl_sd**2
                n_rl += 1
            else: # Gaussian noise
                y[jj,ii] = rng.normal(y0, np.sqrt(r0))
                R[ii][jj,jj] = r0

    return t_obs, y, R




####################################################################################
####################################################################################
####################################################################################
# Reverse lognormal distribution

# Timeout exception
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class reverse_lognormal(rv_continuous):
    "Reverse lognormal distribution"
    # Probability density function
    def _pdf(self, x, mu = 0.0, sigma = 1.0, T = 0.0):
        return np.exp(-(np.log(T-x) - mu)**2/2/sigma**2)/(T-x)/sigma/np.sqrt(2*np.pi)

    # Cumulative probability density function
    def _cdf(self, x, mu = 0.0, sigma = 1.0, T = 0.0):
        return (1.0-erf((np.log(T-x) - mu)/np.sqrt(2)/sigma))/2.0

    # Complement of the cumulative probability density function
    def _sf(self, x, mu = 0.0, sigma = 1.0, T = 0.0): 
        return (1.0+erf((np.log(T-x) - mu)/np.sqrt(2)/sigma))/2.0

    # Inverse of the cumulative probability density function
    def _ppf(self, p, mu = 0.0, sigma = 1.0, T = 0.0):
        return T - np.exp(np.sqrt(2) * sigma * erfinv(1.0-2.0*p) + mu)

    # T and mu can be any value between -oo and +oo
    def _argcheck(self, mu = 0.0, sigma = 1.0, T = 0.0):
        return (sigma > 0)

    # Support of the probability (max = T)
    def _get_support(self, mu = 0.0, sigma = 1.0, T = 0.0):
        return -np.inf, T







####################################################################################
####################################################################################
####################################################################################
# Initial covariances


def create_B_init(case, \
        SV_init = np.array([-10.0,-10.0,20.0]), \
        p = np.array([10.0,28.0,8.0/3.0]), 
        seed = None, ln_vars=[], rl_vars=[]):
    """
    Create the initial guess for the background error covariance matrix 
    for the Lorenz 63 model. 

    #### Required input
    - `case`    ->  Method to generate B-matrix with, either 'rand' or 'nmc'

    #### Optional input
    - `SV_init` ->  Initial value for all state variables, array of size n_SV
    - `p`       ->  Parameters of the Lorenz model, [sigma, rho, beta]
    - `meth`    ->  String indicating the integration method to use
    - `seed`    ->  Seed of the random number generator. Default is random seed
    - `ln_var`  ->  State variables that should be treated lognormally, 
                        as a list of indices between 0 and n_SV-1
    - `rl_var`  ->  State variables that should be treated reverse lognormally, 
                        as a list of indices between 0 and n_SV-1

    #### Output
    - `B`       ->  Background error covariance matrix
    """

    if case == 'rand':
        n_SV = 3
        rng = np.random.default_rng(seed)
        e_b = rng.standard_normal((n_SV))
        B = np.outer(e_b,e_b)

    elif case == 'nmc':
        # Create time values for evaluation
        total_steps = 1000
        dt = 0.01
        t_span = [0.0,dt*total_steps]
        t_eval = np.arange(t_span[0],t_span[1],dt)

        rng = np.random.default_rng(seed)

        # Solve Lorenz model
        SV_1 = LM.sol_L63(t_eval,SV_init + rng.standard_normal(3),p)
        SV_2 = LM.sol_L63(t_eval,SV_init + rng.standard_normal(3),p)

        # Matrix with trajectory values for Gaussian covariance
        X = SV_1 - SV_2

        # Use logaritmic variables for lognormally distributed state variables
        if ln_vars:
            X[ln_vars, :] = np.log(SV_1[ln_vars, :]) - np.log(SV_2[ln_vars, :])


        # Use reverse logaritmic variables for reverse lognormally distributed state variables
        if rl_vars:
            xi1 = np.nanmax(SV_1[2,:])+1.0      # Parameter of the reverse lognormal distribution
            xi2 = np.nanmax(SV_2[2,:])+1.0      # Parameter of the reverse lognormal distribution
            xi = np.max((xi1,xi2))
            X[rl_vars, :] = np.log(xi - SV_1[rl_vars, :]) - np.log(xi - SV_2[rl_vars, :])

        # Updated background error covariance matrix
        B = (X @ X.T)/t_eval.size


    else:
        raise RuntimeError("Initial B-matrix case unknown.")
         

    return B


def create_B_init_n(case, n, \
        SV_init_0 = np.array([-10.0,-10.0,20.0]), \
        p = np.array([10.0,28.0,8.0/3.0]), \
        c = np.array([0.0,0.0,0.0]), 
        seed = None, ln_vars=[], rl_vars=[]):
    """
    Create the initial guess for the background error covariance matrix 
    for the coupled Lorenz 63 model. 

    #### Required input
    - `case`    ->  Method to generate B-matrix with, either 'rand' or 'nmc'

    #### Optional input
    - `SV_init` ->  Initial value for all state variables, array of size n_SV
    - `p`       ->  Parameters of the Lorenz model, [sigma, rho, beta]
    - `meth`    ->  String indicating the integration method to use
    - `seed`    ->  Seed of the random number generator. Default is random seed
    - `ln_var`  ->  State variables that should be treated lognormally, 
                        as a list of indices between 0 and n_SV-1
    - `rl_var`  ->  State variables that should be treated reverse lognormally, 
                        as a list of indices between 0 and n_SV-1

    #### Output
    - `B`       ->  Background error covariance matrix
    """

    if case == 'rand':
        n_SV = 3*n
        rng = np.random.default_rng(seed)
        e_b = rng.standard_normal((n_SV))
        B = np.outer(e_b,e_b)

    elif case == 'nmc':
        # Create time values for evaluation
        total_steps = 1000
        dt = 0.01
        t_span = [0.0,dt*total_steps]
        t_eval = np.arange(t_span[0],t_span[1],dt)

        rng = np.random.default_rng(seed)
        SV_init = np.empty(3*n)
        z_indices = []
        for ii in range(n):
            SV_init[3*ii:3*ii+3] = SV_init_0 + rng.normal(loc = 0.0, scale = 3.0, size = (3))
            z_indices.append(3*ii+2)

        # Solve Lorenz model
        SV_1 = LM.sol_cL63(t_eval,SV_init + rng.standard_normal(3*n),p,c,n)
        SV_2 = LM.sol_cL63(t_eval,SV_init + rng.standard_normal(3*n),p,c,n)

        # Matrix with trajectory values for Gaussian covariance
        X = SV_1 - SV_2

        # Use logaritmic variables for lognormally distributed state variables
        if ln_vars:
            X[ln_vars, :] = np.log(SV_1[ln_vars, :]) - np.log(SV_2[ln_vars, :])

        # Use reverse logaritmic variables for reverse lognormally distributed state variables
        if rl_vars:
            xi1 = np.nanmax(SV_1[z_indices,:])+1.0      # Parameter of the reverse lognormal distribution
            xi2 = np.nanmax(SV_2[z_indices,:])+1.0      # Parameter of the reverse lognormal distribution
            xi = np.max((xi1,xi2))
            X[rl_vars, :] = np.log(xi - SV_1[rl_vars, :]) - np.log(xi - SV_2[rl_vars, :])

        # Updated background error covariance matrix
        B = (X @ X.T)/t_eval.size


    else:
        raise RuntimeError("Initial B-matrix case unknown.")
         

    return B
