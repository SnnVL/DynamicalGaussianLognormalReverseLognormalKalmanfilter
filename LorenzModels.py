"""
Functions needed for solving the different Lorenz models

#### Methods:
- `sol_L63`         ->  Solution to the Lorenz-63 model
- `sol_cL63`        ->  Solution to the coupled Lorenz-63 model

#### Author:
Senne Van Loon
Cooperative Institute for Research in the Atmosphere (CIRA),
Colorado State University,
3925A West Laporte Ave, Fort Collins, CO 80521

#### References and acknowledgements:
* Lorenz, E. N. (1963). Deterministic nonperiodic flow. Journal of atmospheric sciences, 20(2), 130-141.
"""

import numpy as np
from ctypes import c_double, c_int, CDLL

# Load C-library
# The library has to be compiled first, by running:
# cc -fPIC -shared -o C_Lorenz.so models_rk4.c 
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
c_mod = CDLL(dir_path+"/C_Lorenz.so")

def sol_L63(t, x0, p = np.array([10.0,28.0,8.0/3.0])):
    N = 3
    if x0.size != N:
        raise RuntimeError("Incorrect initial conditions!")

    t0 = c_double(t[0])
    dt = c_double(t[1]-t[0])
    nt = t.size

    x0_c = (c_double * N)(*x0)
    sol_c = (c_double * (N*nt))()

    s = c_double(p[0])
    r = c_double(p[1])
    b = c_double(p[2])

    c_mod.sol_L63_(t0,dt,c_int(nt),x0_c,sol_c,s,r,b)

    return np.reshape(np.array(sol_c[:]),(N,nt),order='F')

def sol_cL63(t, x0, p = np.array([10.0,28.0,8.0/3.0]), c = np.array([0.0,0.0,0.0]), n=4):
    N = 3*n
    if x0.size != N:
        raise RuntimeError("Incorrect initial conditions!")

    t0 = c_double(t[0])
    dt = c_double(t[1]-t[0])
    nt = t.size

    x0_c = (c_double * N)(*x0)
    sol_c = (c_double * (N*nt))()

    s = c_double(p[0])
    r = c_double(p[1])
    b = c_double(p[2])
    cx = c_double(c[0])
    cy = c_double(c[1])
    cz = c_double(c[2])

    c_mod.sol_cL63_(t0,dt,c_int(nt),x0_c,sol_c,c_int(n),s,r,b,cx,cy,cz)

    return np.reshape(np.array(sol_c[:]),(N,nt),order='F')
