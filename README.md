# A dynamical Gaussian, lognormal, and reverse lognormal Kalman filter

Code used in Van Loon, S. and Fletcher, S. J. (2023), A dynamical gaussian, lognormal, and reverse lognormal Kalman filter. In review at Quarterly Journal of the Royal Meteorological Society

`LorenzModels.py` is a module file containing the Lorenz-63 and coupled Lorenz-63 models. In order to be able to run it, the C library has to be compiled first:
```cc -fPIC -shared -o C_Lorenz.so models_rk4.c ```

`mod_KalmanDA.py` contains all necessary function to run the nongaussian Kalman filter.

`train_*` files can be used to train the machine learning model to recognize skewness of the z-variable of the Lorenz model

`run_*` files run the experiments described in the manuscript.

`plot_*` files plot the results of the experiments
