=======
SMAC4BB
=======
SMAC4BB is designed for black-box function optimization.


SMAC4BB uses a Gaussian Process (gp) or a set of Gaussian Processes whose hyperparameters are integrated by
MCMC (gp_mcmc) as its surrogate model. SMAC4BB works best on numerical hyperparameter configuration space and should not be applied to the problems with large evaluation budgets (up to 1000 evaluations).

Function optimized by SMAC4BB normally requires the following input

- *cfg*,  the input configuration, example can be found [here](./SMAC4BB_synthetic_function_example.py).

Note: SMAC4BB's optimizer is set default as gp_mcmc, where extra dependencies are required, you need to install SMAC with the following command:


```
pip install smac[gpmcmc]
```
