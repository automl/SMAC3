# 0.2.1

* CI: travis-ci continuous integration on OSX
* ADD: initial design for mulitple configurations, initial design for a 
  random configuration
* MAINT: use sklearn PCA if more than 7 instance features are available (as 
  in SMAC 1 and 2)
* MAINT: use same minimum step size for the stochastic local search as in SMAC2.
* MAINT: use same number of imputation iterations as in SMAC2.
* FIX 98: automatically seed the configuration space object based on the SMAC
  seed.

# 0.2

* ADD 55: Separate modules for the initial design and a more flexible 
  constructor for the SMAC class
* ADD 41: Add ROAR (random online adaptive racing) class
* ADD 82: Add fmin_smac, a scipy.optimize.fmin_l_bfgs_b-like interface to the
  SMAC algorithm
* NEW documentation at https://automl.github.io/SMAC3/stable and 
  https://automl.github.io/SMAC3/dev
* FIX 62: intensification previously used a random seed from np.random 
  instead of from SMAC's own random number generator
* FIX 42: class RunHistory can now be pickled
* FIX 48: stats and runhistory objects are now injected into the target 
  algorithm execution classes
* FIX 72: it is now mandatory to either specify a configuration space or to 
  pass the path to a PCS file
* FIX 49: allow passing a callable directly to SMAC. SMAC will wrap the 
  callable with the appropriate target algorithm runner

# 0.1.3

* FIX 63 using memory limit for function target algorithms (broken since 0.1.1)

# 0.1.2

* FIX 58 output of the final statistics
* FIX 56 using the command line target algorithms (broken since 0.1.1)
* FIX 50 as variance prediction, we use the average predicted variance across the instances

# 0.1.1

* NEW leading ones examples
* NEW raise exception if unknown parameters are given in the scenario file
* FIX 17/26/35/37/38/39/40/46
* CHANGE requirement of ConfigSpace package to 0.2.1
* CHANGE cutoff default is now None instead of 99999999999


# 0.1.0

* Moved to github instead of bitbucket
* ADD further unit tests
* CHANGE Stats object instead of static class
* CHANGE requirement of ConfigSpace package to 0.2.0
* FIX intensify runs at least two challengers
* FIX intensify skips incumbent as challenger
* FIX Function TAE runner passes random seed to target function
* FIX parsing of emtpy lines in scenario file

# 0.0.1

* initial release