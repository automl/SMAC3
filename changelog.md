# 0.2.4

* CI only check code quality for python3
* Perform local search on configurations from previous runs as proposed in the
  original paper from 2011 instead of random configurations as implemented
  before
* CI run travis-ci unit tests with python3.6
* FIX #167, remove an endless loop which occured when using pSMAC

# 0.2.3

* MAINT refactor Intensifcation and adding unit tests
* CHANGE StatusType to Enum
* RM parameter importance package
* FIX ROAR facade bug for cli
* ADD easy access of runhistory within Python
* FIX imputation of censored data
* FIX conversion of runhistory to EPM training data (in particular running time data)
* FIX initial run only added once in runhistory
* MV version number to a separate file
* MAINT more efficient computations in run_history (assumes average as aggregation function across instances)

# 0.2.2

* FIX 124: SMAC could crash if the number of instances was less than seven
* FIX 126: Memory limit was not correctly passed to the target algorithm
  evaluator
* Local search is now started from the configurations with highest EI, drawn by
  random sampling
* Reduce the number of trees to 10 to allow faster predictions (as in SMAC2)
* Do an adaptive number of stochastic local search iterations instead of a fixd
  number (a5914a1d97eed2267ae82f22bd53246c92fe1e2c)
* FIX a bug which didn't make SMAC run at least two configurations per call to
  intensify
* ADD more efficient data structure to update the cost of a configuration
* FIX do only count a challenger as a run if it actually was run
  (and not only considered)(a993c29abdec98c114fc7d456ded1425a6902ce3)

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
