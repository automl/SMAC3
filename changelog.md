# 0.10.0

## Major changes

* ADD further acquisition functions: PI and LCB
* SMAC can now be installed without installing all its dependencies
* Simplify setup.py by moving most thing to setup.cfg

## Bug fixes

* RM typing as requirement
* FIX import of authors in setup.py
* MAINT use json-file as standard pcs format for internal logging 

# 0.9

## Major changes
* ADD multiple optional initial designs: LHC, Factorial Design, Sobol
* ADD fmin interface know uses BORF facade (should perform much better on continuous, low-dimensional functions)
* ADD Hydra (see "Hydra: Automatically Configuring Algorithms for Portfolio-Based Selection" by Xu et al)
* MAINT Not every second configuration is randomly drawn, but SMAC samples configurations randomly with a given probability (default: 0.5)
* MAINT parsing of options

## Interface changes
* ADD two new interfaces to optimize low dimensional continuous functions (w/o instances, docs missing)
  * BORF facade: Initial design + Tuned RF
  * BOGP interface: Initial design + GP 
* ADD options to control acquisition function optimization
* ADD option to transform function values (log, inverse w/ and w/o scaling)
* ADD option to set initial design

## Minor changes
* ADD output of estimated cost of final incumbent
* ADD explanation of "deterministic" option in documentation
* ADD save configspace as json
* ADD random forest with automated HPO (not activated by default)
* ADD optional linear cooldown for interleaving random configurations (not active by default)
* MAINT Maximal cutoff time of pynisher set to UINT16
* MAINT make SMAC deterministic if function is deterministic, the budget is limited and the run objective is quality
* MAINT SLS on acquisition function (plateau walks)
* MAINT README
* FIX abort-on-first-run-crash
* FIX pSMAC input directory parsing
* FIX fmin interface with more than 10 parameters
* FIX no output directory if set to '' (empty string)
* FIX use `np.log` instead of `np.log10`
* FIX No longer use law of total variance for uncertainty prediction for RFs as EPM, but only variance over trees (no variance in trees)
* FIX Marginalize over instances inside of each tree of the forest leads to better uncertainty estimates (motivated by the original SMAC implementation)


# 0.8

## Major changes

* Upgrade to ConfigSpace (0.4.X), which is not backwards compatible. On the plus
  side, the ConfigSpace is about 3-10 times faster, depending on the task.
* FIX #240: improved output directory structure. If the user does not specify
  an output directory a SMAC experiment will have the following structure:
  `smac_/run_<run_id>/*.json`. The user can specify a output directory, e.g.
  `./myExperiment` or `./myExperiment/` which results in
  `./myExperiment/run_<run_id>/*.json`.
* Due to changes in AnaConda's compiler setup we drop the unit tests for
  python3.4.

## Interface changes

* Generalize the interface of the acquisition functions to work with
  ConfigSpaces's configuration objects instead of numpy arrays.
* The acquisition function optimizer can now be passed to the SMBO object.
* A custom SMBO class can now be passed to the SMAC builder object.
* `run_id` is no longer an argument to the Scenario object, making the interface
  a bit cleaner.

## Minor changes

* #333 fixes an incompability with `uncorrelated_mo_rf_with_instances`.
* #323 fixes #324 and #319, which both improve the functioning of the built-in
  validation tools.
* #350 fixes random search, which could accidentaly use configurations found my
  a local acquisition function optimizer.
* #336 makes validation more flexible.


# 0.7.2

* Introduce version upper bound on ConfigSpace dependency (<0.4).

# 0.7.1

* FIX #193, restoring the scenario now possible.
* ADD #271 validation.
* FIX #311 abort on first crash.
* FIX #318, ExecuteTARunOld now always returns a StatusType.

# 0.6

## Major changes

* MAINT documentation (nearly every part was improved and extended, 
  including installation, examples, API).
* ADD EPILS as mode (modified version of ParamILS).
* MAINT minimal required versions of configspace, pyrfr, sklearn increased
  (several issues fixed in new configspace version).
* MAINT for quality scenarios, the user can specify the objective 
  value for crashed runs 
  (returned NaN and Inf are replaced by value for crashed runs).

## Minor changes

* FIX issue #220, do not store external data in runhistory.
* MAINT TAEFunc without pynisher possible.
* MAINT intensification: minimal number of required challengers parameterized.
* FIX saving duplicated (capped) runs.
* FIX handling of ordinal parameters.
* MAINT runobj is now mandatory.
* FIX arguments passed to pyrfr.

# 0.5

## Major changes

* MAINT #192: SMAC uses version 0.4 of the random forest library pyrfr. As a
  side-effect, the library [swig](http://www.swig.org/) is necessary to build
  the random forest.
* MAINT: random samples which are interleaved in the list of challengers are now
  obtained from a generator. This reduces the overhead of sampling random
  configurations.
* FIX #117: only round the cutoff when running a python function as the target
  algorithm.
* MAINT #231: Rename the submodule `smac.smbo` to `smac.optimizer`.
* MAINT #213: Use log(EI) as default acquisition function when optimizing
  running time of an algorithm.
* MAINT #223: updated example of optimizing a random forest with SMAC.
* MAINT #221: refactored the EPM module. The PCA on instance features is now
  part of fitting the EPM instead of reading a scenario. Because of this
  restructuring, the PCA can now take instance features which are external
  data into account.

## Minor changes

* SMAC now outputs scenario options if the log level is `DEBUG` (2f0ceee).
* SMAC logs the command line call if invoked from the command line (3accfc2).
* SMAC explicitly checks that it runs in `python>=3.4`.
* MAINT #226: improve efficientcy when loading the runhistory from a json file.
* FIX #217: adds milliseconds to the output directory names to avoid race.
  conditions when starting multiple runs on a cluster.
* MAINT #209: adds the seed or a pseudo-seed to the output directory name for
  better identifiability of the output directories.
* FIX #216: replace broken call to in EIPS acqusition function.
* MAINT: use codecov.io instead of coveralls.io.
* MAINT: increase minimal required version of the ConfigSpace package to 0.3.2.

# 0.4

* ADD #204: SMAC now always saves runhistory files as `runhistory.json`.
* MAINT #205: the SMAC repository now uses codecov.io instead of coveralls.io.
* ADD #83: support of ACLIB 2.0 parameter configuration space file.
* FIX #206: instances are now explicitly cast to `str`. In case no instance is
  given, a single `None` is used, which is not cast to `str`.
* ADD #200: new convenience function to retrieve an `X`, `y` representation
  of the data to feed it to a new fANOVA implementation.
* MAINT #198: improved pSMAC interface.
* FIX #201: improved handling of boolean arguments to SMAC.
* FIX #194: fixes adaptive capping with re-occurring configurations.
* ADD #190: new argument `intensification_percentage`.
* ADD #187: better dependency injection into main SMAC class to avoid
  ill-configured SMAC objects.
* ADD #161: log scenario object as a file.
* ADD #186: show the difference between old and new incumbent in case of an
  incumbent change.
* MAINT #159: consistent naming of loggers.
* ADD #128: new helper method to get the target algorithm evaluator.
* FIX #165: default value for par = 1.
* MAINT #153: entries in the trajectory logger are now named tuples.
* FIX #155: better handling of SMAC shutdown and crashes if the first
  configuration crashes.

# 0.3

* Major speed improvements when sampling new configurations:
    * Improved conditional hyperparameter imputation (PR #176).
    * Faster generation of the one exchange neighborhood (PR #174).
* FIX #171 potential bug with pSMAC.
* FIX #175 backwards compability for reading runhistory files.

# 0.2.4

* CI only check code quality for python3.
* Perform local search on configurations from previous runs as proposed in the
  original paper from 2011 instead of random configurations as implemented
  before.
* CI run travis-ci unit tests with python3.6.
* FIX #167, remove an endless loop which occured when using pSMAC.

# 0.2.3

* MAINT refactor Intensifcation and adding unit tests.
* CHANGE StatusType to Enum.
* RM parameter importance package.
* FIX ROAR facade bug for cli.
* ADD easy access of runhistory within Python.
* FIX imputation of censored data.
* FIX conversion of runhistory to EPM training data (in particular running
  time data).
* FIX initial run only added once in runhistory.
* MV version number to a separate file.
* MAINT more efficient computations in run_history (assumes average as
  aggregation function across instances).

# 0.2.2

* FIX 124: SMAC could crash if the number of instances was less than seven.
* FIX 126: Memory limit was not correctly passed to the target algorithm
  evaluator.
* Local search is now started from the configurations with highest EI, drawn by
  random sampling.
* Reduce the number of trees to 10 to allow faster predictions (as in SMAC2).
* Do an adaptive number of stochastic local search iterations instead of a fixd
  number (a5914a1d97eed2267ae82f22bd53246c92fe1e2c).
* FIX a bug which didn't make SMAC run at least two configurations per call to
  intensify.
* ADD more efficient data structure to update the cost of a configuration.
* FIX do only count a challenger as a run if it actually was run
  (and not only considered)(a993c29abdec98c114fc7d456ded1425a6902ce3).

# 0.2.1

* CI: travis-ci continuous integration on OSX.
* ADD: initial design for mulitple configurations, initial design for a 
  random configuration.
* MAINT: use sklearn PCA if more than 7 instance features are available (as 
  in SMAC 1 and 2).
* MAINT: use same minimum step size for the stochastic local search as in
  SMAC2.
* MAINT: use same number of imputation iterations as in SMAC2.
* FIX 98: automatically seed the configuration space object based on the SMAC
  seed.

# 0.2

* ADD 55: Separate modules for the initial design and a more flexible 
  constructor for the SMAC class.
* ADD 41: Add ROAR (random online adaptive racing) class.
* ADD 82: Add fmin_smac, a scipy.optimize.fmin_l_bfgs_b-like interface to the
  SMAC algorithm.
* NEW documentation at https://automl.github.io/SMAC3/stable and 
  https://automl.github.io/SMAC3/dev.
* FIX 62: intensification previously used a random seed from np.random 
  instead of from SMAC's own random number generator.
* FIX 42: class RunHistory can now be pickled.
* FIX 48: stats and runhistory objects are now injected into the target 
  algorithm execution classes.
* FIX 72: it is now mandatory to either specify a configuration space or to 
  pass the path to a PCS file.
* FIX 49: allow passing a callable directly to SMAC. SMAC will wrap the 
  callable with the appropriate target algorithm runner.

# 0.1.3

* FIX 63 using memory limit for function target algorithms (broken since 0.1.1).

# 0.1.2

* FIX 58 output of the final statistics.
* FIX 56 using the command line target algorithms (broken since 0.1.1).
* FIX 50 as variance prediction, we use the average predicted variance across
  the instances.

# 0.1.1

* NEW leading ones examples.
* NEW raise exception if unknown parameters are given in the scenario file.
* FIX 17/26/35/37/38/39/40/46.
* CHANGE requirement of ConfigSpace package to 0.2.1.
* CHANGE cutoff default is now None instead of 99999999999.


# 0.1.0

* Moved to github instead of bitbucket.
* ADD further unit tests.
* CHANGE Stats object instead of static class.
* CHANGE requirement of ConfigSpace package to 0.2.0.
* FIX intensify runs at least two challengers.
* FIX intensify skips incumbent as challenger.
* FIX Function TAE runner passes random seed to target function.
* FIX parsing of emtpy lines in scenario file.

# 0.0.1

* initial release.
