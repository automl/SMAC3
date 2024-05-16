# 2.1.0

## Improvements
- Change the surrogate model to be retrained after every iteration by default in the case of blackbox optimization
  (#1106).
- Integrate `LocalAndSortedPriorRandomSearch` functionality into `LocalAndSortedRandomSearch` (#1106).
- Change the way the `LocalAndSortedRandomSearch` works such that the incumbent always is a starting point and that
      random configurations are sampled as the basis of the local search, not in addition (#1106).

## Bugfixes
- Fix path for dask scheduler file (#1055).
- Add OrdinalHyperparameter for random forest imputer (#1065).
- Don't use mutable default argument (#1067).
- Propagate the Scenario random seed to `get_random_design` (#1066).
- Configurations that fail to become incumbents will be added to the rejected lists (#1069).
- SMAC RandomForest doesn't crash when `np.integer` used, i.e. as generated from a `np.random.RandomState` (#1084).
- Fix the handling of n_points/ challengers in the acquisition maximizers, such that this number now functions as the
     number of points that are sampled from the acquisition function to find the next challengers. Now also doesn't
     restrict the config selector to n_retrain many points for finding the max, and instead uses the defaults that are
     defined via facades/ scenarios (#1106).

## Misc
- ci: Update action version (#1072).

## Minor
- When a custom dask client is provided, emit the warning that the `n_workers` parameter is ignored only if it deviates from its default value, `1` ([#1071](https://github.com/automl/SMAC3/pull/1071)).

# 2.0.2

## Improvements
- Add an error when we get an empty dict data_to_scatter so that we can avoid an internal error caused in Dask precautiously.
- Add experimental instruction for installing SMAC in Windows via a WSL.
- More detailed documentation regarding continuing runs.
- Add a new example that demonstrates the use of intensification to speed up cross-validation for machine learning.

## Bugfixes
- Fix bug in the incumbent selection in the case that multi-fidelity is combined with multi-objective (#1019).
- Fix callback order (#1040).
- Handle configspace as dictionary in mlp and parego example.
- Adapt sgd loss to newest scikit-learn version.

# 2.0.1

## Improvements
- Callbacks registration is now a public method of the optimizer and allows callbacks to be inserted at a specific position.
- Adapt developer install instructions to include pre-commit installation
- Add option to pass a dask client to the facade, e.g. enables running on a hpc cluster (#983).
- Added scenario.use_default_config argument/attribute=False, that adds the user's configspace default configuration 
  as an additional_config to the inital design if set to True. This adds one additional configuration to the number of configs 
  originating from the initial design. Since n_trials is still respected, this results in one fewer BO steps
- Adapt developer install instructions to include pre-commit installation.
- Add option to pass a dask client to the facade, e.g. enables running on a hpc cluster (#983).
- Add example for using a callback to log run metadata to a file (#996).
- Move base callback and metadata callback files to own callback directory.
- Add a workaround to be able to pass a dataset via dask.scatter so that serialization/deserialization in Dask becomes much quicker (#993).

## Bugfixes
- The ISB-pair differences over the incumbent's configurations are computed correctly now (#956).
- Adjust amount of configurations in different stages of hyperband brackets to conform to the original paper.
- Fix validation in smbo to use the seed in the scenario.
- Change order of callbacks, intensifier callback for incumbent selection is now the first callback. 
- intensifier.get_state() will now check if the configurations contained in the queue is stored in the runhistory (#997)  


# 2.0.0

## Improvements
- Clarify origin of configurations (#908).
- Random forest with instances predicts the marginalized costs by using a C++ implementation in `pyrfr`, which is much faster (#903).
- Add version to makefile to install correct test release version.
- Add option to disable logging by setting `logging_level=False`. (#947)

## Bugfixes
- Continue run when setting incumbent selection to highest budget when using Successive Halving (#907).
- If integer features are used, they are automatically converted to strings.

## Workflows
- Added workflow to update pre-commit versions (#874).

## Misc
- Added benchmarking procedure to compare to previous releases.


# 2.0.0b1

- Completely reimplemented the intensifiers (including Successive Halving and Hyperband): All intensifiers support multi-fidelity, multi-objective and multi-threading by nature now.
- Expected behaviour for ask-and-tell interface ensured (also for Successive Halving).
- Continuing a run is now fully supported.
- Added more examples.
- Updated documentation based on new implementation.
- Added benchmark to compare different versions.

## Bugfixes
- Correct handling of integer hyperparameters in the initial design (#531)


# 2.0.0a2

## Bugfixes
- Fixed random weight (re-)generalization of multi-objective algorithms: Before the weights were generated for each call to ``build_matrix``, now we only re-generate them for every iteration.
- Optimization may get stuck because of deep copying an iterator for callback: We removed the configuration call from ``on_next_configurations_end``.

## Minor
- Removed example badget in README.
- Added SMAC logo to README.


# 2.0.0a1

## Big Changes
* We redesigned the scenario class completely. The scenario is implemented as a dataclass now and holds only environment variables (like limitations or save directory). Everything else was moved to the components directly.
* We removed runtime optimization completely (no adaptive capping or imputing anymore).
* We removed the command-line interface and restructured everything alongside. Since SMAC was building upon the command-line interface (especially in combination with the scenario), it was complicated to understand the behavior or find specific implementations. With the removal, we re-wrote everything in python and re-implemented the feature of using scripts as target functions.
* Introducing trials: Each config/seed/budget/instance calculation is a trial.
* The configuration chooser is integrated into the SMBO object now. Therefore, SMBO finally implements an ask-tell interface now.
* Facades are redesigned so that they accept instantiated components directly. If a component is not passed, a default component is used, which is specified for each facade individually in the form of static methods. You can use those static methods directly to adapt a component to your choice.
* A lot of API changes and renamings (e.g., RandomConfigurationChooser -> RandomDesign, Runhistory2EPM -> RunHistoryEncoder).
* Ambiguous variables are renamed and unified across files.
* Dependencies of modules are reduced drastically.
* We incorporated Pynisher 1.0, which ensures limitations cross-platform.
* We incorporated ConfigSpace 0.6, which simplified our examples.
* Examples and documentation are completely reworked. Examples use the new ConfigSpace, and the documentation is adapted to version 2.0.
* Transparent target function signatures: SMAC checks now explicitly if an argument is available (the required arguments are now specified in the intensifier). If there are more arguments that are not passed by SMAC, a warning is raised.
* Components implement a ``meta`` property now, all of which describe the initial state of SMAC. The facade collects all metadata and saves the initial state of the scenario.
* Improved multi-objective in general: RunHistory (in addition to RunHistoryEncoder) both incorporates the multi-objective algorithm. In other words, if the multi-objective algorithm changes the output, it directly affects the optimization process.
* Configspace is saved in json only
* StatusType is saved as integer and not as dict anymore
* We changed the behavior of continuing a run:
    * SMAC automatically checks if a scenario was saved earlier. If there exists a scenario and the initial state is the same, SMAC automatically loads the previous data. However, continuing from that run is not possible yet.
    * If there was a scenario earlier, but the initial state is different, then the user is asked to overwrite the run or to still continue the run although the state is different (Note that this only can happen if the name specified in the scenario is the same). Alternatively, an `old` to the old run is added (e.g., the name was test, it becomes test-old).
    * The initial state of the SMAC run also specifies the name (if no name in the scenario is specified). If the user changes something in the code base or in the scenario, the name and, therefore, the save location automatically changes.

## New Features
* Added a new termination feature: Use `terminate_cost_threshold` in the scenario to stop the optimization after a configuration was evaluated with a cost lower than the threshold.
* Callbacks are completely redesigned. Added callbacks to the facade are called in different positions in the Bayesian optimization loop.
* The multi-objective algorithm `MeanAggregationStrategy` supports objective weights now.
* RunHistory got more methods like ``get_incumbent`` or ``get_pareto_front``.

## Fixes
* You ever noticed that the third configuration has no origin? It's fixed now.
* We fixed ParEGO (it updates every time training is performed now).

## Optimization Changes
* Changed initial design behavior
    * You can add additional configurations now.
    * ``max_ratio`` will limit both ``n_configs`` and ``n_configs_per_hyperparameter`` but not additional configurations
    * Reduced default ``max_ratio`` to 0.1.

## Code Related
* Converted all unittests to pytests.
* Instances, seeds, and budgets can be set to none now. However, mixing none and non-none will throw an exception.


# 1.4.0

## Features
* [BOinG](https://arxiv.org/abs/2111.05834): A two-stage bayesian optimization approach to allow the 
optimizer to focus on the most promising regions.
* [TurBO](https://arxiv.org/abs/1910.01739): Reimplementaion of TurBO-1 algorithm.
* Updated pSMAC: Can pass arbitrary SMAC facades now. Added example and fixed tests.

## Improvements
* Enabled caching for multi-objectives (#872). Costs are now normalized in `get_cost` 
or optionally in `average_cost`/`sum_cost`/`min_cost` to receive a single float value. Therefore,
the cached cost values do not need to be updated everytime a new entry to the runhistory was added.

## Interface changes
* We changed the location of gaussian processes and random forests. They are in the folders
epm/gaussian_process and epm/random_forest now.
* Also, we restructured the optimizer folder and therefore the location of the acquisition functions
and configuration chooser.
* Multi-objective functions are located in the folder `multi_objective`.
* pSMAC facade was moved to the facade directory.


# 1.3.4
* Added reference to JMLR paper.
* Typos in documentations.
* Code more readable since all typings are imported at the beginning of the file.
* Updated stale bot options.


# 1.3.3
* Hotfix: Since multi-objective implementation depends on normalized costs, it now is ensured that the
cached costs are updated everytime a new entry is added.
* Removed mac-specific files.
* Added entry point for cli.
* Added `ConfigSpace` to third known parties s.t. sorting should be the same across different
operating systems.
* Fixed bugs in makefile in which tools were specified incorrectly.
* Executed isort/black on examples and tests.
* Updated README.
* Fixed a problem, which incremented time twice before taking log (#833).
* New wrapper for multi-objective models (base_uncorrelated_mo_model). Makes it easier for
developing new multi-objective models.
* Raise error if acquisition function is incompatible with the epm models.
* Restricting pynisher.


# 1.3.2
* Added stale bot support.
* If package version 0.0.0 via `get_distribution` is found, the version of the module is used
instead.
* Removed `tox.ini`.
* Moved `requirements.txt` to `setup.py`.
* Added multi-objective support for ROAR.
* Added notes in documentation that `SMAC4MF` is the closest implementation to BOHB/HpBandSter.


# 1.3.1
* Added Python 3.7 support again.


# 1.3

## Features
* [PiBO](https://openreview.net/forum?id=MMAeCXIa89): Augment the acquisition function by multiplying by a pdf given by the user.
The prior then decays over time, allowing for the optimization to carry on as per default.
* The `RunHistory` can now act as a `Mapping` in that you can use the usual methods you
can use on dicts, i.e. `len(rh)`, `rh.items()`, `rh[key]`. Previously this was usually done by
accessing `rh.data` which is still possible.

## Minor Changes
* Updated the signature of the `ROAR` facade to match with it's parent class `SMAC4AC`.
Anyone relying on the output directory **without** specifying an explicit `run_id` to a `ROAR`
facade should now expect to see the output directory at `run_0` instead of `run_1`. See #827

## Code-Quality
* Updated and integrated flake8, mypy, black, and isort.

## Documentation
* SMAC uses [automl_sphinx_theme](https://github.com/automl/automl_sphinx_theme) now and therefore
the API is displayed nicer.


# 1.2

## Features
* Added multi-objective optimization via Mean-Aggregation or Par-EGO (#817, #818). Both approaches normalize
the costs objective-wise based on all data in the history.

## Major Changes
* Results are instantly saved by default now. That means, runhistory.json is saved every time
a trial is added.
* Determinstic behaviour (defined in scenario) is default now. Calling a function/TAE with the same
seed and configuration is expected to be the same.
* Limit resources behaviour is by default false now. This is particually important because pynisher
does not work on all machines (e.g. Colab, Mac, Windows, ...) properly.
* Renamed scenario object `save_results_instantly` to `save_instantly`.
* Added `multi_objectives` as scenario argument.
* Expanded `cost_for_crash` for multi-objective support.

## Examples
* Integrated spear_qcp example for commandline.
* Python examples are now executed so that the output in the documentation is shown.
* Added multi-objective example.

## Documentation
* Added runhistory page.

## Workflow Clean-up
* Adds PEP 561 compliance (exports types so other packages can be aware of them).
* Allow manual workflow_dispatch on actions that might require it (can manually trigger them from github UI).
* Prevent the double trigger of actions by making push and pull_request and more strict.
* A push to a pull request should no longer cause double the amount of tests to run (along with the other workflows that had on: [push, pull_request].
* Some general cleanup, giving names to some actions, adding some linebreaks to break up things, ...
* Command-line examples are tested again.
* pytest.yaml:
  * Now scheduled to auto run everyday instead of every week.
  * Clean up the body of the steps and move some things to env var.
  * Scaffold for matrix that includes windows and mac testing (currently excluded, see comments).
  * Includes tests for Python 3.10.
  * Changed the boolean flags in the matrix to just be a categorical, easier to read.

## Minor Changes
* Specified that dask should not cache functions/results (#803) .
* Handles invalid configuration vectors gracefully (#776).
* Specified scenario docs that also SMAC options can be used.
* Docs display init methods now.
* Parameters in the docs are shown first now.
* Successive Halving only warns you once if one worker is used only.
* Statistics are better readable now.
* Sobol sequence does not print warnings anymore.


# 1.1.1

## Minor Changes
* Added comparison between SMAC and similar tools.
* Updated installation guide.
* Added a warning that CLI is only available when installing from GitHub.


# 1.1

## Features
* Option to use an own stopping strategy using `IncorporateRunResultCallback`.


## Major Changes
* Documentation was updated thoroughly. A new theme with a new structure is provided and all pages
  have been updated. Also, the examples revised and up-to-date.
* Changed `scripts/smac` to `scripts/smac.py`.

## Minor Changes
* `README.md` updated.
* `CITATION.cff` added.
* Made `smac-validate.py` consistent with runhistory and tae. (#762)
* `minR`, `maxR` and `use_ta_time` can now be initialized by the scenario. (#775)
* `ConfigSpace.util.get_one_exchange_neighborhood`'s invalid configurations are ignored. (#773)

## Bug Fixes
* Fixed an incorrect adaptive capping behaviour. (#749)
* Avoid the potential `ValueError` raised by `LocalSearch._do_search`. (#773)


# 1.0.1

## Minor Changes
* Added license information to every file.
* Fixed a display bug inside usage recommendation. 


# 1.0.0

The main purpose of this release is to be synchronized with our upcoming paper.
Since many urgent features were already taken care of in 0.14.0, this release mainly focuses on better documentation and examples.

## Features
* Examples and quickstart guide can now be generated by [sphinx-gallry](https://sphinx-gallery.github.io/stable/index.html).
* Added make command `make doc` and `make doc-with-examples`.

## Major changes
* Examples are separated into categories.
* Renamed facade SMAC4BO to SMAC4BB (black-box).
* Add thompson sampling as a new acquisition function

## Minor Changes
* Included linkcheck and buildapi to the `make doc` command.
* `quickstart.rst` was converted to `quickstart_example.py` to be processed by sphinx-gallery.
* Examples renamed from `*.py` to `*_example.py`, unless file name was `*_func.py`, in which case it was unchanged.
* Flake8 fixes for spear_qcp as there were a lot of complaints running `pre-commit`.
* Fixes pydoc issues.
* Fixed links in the README.
* Fixed warnings given during the doc build.
* Fixed inconsistent output shape described in `smac.epm.gaussian_process.GaussianProcess.sample_functions`
* Examples are wrapped inside `if __name__ == "__main__"`, fixing problems on mac.


# 0.14.0

## Breaking Changes
* `BOHB4HPO` facade has been renamed to `SMAC4MF` facade (#738)
* Require `scipy` >= 1.7 (#729)
* Require `emcee` >= 3.0.0 (#723)

## Major Changes
* Drop support for Python 3.6 (#726)
* Added Colab to try SMAC in your browser! (#697)

## Minor Changes
* Added gradient boosting example, removed random forest example (#722)
* `lazy_import` dependency dropped (#741)
* Replaced `pyDOE` requirement with `scipy` for LHD design (#735)
* Uses scrambled Sobol Sequence (#733)
* Moved to Github actions (#715)
* Improved testing (#720, #723, #739, #743)
* Added option `save_results_instantly` in scenario object to save results instantly (#728)
* Changed level of intensification messages to debug (#724)

## Bug Fixes
* Github badges updated (#732)
* Fixed memory limit issue for `pynisher` (#717)
* More robust multiprocessing (#709, #712)
* Fixed serialization with runhistory entries (#706)
* Separated evaluation from get next challengers in intensification (#734)
* Doc fixes (#727, #714)


# 0.13.1

## Minor Changes
* Improve error message for first run crashed (#694).
* Experimental: add callback mechanism (#703).

## Bug fixes
* Fix a bug which could make successive halving fail if run in parallel (#695).
* Fix a bug which could cause hyperband to ignore the lowest budget (#701).


# 0.13.0

## Major Changes
* Separated evaluation from get next challengers in intensification (#663)
* Implemented parallel SMAC using dask (#675, #677, #681, #685, #686)
* Drop support for Python 3.5

## Minor Changes
* Update Readme
* Remove runhistory from TAE (#663)
* Store SMAC's internal config id in the configuration object (#679)
* Introduce Status Type STOP (#690)

## Bug Fixes
* Only validate restriction of Sobol Sequence when choosing Sobol Sequence (#664)
* Fix wrong initialization of list in local search (#680)
* Fix setting random seed with a too small range in Latin Hypercube design (#688)


# 0.12.3

## Minor Changes

* Use Scipy's Sobol sequence for the initial design instead of a 3rd-party package (#600)
* Store start and end time of function evaluation (#647)

## Bug Fixes

* Fixes an issue in the Bayesian optimization facade which triggered an exception when tuning categorical 
  hyperparameters (#666)
* Fixes an issue in the Gaussian process MCMC which resulted in reduced execution speed and reduced performance (#666)


# 0.12.2

## Bug Fixes

* Fixes the docstring of SMAC's default acquisition function optimizer (#653)
* Correctly attributes the configurations' origin if using the `FixedSet` acquisition function optimizer (#653)
* Fixes an infinite loop which could occur if using only a single configuration per iteration (#654)
* Fixes a bug in the kernel construction of the `BOFacade` (#655)


# 0.12.1

## Minor Changes

* Upgrade the minimal scikit-learn dependency to 0.22.X.
* Make GP predictions faster (#638)
* Allow passing `tae_runner_kwargs` to `ROAR`.
* Add a new StatusType `DONOTADVANCE` for runs that would not benefit from a higher budgets. Such runs are always used 
  to build a model for SH/HB (#632)
* Add facades/examples for HB/SH (#610)
* Compute acquisition function only if necessary (#627,#629)

## Bug Fixes
* Fixes a bug which caused SH/HB to consider TIMEOUTS on all budgets for model building (#632)
* Fixed a bug in adaptive capping for SH (#619,#622)


# 0.12.0

## Major Changes

* Support for Successive Halving and Hyperband as new instensification/racing strategies.
* Improve the SMAC architecture by moving from an architecture where new candidates are passed to the racing algorithm 
  to an architecture where the racing algorithm requests new candidates, which is necessary to implement the
  [BOHB](http://proceedings.mlr.press/v80/falkner18a.html) algorithm (#551).
* Source code is now PEP8 compliant. PEP8 compliance is checked by travis-ci (#565).
* Source code is now annotated with type annotation and checked with mypy.

## Minor Changes

* New argument to directly control the size of the initial design (#553).
* Acquisition function is fed additional arguments at update time (#557).
* Adds new acquisition function maximizer which goes through a list of pre-specified configurations (#558).
* Document that the dependency pyrfr does not work with SWIG 4.X (#599).
* Improved error message for objects which cannot be serialized to json (#453).
* Dropped the random forest with HPO surrogate which was added in 0.9.
* Dropped the EPILS facade which was added in 0.6.
* Simplified the interface for constructing a runhistory object.
* removed the default rng from the Gaussian process priors (#554).
* Adds the possibility to specify the acquisition function optimizer for the random search (ROAR) facade (#563).
* Bump minimal version of `ConfigSpace` requirement to 0.4.9 (#578).
* Examples are now rendered on the website using sphinx gallery (#567).

## Bug fixes

* Fixes a bug which caused SMAC to fail for Python function if `use_pynisher=False` and an exception was raised
  (#437).
* Fixes a bug in which samples from a Gaussian process were shaped differently based on the number of dimesions of
  the `y`-array used for fitting the GP (#556).
* Fixes a bug with respect saving data as json (#555).
* Better error message for a sobol initial design of size `>40` ( #564).
* Add a missing return statement to `GaussianProcess._train`.


# 0.11.1

## Changes

* Updated the default hyperparameters of the Gaussian process facade to follow recent research (#529)
* Enabled `flake8` code style checks for newly merged code (#525)

# 0.11.0

## Major changes

* Local search now starts from observed configurations with high acquisition function values, low cost and the from 
  unobserved configurations with high acquisition function values found by random search (#509)
* Reduces the number of mandatory requirements (#516)
* Make Gaussian processes more resilient to linalg error by more aggressively adding noise to the diagonal (#511)
* Inactive hyperparameters are now imputed with a value outside of the modeled range (-1) (#508)
* Replace the GP library George by scikit-learn (#505)
* Renames facades to better reflect their use cases (#492), and adds a table to help deciding which facade to use (#495)
* SMAC facades now accept class arguments instead of object arguments (#486)

## Minor changes

* Vectorize local search for improved speed (#500)
* Extend the Sobol and LHD initial design to work for non-continuous hyperparameters as well applying an idea similar
  to inverse transform sampling (#494)
  
## Bug fixes

* Fixes a regression in the validation scripts (#519)
* Fixes a unit test regression with numpy 1.17 (#523)
* Fixes an error message (#510)
* Fixes an error making random search behave identical for all seeds

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
