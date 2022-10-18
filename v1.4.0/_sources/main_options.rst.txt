:hydra_iterations: number of hydra iterations. Only active if mode is set to Hydra Default: 3.
:hydra_validation: set to validate incumbents on. valX => validation set of size training_set * 0.X Default: train.
:incumbents_per_round: number of configurations to keep per psmac/hydra iteration. Default: 1.
:mode: Configuration mode. Default: SMAC4AC.
:n_optimizers: number of optimizers to run in parallel per psmac/hydra iteration. Default: 1.
:psmac_validate: Validate all psmac configurations.
:random_configuration_chooser: path to a python module containing a class `RandomConfigurationChooserImpl`implementing the interface of `RandomConfigurationChooser`
:restore_state: Path to directory with SMAC-files.
:scenario_file: Scenario file in AClib format.
:seed: Random Seed. Default: 1.
:verbose_level: Verbosity level. Default: 20.


