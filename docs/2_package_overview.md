# Package Overview

SMAC supports you in determining well-performing hyperparameter configurations for your algorithms. By being a robust and flexible framework for [Bayesian Optimization][BayesianOptimization], SMAC can improve performance within a few function evaluations. It offers several entry points and pre-sets for typical use cases, such as optimizing hyperparameters, solving low dimensional continuous (artificial) global optimization problems and configuring algorithms to perform well across multiple problem [instances][Instances].

## Features

SMAC has the following characteristics and capabilities:

#### Global Optimizer
[Bayesian Optimization][BayesianOptimization] is used for sample-efficient optimization.

#### Optimize [Black-Box][Black-Box] Functions
Optimization is only aware of input and output. It is agnostic to internals of the function.

#### Flexible Hyperparameters
Use categorical, continuous, hierarchical and/or conditional hyperparameters with the well-integrated [ConfigurationSpace](https://automl.github.io/ConfigSpace). SMAC can optimize *up to 100 hyperparameters* efficiently.

#### Any Objectives
Optimization with any [objective][Objective] (e.g., accuracy, runtime, cross-validation, ...) is possible.

#### [Multi-Objective][Multi-Objective] Optimization
Optimize arbitrary number of objectives using scalarized multi-objective algorithms. Both ParEGO [[Know06][Know06]] and mean aggregation strategies are supported.

#### [Multi-Fidelity][Multi-Fidelity] Optimization
Judge configurations on multiple [budgets][Budget] to discard unsuitable configurations early on. This will result in a massive speed-up, depending on the budgets.

#### [Instances][Instances]
Find well-performing hyperparameter configurations not only for one instance (e.g. dataset) of an algorithm, but for many.

#### Command-Line Interface
SMAC can not only be executed within a python file but also from the command line. Consequently, not only algorithms in python can be optimized, but implementations in other languages as well.

!!! note
    Command-line interface has been temporarily disabled in v2.0. Please fall back to v1.4 if you need it.

## Comparison

The following table provides an overview of SMAC's capabilities in comparison with other optimization tools.

| Package      | Complex Hyperparameter Space | [Multi-Objective][Multi-Objective] | [Multi-Fidelity][Multi-Fidelity] | [Instances][Instances] | Command-Line Interface | Parallelism |
|--------------|------------------------------|----------------------|---------------------|----------------|------------------------|-------------|
| HyperMapper  | ✅                            | ✅                    | ❌                   | ❌              | ❌                      | ❌           |
| Optuna       | ✅                            | ✅                    | ✅                   | ❌              | ✅                      | ✅           |
| Hyperopt     | ✅                            | ❌                    | ❌                   | ❌              | ✅                      | ✅           |
| BoTorch      | ❌                            | ✅                    | ✅                   | ❌              | ❌                      | ✅           |
| OpenBox      | ✅                            | ✅                    | ❌                   | ❌              | ❌                      | ✅           |
| HpBandSter   | ✅                            | ❌                    | ✅                   | ❌              | ❌                      | ✅           |
| SMAC         | ✅                            | ✅                    | ✅                   | ✅              | ✅                      | ✅           |
