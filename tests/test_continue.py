from __future__ import annotations

import itertools
import pytest
import warnings

import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.acquisition.maximizers import RandomSearch
from smac.acquisition.functions import EI

from smac import HyperparameterFacade, Scenario, MultiFidelityFacade, BlackBoxFacade
from smac.intensification import SuccessiveHalving

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class DigitsDataset:
    def __init__(self) -> None:
        self._data = datasets.load_digits()

    def get_instances(self, n=45) -> list[str]:
        """Create instances from the dataset which include two classes only."""
        return [f"{classA}-{classB}" for classA, classB in itertools.combinations(self._data.target_names, 2)][:n]

    def get_instance_features(self, n=45) -> dict[str, list[int | float]]:
        """Returns the mean and variance of all instances as features."""
        features = {}
        for instance in self.get_instances(n):
            data, _ = self.get_instance_data(instance)
            features[instance] = [np.mean(data), np.var(data)]

        return features

    def get_instance_data(self, instance: str) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve data from the passed instance."""
        # We split the dataset into two classes
        classA, classB = instance.split("-")
        indices = np.where(np.logical_or(int(classA) == self._data.target, int(classB) == self._data.target))

        data = self._data.data[indices]
        target = self._data.target[indices]

        return data, target


class SGD:
    def __init__(self, dataset: DigitsDataset) -> None:
        self.dataset = dataset

    @property
    def configspace(self) -> ConfigurationSpace:
        """Build the configuration space which defines all parameters and their ranges for the SGD classifier."""
        cs = ConfigurationSpace(seed=0)

        # We define a few possible parameters for the SGD classifier
        alpha = Float("alpha", (0, 1), default=1.0)
        l1_ratio = Float("l1_ratio", (0, 1), default=0.5)
        learning_rate = Categorical("learning_rate", ["constant", "invscaling", "adaptive"], default="constant")
        eta0 = Float("eta0", (0.00001, 1), default=0.1, log=True)

        # Add the parameters to configuration space
        cs.add_hyperparameters([alpha, l1_ratio, learning_rate, eta0])

        return cs

    def train(self, config: Configuration, instance: str = "0-1", budget: float = 30, seed: int = 0) -> float:
        """Creates a SGD classifier based on a configuration and evaluates it on the
        digits dataset using cross-validation."""

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # SGD classifier using given configuration
            clf = SGDClassifier(
                loss="log",
                penalty="elasticnet",
                alpha=config["alpha"],
                l1_ratio=config["l1_ratio"],
                learning_rate=config["learning_rate"],
                eta0=config["eta0"],
                max_iter=budget,
                early_stopping=True,
                random_state=seed,
            )

            # Get instance
            data, target = self.dataset.get_instance_data(instance)

            cv = StratifiedKFold(n_splits=4, random_state=seed, shuffle=True)  # to make CV splits consistent
            scores = cross_val_score(clf, data, target, cv=cv)

        return 1 - np.mean(scores)


# def test_tell_one_seed():
#     dataset = DigitsDataset()
#     model = SGD(dataset)
#     seed = 462

#     scenario = Scenario(
#         model.configspace,
#         deterministic=True,
#         n_trials=10,  # We want to try max 5000 different configurations
#         # min_budget=1,  # Use min one instance
#         # max_budget=45,  # Use max 45 instances (if we have a lot of instances we could constraint it)
#         # instances=dataset.get_instances(),
#         # instance_features=dataset.get_instance_features(),
#     )

#     # Create our SMAC object and pass the scenario and the train method
#     smac = HyperparameterFacade(
#         scenario,
#         model.train,
#         initial_design=HyperparameterFacade.get_initial_design(scenario, n_configs=5, max_ratio=1),
#         intensifier=HyperparameterFacade.get_intensifier(scenario, max_config_calls=1),
#         logging_level=0,
#         overwrite=True,
#     )

#     # We can provide SMAC with custom configurations first
#     for config in model.configspace.sample_configuration(10):
#         cost = model.train(config, seed=seed)

#         trial_info = TrialInfo(config, seed=seed)
#         trial_value = TrialValue(cost=cost, time=0.5)

#         smac.tell(trial_info, trial_value)

#     assert smac.stats.finished == 10
#     assert smac.stats.submitted == 0  # We have 0 submittions because we don't call the ask method

#     smac.optimize()

#     # After optimization we expect to have +10 finished
#     assert smac.stats.finished == 10 + 10
#     assert smac.stats.submitted == 11  # We have one submittion which is skipped
#     assert len(smac.runhistory) == 20  # However, the skipped one is not saved anymore

#     # We expect SMAC to use the same seed if configs with a seed were passed
#     for k in smac.runhistory.keys():
#         assert k.seed == seed


# def test_tell_multiple_seeds():
#     dataset = DigitsDataset()
#     model = SGD(dataset)
#     seeds = [82, 9444, 726]

#     scenario = Scenario(
#         model.configspace,
#         deterministic=True,
#         n_trials=30,  # We want to try max 5000 different configurations
#         # min_budget=1,  # Use min one instance
#         # max_budget=45,  # Use max 45 instances (if we have a lot of instances we could constraint it)
#         # instances=dataset.get_instances(),
#         # instance_features=dataset.get_instance_features(),
#     )

#     # Create our SMAC object and pass the scenario and the train method
#     smac = HyperparameterFacade(
#         scenario,
#         model.train,
#         initial_design=HyperparameterFacade.get_initial_design(scenario, n_configs=5, max_ratio=1),
#         intensifier=HyperparameterFacade.get_intensifier(scenario, max_config_calls=len(seeds)),
#         logging_level=0,
#         overwrite=True,
#     )

#     # We can provide SMAC with custom configurations first
#     for config in model.configspace.sample_configuration(10):
#         for seed in seeds:
#             cost = model.train(config, seed=seed)

#             trial_info = TrialInfo(config, seed=seed)
#             trial_value = TrialValue(cost=cost, time=0.5)

#             smac.tell(trial_info, trial_value)

#     assert smac.stats.finished == 10 * len(seeds)
#     assert smac.stats.submitted == 0  # We have 0 submittions because we don't call the ask method

#     smac.optimize()

#     # After optimization we expect to have +30 finished
#     assert smac.stats.finished == 10 * len(seeds) + 30
#     assert smac.stats.submitted == 31  # We have one submittion which is skipped
#     assert len(smac.runhistory) == 10 * len(seeds) + 30  # However, the skipped one is not saved anymore

#     # We expect SMAC to use the same seed if configs with a seed were passed
#     for k in smac.runhistory.keys():
#         assert k.seed in seeds


# def test_get_target_function_seeds():
#     dataset = DigitsDataset()
#     model = SGD(dataset)

#     scenario = Scenario(
#         model.configspace,
#         deterministic=False,
#         n_trials=100,  # We want to try max 5000 different configurations
#         # min_budget=1,  # Use min one instance
#         # max_budget=45,  # Use max 45 instances (if we have a lot of instances we could constraint it)
#         # instances=dataset.get_instances(10),
#         # instance_features=dataset.get_instance_features(10),
#     )

#     # Create our SMAC object and pass the scenario and the train method
#     smac = HyperparameterFacade(
#         scenario,
#         model.train,
#         # model=BlackBoxFacade.get_model(scenario),
#         initial_design=HyperparameterFacade.get_initial_design(scenario, n_configs=5, max_ratio=1),
#         intensifier=HyperparameterFacade.get_intensifier(
#             scenario,
#             max_config_calls=10,
#             intensify_percentage=0.0,  # Make it reproducible
#         ),
#         overwrite=True,
#     )

#     smac.optimize()
#     seeds = []
#     for k in smac.runhistory.keys():
#         if k.seed not in seeds:

#             seeds += [k.seed]

#     # We expect max 10 seeds
#     assert len(seeds) == 10


# def test_get_target_function_seeds_with_instances():
#     dataset = DigitsDataset()
#     model = SGD(dataset)

#     scenario = Scenario(
#         model.configspace,
#         deterministic=False,
#         n_trials=100,  # We want to try max 5000 different configurations
#         # min_budget=1,  # Use min one instance
#         # max_budget=45,  # Use max 45 instances (if we have a lot of instances we could constraint it)
#         instances=dataset.get_instances(2),
#         # instance_features=dataset.get_instance_features(10),
#     )

#     # Create our SMAC object and pass the scenario and the train method
#     smac = HyperparameterFacade(
#         scenario,
#         model.train,
#         # model=BlackBoxFacade.get_model(scenario),
#         initial_design=HyperparameterFacade.get_initial_design(scenario, n_configs=5, max_ratio=1),
#         intensifier=HyperparameterFacade.get_intensifier(
#             scenario,
#             max_config_calls=10,
#             intensify_percentage=0.0,  # Make it reproducible
#         ),
#         overwrite=True,
#     )

#     # We expect `max_config_calls` seeds here
#     assert len(smac._intensifier.get_target_function_seeds()) == 10

#     smac.optimize()
#     seeds = []
#     for k in smac.runhistory.keys():
#         if k.seed not in seeds:

#             seeds += [k.seed]

#     # We expect max 10 seeds
#     assert len(seeds) == 10


def test_tell_sh():
    """Fails if using tell without ask in SH."""
    dataset = DigitsDataset()
    model = SGD(dataset)
    seeds = [35, 82, 9444]

    scenario = Scenario(
        model.configspace,
        deterministic=False,
        n_trials=50,
        min_budget=1,
        instances=dataset.get_instances(2),
    )

    # Create our SMAC object and pass the scenario and the train method
    smac = MultiFidelityFacade(
        scenario,
        model.train,
        initial_design=MultiFidelityFacade.get_initial_design(scenario, n_configs=5, max_ratio=1),
        intensifier=MultiFidelityFacade.get_intensifier(scenario, n_seeds=5),
        overwrite=True,
    )

    for config in model.configspace.sample_configuration(10):
        for seed in seeds:
            cost = model.train(config, seed=seed)

            trial_info = TrialInfo(config, seed=seed)
            trial_value = TrialValue(cost=cost, time=0.5)

            # Screw it, it does not work ...
            with pytest.raises(RuntimeError, match="Successive Halving does not work with.*"):
                smac.tell(trial_info, trial_value)

    # Now we try to ask first
    # Create our SMAC object and pass the scenario and the train method
    smac = MultiFidelityFacade(
        scenario,
        model.train,
        initial_design=MultiFidelityFacade.get_initial_design(scenario, n_configs=5, max_ratio=1),
        intensifier=SuccessiveHalving(scenario, n_seeds=5),
        overwrite=True,
    )

    for _ in range(50):
        info = smac.ask()
        value = TrialValue(cost=cost, time=0.5)

        # That should work :)
        smac.tell(info, value)

    # Just check if we registered something
    assert (smac._intensifier._intensifier_instances[0]._run_tracker) > 0

    # And now even optimize should work
    smac.optimize()
    assert smac.stats.finished == 50  # The tell does not increase the counter
