"""
Callback for logging run metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example for using a callback to log run metadata to a file. Any arguments passed to the callback will be logged
to a json file at the beginning of the SMAC run (arguments must be json serializable).

Instead of editing the Git-related information (repository, branch, commit) by hand each time they change,
this information can also be added automatically using GitPython (install via "pip install GitPython").
There is an example for obtaining the information via GitPython below:
    from git import Repo
    repo = Repo(".", search_parent_directories=True)
    MetadataCallback(
        repository=repo.working_tree_dir.split("/")[-1],
        branch=str(repo.active_branch),
        commit=str(repo.head.commit),
        command=" ".join([sys.argv[0][len(repo.working_tree_dir) + 1:]] + sys.argv[1:]),
    )
"""

import sys

from ConfigSpace import Configuration, ConfigurationSpace, Float

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.callback.metadata_callback import MetadataCallback

__copyright__ = "Copyright 2023, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class Rosenbrock2D:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x0 = Float("x0", (-5, 10), default=-3)
        x1 = Float("x1", (-5, 10), default=-4)
        cs.add_hyperparameters([x0, x1])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        x1 = config["x0"]
        x2 = config["x1"]

        cost = 100.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
        return cost


if __name__ == "__main__":
    model = Rosenbrock2D()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, n_trials=200)

    # Now we use SMAC to find the best hyperparameters and add the metadata callback defined above
    HPOFacade(
        scenario,
        model.train,
        overwrite=True,
        callbacks=[
            MetadataCallback(
                project_name="My Project Name",
                repository="My Repository Name",
                branch="Name of Active Branch",
                commit="Commit Hash",
                command=" ".join(sys.argv),
                additional_information="Some Additional Information",
            )
        ],
        logging_level=999999,
    ).optimize()
