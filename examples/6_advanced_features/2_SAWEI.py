"""
Self-Adjusting Weighted Expected Improvement (SAWEI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of applying SMAC to optimize a quadratic function using SAWEI [Benjamins et al, 2023].

[Benjamins et al., 2023] C. Benjamins, E. Raponi, A. Jankovic, C. Doerr and M. Lindauer. 
                Self-Adjusting Weighted Expected Improvement for Bayesian Optimization.
                AutoML Conference 2023.
"""

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt

from smac import BlackBoxFacade
from smac import RunHistory, Scenario

import smac
import numpy as np
import pandas as pd
import seaborn as sns
from smac.callback.sawei_callback import get_sawei_kwargs, WEITracker, UpperBoundRegretCallback


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class QuadraticFunction:
    @property
    def configspace(self) -> ConfigurationSpace:
        """Configuration/search space of the problem

        Returns:
            ConfigurationSpace: Configuration space.
        """
        cs = ConfigurationSpace(seed=0)
        x = Float("x", (-5, 5), default=-5)
        cs.add_hyperparameters([x])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        x = config["x"]
        return x**2


def plot(runhistory: RunHistory, incumbent: Configuration) -> None:
    """Plot the objective function, all trials and the incumbent

    Args:
        runhistory (RunHistory): Runhistory object after optimization
        incumbent (Configuration): Best configuration found by SMAC
    """
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    plt.figure()

    # Plot ground truth
    x = list(np.linspace(-5, 5, 100))
    y = [xi * xi for xi in x]
    plt.plot(x, y)

    # Plot all trials
    for k, v in runhistory.items():
        config = runhistory.get_config(k.config_id)
        x = config["x"]
        y = v.cost  # type: ignore
        plt.scatter(x, y, c="blue", alpha=0.1, zorder=9999, marker="o")

    # Plot incumbent
    plt.scatter(incumbent["x"], incumbent["x"] * incumbent["x"], c="red", zorder=10000, marker="x")

    plt.show()

def plot_sawei(facade: smac.facade.abstract_facade.AbstractFacade) -> None:
    """Plot the interpolation weight alpha of SAWEI and the UBR (Upper Bound Regret)

    Args:
        facade (smac.facade.abstract_facade.AbstractFacade): SMAC facade object after optimization
    """

    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    
    weitracker = None
    ubrtracker = None
    for callback in facade._callbacks:
        if isinstance(callback, WEITracker):
            weitracker = callback
        if isinstance(callback, UpperBoundRegretCallback):
            ubrtracker = callback

    # The data also lies in the output folder as a csv file
    df_alpha = pd.DataFrame(weitracker.history)
    df_ubr = pd.DataFrame(ubrtracker.history)
    df = pd.merge(df_alpha, df_ubr, on="n_evaluated")

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax = sns.lineplot(data=df, x="n_evaluated", y="alpha", ax=ax, label="alpha", color=sns.color_palette()[0])
    ax2 = sns.lineplot(data=df, x="n_evaluated", y="ubr", ax=ax2, label="ubr", color=sns.color_palette()[1])
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_ylabel("alpha")
    ax2.set_ylabel("ubr")
    ax.set_xlabel("n_evaluated")
    plt.show()
    
    

if __name__ == "__main__":
    model = QuadraticFunction()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, deterministic=True, n_trials=50, seed=np.random.randint(low=0,high=10000))

    # Get the kwargs necessary to use SAWEI
    # SAWEI is implemented as a chain of callbacks
    sawei_kwargs = get_sawei_kwargs()

    # Now we use SMAC to find the best hyperparameters
    optimizer = BlackBoxFacade(
        scenario,
        model.train,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        **sawei_kwargs  # Add SAWEI
    )

    incumbent = optimizer.optimize()

    # Get cost of default configuration
    default_cost = optimizer.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = optimizer.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")

    # Let's plot it too
    plot(optimizer.runhistory, incumbent)

    # Let's plot the interpolation weight alpha of SAWEI and the UBR (Upper Bound Regret)
    plot_sawei(optimizer)
