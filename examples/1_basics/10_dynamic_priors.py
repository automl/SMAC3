"""Stating User Priors During a Run
# Flags: doc-Runnable

A user learns things about their problem while the optimization is running: a first look at the
results suggests where to concentrate, or a colleague mentions what worked for them last week.
Example 6 shows how to state such a belief before the run starts; this one states beliefs while it
is going on, with ``smac.add_prior``.

Each belief fades on its own schedule, measured from the trial at which it was stated, so one
stated late in the run arrives at full strength instead of inheriting the exponent an earlier one
has already decayed to. Beliefs accumulate, and any of them may be withdrawn again. See "Dynamic
Priors in Bayesian Optimization for Hyperparameter Optimization" by Lukas Fehring et al.

We optimize a two dimensional function whose optimum sits off centre, drive the loop with ask and
tell so that the beliefs can be stated in between, and print what the run made of them.
"""

from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float

from smac import BlackBoxFacade, Scenario
from smac.acquisition.weight import IncumbentComparisonPolicy
from smac.runhistory import TrialValue

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


OPTIMUM = {"x": 0.85, "y": 0.15}


class Bowl:
    @property
    def configspace(self) -> ConfigurationSpace:
        configspace = ConfigurationSpace(seed=0)
        configspace.add(Float("x", (0.0, 1.0)), Float("y", (0.0, 1.0)))

        return configspace

    def train(self, config: Configuration, seed: int = 0) -> float:
        return float((config["x"] - OPTIMUM["x"]) ** 2 + (config["y"] - OPTIMUM["y"]) ** 2)


if __name__ == "__main__":
    model = Bowl()

    scenario = Scenario(model.configspace, n_trials=60, deterministic=True, seed=0)
    smac = BlackBoxFacade(scenario, model.train, overwrite=True)

    for trial in range(scenario.n_trials):
        if trial == 20:
            # A rough belief, stated as plain values. Only the hyperparameters named get one; anything
            # left out keeps its uniform distribution, so a belief about part of the space is simply a
            # shorter dictionary.
            rough = smac.add_prior({"x": 0.8}, key="a first look at the results")
            print(f"Stated a belief about x at trial {trial}: {rough}")

        if trial == 35:
            # A sharper belief, stated later. It arrives at full strength, while the first one has
            # already faded. The safeguard checks it against the surrogate before it is acted on, and
            # returns None if it looks implausible.
            sharp = smac.add_prior(
                OPTIMUM,
                key="what worked for a colleague",
                acceptance_policy=IncumbentComparisonPolicy(),
            )
            print(f"Stated a belief about both at trial {trial}: {sharp}")

            if sharp is None:
                print("  ... which the surrogate did not find plausible, so nothing changed.")

        info = smac.ask()
        smac.tell(info, TrialValue(cost=model.train(info.config, info.seed), time=0.0))

    incumbent = smac.intensifier.get_incumbent()
    print(f"\nIncumbent: {dict(incumbent)}")
    print(f"Optimum:   {OPTIMUM}")

    # Each belief carries its own anchor and its own decay, so the older one has less to say by now.
    for key, prior in smac.priors.items():
        print(f"\n{key}: stated at trial {prior.t0}, exponent now {prior.decay(prior.steps):.3f}")

    # A belief can be withdrawn again, which takes effect on the next configuration asked for.
    for key in list(smac.priors):
        smac.remove_prior(key)

    assert smac.priors == {}
    assert np.isclose(incumbent["x"], OPTIMUM["x"], atol=0.2)
