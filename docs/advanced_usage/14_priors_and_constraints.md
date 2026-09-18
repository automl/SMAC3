# Steering the Search

Two things in SMAC change where the search looks without changing what the surrogate model believes about the
objective: a **user prior** over the optimum, and an **output constraint** on a measured quantity. Both are
multiplied into the acquisition function, so they share one mechanism and can be used together.

$$a_w(\mathbf{X}) = a(\mathbf{X}) \prod_i w_i(\mathbf{X})$$

Each $w_i$ is an [`AbstractAcquisitionWeight`][smac.acquisition.weight.abstract_weight.AbstractAcquisitionWeight],
and [`WeightedAcquisitionFunction`][smac.acquisition.function.weighted_acquisition_function.WeightedAcquisitionFunction]
applies the list of them. Weighting rather than penalizing keeps the objective model clean: a penalty would make
the surrogate fit a cliff that is not a feature of the objective, degrading its predictions everywhere.

## Output constraints

Bound a quantity the target function reports, rather than one the configuration space can express. Declare the
bounds on the scenario and return the measured values alongside the cost:

```python
scenario = Scenario(configspace, objectives="error", constraints=["latency <= 100"])

def train(config, seed=0):
    return error, {"latency": latency_ms}
```

Each constrained output is modelled separately, and the acquisition function is weighted by the probability that
every bound holds [[GKZ+14][GKZ+14]]. The reported incumbent is the best *feasible* configuration, and while
nothing feasible has been observed the search maximizes the probability of feasibility alone.

## User priors over the optimum

A prior is a density over the search space saying where you think the optimum is. Its influence fades as the
surrogate learns, so it speeds up a good run without preventing a wrong belief from being corrected
[[HSSL22][HSSL22]].

State one before the run by placing distributions on the configuration space (see the *User Priors over the
Optimum* example), or during the run:

```python
smac = HyperparameterOptimizationFacade(scenario, train)

key = smac.add_prior({"learning_rate": 0.01, "n_layers": 3})
```

Only the hyperparameters named get a belief; the rest keep their uniform distribution, so a belief about part of
the search space is simply a shorter dictionary. `add_prior` accepts a `ConfigurationSpace` carrying
distributions, or an [`AbstractInputPrior`][smac.acquisition.weight.prior.AbstractInputPrior], as well.

The belief takes effect on the next configuration asked for, and `smac.remove_prior(key)` withdraws it again.

### Beliefs stated during a run

Every belief is anchored at the trial count when it was stated, and decays from there
[[FWS+25][FWS+25]]. A belief stated at trial 120 therefore arrives at full strength rather than inheriting the
near-flat exponent an earlier belief has already decayed to. Beliefs accumulate, and are combined by summing, so
the ensemble reads as "any of these regions is worth a look" - multiplying would let two beliefs about different
regions cancel each other out.

!!! note

    Summing has a consequence worth knowing. A fully decayed belief contributes about one everywhere, while a
    freshly stated sharp belief contributes far less than one almost everywhere, so a pile of stale beliefs can
    drown out the newest and most informative one. `PriorEnsemble(prune_exponent=...)` drops beliefs whose
    influence has faded, and `combination="max"` lets the most enthusiastic belief decide instead.

Candidates are also drawn from each belief, not only ranked by it. Weighting alone leaves a sharply peaked belief
unreachable, because the acquisition maximizer would never sample near the peak in the first place.

### Rejecting an implausible belief

By default every stated belief is acted on: the user is in charge. An automated caller with nobody watching can
ask for a safeguard, which checks the belief against the surrogate before acting on it:

```python
from smac.acquisition.weight import IncumbentComparisonPolicy

key = smac.add_prior(values, acceptance_policy=IncumbentComparisonPolicy())

if key is None:
    ...  # the surrogate did not find the belief plausible, and nothing changed
```

The policy draws configurations from the belief and from a belief-shaped neighbourhood of the current incumbent,
scores both under the current model, and rejects the belief when its region looks clearly worse
[[FWS+25][FWS+25]]. Its threshold is in raw objective units, so set it for the objective at hand.

## Using both together

Nothing special is needed. Declaring constraints adds a feasibility weight to whatever the acquisition function
already carries:

```python
scenario = Scenario(configspace, objectives="error", constraints=["latency <= 100"])
smac = HyperparameterOptimizationFacade(scenario, train)

smac.add_prior({"learning_rate": 0.01})
```

!!! warning

    Wrapping one weighted acquisition function in another is refused. Both would shift the values of a confidence
    bound by the incumbent, so the shift would land twice and the ranking would be meaningless. Add the weights to
    one wrapper instead, which is what `add_prior` and the facade both do.
