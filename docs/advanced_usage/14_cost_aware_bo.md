# Cost-Aware Bayesian Optimization

Cost-aware BO extends standard SMAC by accounting for the **resource cost**
of each configuration evaluation (e.g. wall-clock time, API credits, energy).
Instead of treating all evaluations as equally expensive, the optimizer
preferentially selects cheap configurations early and shifts to
performance-guided search as the budget is spent.

## When to use it

Use `CostAwareFacade` when:
- Evaluations differ significantly in cost (e.g. training a small vs. large model).
- You have a fixed total resource budget rather than a fixed trial count.
- You want to maximize performance-per-unit-cost rather than raw performance.

## How it works

The facade uses two cost-aware components:

**CostAwareInitialDesign** samples an initial set of configurations that are
both diverse and cheap, staying within an `initial_budget` fraction of the
total resource budget.

**CostAwareAcquisitionFunction (EI-Cool)** wraps any acquisition function
with a cost penalty that decreases as the budget is consumed:

$$\mathrm{Acq}(x)_{\text{cost-aware}} = \frac{\mathrm{Acq}(x)}{c(x)^\alpha}$$

where $\alpha$ decreases from 1 to 0 as the cumulative cost approaches the
total budget.

## Usage

```python
from smac.facade.cost_aware_facade import CostAwareFacade
from smac.scenario import Scenario

def target_function(config, seed=0):
    # Must return a dict with "performance" and "cost" keys
    return {"performance": ..., "cost": ...}

smac = CostAwareFacade(
    scenario=scenario,
    target_function=target_function,
    total_resource_budget=50.0,
    cost_formula=lambda config: my_cost_estimate(config),
)
smac.optimize()
```

See the full example [`examples/4_advanced_optimizer/6_cost_aware_facade.py`](../examples/4%20Advanced%20Topics/6_cost_aware_facade.md).