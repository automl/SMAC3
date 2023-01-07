from smac.runhistory import RunHistory
from smac.runhistory.dataclasses import InstanceSeedBudgetKey
from smac.utils.pareto_front import calculate_pareto_front, sort_by_crowding_distance


def test_pareto_front(configspace_small):
    """Tests whether the pareto front is correctly calculated."""
    rh = RunHistory()
    isb_key = InstanceSeedBudgetKey(instance=None, seed=0, budget=None)
    configs = configspace_small.sample_configuration(20)
    config_instance_seed_budget_keys = [[isb_key]] * 20

    # Add points on pareto
    rh.add(configs[0], cost=[5, 5], instance=isb_key.instance, budget=isb_key.budget, seed=isb_key.seed)
    rh.add(configs[1], cost=[4, 6], instance=isb_key.instance, budget=isb_key.budget, seed=isb_key.seed)

    # Add points not on pareto
    rh.add(configs[2], cost=[5, 6], instance=isb_key.instance, budget=isb_key.budget, seed=isb_key.seed)
    rh.add(configs[3], cost=[5, 6], instance=isb_key.instance, budget=isb_key.budget, seed=isb_key.seed)

    # Calculate pareto front
    configs = calculate_pareto_front(rh, configs[:4], config_instance_seed_budget_keys[:4])
    assert len(configs) == 2


def test_crowding_distance(configspace_small):
    """Tests whether the configs are correctly sorted by the crowding distance."""
    rh = RunHistory()
    isb_key = InstanceSeedBudgetKey(instance=None, seed=0, budget=None)
    configs = configspace_small.sample_configuration(20)
    config_instance_seed_budget_keys = [[isb_key]] * 20

    # Add points on pareto
    rh.add(configs[0], cost=[5, 5], instance=isb_key.instance, budget=isb_key.budget, seed=isb_key.seed)
    rh.add(configs[1], cost=[4, 6], instance=isb_key.instance, budget=isb_key.budget, seed=isb_key.seed)

    # Add points not on pareto
    rh.add(configs[2], cost=[5, 6], instance=isb_key.instance, budget=isb_key.budget, seed=isb_key.seed)
    rh.add(configs[3], cost=[5, 6], instance=isb_key.instance, budget=isb_key.budget, seed=isb_key.seed)

    # Calculate pareto front
    incumbents = calculate_pareto_front(rh, configs[:4], config_instance_seed_budget_keys[:4])
    sorted_configs = sort_by_crowding_distance(rh, incumbents, config_instance_seed_budget_keys[: len(incumbents)])
    # Nothing should happen if we only have two points on the pareto front
    assert sorted_configs == incumbents

    # Now we add another point on the pareto front
    rh.add(configs[4], cost=[3.0, 6.0001], instance=isb_key.instance, budget=isb_key.budget, seed=isb_key.seed)
    incumbents = calculate_pareto_front(rh, configs[:5], config_instance_seed_budget_keys[:5])
    assert len(incumbents) == 3

    # We want to remove the second config now
    sorted_configs = sort_by_crowding_distance(rh, incumbents, config_instance_seed_budget_keys[: len(incumbents)])
    assert configs[0] in sorted_configs
    assert configs[4] in sorted_configs
    assert configs[1] in sorted_configs

    # configs[1] should be last
    sorted_configs = sorted_configs[:2]
    assert configs[1] not in sorted_configs
