from smac.runhistory import InstanceSeedBudgetKey


k1 = InstanceSeedBudgetKey(instance=1, seed=1, budget=5)
k2 = InstanceSeedBudgetKey(instance=1, seed=5, budget=5)
k3 = InstanceSeedBudgetKey(instance=7, seed=5, budget=5)

s1 = set([k1])
s2 = set([k1, k2, k3])

print(s2 - s1)
sorted(s2 - s1)
