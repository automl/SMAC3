import numpy as np

seed = 0

rng = np.random.RandomState(seed)


blub = ["a", "b", "c"]

rng.shuffle(blub)
print(blub)
