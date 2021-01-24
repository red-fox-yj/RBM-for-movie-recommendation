import numpy as np

np_rng = np.random.RandomState(1234)
weights = np.asarray(
    np_rng.uniform(
        low=-0.1 * np.sqrt(6.0 / (6 + 2)),
        high=0.1 * np.sqrt(6.0 / (6 + 2)),
        size=(6, 2),
    )
)
weights = np.insert(weights, 0, 0, axis=0)
weights = np.insert(weights, 0, 0, axis=1)
n = np.random.rand(6, 3)
print(n)