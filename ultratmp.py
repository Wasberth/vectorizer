import numpy as np

algo = np.zeros((4,2))
otro = np.ones((2,2))

print(algo.shape)
algo = np.concatenate((algo, otro))
print(algo.shape)
print(algo)