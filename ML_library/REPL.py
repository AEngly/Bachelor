import numpy as np

test1 = np.array([1, 2, 7, 4, 5])
test3 = np.array([1, 2, 3, 4, 5])
test2 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 5]])

# print((np.argwhere(test1) == 1).flatten()[1])
# print(np.unique(test2))
# print(np.bincount(test1))

# feat_idxs = np.random.choice(5, 5, replace=False)
# print(feat_idxs)

print(np.sum(test1 == test3))
