# IMPORTING SCRIPTS

from Model import Model

# IMPORTING PACKAGES

import numpy as np
from collections import Counter

# THEORY SECTION

# Strategies for converting a binary classifier into a multiclass classifier:
# 1. One vs. All (creates k subproblems, take one with highest confidence)
#    However, we increase the risk of misclassification.
# 2. All-Pairs (consider all pairs of classes - choose(k,2) subproblems)
#    Use vote system. Pick the class with the most votes.
# 3. Error Correcting Output Codes (Dietterich and Bakiri)
#    Create coding matrix.

# However, it is not needed for decision trees.


# Entropy in relation to random variables is a concept invented by Claude Shannon
def entropy(y):

    hist = np.bincount(y)
    p_list = hist / len(y)
    return -np.sum([p*np.log(p, 2) for p in p_list if p > 0])


class Node():

    def __init__(self, feature=None, left_child=None, right_child=None, treshold=None, value=None):

        self.feature = feature
        self.left_child = left_child
        self.right_child = right_child
        self.treshold = treshold
        self.value = value

    def is_leaf(self):
        return self.value is not None


class Decision_Tree(Model):

    def guide():
        print("Just use it!")

    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features

    def fit(self, X, y):

        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_three(X, y)  # Underscore means intended for internal use only
        print(X)

    # Underscore means intended for internal use only
    def _grow_tree(self, X, y, depth=0):

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1  # this means that there is only one type label back in the sample passed
                or n_samples < self.min_samples_split):  # if less than one sample, it makes no sense to split it
            # in case it is due to depth, assign most frequent label
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # if it is not meeting the criterias for stopping, then get list of indices
        feat_idxs = np.array([i for i in range(n_features)])

        # greedy search
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    # The below method tries all thresholds and pick the one with the most information gain.
    # IG = Entropy(parent) - Entropy(children)
    # Entropy(parent) = - summation(p(x)*log_2(p(x)) for each x in X) where p(x) = #x/n
    # Entropy(children) = same and then the weighted arithmetic mean of both

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)  # split based on each variable
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threh = threshold

        return split_idx, split_threh

    def _information_gain(self, y, X_column, split_threh):

        parent_entropy = entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_threh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        return parent_entropy - child_entropy

    def _split(self, X_column, split_threh):

        # Returning list of indices where values are below and above threshold
        left_idxs = np.argwhere(X_column <= split_threh).flatten()
        right_idxs = np.argwhere(X_column > split_threh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):

        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):

        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def score(self, y_pred, y_true):

        if len(y_pred) != len(y_true):
            print("The testing data must have the same dimension as the prediction!")
            return None
        else:
            n_correct = np.sum(y_pred == y_true)
            n_total = len(y_pred)
            return n_correct/n_total
