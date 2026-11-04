# Chapter 1: Decision Trees & Supervised Learning Fundamentals

## 1. The Core Objective of Machine Learning
At its foundational level, machine learning is about predicting the future (unseen ratings or outcomes) based on past experience (historical data)[cite: 2].

* **Examples:** The specific objects about which our algorithm makes predictions[cite: 2]. In a course recommender system, an example is a specific Student/Course pair, such as `Alice/Algorithms`[cite: 2].
* **Generalization vs. Memorization:**
  * **Memorization (Recalling Past Experience):** If Alice already took *Artificial Intelligence* last year and rated it `+2 (awesome)`, predicting she will like it does not prove the system has learned—it is merely recalling past experience[cite: 2].
  * **Unfair Extrapolation (Beyond Experience):** Asking the system if Alice will enjoy *History of Pottery* when the system has zero prior experience or data about what *History of Pottery* is[cite: 2].
  * **True Generalization:** Using historical data (e.g., knowing Alice liked *Artificial Intelligence*) to induce a rule that accurately predicts her rating for a related unseen example (e.g., `Alice/Machine Learning`)[cite: 2].

---

## 2. The General Framework of Induction

Supervised learning follows a formal pipeline known as the **framework of induction**[cite: 2]:

1. **Training Data:** Historical examples paired with known labels (e.g., past courses Alice took and the scores from `-2` to `+2` she gave them)[cite: 2].
2. **Learning Algorithm:** The procedure that reads in the training data and computes a learned function $f$[cite: 2].
3. **Induced Function ($f$):** A mapping function that takes a new, unseen example and computes a corresponding prediction (e.g., $f(\text{Alice/Machine Learning}) = \text{high rating}$)[cite: 2].
4. **Test Set:** The collection of unseen examples on which we evaluate the predictive performance of our learned function $f$[cite: 2].

### Induction Pipeline Diagram (ASCII Representation)

    +------------------------+         +--------------------+
    |     Training Data      | ------> | Learning Algorithm |
    | (Examples + Labels)    |         +--------------------+
    +------------------------+                   |
                                                 | Induces
                                                 v
    +------------------------+         +--------------------+
    |      Test Example      | ------> | Learned Function f |
    | (Unseen Student/Course)|         +--------------------+
    +------------------------+                   |
                ^                                | Outputs
                |                                v
                +----------------------- Predicted Label

---

## 3. Step-by-Step Decision Tree Algorithm Mechanics

A Decision Tree is a hierarchical supervised learning model that represents the learned function $f$ as a tree of conditional feature checks.

### Tree Structure & Splitting Flow

### A. Formal Definitions & Representation
* **Input Space ($\mathcal{X}$):** The universe of all possible unseen or seen objects/examples.
* **Feature Vector ($\mathbf{x}$):** Observed properties extracted from an example so the algorithm can compare objects (e.g., `Course Department`, `Has Programming Assignments`, `Workload Level`, `Instructor`).
* **Output Space ($\mathcal{Y}$):** The set of possible target predictions or labels (e.g., discrete ratings $\{-2, -1, 0, +1, +2\}$ or binary classes $\{\text{Like}, \text{Dislike}\}$)[cite: 2].

### B. Greedy Splitting Algorithm (ID3 / CART)
To construct the tree from training data $D$, the algorithm executes the following recursive steps:

1. **Check Base Cases (Stopping Criteria):**
   * If all examples in $D$ have the exact same label $y$, return a **Leaf Node** predicting $y$.
   * If no features remain to split on (or `max_depth` is reached), return a **Leaf Node** predicting the **majority class** of $D$.
2. **Evaluate All Candidate Splits:**
   * Loop over every available feature $j \in \{1, \dots, d\}$.
   * Partition the current dataset $D$ into subsets based on the values of feature $j$.
3. **Measure Impurity & Information Gain:**
   * Calculate **Shannon Entropy** $H(D)$ to measure label uncertainty in the current node:
     $$H(D) = -\sum_{y \in \mathcal{Y}} P(y \mid D) \log_2 P(y \mid D)$$
   * Or calculate **Gini Impurity**:
     $$\text{Gini}(D) = 1 - \sum_{y \in \mathcal{Y}} P(y \mid D)^2$$
   * Compute **Information Gain ($IG$)** for splitting dataset $D$ on feature $j$:
     $$IG(D, j) = H(D) - \sum_{v \in \text{Values}(j)} \frac{\vert{}D_v\vert{}}{\vert{}D\vert{}} H(D_v)$$
4. **Select Best Split & Recurse:**
   * Choose feature $j^* = \arg\max_j IG(D, j)$.
   * Create a decision node for $j^*$ and recursively call the tree-building function on each child subset $D_v$.

---

## 4. Worked Numerical Example: Course Recommender

Suppose we have Alice's historical course ratings (Training Set $D$ with $\vert{}D\vert{} = 4$ examples)[cite: 2]:

| Example (Course) | Feature 1: CS Dept? | Feature 2: Math Heavy? | Known Label (Liked?) |
| :--- | :--- | :--- | :--- |
| Artificial Intelligence | Yes | Yes | Yes (+1) |
| Compilers | Yes | No | Yes (+1) |
| Calculus | No | Yes | No (-1) |
| World History | No | No | No (-1) |

### Step 1: Compute Root Entropy $H(D)$
* Total examples: $4$ ($2$ `Yes`, $2$ `No`).
* $P(\text{Yes}) = 0.5$, $P(\text{No}) = 0.5$.
* $$H(D) = -\left(0.5 \log_2(0.5) + 0.5 \log_2(0.5)\right) = 1.0 \text{ bit}$$

### Step 2: Evaluate Split on `Feature 1: CS Dept?`
* **Branch `CS Dept = Yes`:** Contains `{AI, Compilers}` $\rightarrow$ $2$ `Yes`, $0$ `No` $\rightarrow$ Entropy $H(D_{\text{Yes}}) = 0.0$.
* **Branch `CS Dept = No`:** Contains `{Calculus, World History}` $\rightarrow$ $0$ `Yes`, $2$ `No` $\rightarrow$ Entropy $H(D_{\text{No}}) = 0.0$.
* **Information Gain:**
  $$IG(D, \text{CS Dept}) = 1.0 - \left(\frac{2}{4}(0) + \frac{2}{4}(0)\right) = 1.0 \text{ bit}$$

### Step 3: Evaluate Split on `Feature 2: Math Heavy?`
* **Branch `Math Heavy = Yes`:** Contains `{AI, Calculus}` $\rightarrow$ $1$ `Yes`, $1$ `No` $\rightarrow$ Entropy $H(D_{\text{Yes}}) = 1.0$.
* **Branch `Math Heavy = No`:** Contains `{Compilers, World History}` $\rightarrow$ $1$ `Yes`, $1$ `No` $\rightarrow$ Entropy $H(D_{\text{No}}) = 1.0$.
* **Information Gain:**
  $$IG(D, \text{Math Heavy}) = 1.0 - \left(\frac{2}{4}(1.0) + \frac{2}{4}(1.0)\right) = 0.0 \text{ bits}$$

### Step 4: Final Induced Function $f$
Since `CS Dept?` yields the maximum Information Gain ($1.0\text{ bit}$), it becomes the Root Node, immediately producing two pure leaf nodes:
* **If** `CS Dept == Yes` $\rightarrow$ Predict **Yes (+1)**
* **If** `CS Dept == No` $\rightarrow$ Predict **No (-1)**
* **Test Prediction:** For a new test example `Alice/Algorithms` where `CS Dept = Yes`, $f(\text{Alice/Algorithms}) = \text{Yes (+1)}$[cite: 2].

---

## 5. Minimal Python Implementation from Scratch

[Root Node: Full Training Set]
                                |
               Evaluate all features & thresholds
               Select split with Max Information Gain
                     /                     \
         (Feature A == Yes)           (Feature A == No)
                /                               \
    [Left Child: Subset 1]             [Right Child: Subset 2]
              |                                   |
   Pure Node (All Class +2)            Impure Node (Mixed Classes)
              |                                   |
   [Leaf Node: Predict +2]             Split on next best Feature B
                                             /            \
                                    (True)  /              \ (False)
                                [Leaf: Predict +1]   [Leaf: Predict -2]

```python
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value for split
        self.left = left                # Left child Node
        self.right = right              # Right child Node
        self.value = value              # Class prediction if Leaf Node

    def is_leaf(self):
        return self.value is not None

class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Find best split greedily
        best_feat, best_thresh = self._best_split(X, y, n_features)
        if best_feat is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Partition data and recurse
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs = X_column <= threshold
        right_idxs = X_column > threshold

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
```


