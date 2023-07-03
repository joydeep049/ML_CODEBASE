"""## Ensemble Learning
Bagging:
It does with any model what random forest does with decision tree.
Creates new datasets using Resampling with Replacement technique.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

scores = cross_val_score(DecisionTreeClassifier(), digits.data, digits.target, cv=5)

scores.mean()

"""### Bagging
A Bagging classifier.

A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [1]. If samples are drawn with replacement, then the method is known as Bagging [2]. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [3]. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches [4].
"""

from sklearn.ensemble import BaggingClassifier
bag= BaggingClassifier(
    estimator= DecisionTreeClassifier(),
    n_estimators= 100,
    max_samples=0.8,
    oob_score= True

)
# When we use resampling with replacement technique, we might miss some elements altogether .. meaning some might not be in the any of the
# bags created. Those elements are called out-of-bag (oob) elements

bag.fit(train_x, train_y)

bag.oob_score_

bag.score(test_x, test_y)
# As we can see , the bagged decision tree has very high accuracy comparing to the normal one.
