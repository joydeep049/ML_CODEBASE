"""## GridsearchCV
Lets us cross validate a given model for a given set of hyperparameter values.
"""

from sklearn.svm import SVC
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
iris= load_iris()

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(SVC(gamma= 'auto'),
                   {
                       "C": [1,10,20],
                       "kernel":["rbf", 'linear']},
                    cv=5, return_train_score = False)
clf.fit(iris.data, iris.target)
clf.cv_results_

result_df= pd.DataFrame(clf.cv_results_)
result_df.head()

clf.best_score_

clf.best_params_m
