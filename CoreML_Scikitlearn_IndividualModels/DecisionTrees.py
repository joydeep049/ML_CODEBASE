"""## Decision Trees
They are algorithms which is used to solve complicated classification problems by generating multiple decision boundaries.
It works by applying multiple conditionals to the data in order to separate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

from sklearn import tree
model= tree.DecisionTreeClassifier()

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()

sl= pd.read_csv('salaries.csv')
sl.head()

from sklearn.preprocessing import LabelEncoder
le_company= LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()

sl['company_n']= le_company.fit_transform(sl['company'])
sl['job_n']= le_job.fit_transform(sl['job'])
sl['degree_n']= le_degree.fit_transform(sl['degree'])

sl.head()

sl.drop(['company','job','degree'], axis='columns')

from sklearn.model_selection import train_test_split
train_x, test_x, train_y,test_y =train_test_split(sl[['company_n','job_n','degree_n']], sl['salary_more_then_100k'], test_size= 0.2)

model.fit(train_x, train_y)

model.score(test_x, test_y)
