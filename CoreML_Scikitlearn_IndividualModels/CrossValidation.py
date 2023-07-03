"""## K Fold Cross Validation
This technique takes the average accuracy of multiple iterations to solve the same problem.
"""

!pip install sklearn.model_selection.StratifiedFold

from sklearn.datasets import load_digits
digit= load_digits()

from sklearn.model_selection import KFold
f= KFold (n_splits= 3)

for a,b in f.split([1,2,3,4,5,6]):
  print(a,b)

train_x.shape

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_score(model, train_x, test_x, train_y , test_y):
  model.fit(train_x, train_y)
  return model.score(test_x, test_y)

score_log= []
score_svc=[]
score_rfc= []
for a,b in f.split(list(digit.data)):
  train_x, test_x, train_y , test_y = digit.data[a], digit.data[b], digit.target[a], digit.target[b]
  score_log.append(get_score(LogisticRegression(),train_x, test_x, train_y , test_y ))
  score_svc.append(get_score(SVC(),train_x, test_x, train_y , test_y ))
  score_rfc.append(get_score(RandomForestClassifier(),train_x, test_x, train_y , test_y ))

print("Logistic regression",score_log)
print("Support vector classifiers",score_svc)
print("Random Forest",score_rfc)

# Using sklearn API for auto cross validation
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(),digit.data, digit.target)

cross_val_score(SVC(),digit.data, digit.target)

cross_val_score(RandomForestClassifier(),digit.data, digit.target)

