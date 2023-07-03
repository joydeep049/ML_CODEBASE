"""## Randomized Search
Lets us cross validate our model from random values.
"""

from sklearn.model_selection import RandomizedSearchCV
rs= RandomizedSearchCV(SVC(gamma='auto'),
                       {
                           "C": [1,5,10],
                           "kernel": ["rbf", 'linear']
                       }, cv= 5 , return_train_score= False, n_iter= 2)
rs.fit(iris.data, iris.target)
rs.cv_results_

rs_df= pd.DataFrame(rs.cv_results_)
rs_df.head()

rs.best_score_

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

models= {
    "SVM": {
        "model": SVC(gamma='auto'),
        "params": {
            "C": [x for x in range(50)],
            "kernel": ["rbf",'linear']
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [x for x in range(50)]
        }
    },
    "Logistic Regressiom":{
        "model": LogisticRegression(),
        "params": {
            "C": [x for x in range(50)]
        }
    },


}

results= []
for model_name, mp in models.items():
  clf= GridSearchCV(mp['model'],mp['params'], cv= 5, return_train_score= False )
  clf.fit(iris.data,iris.target)
  results.append({
      "model": model_name,
      "Best Score": clf.best_score_,
      "Best Parameters": clf.best_params_
  })
results

rs_df= pd.DataFrame(results, columns= ['Model',"Best Scores", "Best Parameters"])
