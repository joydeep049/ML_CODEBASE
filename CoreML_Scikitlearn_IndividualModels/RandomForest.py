"""## Random Forest
This type of an alogrithm uses multiple decision trees as estimators, and gives out the result as the majority of the result given by those decision trees.
"""

from sklearn.ensemble import RandomForestClassifier
model2= RandomForestClassifier(n_estimators= 33)

model2.fit(x_train, y_train)

model2.score(x_test, y_test)

acc= []
max=0
for i in range(100):
  model2= RandomForestClassifier(n_estimators=i+1)
  model2.fit(x_train,y_train)
  acc.append(model2.score(x_test,y_test))
  if(acc[i]> max):
    max= acc[i]
    j=i

plt.plot(acc)

print(j)

"""### Wine dataset"""

from sklearn.datasets import load_wine
wine= load_wine()

dir(wine)

wine.DESCR

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y=  train_test_split(wine.data, wine.target, test_size= 0.2)

test_x.shape

from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators= 40)

rfc.fit(train_x, train_y)

rfc.score(test_x, test_y)


