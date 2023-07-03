"""## Support Vector Machines
This is a highly efficient classification algorithm which aims at Maximising the margin i.e maximizing the distance of the possible classsification hyperplanes or lines from the nearby points.
"""

from sklearn.datasets import load_iris
iris= load_iris()

dir(iris)

iris.data.shape

iris.target.shape

import matplotlib.pyplot as plt
import pandas as pd
df= pd.DataFrame(iris.data, columns= iris.feature_names)

df['target']=iris.target
df.head()

df0= df[df.target==0]
df1= df[df.target==1]
df2= df[df.target==2]
df2.head()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color= 'green', marker='+', label="0 target")
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color= 'blue', marker='.',  label="1 target")
# plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'],color= 'yellow', label="2 target")
plt.legend()
plt.show()

X= df.drop(['target'], axis= 'columns')
X.head()

Y= df.target
Y.head()

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(X, Y, test_size= 0.2)

from sklearn.svm import SVC
model= SVC()

# Training the model
model.fit(x_train,y_train)

model.score(x_test, y_test)

### Digits Dataset
from sklearn.datasets import load_digits
digits= load_digits()

dir(digits)

plt.imshow(digits.images[87])
print(digits.target[87])
plt.show()

digits.data[0]

from sklearn.model_selection import train_test_split
x_train,x_test, y_train ,y_test= train_test_split(digits.data, digits.target, test_size=0.2)

from sklearn.svm import SVC
model1= SVC()

model1.fit(x_train, y_train)

model1.score(x_test, y_test)

y_predicted= model1.predict(x_test)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix( y_test,y_predicted )
cm

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
