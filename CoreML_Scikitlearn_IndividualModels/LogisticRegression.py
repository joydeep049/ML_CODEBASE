"""## Logistic Regression"""

data= pd.read_csv("insurance_data.csv")
data.head()

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.model_selection import train_test_split

# Here passing age as 2D array , because features are passed as a multidimensional tensor.
X_train, X_test, Y_train, Y_test= train_test_split(data[['age']],data.bought_insurance,test_size= 0.1)

model.fit(X_train, Y_train)

model.score(X_test,Y_test)
# 100% accuracy

"""## Multiclass Classification

### Digits Dataset
"""

# This dataset contains 1727 grayscale images of digits of size 8x8 .
# So each image has a feature vector of size 64.
from sklearn.datasets import load_digits

digits= load_digits()

dir(digits)

digits.data[0]
# Each image is a feature vector of length 64.

# Visulaize the image
plt.gray()
plt.matshow(digits.images[34])
print(digits.target[34])

from sklearn.model_selection import train_test_split

train_x, test_x, train_y , test_y= train_test_split(digits.data, digits.target, test_size= 0.2)

from sklearn.linear_model import LogisticRegression

model= LogisticRegression()

model.fit(train_x, train_y)

model.score(test_x, test_y)

plt.gray()
plt.matshow(digits.images[64])
print(digits.target[64])

model.predict([digits.data[64]])

"""#### Confusion Matrix"""

y_predicted= model.predict(test_x)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, y_predicted)
cm

import seaborn as sn
plt.figure(figsize=(10,10))
sn.heatmap(cm,annot= True)
plt.xlabel('Predicted')
plt.ylabel('Actual')

"""### Iris Dataset"""

from sklearn.datasets import load_iris
iris= load_iris()

dir(iris)



from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y= train_test_split(iris.data, iris.target,test_size=0.2)

test_y.shape

from sklearn.linear_model import LogisticRegression
log= LogisticRegression()

log.fit(train_x, train_y)

log.score(test_x, test_y)

"""#### Confusion Matrix"""

# Plot Confusion Matrix
y_predicted = log.predict(test_x)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(test_y, y_predicted)
cm

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
