## Naive Bayes Classifier
"""

tita= pd.read_csv("titanic.csv")
tita.head()

tita.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis='columns',inplace= True )
tita.head()

inputs= tita.drop(['Survived'], axis= 'columns')
inputs.head(3)

Y = tita['Survived']
Y.head(
)

dummies = pd.get_dummies(inputs.Sex)
dummies.head()

inputs= pd.concat([inputs, dummies], axis= 'columns')
inputs.head(3)

inputs.drop("Sex", axis='columns', inplace = True)
inputs.head(3)

"""### To check if their are any Null Values In any of the columns."""

inputs.columns[inputs.isna().any()]

"""### To fill the Null values if present"""

inputs.Age= inputs.Age.fillna(inputs.Age.mean())
inputs.Age

"""### To print number of null values in each column"""

inputs.isna().sum()

Y

"""### Creating the model and training the data"""

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y= train_test_split(inputs, Y , test_size=0.2)

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()

nb.fit(train_x, train_y)

nb.score(test_x, test_y)

"""### Plotting the confusion matrix."""

y_predicted = nb.predict(test_x)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y,y_predicted)
cm

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot= True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.legend()
plt.show()
