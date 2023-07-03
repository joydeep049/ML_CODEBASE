"""## BASICS

output= model(x_test)
print(output[4])
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from word2number import w2n

reg= linear_model.LinearRegression()

df= pd.read_csv("my_file.csv")

df.sample(7
          )

df.Bedrooms

reg.fit(df[["Bedrooms"]], df.Price)

reg.predict([[3000]])

pred= pd.read_csv("my_file_predict.csv")
pred.head()

df1= pd.read_csv("exercise.csv")

df1.head(10)

df1.experience= df1.experience.fillna("zero")

df1.isna().any()

score_median= df1["test_score(out of 10)"].median()
score_median

df1["test_score(out of 10)"]= df1["test_score(out of 10)"].fillna(score_median)

df1.head(10)

df1.isna().any()

df1.experience

"""### Word To Number

"""

df1.experience= df1.experience.apply(w2n.word_to_num)

df1.head()

n_reg= linear_model.LinearRegression()

n_reg.fit(df1[["experience","test_score(out of 10)","interview_score(out of 10)"]], df1["salary($)"])

n_reg.predict([[2,9,6]])

def gradient_descent(X,Y):
  iterations = 1000
  m_curr= c_curr= 0
  learning_rate= 0.01
  n= len(X)
  for i in range(iterations):
    y_predicted= m_curr * X + c_curr
    md = -(2/n) * sum(X * (Y- y_predicted))
    cd= -(2/n) * sum( Y- y_predicted)
    m_curr = m_curr - learning_rate * md
    c_curr= c_curr- learning_rate * cd
    print("m {} ,c {} ,Iteration {}".format(m_curr, c_curr, i+1))
  print("Final Value of M={} and C={}is ".format(m_curr, c_curr))

# Y= 3*X + 1
X= np.array([1,2,3,4,5,6])
Y= np.array([4,7,10,13,16,19])
gradient_descent(X,Y)

"""### Joblib is a library of sklearn used to save and load models."""

import joblib
joblib.dump(reg,'filename')

model1= joblib.load("filename")

model1.predict([[4000]])

df2= pd.read_csv("myfile__.csv")

df2.head()

df2['gender1']= df2['Gender'].map({'Male':0 , 'Female':1})

df2.head()

df2['gender1']=df2['gender1'].fillna(0)

df2.head()

reg1= linear_model.LinearRegression()

reg1.fit(df2[["Age","gender1"]], df2["Salary"])

reg1.predict([[12,0]])

hp= pd.read_csv("houseprices.csv")

hp.head()

# Dummies
new=pd.get_dummies(hp.town)
new

merged= pd.concat([hp,new],axis="columns")
merged



"""### Using Label Encoder

"""

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

hp1=hp
hp1.town= le.fit_transform(hp1.town)
hp1

X= hp1[['town', 'area']].values
X

Y= hp1.price
Y

"""### One Hot Encoder"""

from sklearn.preprocessing import OneHotEncoder
ohe= OneHotEncoder()

ohe.fit_transform(X).toarray()

cp= pd.read_csv("carprices.csv")
cp

plt.scatter(cp['Mileage'],cp['Sell Price($)'])
# Linear Relationship

plt.scatter(cp['Age(yrs)'],cp['Sell Price($)'])
# Linear Relationship

X= cp[['Age(yrs)','Mileage']]
Y= cp['Sell Price($)']

from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y= train_test_split(X,Y,test_size= 0.2)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(train_X, train_Y)

model.score(test_X, test_Y)
