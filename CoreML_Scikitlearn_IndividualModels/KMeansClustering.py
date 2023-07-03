## K-Means Clustering


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv("income.csv")
df.head()

plt.scatter(df['Age'], df['Income($)'])

km= KMeans(n_clusters= 3)

y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted

df['cluster']= y_predicted

df.head()

df1= df[df.cluster==0]
df2= df[df.cluster==1]
df3= df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],marker='.',color= "red" ,label = "class 1")
plt.scatter(df2.Age,df2['Income($)'],marker='.',color= "blue" ,label = "class 2")
plt.scatter(df3.Age,df3['Income($)'], marker='.',color= "red",label = "class 3")
plt.xlabel("Age")
plt.ylabel("Income($)")
plt.legend()
# As we can see the clusters formed arent perfect. This is because our data is not scaled properly. Y axis has a very high range
# whereas x-axis has a very narrow range.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df['Income($)']= scaler.fit_transform(df[['Income($)']])

df.head()

df1= df[df.cluster==0]
df2= df[df.cluster==1]
df3= df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],marker='.',color= "red" ,label = "class 1")
plt.scatter(df2.Age,df2['Income($)'],marker='.',color= "blue" ,label = "class 2")
plt.scatter(df3.Age,df3['Income($)'], marker='.',color= "red",label = "class 3")
plt.xlabel("Age")
plt.ylabel("Income($)")
plt.legend()

df['Age']= scaler.fit_transform(df[['Age']])

y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted

df['cluster']= y_predicted

df1= df[df.cluster==0]
df2= df[df.cluster==1]
df3= df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],marker='.',color= "red" ,label = "class 1")
plt.scatter(df2.Age,df2['Income($)'],marker='.',color= "blue" ,label = "class 2")
plt.scatter(df3.Age,df3['Income($)'], marker='.',color= "green",label = "class 3")
plt.xlabel("Age")
plt.ylabel("Income($)")
plt.legend()
