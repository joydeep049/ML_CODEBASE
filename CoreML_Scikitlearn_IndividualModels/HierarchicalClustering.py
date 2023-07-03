"""## Hierarchical Clustering
A different kind of clustering which works on the principle of euclidean distance, but the approach is very different from K-Means.

The optimum number of clusters can be found by finding the longest vertical line that does not cross any extended horizontal Line in a dendogram.
The number of points of intersection of a horizontal line through the vertical line found above gives the optimum number of clusters formed.

### Visualizing The data
"""

cust= pd.read_csv("Mall_Customers.csv")
cust.head()

plt.scatter(cust['Annual Income (k$)'],cust['Spending Score (1-100)'] )
plt.ylabel('Spending Score (1-100)')
plt.xlabel('Annual Income (k$)')
plt.legend()
plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

cust['Annual Income (k$)']= scaler.fit_transform(cust[['Annual Income (k$)']])
cust['Spending Score (1-100)'] =scaler.fit_transform(cust[['Spending Score (1-100)']] )
cust.head()

plt.scatter(cust['Annual Income (k$)'],cust['Spending Score (1-100)'] )
plt.ylabel('Spending Score (1-100)')
plt.xlabel('Annual Income (k$)')
plt.legend()
plt.show()

X = cust.iloc[:,[3,4]].values

"""### Using the dendrogram to find the optimum number of clusters"""

import scipy.cluster.hierarchy as sch
dendrogram= sch.dendrogram(sch.linkage(X, method= "ward")) # Ward- Method of minimum variance between clusters.
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances")
plt.show()

"""### Training the hierarchical clustering model on the dataset."""

from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters= 5, affinity='euclidean', linkage='ward')

y_hc= hc.fit_predict(X)

y_hc

"""### Printing out the result of the model."""

plt.scatter(X[y_hc==0,0], X[y_hc==0,1], color= 'red', label= 'Cluster 1')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], color= 'magenta', label= 'Cluster 2')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], color= 'cyan', label= 'Cluster 3')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], color= 'blue', label= 'Cluster 4')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], color= 'green', label= 'Cluster 5')
plt.legend()
