"""## Principal Component Analysis(PCA)
It involves reducing the dimensionality of a particular problem by analysing the components(features) with the most amount of variance.

"""

from sklearn.datasets import load_digits
digits= load_digits()

dir(digits)

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

digits.data[0].reshape(-1,8)

plt.gray()
plt.matshow(digits.data[0].reshape(-1,8))
print(digits.target[0])

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y= train_test_split(digits.data, digits.target, test_size= 0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators= 5)

model.fit(train_x, train_y)

model.score(test_x, test_y)

import pandas as pd
df= pd.DataFrame(digits.data, columns= digits.feature_names)
df.head()

X= df
Y= digits.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled= scaler.fit_transform(X)

# Now Apply PCA
from sklearn.decomposition import PCA
pc= PCA(0.90)
X_scaled_pc= pc.fit_transform(X_scaled)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y , test_y = train_test_split(X_scaled_pc, Y , test_size= 0.3 )

model.fit(train_x, train_y)

model.score(test_x, test_y)
# Score is lessser. which means we removed some important Features.
