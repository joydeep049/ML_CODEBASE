"""### Spam Detection"""

spam = pd.read_csv("spam.csv")
spam.head()

spam['spam']= spam['Category'].apply(lambda x : 1 if x=='spam' else 0)
# spam['spam'] = spam['Category'].map({"spam":1, "ham":0})
spam.head()

train_x, test_x, train_y, test_y= train_test_split(spam['Category'], spam.spam, test_size= 0.25)

"""#### Using countvectorizer to vectorize the sentences."""

from sklearn.feature_extraction.text import CountVectorizer
v= CountVectorizer()

type(train_x)

train_x_values = v.fit_transform(train_x.values)

"""#### Using Multinomial Bayes ."""

from sklearn.naive_bayes import MultinomialNB
model= MultinomialNB()
model.fit(train_x_values, train_y)

model.score(v.fit_transform(test_x), test_y)

model.predict(train_x_values[2])

"""#### Using cross-validation score to visualize the behaviour of the model using during splits of inputs"""

from sklearn.model_selection import cross_val_score
cross_val_score(MultinomialNB(), v.fit_transform(spam['Category']),spam.spam, cv= 10)

"""### Creating our own data Pipeline"""

from sklearn.pipeline import Pipeline
clf= Pipeline([
    ("Vectorizer", CountVectorizer()),
    ("nb", MultinomialNB())
])
clf.fit(train_x, train_y)

clf.score(test_x, test_y)
