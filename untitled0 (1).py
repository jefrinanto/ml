
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["I love this product", "This is terrible", "Very happy", "I hate it"]
labels = ["positive", "negative", "positive", "negative"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

print(model.predict(vectorizer.transform(["I love it"])))
