import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = []

with open("train_data.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.split(" ::: ")
        if len(parts) == 4:
            id_, title, genre, description = parts
            data.append([title, genre, description])

df = pd.DataFrame(data, columns=["Title", "Genre", "Description"])

X = df["Description"]
y = df["Genre"]

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

sample = ["A hero saves the world from aliens using advanced technology"]

sample_vector = vectorizer.transform(sample)

prediction = model.predict(sample_vector)

print("Predicted Genre:", prediction[0])