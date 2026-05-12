import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only needed columns
df = df[['v1', 'v2']]
df.columns = ['Label', 'Message']

# Convert labels to numbers
df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

X = df['Message']
y = df['Label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Test sample
sample = ["Congratulations! You won a free iPhone. Click now!"]

sample_vectorized = vectorizer.transform(sample)

prediction = model.predict(sample_vectorized)

if prediction[0] == 1:
    print("Prediction: Spam")
else:
    print("Prediction: Legitimate")