import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a sample hotel review dataset
data = {
    "review": [
        "The room was clean and comfortable",
        "Excellent service and friendly staff",
        "The food was delicious",
        "Wonderful hotel, I enjoyed my stay",
        "The location was perfect",
        "The staff were helpful and polite",
        "Amazing breakfast and beautiful rooms",
        "I would definitely recommend this hotel",
        "Very pleasant experience",
        "Everything was excellent",
        "The room was dirty and uncomfortable",
        "Terrible service from the staff",
        "The food was cold and tasteless",
        "I hated my stay at this hotel",
        "The location was disappointing",
        "The staff were rude and unhelpful",
        "The room smelled bad",
        "Very poor experience",
        "The hotel was noisy and dirty",
        "I would never recommend this hotel",
    ],
    "sentiment": [
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
    ],
}

df = pd.DataFrame(data)

# Step 2: Separate inputs (X) and labels (y)
X = df["review"]
y = df["sentiment"]

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30, random_state=42,
    stratify=y,
)

# Step 4: Create the ML pipeline
model = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Evaluate the model
predictions = model.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Step 7: Predict new hotel reviews
new_reviews = [
    "The room was fantastic and the staff were wonderful",
    "Dirty bathroom and terrible customer service",
    "The food was good but the room was noisy",
]

results = model.predict(new_reviews)

print("\nNew review predictions:")
for review, sentiment in zip(new_reviews, results):
    print(f"Review: {review}")
    print(f"Predicted sentiment: {sentiment}")
    print("-" * 50)

# Step 8: Save the trained model
import joblib

joblib.dump(model, "hotel_sentiment_model.joblib")
print("Model saved successfully!")
