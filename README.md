# Hotel Sentiment Analysis

This project trains a simple machine learning model to classify hotel reviews as either positive or negative sentiment.

## Overview

The script uses:
- `pandas` for dataset creation and handling
- `TfidfVectorizer` to convert review text into numerical features
- `LogisticRegression` as the classifier
- `train_test_split` to validate model performance
- `joblib` to save the trained model

The model is trained on a small synthetic hotel review dataset and evaluated on a test set before being used to predict sentiment for new reviews.

## Project files

- `sentiment_analysis.py` — training, evaluation, prediction, and model saving logic
- `hotel_sentiment_model.joblib` — saved trained model artifact

## Requirements

Python 3.9+ is recommended.

Install the required packages:

```bash
pip install pandas scikit-learn joblib
```

## Run the project

From the project root:

```bash
python sentiment_analysis.py
```

This will:
1. Build the sample hotel review dataset
2. Train the sentiment model
3. Print the accuracy and classification report
4. Predict sentiment for sample new reviews
5. Save the model as `hotel_sentiment_model.joblib`

## Example output

The script prints accuracy metrics and a classification report similar to:

```text
Test accuracy: 1.00

Classification report:
              precision    recall  f1-score   support

    negative       1.00      1.00      1.00       3
    positive       1.00      1.00      1.00       3
```

## Notes

This is a lightweight example for learning text classification basics. For production use, you would typically replace the handcrafted dataset with a larger and more diverse labeled dataset and evaluate more robust models.
