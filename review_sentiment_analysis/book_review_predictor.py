import pandas as pd
import joblib
from .book_review_model_creator import preprocess_text

#Function to predict sentiment for a new review
def predict_sentiment(review):
    model = joblib.load('review_sentiment_analysis/models/book_review_model.joblib')
    vectorizer = joblib.load('review_sentiment_analysis/models/book_review_vectorizer.joblib')
    
    processed_review = preprocess_text(review)
    X = vectorizer.transform([processed_review])
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    return {
        'sentiment': 'positive' if prediction == 1 else 'negative',
        'confidence': probability[1] if prediction == 1 else probability[0]
    }