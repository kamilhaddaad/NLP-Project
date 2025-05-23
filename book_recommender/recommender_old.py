import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load and preprocess dataset
df = pd.read_csv("bookrecommendor_data/Books_Preprocessed.csv")
df = df[["ISBN", "Title", "Author", "Year", "Publisher", "Processed_Title"]].dropna()
df.columns = ["ISBN", "Title", "Author", "Year", "Publisher", "Processed_Title"]
df = df.drop_duplicates(subset=["Title"])

# Vectorizer and similarity matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Processed_Title"])
cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def recommend_books(book_title, num_recommendations=15):
    processed_input = preprocess_text(book_title)
    input_vector = vectorizer.transform([processed_input])
    
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    similar_indices = similarities.argsort()[::-1][1:num_recommendations+1]
    
    recommended_books = df.iloc[similar_indices][["Title", "Author", "Year"]]
    return recommended_books