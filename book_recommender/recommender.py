
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


books_data_df = pd.read_csv("book_recommender/data/books_data.csv")
book_embeddings = np.load('book_recommender/models/book_embeddings.npy')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load and preprocess dataset
books_preprocessed_df = pd.read_csv("book_recommender/data/Books_Preprocessed.csv")
books_preprocessed_df = books_preprocessed_df[["ISBN", "Title", "Author", "Year", "Publisher", "Processed_Title"]].dropna()
books_preprocessed_df.columns = ["ISBN", "Title", "Author", "Year", "Publisher", "Processed_Title"]
books_preprocessed_df = books_preprocessed_df.drop_duplicates(subset=["Title"])

# Vectorizer and similarity matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(books_preprocessed_df["Processed_Title"])
cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Calculate cosine similarities (for book recommendor feature)
cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

def recommend_books_description_based(book_title, num_recommendations=15):
    try:
        book_index = books_data_df[books_data_df['title'].str.lower() == book_title.lower()].index[0]
    except IndexError:
        print(f"Book with title '{book_title}' not found in the dataset.")
        return pd.DataFrame(columns=["title", "summary"])

    input_embedding = book_embeddings[book_index].reshape(1, -1)

    # Calculate cosine similarity between the input book and all other books
    similarities = cosine_similarity(input_embedding, book_embeddings).flatten()

    # Get the indices of the most similar books
    similar_indices = similarities.argsort()[::-1][1:num_recommendations + 1]

    return books_data_df.iloc[similar_indices][["title", "summary"]]

# Function for book recommendor feature
def recommend_books_title_based(book_title, num_recommendations=15):
    processed_input = preprocess_text(book_title)
    input_vector = vectorizer.transform([processed_input])
    
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    
    similar_indices = similarities.argsort()[::-1][1:num_recommendations+1]
    
    recommended_books = books_preprocessed_df.iloc[similar_indices][["Title", "Author", "Year"]]
    return recommended_books