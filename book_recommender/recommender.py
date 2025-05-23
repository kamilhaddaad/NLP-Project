
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("book_recommender/data/books_data.csv")
book_embeddings = np.load('book_recommender./models/book_embeddings.npy')

def recommend_books(book_title, num_recommendations=15):
    try:
        # Get the index of the book (case-insensitive match)
        book_index = df[df['title'].str.lower() == book_title.lower()].index[0]
    except IndexError:
        print(f"Book with title '{book_title}' not found in the dataset.")
        return pd.DataFrame(columns=["title", "summary"]) # Return empty DataFrame

    # Get the embedding of the input book
    input_embedding = book_embeddings[book_index].reshape(1, -1)

    # Calculate cosine similarity between the input book and all other books
    similarities = cosine_similarity(input_embedding, book_embeddings).flatten()

    # Get the indices of the most similar books
    # We exclude the first one because it will be the book itself
    similar_indices = similarities.argsort()[::-1][1:num_recommendations + 1]

    return df.iloc[similar_indices][["title", "summary"]]