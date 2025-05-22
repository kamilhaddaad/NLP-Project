from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

topic_model = BERTopic.load("models/bertopic_model")
embeddings = np.load("models/embeddings.npy", allow_pickle=True)
df = pd.read_pickle("data/df_with_topics.pkl")

def recommend_books(input_title, top_n=5):
    input_row = df[df["title"].str.lower() == input_title.lower()]
    if input_row.empty:
        print("Tytuł nie znaleziony.")
        return pd.DataFrame(columns=["title", "summary"])
    
    idx = input_row.index[0]
    input_topic = df.loc[idx, "topic"]
    input_embedding = df.loc[idx, "embedding"]
    input_genre = df.loc[idx, "genre"]

    similarities = cosine_similarity([input_embedding], list(df["embedding"]))[0]

    scores = []
    for i in range(len(df)):
        if df.iloc[i]["title"].lower() == input_title.lower():
            continue
        score = 0
        if input_title.split()[0].lower() in df.iloc[i]["title"].lower():
            score += 2
        if df.iloc[i]["topic"] == input_topic:
            score += 2
        if df.iloc[i]["genre"].lower() == input_genre.lower():
            score += 1
        score += similarities[i]
        scores.append((i, score))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    similar_indices = [i for i, _ in sorted_scores]
    recommended_books_df = df.iloc[similar_indices][["title", "summary"]]

    return recommended_books_df