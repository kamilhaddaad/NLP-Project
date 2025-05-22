# import pandas as pd
# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Load and preprocess dataset
# df = pd.read_csv("bookrecommendor_data/Books_Preprocessed.csv")
# df = df[["ISBN", "Title", "Author", "Year", "Publisher", "Processed_Title"]].dropna()
# df.columns = ["ISBN", "Title", "Author", "Year", "Publisher", "Processed_Title"]
# df = df.drop_duplicates(subset=["Title"])

# # Vectorizer and similarity matrix
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df["Processed_Title"])
# cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

# def preprocess_text(text):
#     doc = nlp(text.lower())
#     tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
#     return " ".join(tokens)

# def recommend_books(book_title, num_recommendations=15):
#     processed_input = preprocess_text(book_title)
#     input_vector = vectorizer.transform([processed_input])
    
#     similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
#     similar_indices = similarities.argsort()[::-1][1:num_recommendations+1]
    
#     recommended_books = df.iloc[similar_indices][["Title", "Author", "Year"]]
#     return recommended_books


import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re  

nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


print("Loading dataset...")
dataset = load_dataset("booksouls/goodreads-book-descriptions", split="train")

print("Filtering data...")
books_filtered = dataset.filter(lambda x: x['description'] and x['title'])
books_sample = books_filtered.select(range(min(10000, len(books_filtered))))

df = pd.DataFrame({
    "Title": books_sample["title"],
    "Description": books_sample["description"]
})
print(f"Loaded {len(df)} books into DataFrame.")

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words_english = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words_english and len(word) > 1]
    return " ".join(tokens)

print("Processing Descriptions...")
df["Processed_Description"] = df["Description"].apply(preprocess_text)

df = df[df["Processed_Description"].str.len() > 0].reset_index(drop=True)
print(f"Number of books after processing: {len(df)}")

if df.empty:
    print("WARNING! DataFrame is empty!!!")

# vectorization
print("Matrix TF-IDF...")
vectorizer = None
tfidf_matrix = None
lsa = None
lsa_matrix = None

if df.empty:
    print("WARNING! DataFrame is empty!!!.")
else:
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )
    tfidf_matrix = vectorizer.fit_transform(df["Processed_Description"])

    print("Dimensionality reduction LSA (TruncatedSVD)...")
    n_samples, n_features = tfidf_matrix.shape
    n_components_lsa = min(150, max(n_samples - 1, 1), max(n_features - 1, 1))

    if n_components_lsa <= 0:
        print(f"Cannot use LSA z n_components={n_components_lsa}. Not enough data.")
        lsa_matrix = None
        lsa = None
    else:
        lsa = TruncatedSVD(n_components=n_components_lsa, random_state=42)
        lsa_matrix = lsa.fit_transform(tfidf_matrix)
        print(f"LSA Matrix size: {lsa_matrix.shape if lsa_matrix is not None else 'Brak'}")

### RECOMMENDATION
def recommend_books(description: str, num_recommendations: int = 15) -> pd.DataFrame:
    empty_df_return = pd.DataFrame(columns=["Title", "Description"])

    if df.empty or lsa_matrix is None or lsa is None or vectorizer is None:
        print("WARNING! DataFrame is empty!!!")
        return empty_df_return

    processed_input_description = preprocess_text(description)
    if not processed_input_description:
        print("The input description is empty after processing. Cannot generate recommendation.")
        return empty_df_return

    try:
        input_vector_tfidf = vectorizer.transform([processed_input_description])
        input_vector_lsa = lsa.transform(input_vector_tfidf)
    except Exception as e:
        print(f"Error while transforming input description: {e}")
        return empty_df_return

    similarities = cosine_similarity(input_vector_lsa, lsa_matrix).flatten()

    if len(similarities) <= 1:
        if len(similarities) > 0 and num_recommendations > 0:
            indices_to_take = min(num_recommendations, len(similarities))
            similar_indices = similarities.argsort()[::-1][:indices_to_take]
        else:
            return empty_df_return
    else:
        max_possible_recommendations = len(similarities) - 1
        actual_recommendations_count = min(num_recommendations, max_possible_recommendations)
        if actual_recommendations_count <= 0:
            return empty_df_return
        similar_indices = similarities.argsort()[::-1][1:actual_recommendations_count + 1]

    if len(similar_indices) == 0:
        return empty_df_return

    recommended_books_df = df.iloc[similar_indices][["Title", "Description"]]
    return recommended_books_df

