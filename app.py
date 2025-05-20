from flask import Flask, render_template, request, jsonify, send_from_directory
from review_sentiment_analysis.book_review_predictor import predict_sentiment
from genre_predictor.book_analyzer import BookAnalyzer
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5ForConditionalGeneration, T5Tokenizer
import torch
from summary_generator.summary_generator import SummaryGenerator

# Create Flask app
app = Flask(__name__)

# Initialize book recommendor feature
df = pd.read_csv("bookrecommendor_data/Books_Preprocessed.csv")
df = df[["ISBN", "Title", "Author", "Year", "Publisher", "Processed_Title"]].dropna()
df.columns = ["ISBN", "Title", "Author", "Year", "Publisher", "Processed_Title"]
df = df.drop_duplicates(subset=["Title"])

# Initialize book analyzer feature
book_analyzer = BookAnalyzer()

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# Load & initialize generate description feature
gpt2_blurb_model = None
gpt2_blurb_tokenizer = None

# Define the path to the saved model
MODEL_SAVE_PATH = "title_generator/models/book_title_generator_t5_model"

# Load the fine-tuned model and tokenizer
try:
    title_tokenizer = T5Tokenizer.from_pretrained(MODEL_SAVE_PATH)
    title_model = T5ForConditionalGeneration.from_pretrained(MODEL_SAVE_PATH)
    title_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    title_model.to(title_device)
except Exception as e:
    print(f"Error loading Book Title Generation model: {e}")
    title_tokenizer = None
    title_model = None
    title_device = None

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Create TfidfVectorizer & TF-IDF matrix (for book recommendor feature)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Processed_Title"])

# Calculate cosine similarities (for book recommendor feature)
cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

# Function for book recommendor feature
def recommend_books(book_title, num_recommendations=15):
    processed_input = preprocess_text(book_title)
    input_vector = vectorizer.transform([processed_input])
    
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    
    similar_indices = similarities.argsort()[::-1][1:num_recommendations+1]
    
    recommended_books = df.iloc[similar_indices][["Title", "Author", "Year"]]
    return recommended_books

# Function for description generator feature
def generate_blurb(topic, genre, max_length=150):
    global gpt2_blurb_model, gpt2_blurb_tokenizer
    if gpt2_blurb_model is None or gpt2_blurb_tokenizer is None:
        gpt2_blurb_tokenizer = GPT2Tokenizer.from_pretrained('blurb_generator/gpt2_blurb_finetuned')
        gpt2_blurb_model = GPT2LMHeadModel.from_pretrained('blurb_generator/gpt2_blurb_finetuned')

    prompt = f"Topic: {topic}. Genre: {genre}. Description:"
    input_ids = gpt2_blurb_tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output = gpt2_blurb_model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            pad_token_id=gpt2_blurb_tokenizer.eos_token_id,
            eos_token_id=gpt2_blurb_tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )

    generated = gpt2_blurb_tokenizer.decode(output[0], skip_special_tokens=True)
    result = generated[len(prompt):].strip()

    last_punct = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
    if last_punct != -1:
        result = result[:last_punct + 1]

    return result

# Function for title generator feature
def generate_book_title(summary, model, tokenizer, device, num_return_sequences=3, max_length=64):
    if model is None or tokenizer is None:
        return ["Model not loaded."]

    input_text = "generate title: " + summary
    input_encoding = tokenizer(
        input_text,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    outputs = model.generate(
        input_ids=input_encoding['input_ids'],
        attention_mask=input_encoding['attention_mask'],
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        num_return_sequences=num_return_sequences
    )

    generated_titles = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_titles

# Instantiate summary generator
summary_generator = SummaryGenerator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    book_title = request.form.get('book_title')
    if not book_title:
        return jsonify({'error': 'Please enter a book title'}), 400
    
    try:
        recommendations = recommend_books(book_title)
        return jsonify({
            'recommendations': recommendations.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.form.get('review')
    if not review:
        return jsonify({'error': 'Please enter a review'}), 400
    
    try:
        result = predict_sentiment(review)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-book', methods=['POST'])
def analyze_book():
    description = request.form.get('description')
    if not description:
        return jsonify({'error': 'Please enter a book description'}), 400
    
    try:
        analysis = book_analyzer.analyze_book(description)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-blurb', methods=['POST'])
def generate_blurb_api():
    topic = request.form.get('title')
    genre = request.form.get('genre')
    if not topic or not genre:
        return jsonify({'error': 'Please enter both topic and genre'}), 400
    try:
        blurb = generate_blurb(topic, genre)
        return jsonify({'blurb': blurb})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_title', methods=['POST'])
def generate_title_route():
    if request.method == 'POST':
        data = request.get_json()
        summary = data.get('summary', '')

        if not summary:
            return jsonify({"error": "No summary provided"}), 400

        generated_titles = generate_book_title(summary, title_model, title_tokenizer, title_device)

        return jsonify({"titles": generated_titles})
        
@app.route('/generate-summary', methods=['POST'])
def generate_summary_api():
    title = request.form.get('title')
    additionalInfo = request.form.get('additionalInfo')
    if not title or not additionalInfo:
        return jsonify({'error': 'Please enter both title and additional informations that should be included in the summary'}), 400
    try:
        summary = summary_generator.generate_summary(title, additionalInfo)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True) 
    
#Command to run flask app is: 
#$env:FLASK_APP="app.py"; flask run --no-reload