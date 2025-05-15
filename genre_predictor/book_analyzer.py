import torch
from transformers import BertTokenizer, BertForSequenceClassification
import spacy
import json
import os

class BookAnalyzer:
    def __init__(self, model_dir='genre_predictor/models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load saved model and tokenizer
        model_path = 'genre_predictor/models/book_analyzer_model'
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.genre_classifier = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.genre_classifier.eval()  # Set to evaluation mode
        
        # Load genres
        with open(os.path.join(model_dir, 'genres.json'), 'r') as f:
            self.genres = json.load(f)
        
        # Load spaCy model for theme analysis
        self.nlp = spacy.load('en_core_web_lg')

    # function for genre predictor feature
    def analyze_book(self, description):
        """Analyze a book's description and return genre, themes, and emotional tone"""
        # Genre prediction
        inputs = self.tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.genre_classifier(**inputs)
            genre_probs = torch.softmax(outputs.logits, dim=1)
            predicted_genres = torch.topk(genre_probs, k=3)
            
        # Theme analysis using spaCy
        doc = self.nlp(description)
        
        # Extract key themes (noun phrases and named entities)
        themes = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Only multi-word phrases
                themes.append(chunk.text.lower())
        
        # Extract named entities
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
        
        # Emotional tone analysis
        sentiment_score = doc.sentiment
        
        return {
            'genres': [
                {
                    'genre': self.genres[idx],
                    'confidence': float(prob)
                }
                for prob, idx in zip(predicted_genres.values[0], predicted_genres.indices[0])
            ],
            'themes': list(set(themes))[:5],
            'entities': list(set(entities))[:5],
            'sentiment': {
                'score': float(sentiment_score),
                'label': 'Positive' if sentiment_score > 0 else 'Negative'
            }
        } 