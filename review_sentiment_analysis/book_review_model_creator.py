import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import re

#Function to preprocess the text(convert to lowercase, remove special characters and numbers)
def preprocess_text(text):
    """Preprocess text by removing special characters and converting to lowercase"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

#Function to load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['processed_review'] = df['reviewText'].apply(preprocess_text)
    
    # Remove neutral ratings (3 stars)
    df = df[df['rating'] != 3]
    df['sentiment'] = (df['rating'] >= 4).astype(int)
    return df

#Main function to train the model
def main():
    print("Loading data...")
    df = load_data('review_sentiment_analysis/data/all_kindle_review.csv')
    
    # First split: 80% train+dev, 20% test
    train_dev, test = train_test_split(df, test_size=0.2, random_state=42)
    # Second split: 75% train, 25% dev (of the 80%)
    train, dev = train_test_split(train_dev, test_size=0.25, random_state=42)
    
    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"Training set: {len(train)} samples")
    print(f"Development set: {len(dev)} samples")
    print(f"Test set: {len(test)} samples")
    
    # Print class distribution
    print("\nClass distribution:")
    for name, dataset in [("Training", train), ("Development", dev), ("Test", test)]:
        pos_percentage = (dataset['sentiment'] == 1).mean() * 100
        print(f"{name} set - Positive reviews: {pos_percentage:.1f}%")
    
    # Create TF-IDF vectorizer
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        stop_words='english',
        #Include bigrams, and trigrams
        ngram_range=(1, 3)
    )
    
    # Fit and transform the training data
    X_train = vectorizer.fit_transform(train['processed_review'])
    X_dev = vectorizer.transform(dev['processed_review'])
    X_test = vectorizer.transform(test['processed_review'])
    
    y_train = train['sentiment']
    y_dev = dev['sentiment']
    y_test = test['sentiment']
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    best_model = None
    best_score_dev = 0
    best_score_train = 0
    
    #Testing different depths for Decision Tree
    print("\nTraining Decision Tree models...")
    for max_depth in [3, 4, 5]:
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt.fit(X_train, y_train)
        
        train_auc = roc_auc_score(y_train, dt.predict_proba(X_train)[:, 1])
        dev_auc = roc_auc_score(y_dev, dt.predict_proba(X_dev)[:, 1])
        if dev_auc > best_score_dev and train_auc > best_score_train:
            best_score_dev = dev_auc
            best_score_train = train_auc
            best_model = dt
        
        print(f"\nDecision Tree (max_depth={max_depth}):")
        print(f"Training AUC: {train_auc:.4f}")
        print(f"Development AUC: {dev_auc:.4f}")
    
    # Testing different number of trees for Random Forest
    print("\nTraining Random Forest models...")
    for n_trees in [20, 100]:
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate on training and development sets
        train_auc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
        dev_auc = roc_auc_score(y_dev, rf.predict_proba(X_dev)[:, 1])
        if dev_auc > best_score_dev and train_auc > best_score_train:
            best_score_dev = dev_auc
            best_score_train = train_auc
            best_model = rf
        
        print(f"\nRandom Forest (n_trees={n_trees}):")
        print(f"Training AUC: {train_auc:.4f}")
        print(f"Development AUC: {dev_auc:.4f}")
    
    # Choosing the best model
    print("\nComparing models to find the best one and training it on combined train+dev data...")
    X_train_dev = vectorizer.fit_transform(pd.concat([train['processed_review'], dev['processed_review']]))
    y_train_dev = pd.concat([y_train, y_dev])
    
    final_model = best_model
    final_model.fit(X_train_dev, y_train_dev)
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    test_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nFinal model {best_model.__class__.__name__} on test set:")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': vectorizer.get_feature_names_out(),
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important words:")
    print(feature_importance.head(10))
    
    # Save the model and vectorizer
    import joblib
    joblib.dump(final_model, 'models/book_review_model.joblib')
    joblib.dump(vectorizer, 'models/book_review_vectorizer.joblib')
    print("\nModel and vectorizer saved to models/book_review_model.joblib and models/book_review_vectorizer.joblib")

if __name__ == "__main__":
    main()