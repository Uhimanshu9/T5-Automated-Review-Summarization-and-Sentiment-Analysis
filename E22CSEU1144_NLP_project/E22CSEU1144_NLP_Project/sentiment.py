import nltk
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.datasets import imdb

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to clean text
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = nltk.word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return ' '.join(tokens)

# Function to decode IMDb reviews (specific to IMDb dataset)
def decode_review(review, reverse_word_index):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in review])

# Load IMDb dataset
def load_imdb_dataset():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}

    # Decode reviews
    train_texts = [decode_review(review, reverse_word_index) for review in train_data]
    test_texts = [decode_review(review, reverse_word_index) for review in test_data]

    return train_texts, train_labels, test_texts, test_labels

# Save the model and vectorizer
def save_model_and_vectorizer(model, vectorizer, model_path="sentiment_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

# Load the model and vectorizer
def load_model_and_vectorizer(model_path="sentiment_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)
    print(f"Model loaded from {model_path}")
    print(f"Vectorizer loaded from {vectorizer_path}")
    return loaded_model, loaded_vectorizer

# Load dataset (IMDb here but can be replaced with other datasets)
train_texts, train_labels, test_texts, test_labels = load_imdb_dataset()

# Preprocess the texts
train_texts = [clean_text(text) for text in train_texts]
test_texts = [clean_text(text) for text in test_texts]

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, train_labels)

# Save the trained model and vectorizer
save_model_and_vectorizer(model, vectorizer)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='binary')
recall = recall_score(test_labels, predictions, average='binary')
f1 = f1_score(test_labels, predictions, average='binary')
conf_matrix = confusion_matrix(test_labels, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Predict sentiment for new inputs using the original model
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Example usage for new text
new_text = "This movie was absolutely fantastic! Loved every bit of it."
print(f'Predicted Sentiment: {predict_sentiment(new_text)}')

new_text = "It was the worst movie I have ever seen."
print(f'Predicted Sentiment: {predict_sentiment(new_text)}')

# Load the model and vectorizer (example usage)
loaded_model, loaded_vectorizer = load_model_and_vectorizer()

# Predict sentiment for new inputs using the loaded model and vectorizer
def predict_sentiment_with_loaded_model(text, loaded_model, loaded_vectorizer):
    cleaned_text = clean_text(text)
    vectorized_text = loaded_vectorizer.transform([cleaned_text])
    prediction = loaded_model.predict(vectorized_text)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Example usage for new text with loaded model
new_text = "The storyline was intriguing and the acting was superb!"
print(f'Predicted Sentiment: {predict_sentiment_with_loaded_model(new_text, loaded_model, loaded_vectorizer)}')

new_text = "I found it boring and predictable."
print(f'Predicted Sentiment: {predict_sentiment_with_loaded_model(new_text, loaded_model, loaded_vectorizer)}')