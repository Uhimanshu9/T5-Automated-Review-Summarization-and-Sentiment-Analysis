import os
import pandas as pd
import re
import nltk
import joblib
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Configuration
MODEL_NAME = "t5-small"  # T5 model for summarization
MAX_LEN = 200  # Maximum sequence length
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')

# Load the trained summarization model and tokenizer
@st.cache_resource
def load_summarization_model():
    tokenizer = T5Tokenizer.from_pretrained("my_folder")
    model = T5ForConditionalGeneration.from_pretrained("my_folder").to(DEVICE)
    return tokenizer, model


# Summarization function
def summarize(text, model, tokenizer, max_len=MAX_LEN):
    """Generate summary for a given text."""
    input_text = f"summarize: {text}"
    inputs = tokenizer(
        input_text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(DEVICE)

    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Load the sentiment analysis model and vectorizer
@st.cache_resource
def load_sentiment_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


# Function to clean text
def clean_text(text):
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\W+", " ", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = nltk.word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words("english")]  # Remove stopwords
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return " ".join(tokens)


# Predict sentiment
def predict_sentiment(text, model, vectorizer):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return "Positive" if prediction[0] == 1 else "Negative"


# Streamlit GUI
def main():
    st.title("Text Summarization and Sentiment Analysis Tool")
    st.write(
        "Enter a text below. The tool will generate a concise summary and analyze the sentiment."
    )

    # User input
    user_input = st.text_area("Input Text", placeholder="Enter text here...")
    generate_button = st.button("Process")

    # Load models
    tokenizer, summarization_model = load_summarization_model()
    sentiment_model, vectorizer = load_sentiment_model()

    if generate_button:
        if user_input.strip():
            with st.spinner("Processing..."):
                # Generate summary
                summary = summarize(user_input, summarization_model, tokenizer)

                # Perform sentiment analysis
                sentiment = predict_sentiment(user_input, sentiment_model, vectorizer)

            st.success("Processing complete!")
            
            # Display results
            st.subheader("Generated Summary:")
            st.write(summary)
            
            st.subheader("Sentiment Analysis Result:")
            st.write(f"The sentiment is {sentiment}.")
        else:
            st.warning("Please enter some text to process.")


if _name_ == "_main_":
    main()