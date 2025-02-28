# T5Bayes: Automated Review Summarization and Sentiment Analysis


## Overview

T5Bayes is an integrated text analysis system combining T5-based text summarization and Naive Bayes sentiment analysis to efficiently process and analyze textual review data. This system provides automated insights from customer reviews by generating concise summaries and sentiment predictions through a user-friendly interface.

## Project Structure

```
T5Bayes/
│
├── final.py           # Main execution script
├── sentiment.py       # Naive Bayes sentiment analysis implementation
├── summariser.py      # T5 summarization implementation
├── report.docx        # Project documentation
└── t5bayes-readme.md  # This README file
```

## Features

- **Text Summarization**: Fine-tuned T5 transformer model that generates abstraction-based summaries of reviews
- **Sentiment Analysis**: Multinomial Naive Bayes classifier that predicts positive/negative sentiment
- **Integrated Pipeline**: Combined workflow for processing review data
- **High Performance**: 85%+ sentiment classification accuracy and high-quality summaries

## Installation

```bash
# Clone the repository
git clone https://github.com/Uhimanshu9/T5Bayes-Automated-Review-Summarization-and-Sentiment-Analysis.git
cd T5Bayes-Automated-Review-Summarization-and-Sentiment-Analysis

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Required Dependencies

```
transformers
tensorflow
scikit-learn
pandas
numpy
nltk
```

## Usage

The project consists of three main Python files:

1. `sentiment.py`: Implements the Naive Bayes sentiment analysis model
2. `summariser.py`: Implements the T5-based text summarization model
3. `final.py`: Main script that integrates both models and should be run last

### Running the Application

```bash
# Execute the main script
python final.py
```

### Example Code

```python
# Example usage in your own code
from sentiment import SentimentAnalyzer
from summariser import TextSummarizer

# Initialize models
sentiment_analyzer = SentimentAnalyzer()
text_summarizer = TextSummarizer()

# Process a review
review = "This product exceeded my expectations. The quality is outstanding and it works exactly as advertised. I would highly recommend it to anyone looking for a reliable solution."
summary = text_summarizer.summarize(review)
sentiment = sentiment_analyzer.predict(review)

print(f"Summary: {summary}")
print(f"Sentiment: {sentiment}")
```

## Methodology

### Data Processing

- **Summarization**: Fine-tuned on product review datasets
- **Sentiment Analysis**: Trained on IMDb movie review dataset
- **Preprocessing**: Removal of stopwords, punctuation, and HTML tags, followed by tokenization

### Models

- **T5 Summarizer**: Abstraction-based summarization with high ROUGE and BLEU scores
- **Naive Bayes Classifier**: Fast and efficient sentiment classification with 85%+ accuracy

## Future Scope

- Multilingual support for reviews in multiple languages
- Integration of transformer-based sentiment analysis for more nuanced detection
- Web API deployment for broader accessibility
- Domain-specific adaptations for healthcare, legal, and other specialized applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Himanshu Dahiya - E22CSEU1144

Project Link: [https://github.com/Uhimanshu9/T5-Automated-Review-Summarization-and-Sentiment-Analysis](https://github.com/Uhimanshu9/T5-Automated-Review-Summarization-and-Sentiment-Analysis)

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for T5 implementation
- [IMDb](https://www.imdb.com/) for the sentiment analysis dataset
- [NLTK](https://www.nltk.org/) for natural language processing tools
