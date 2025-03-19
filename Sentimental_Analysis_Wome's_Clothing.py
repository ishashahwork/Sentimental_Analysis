# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 20:38:22 2025

@author: ishah
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import re
import matplotlib.pyplot as plt
import seaborn as sns


# Download necessary NLTK resources (do this once)
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("C:\\Users\\ishah\\Downloads\\Womens_Clothing_E-Commerce_Reviews.csv")  # Replace with your file path

# Data Cleaning and Preprocessing
def clean_text(text):
    if isinstance(text, str):  # Check if it's a string, handle potential missing values
        text = re.sub(r'[^\w\s]', '', text, re.UNICODE)  # Remove punctuation
        text = text.lower()  # Lowercase
        return text
    return ""  # Return empty string if it's not a string

df['Review Text'] = df['Review Text'].apply(clean_text)  # Apply cleaning to the review text

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if isinstance(text, str) and text:  # Check for string AND non-empty string
        scores = analyzer.polarity_scores(text)
        return scores['compound']
    return 0  # Return a neutral score (0) for missing or invalid text

df['Sentiment_Score'] = df['Review Text'].apply(analyze_sentiment)

# Categorize Sentiment
def categorize_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment_Category'] = df['Sentiment_Score'].apply(categorize_sentiment)

# --- Analysis and Visualization ---

# Define a custom calm palette
calming_palette = ["#A8D5BA", "#B2DFDB", "#81C784", "#4DB6AC", "#80CBC4"]

# Set Seaborn style and apply the custom palette
sns.set(style="whitegrid", palette=calming_palette)

# Set Seaborn style and custom calm palette for a soothing look



# Sentiment vs. Rating (Box Plot)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Rating', y='Sentiment_Score', data=df, palette=calming_palette)
plt.title('Sentiment Score vs. Product Rating', fontsize=14)
plt.xlabel('Product Rating', fontsize=12)
plt.ylabel('Sentiment Score', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# Sentiment Category Distribution (Count Plot)
plt.figure(figsize=(6, 4))
sns.countplot(x='Sentiment_Category', data=df, order=['Positive', 'Neutral', 'Negative'], palette=calming_palette)
plt.title('Distribution of Sentiment Categories', fontsize=14)
plt.xlabel('Sentiment Category', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# Sentiment by Division (if available)
if 'Department Name' in df.columns:  # Check if the column exists
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Department Name', hue='Sentiment_Category', data=df, palette=calming_palette)
    plt.title('Sentiment by Department', fontsize=14)
    plt.xlabel('Department Name', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# Display the first few rows with sentiment analysis results
print(df[['Review Text', 'Sentiment_Score', 'Sentiment_Category']].head())

# --- Further Analysis (Word Frequencies) --

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer



def get_top_words_by_sentiment(text, sentiment, n=20):

    # 1. Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]

    # 2. Calculate TF-IDF scores
    vectorizer = TfidfVectorizer(stop_words='english') 
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]


    # Create a dictionary of word-score pairs
    word_scores = dict(zip(feature_names, tfidf_scores))

    # Sort words by score in descending order
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_words[:n]

# Find top words in positive reviews
positive_reviews_text = " ".join(df[df['Sentiment_Category'] == 'Positive']['Review Text'].dropna())
top_positive_words = get_top_words_by_sentiment(positive_reviews_text, 'Positive')

# Find top words in negative reviews
negative_reviews_text = " ".join(df[df['Sentiment_Category'] == 'Negative']['Review Text'].dropna())
top_negative_words = get_top_words_by_sentiment(negative_reviews_text, 'Negative')

print("\nTop Positive Words:")
print(top_positive_words)

print("\nTop Negative Words:")
print(top_negative_words)

# Find top positive and negative phrases
# Function to extract phrases
def extract_phrases(text):
    tokens = word_tokenize(text)
    bigrams = list(zip(tokens, tokens[1:])) 
    return [" ".join(bigram) for bigram in bigrams]

# Analyze sentiment at the phrase level
def analyze_phrase_sentiment(text):
    phrases = extract_phrases(text)
    phrase_sentiments = {}
    for phrase in phrases:
        score = analyzer.polarity_scores(phrase)['compound']
        phrase_sentiments[phrase] = score
    return phrase_sentiments

df['Phrase_Sentiments'] = df['Review Text'].apply(analyze_phrase_sentiment)

# Find top positive and negative phrases
def get_top_phrases(sentiment, n=20):
    all_phrases = []
    for index, row in df[df['Sentiment_Category'] == sentiment].iterrows(): 
        phrases = row['Phrase_Sentiments'] 
        if isinstance(phrases, dict): 
            all_phrases.extend(list(phrases.items())) 
    
    top_phrases = Counter(all_phrases).most_common(n) 
    return top_phrases

top_positive_phrases = get_top_phrases('Positive')
top_negative_phrases = get_top_phrases('Negative')

print("\nTop Positive Phrases:")
print(top_positive_phrases)

print("\nTop Negative Phrases:")
print(top_negative_phrases)

