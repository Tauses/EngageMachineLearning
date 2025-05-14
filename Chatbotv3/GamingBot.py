import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict
import random

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load and preprocess the dataset
df = pd.read_csv("online_gaming_behavior_dataset.csv")

# Select features for clustering
features = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek',
            'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# NLP setup
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Chat history
conversation_log = []
sentiment_log = []
context_memory = defaultdict(list)


# NLP pre-processing
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmas = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    return [word for word in lemmas if word not in stop_words]


# Handle user input
def analyze_input(user_input):
    tokens = preprocess_text(user_input)
    sentiment_score = sia.polarity_scores(user_input)['compound']
    sentiment_log.append(sentiment_score)
    conversation_log.extend(tokens)
    context_memory['recent'].extend(tokens)
    return tokens, sentiment_score


# Generate natural, context-based response
def generate_response(tokens, sentiment):
    topics = set(tokens)

    if 'game' in topics:
        return "What kind of games do you enjoy the most?"
    elif 'stress' in topics or 'angry' in topics:
        return "I understand gaming can sometimes be frustrating. Want to talk about what happened?"
    elif 'fun' in topics or 'happy' in topics:
        return "That's great! Tell me what made it so enjoyable."
    elif sentiment > 0.5:
        return "You sound positive! That’s nice to hear."
    elif sentiment < -0.5:
        return "Sounds like something’s off. Want to share more?"
    elif 'achievement' in topics or 'level' in topics:
        return "Nice! Reaching new levels or unlocking achievements feels good, doesn’t it?"
    else:
        context = ' '.join(context_memory['recent'][-10:])
        return f"Interesting... you mentioned '{context}'. Want to continue?"


# Summary at the end
def summarize_conversation():
    print("\n--- Chat Summary ---")
    word_freq = nltk.FreqDist([w for w in conversation_log if w not in stop_words])
    top_words = word_freq.most_common(5)
    print("Main topics:", [w for w, _ in top_words])

    avg_sentiment = np.mean(sentiment_log)
    mood = (
        "very positive 😊" if avg_sentiment > 0.5 else
        "very negative 😟" if avg_sentiment < -0.5 else
        "neutral 😐"
    )
    print(f"Overall mood: {mood} (avg sentiment score: {avg_sentiment:.2f})")


# Main chat loop
def start_chat():
    print("🎮 Gamer Psychologist Chatbot (type 'bye' or 'quit' to exit)\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ['bye', 'quit']:
            summarize_conversation()
            break
        tokens, sentiment = analyze_input(user_input)
        reply = generate_response(tokens, sentiment)
        print("Bot:", reply)


start_chat()
