import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, download
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

# Load and cluster dataset
def cluster_dataset(filepath, n_clusters=5):
    df = pd.read_csv(filepath)
    # Select numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Encode categorical features
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    cat_encoded = encoder.fit_transform(df[cat_cols])
    # Scale numeric features
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(df[numeric_cols])
    # Combine for clustering
    X = np.hstack([numeric_scaled, cat_encoded])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    return kmeans, encoder, scaler, df

# Chatbot implementation
class Chatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.conversation = []  # List of (user_input, sentiment)
        # Prepare summarization and generation pipelines
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.generator = pipeline('text-generation', model='gpt2')

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmas)

    def analyze_sentiment(self, text):
        scores = self.sentiment_analyzer.polarity_scores(text)
        comp = scores['compound']
        if comp >= 0.05:
            sentiment = 'positive'
        elif comp <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return sentiment, comp

    def generate_response(self, text):
        # Use conversation context + current input as prompt
        history = " ".join([u for u, _ in self.conversation])
        prompt = (history + ' ' + text).strip()
        out = self.generator(prompt, max_length=len(prompt.split())+50, do_sample=True, top_p=0.9)
        response = out[0]['generated_text'][len(prompt):].strip()
        return response

    def summarize_conversation(self):
        full_text = " ".join([u for u, _ in self.conversation])
        summary = self.summarizer(full_text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
        # Sentiment summary
        sentiments = [s for _, s in self.conversation]
        pos = sentiments.count('positive')
        neg = sentiments.count('negative')
        neu = sentiments.count('neutral')
        sentiment_summary = f"Sentiment breakdown: {pos} positive, {neg} negative, {neu} neutral out of {len(sentiments)} messages."
        return summary + '\n' + sentiment_summary

    def chat(self):
        print("Chatbot (English only). Type 'bye' or 'quit' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['bye', 'quit']:
                print("\nConversation summary:")
                print(self.summarize_conversation())
                break
            proc = self.preprocess(user_input)
            sentiment, score = self.analyze_sentiment(user_input)
            self.conversation.append((user_input, sentiment))
            response = self.generate_response(proc)
            print("Bot:", response)

if __name__ == "__main__":
    # Cluster the dataset first
    kmeans_model, encoder, scaler, clustered_df = cluster_dataset('online_gaming_behavior_dataset.csv', n_clusters=5)
    # You can inspect clustered_df or save to file if needed
    # Start chatting
    bot = Chatbot()
    bot.chat()