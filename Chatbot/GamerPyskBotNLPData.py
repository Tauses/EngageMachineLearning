import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# NLTK setup
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

# Load your dataset
df = pd.read_csv("GamingStudy_data.csv")

# NLP tools
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# Preprocess text input
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]

# Track sentiment
def track_sentiment(text):
    return sia.polarity_scores(text)['compound']

# Simple insights for feedback
avg_playtime = df['PlayTimeHours'].mean()
most_common_genre = df['GameGenre'].mode()[0]
avg_sessions = df['SessionsPerWeek'].mean()
avg_duration = df['AvgSessionDurationMinutes'].mean()

# Dynamic response generator
def generate_response(tokens, sentiment):
    if 'addict' in tokens or 'hooked' in tokens:
        return "It’s okay to feel that way. A lot of players report high engagement—maybe setting small breaks might help."
    elif 'alone' in tokens or 'lonely' in tokens:
        return "You’re not alone in feeling that. Many gamers mention the social side of games helps them feel connected."
    elif 'strategy' in tokens:
        return f"Strategy games are actually the most popular genre in our dataset—you're not alone in loving them!"
    elif 'help' in tokens or 'stop' in tokens:
        return "It’s strong of you to recognize that. Would setting a daily limit or journaling how you feel help?"
    elif sentiment < -0.4:
        return "I'm picking up some distress. Want to talk about what’s really on your mind?"
    elif sentiment > 0.4:
        return "Glad you're feeling good about gaming today! What’s been the highlight for you?"
    else:
        return "Interesting. How does that make you feel, personally?"

# Main chatbot
def chatbot():
    print("🎮 Hello! I’m your gamer psychologist chatbot.")
    print(f"📊 Average player plays {avg_playtime:.1f} hours/week. Most common genre: {most_common_genre}.")
    print("🧠 What’s been your experience lately with gaming? Type 'exit' to quit.\n")

    full_conversation = []
    total_sentiment = 0
    turns = 0

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("👋 Thanks for opening up. Let's reflect on what we talked about...")
            break

        tokens = preprocess_text(user_input)
        sentiment = track_sentiment(user_input)
        total_sentiment += sentiment
        turns += 1
        full_conversation.extend(tokens)

        response = generate_response(tokens, sentiment)
        print(f"Chatbot: {response}")

    # Summary
    print("\n📊 Conversation Summary")
    text_blob = ' '.join(full_conversation)
    wordcloud = WordCloud(stopwords=stop_words, background_color='white').generate(text_blob)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("🧩 Word Cloud of Our Conversation")
    plt.show()

    # Frequency analysis
    word_freq = Counter(full_conversation)
    print("\n🗨️ Top 10 words you used:")
    for word, freq in word_freq.most_common(10):
        print(f"{word}: {freq}")

    # Sentiment result
    avg_sentiment = total_sentiment / turns if turns else 0
    if avg_sentiment > 0.3:
        mood = "😄 Mostly positive"
    elif avg_sentiment < -0.3:
        mood = "😔 Some negative feelings came up"
    else:
        mood = "😐 Fairly balanced mood"
    print(f"\n🧠 Emotional tone during chat: {mood}")

# Run the chatbot
chatbot()
