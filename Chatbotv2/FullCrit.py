import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

# Init tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# Load dataset
df = pd.read_csv("GamingStudy_data.csv", encoding='cp1252')  # Replace with 'latin1' if needed

# Simulate burnout_risk based on heuristics
def calculate_burnout(row):
    score = 0
    if row['PlayTimeHours'] > 40:
        score += 1
    if row['GameDifficulty'] > 7:
        score += 1
    if row['SessionsPerWeek'] > 6:
        score += 1
    return ['Low', 'Medium', 'High'][score]

df['burnout_risk'] = df.apply(calculate_burnout, axis=1)

# Train model
features = ['PlayTimeHours', 'InGamePurchases', 'GameDifficulty', 'SessionsPerWeek',
            'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']

X = df[features]
y = LabelEncoder().fit_transform(df['burnout_risk'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Text preprocessing
def preprocess_text(text):
    words = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(w) for w in words if w.isalpha() and w not in stop_words]

# Topic detection
def detect_topic(tokens):
    themes = {
        'addiction': ['addicted', 'hooked', 'obsessed'],
        'stress': ['tired', 'burned', 'overwhelmed', 'stressed'],
        'fun': ['enjoy', 'fun', 'happy'],
        'balance': ['balance', 'routine', 'limit'],
        'social': ['friend', 'alone', 'community', 'multiplayer'],
    }
    for topic, keywords in themes.items():
        if any(word in keywords for word in tokens):
            return topic
    return None

# Burnout prediction
def predict_burnout():
    print("📊 Let's assess your burnout risk. Answer a few quick questions:")
    try:
        hp = float(input("→ Hours played per week: "))
        purchases = int(input("→ In-game purchases per month: "))
        difficulty = int(input("→ Preferred game difficulty (1-10): "))
        sessions = int(input("→ Gaming sessions per week: "))
        avg_duration = float(input("→ Average session duration (minutes): "))
        level = int(input("→ Current player level (0-100): "))
        achievements = int(input("→ Achievements unlocked (0-100): "))

        user_data = pd.DataFrame([[hp, purchases, difficulty, sessions, avg_duration, level, achievements]],
                                 columns=features)
        prediction = model.predict(user_data)[0]
        decoded = {0: 'Low', 1: 'Medium', 2: 'High'}[prediction]
        print(f"🧠 Based on your gaming habits, your **burnout risk** is: **{decoded}**")
    except:
        print("⚠️ Invalid input. Let's just continue our chat.")

# Chatbot
def chatbot():
    print("🎮 Hello! I’m your gamer psychologist. What’s on your mind today?")
    sentiment_score = 0
    full_tokens = []

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("👋 Take care of yourself. Thanks for the talk!")
            break

        if "burnout" in user_input.lower():
            predict_burnout()
            continue

        tokens = preprocess_text(user_input)
        full_tokens.extend(tokens)

        topic = detect_topic(tokens)
        sentiment = sia.polarity_scores(user_input)['compound']
        sentiment_score += sentiment

        # Simple response system
        if topic == 'addiction':
            print("Chatbot: It sounds like you’re reflecting deeply on your gaming habits.")
        elif topic == 'stress':
            print("Chatbot: That sounds like a lot to carry. Want to talk about what’s overwhelming you?")
        elif topic == 'balance':
            print("Chatbot: It’s good that you’re thinking about balance. Any changes you’re considering?")
        elif topic == 'social':
            print("Chatbot: Having a community online can help. Do you feel supported there?")
        elif topic == 'fun':
            print("Chatbot: That’s great to hear! Joy in gaming is so important.")
        else:
            print("Chatbot: I'm listening. Tell me more...")

    # Wrap-up summary
    print("\n📊 Summary of our conversation:")
    wordcloud = WordCloud(stopwords=stop_words, background_color='white').generate(' '.join(full_tokens))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    print("🗨️ Top words:")
    for word, count in Counter(full_tokens).most_common(10):
        print(f"{word}: {count}")

    avg_sent = sentiment_score / (len(full_tokens) // 5 + 1)
    if avg_sent > 0.3:
        print("\n😄 You seemed mostly positive today.")
    elif avg_sent < -0.3:
        print("\n😔 You shared some heavy thoughts. I'm glad you talked about them.")
    else:
        print("\n😐 Your mood seemed pretty balanced overall.")

# Run it
chatbot()
