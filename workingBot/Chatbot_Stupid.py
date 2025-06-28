import pickle
import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter

# --- Download required NLTK data (run once) ---
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')

# --- Load dataset and models ---
# Dataset not directly used in chat, but models trained on it
with open('rf_model_with_clusters.pkl', 'rb') as f:
    profile_pipeline = pickle.load(f)
with open('gamer_profile_model_k6.pkl', 'rb') as f:
    rf_pipeline = pickle.load(f)

# --- Initialize NLP tools ---
sent_analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Predefined entities for gaming domain ---
genres = ["action", "strategy", "rpg", "shooter", "moba", "adventure", "sports"]
difficulties = ["easy", "medium", "hard"]

# --- Chat state ---
chat_history = []
sentiments = []
keywords = []
user_features = {
    "Gender": "Male",
    "Location": "Europe",
    "Age": 30,
    "GameGenre": None,
    "PlayTimeHours": None,
    "InGamePurchases": 0,
    "GameDifficulty": None,
    "SessionsPerWeek": None,
    "AvgSessionDurationMinutes": None,
    "PlayerLevel": None,
    "AchievementsUnlocked": None,
    "EngagementLevel": None,
    "ClusterID": None
}


# --- Functions ---
def clean_input(text):
    tokens = nltk.word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(t.lower()) for t in tokens if t.isalpha() and t.lower() not in stop_words]
    keywords.extend(lemmas)
    return lemmas


def extract_entities(text):  # Fanger alle mine user features
    text_low = text.lower()

    # Genre
    for g in genres:
        if g in text_low:
            user_features['GameGenre'] = g.capitalize()

    # Difficulty
    for d in difficulties:
        if d in text_low:
            user_features['GameDifficulty'] = d.capitalize()

    # Play time hours
    m = re.search(r"(\d+)[ ]?(hours|hrs|h)", text_low)
    if m:
        user_features['PlayTimeHours'] = int(m.group(1))

    # Sessions per week
    m2 = re.search(r"(\d+)[ ]?(sessions per week|sessions a week|times a week)", text_low)
    if m2:
        user_features['SessionsPerWeek'] = int(m2.group(1))

    # Session duration
    m3 = re.search(r"(\d+)[ ]?(minutes per session|min per session)", text_low)
    if m3:
        user_features['AvgSessionDurationMinutes'] = int(m3.group(1))

    # Player level
    m4 = re.search(r"level[ ]?(\d+)", text_low)
    if m4:
        user_features['PlayerLevel'] = int(m4.group(1))

    # Achievements
    m5 = re.search(r"(\d+)[ ]?(achievements|trophies)", text_low)
    if m5:
        user_features['AchievementsUnlocked'] = int(m5.group(1))

    # In-game purchases
    if "buy" in text_low or "purchases" in text_low or "skins" in text_low:
        user_features['InGamePurchases'] = 1

    # Gender
    if "i am male" in text_low or "i'm male" in text_low or "i am a man" in text_low:
        user_features['Gender'] = "Male"
    elif "i am female" in text_low or "i'm female" in text_low or "i am a woman" in text_low:
        user_features['Gender'] = "Female"

    # Age
    m6 = re.search(r"i am (\d{1,2}) years old", text_low)
    if m6:
        user_features['Age'] = int(m6.group(1))

    # Location
    locations = ["europe", "usa", "united states", "canada", "asia", "africa", "australia", "south america", "germany", "denmark", "sweden"]
    for loc in locations:
        if loc in text_low:
            user_features['Location'] = loc.capitalize()




def user_intent(text):
    low = text.lower()
    if any(w in low for w in ["recommend", "suggest"]):
        return 'recommend'
    if any(w in low for w in ["what kind of gamer", "gamer type"]):
        return 'profile'
    return 'chat'


def sentiment_response(text):
    score = sent_analyzer.polarity_scores(text)['compound']
    sentiments.append(score)
    if score >= 0.5:
        return "You seem quite positive about gaming!"
    if score <= -0.5:
        return "You sound concerned – is something frustrating you in games?"
    return "Tell me more about your gaming habits."


def predict_profile():
    df = pd.DataFrame([user_features])
    cluster = profile_pipeline.predict(df)[0]
    user_features['ClusterID'] = cluster  # <- nødvendigt for RF
    return cluster



def predict_preference():
    df = pd.DataFrame([user_features])
    return rf_pipeline.predict(df)[0]


def recommendations(cluster, pref):
    recs = {
        0: { 'High': 'Try fast-paced FPS like Apex Legends or Valorant.',
             'Medium': 'Check out co-op shooters like Deep Rock Galactic.',
             'Low': 'Casual arcade shooters might suit you.'},
        1: { 'High': 'Explore strategy games like Civilization VI or Stellaris.',
             'Medium': 'Try turn-based games like XCOM 2.',
             'Low': 'Mobile strategy like Clash Royale could fit.'},
        2: { 'High': 'Play MOBAs like League of Legends or Dota 2.',
             'Medium': 'Arena brawlers or team games are fun.',
             'Low': 'Watching esports might be engaging.'},
        3: { 'High': 'Dive into RPGs like Witcher 3 or Mass Effect.',
             'Medium': 'Narrative games like Life is Strange.',
             'Low': 'Short indie story games are nice.'}
    }
    return recs.get(cluster, {}).get(pref, 'Explore different genres to find your favorite!')

def check_required_features():
    missing = []
    if not user_features['GameGenre']:
        missing.append("Game genre")
    if user_features['PlayTimeHours'] is None:
        missing.append("Play time hours")
    return missing

def summarize():
    avg = sum(sentiments)/len(sentiments) if sentiments else 0
    mood = 'positive' if avg>0 else 'negative' if avg<0 else 'neutral'
    freq = Counter(keywords).most_common(10)
    print("\n--- Conversation Summary ---")
    print(f"Overall mood: {mood}")
    print("Top keywords:")
    for w,c in freq:
        print(f" - {w}: {c}")
    print("\n--- USER FEATURES ---")
    for k, v in user_features.items():
        print(f"{k}: {v}")

# --- Chat loop ---
print("Welcome to the Gaming Expert Bot! Ask me about gamer profiles or game recommendations.")
print("Type 'exit' to end and see a summary.")
while True:
    text = input('You: ')
    if text.lower() in ['exit','quit','bye']:
        summarize()
        break

    chat_history.append(text)
    clean_input(text)
    extract_entities(text)
    print('Bot:', sentiment_response(text))

    intent = user_intent(text)
    if intent in ['profile', 'recommend']:
        missing = check_required_features()
        if missing:
            print(f"Bot: Please provide: {', '.join(missing)}.")
            continue

        cluster = predict_profile()
        pref = predict_preference()
        rec = recommendations(cluster, pref)

        if intent == 'profile':
            print(f"Bot: You belong to gamer cluster #{cluster}.")
        print(f"Bot: You are likely to enjoy {pref.lower()} games.")
        print(f"Bot: Recommendation: {rec}")


