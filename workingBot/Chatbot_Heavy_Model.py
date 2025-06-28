import pickle
import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

#########################################################
#       CHATBOT 'HEAVY' MODEL                           #
#       DOES NOT NEED KEYWORDS                          #
#       USES RANDOMFOREST AND K-MEANS ML                #
#       'AUTOMATICALLY'                                 #
#       ANSWERS TAKE LONGER HERE                        #
#                                                       #
#           MADE BY TAUS ENGELSTOFT SCHADE              #
#                                                       #
#########################################################

# --- Setup cache/model path ---
d_cache = r"D:\PycharmProjects\MLExam\models\hf_cache"
os.makedirs(d_cache, exist_ok=True)

hf_file = hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_0.gguf",
    cache_dir=d_cache
)

target = r"D:\PycharmProjects\MLExam\models\llama-2-7b-chat.Q4_0.gguf"
os.makedirs(os.path.dirname(target), exist_ok=True)
if not os.path.exists(target):
    os.replace(hf_file, target)

# --- Download required NLTK data ---
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# -- LOAD updated models --
with open('gamer_profile_model_k6.pkl', 'rb') as f:  # NEW KMeans
    profile_pipeline = pickle.load(f)

with open('rf_model_with_clusters.pkl', 'rb') as f:  # NEW RF
    rf_pipeline = pickle.load(f)


# --- NLP setup ---
sent_analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- LLM setup ---
llm = Llama(
    model_path=target,
    n_ctx=4096 # dette er max. må ikke sættes højere
)

# --- Constants ---
genres = ["action", "strategy", "rpg", "shooter", "moba", "adventure", "sports"]
difficulties = ["easy", "medium", "hard"]

# --- State ---
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
    "EngagementLevel": None
}

# --- Helpers ---
def clean_input(text):  # Gemmer nøgleord i en keywords liste
    tokens = nltk.word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(t.lower()) for t in tokens if t.isalpha() and t.lower() not in stop_words]
    keywords.extend(lemmas)

def extract_entities(text): # Tjekker tekst for alle user features
    txt = text.lower()

    # Genre
    #for g in genres:                                       # GAMMEL MÅDE AT GØRE DET PÅ
    #    if g in txt:                                       # ERSTATTET GRUNDET SKRØBELIGHED OG UFLEKSIBILITET
    #        user_features['GameGenre'] = g.capitalize()
    GENRE_ALIASES = {                                       # BRUGER NU SEMANTISK MATCHING
        "action": ["action", "fast-paced", "combat-heavy"], # MERE FLEKSIBEL OG FANGER FLERE ORD
        "strategy": ["strategy", "tactics", "planning", "rts", "turn-based"],
        "rpg": ["rpg", "role playing", "role-playing", "quests"],
        "shooter": ["shooter", "fps", "shooting", "gunplay"],
        "moba": ["moba", "multiplayer online battle arena", "league of legends"],
        "adventure": ["adventure", "story-driven", "exploration", "narrative"],
        "sports": ["sports", "football", "soccer", "nba", "fifa"]
    }

    for genre, keywords in GENRE_ALIASES.items():
        if any(k in txt for k in keywords):
            user_features['GameGenre'] = genre.capitalize()

    # Difficulty
    for d in difficulties:
        if d in txt:
            user_features['GameDifficulty'] = d.capitalize()

    # Hours played
    m = re.search(r"(\d+)[-–]?\d*\s?(hours|hrs|h)?", txt)    # Udvidet regex
    if m:                                                           # leder stadig efter keywords
        hours = int(m.group(1))                                     # på en heuristisk metode
        if "per day" in txt or "every day" in txt:                  # dvs. if'en her prøver sig på
            user_features['PlayTimeHours'] = hours * 7              # indtil den finder en løsning (heri er løsning keywordet)
        elif "per week" in txt or "weekly" in txt:
            user_features['PlayTimeHours'] = hours
        elif "a few hours" in txt:
            user_features['PlayTimeHours'] = 5  # sætter standard for 'a few hours' og all day
        elif "all day" in txt:
            user_features['PlayTimeHours'] = 35
        else:
            user_features['PlayTimeHours'] = hours  # fallback, hvis ingen matches tager vi
                                                    # 'X hours' i brugerinputtet
    # Sessions per week
    m2 = re.search(r"(\d+)[ ]?(sessions per week|sessions a week|times a week)", txt)
    if m2:
        user_features['SessionsPerWeek'] = int(m2.group(1))
    elif "every day" in txt or "everyday" in txt:
        user_features['SessionsPerWeek'] = 7

    # Avg session duration
    m3 = re.search(r"(\d+)[ ]?(minutes|mins|min) (per session|each session)?", txt)
    if m3:
        user_features['AvgSessionDurationMinutes'] = int(m3.group(1))

    # Player level
    m4 = re.search(r"(level|lvl)[ ]?(\d+)", txt) # regex
    if m4:
        user_features['PlayerLevel'] = int(m4.group(2))

    # Achievements unlocked
    m5 = re.search(r"(\d+)[ ]?(achievements|trophies|unlocked)", txt) # regex
    if m5:
        user_features['AchievementsUnlocked'] = int(m5.group(1))

    # In-game purchases
    if "buy skins" in txt or "in-game purchase" in txt or "i spend money" in txt or "microtransactions" in txt: # kunne godt forbedres ved
        user_features['InGamePurchases'] = 1                                                                    # ikke at gøre det til keywords
                                            # Bliver kun sat til 1 hvis fundet og ikke antallet af køb
    # Gender
    if "i am a woman" in txt or "i'm a woman" in txt or "female" in txt:    # også heuristisk metode her
        user_features['Gender'] = "Female"
    elif "i am a man" in txt or "i'm a man" in txt or "male" in txt:
        user_features['Gender'] = "Male"

    # Age
    age_match = re.search(r"(\d{2})\s?(years? old|yo)?", txt) # Bruger regex til at søge efter user features
    if age_match:
        age = int(age_match.group(1))
        if 5 < age < 100:
            user_features['Age'] = age

    # Location (basic check)
    countries = ["denmark", "sweden", "norway", "germany", "france", "usa", "canada"] # begrænset til disse lande
    for c in countries:
        if c in txt:
            user_features['Location'] = c.capitalize()


def sentiment_response(text):
    score = sent_analyzer.polarity_scores(text)['compound']
    sentiments.append(score)
    if score >= 0.5:
        return "You seem quite positive about gaming!"
    if score <= -0.5:
        return "You seem concerned – anything frustrating you?"
    return "Tell me more about your gaming habits."

def predict_preference():
    df = pd.DataFrame([user_features])
    cluster = profile_pipeline.predict(df)[0]
    df["ClusterID"] = cluster  # <- Tilføj ClusterID før RF
    user_features['EngagementLevel'] = rf_pipeline.predict(df)[0]
    return user_features['EngagementLevel'] # Engagement level bruges til at udlede mit ClusterID


def predict_profile():
    df = pd.DataFrame([user_features])
    return profile_pipeline.predict(df)[0]


def recommendations(cluster, pref):
    recs = {
        0: {'High': 'Build epic cities in Anno 1800 or Starcraft II.',
            'Medium': 'Try Plague Inc or Civilization Lite.',
            'Low': 'Mini strategy games on mobile might suit you.'},
        1: {'High': 'Wordscapes or Monument Valley are great.',
            'Medium': 'Play casual puzzle games like Candy Crush.',
            'Low': 'Short brain teasers might be enough.'},
        2: {'High': 'Play Apex Legends or CS:GO for adrenaline.',
            'Medium': 'Try Fortnite or Overwatch casually.',
            'Low': 'Maybe try single-player FPS like Doom Eternal.'},
        3: {'High': 'Witcher 3 or Divinity: Original Sin II.',
            'Medium': 'Try Skyrim or Dragon Age.',
            'Low': 'Casual fantasy RPGs like Fable.'},
        4: {'High': 'Try indie hits like Hollow Knight or Hades.',
            'Medium': 'Portal 2 and Stardew Valley are relaxing.',
            'Low': 'Try browser games or short Steam titles.'},
        5: {'High': 'Play It Takes Two or Phasmaphobia with friends.',
            'Medium': 'Try Among Us or Fall Guys.',
            'Low': 'Maybe play Jackbox Games at parties.'},
    }
    return recs.get(cluster, {}).get(pref, 'Explore different genres to find your favorite!')


# --- Main Chat ---
def chat():
    print("🎮 Welcome to Hybrid GamingBot! Type 'exit' to quit.")
    llm_history = [
        {"role": "system", "content": "You are a gaming expert who helps users find games based on their habits. Integrate any predictions into the conversation naturally."}
    ]

    while True:
        user = input("You: ") # Her håndteres brugerinputtet
        if user.lower() in ('exit', 'quit', 'bye'):
            break

        chat_history.append(user)
        clean_input(user)
        extract_entities(user)  # Kontra V2, vil vi ikke printe manglende user features
        sentiment_text = sentiment_response(user)   # Generere stemningskommentar på VADER scoren

        user_context = f"{user}\n(Sentiment analysis: {sentiment_text})"
        ml_output_text = ""

        # Auto ML prediction
        required_fields = ["GameGenre", "PlayTimeHours", "SessionsPerWeek", "GameDifficulty",   # Ny tilføjelse. Ændret fra at opfylde to features
                           "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"]  # som var playtimehours og gamegenre, men det var fejlbehæftet pg skrøbeligt
        filled = [f for f in required_fields if user_features[f] is not None]                   # erstattet med at den minimum skal have tre udfyldte features for at
                                                                                                # forudsige cluster
        if len(filled) >= 3:                # <- fx: Kør prediction hvis mindst 3 vigtige features er sat
            pref = predict_preference()     # kan justeres
            cluster = predict_profile()     # Vil ikke forudsige uden min 3 user features
            rec = recommendations(cluster, pref)
            ml_output_text = (
                f"My predicted engagement level is '{pref}', and your gaming cluster is #{cluster}. "
                f"Suggested game: {rec}"
            )
            user_context += f"\n\n[Profile Insight] {ml_output_text}"   # inkluderes i svar

        llm_history.append({"role": "user", "content": user_context})   # sendes til llama

        resp = llm.create_chat_completion(
            messages=llm_history,
            max_tokens=512,     # Chatbottens MAX svar længde. kan justeres på eget ansvar, anbefales ikke højere 512 tokens = 300 - 400 ord
            temperature=0.7,    # Styrer tilfældigheden af output, dvs. højere = kreative, forudsigelige lavere = faktuelle og forudsigelige
            top_p=0.9           # Model til at vælge mest sandsynlige ord. så 0.9 = 90% sandsynlighed
        )

        assistant = resp['choices'][0]['message']['content']
        print("Assistant:", assistant)
        llm_history.append({"role": "assistant", "content": assistant})

    # --- Summary ---
    avg = sum(sentiments) / len(sentiments) if sentiments else 0
    mood = 'positive' if avg > 0 else 'negative' if avg < 0 else 'neutral'
    freq = Counter(keywords).most_common(10)

    print("\n--- Chat Summary ---")
    print(f"Mood: {mood}")
    print("Keywords:")
    for w, c in freq:
        print(f" - {w}: {c}")

    print("\n--- Final user profile ---")
    for k, v in user_features.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    chat()
