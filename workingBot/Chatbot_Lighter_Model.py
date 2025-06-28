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
#       CHATBOT 'LIGHT' MODEL                           #
#       USES KEYWORDS TO USE ML MODELS                  #
#       USE 'recommend' for RANDOMFOREST ML             #
#       USE 'profile' for K-MEANS ML                    #
#                                                       #
#           MADE BY TAUS ENGELSTOFT SCHADE              #
#                                                       #
#########################################################


# ensure your D: cache folder exists
d_cache = r"D:\PycharmProjects\MLExam\models\hf_cache"
os.makedirs(d_cache, exist_ok=True)

hf_file = hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_0.gguf",
    cache_dir=d_cache
)

# move into your models/ folder if needed
target = r"D:\PycharmProjects\MLExam\models\llama-2-7b-chat.Q4_0.gguf"
os.makedirs(os.path.dirname(target), exist_ok=True)
if not os.path.exists(target):
    os.replace(hf_file, target)

# --- Download required NLTK data (run once) ---
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')

# --- Load updated ML pipelines ---
with open('gamer_profile_model_k6.pkl', 'rb') as f:
    profile_pipeline = pickle.load(f)

with open('rf_model_with_clusters.pkl', 'rb') as f:
    rf_pipeline = pickle.load(f)



# --- Initialize NLTK tools ---
sent_analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Initialize local LLM via llama.cpp ---
llm = Llama(
    model_path=r"D:\PycharmProjects\MLExam\models\llama-2-7b-chat.Q4_0.gguf",
    n_ctx = 4096 # dette er max. må ikke sættes højere
)

# --- Domain entities ---
genres = ["action", "strategy", "rpg", "shooter", "moba", "adventure", "sports"]
difficulties = ["easy", "medium", "hard"]

# --- Chat state and user features ---
chat_history = []
sentiments = []
keywords = []
user_features = {
    "Gender": "Male",       # default
    "Location": "Europe",   # default
    "Age": 30,              # default
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

# --- NLP helper functions ---
def clean_input(text):
    tokens = nltk.word_tokenize(text)   # Splitter tekst op i ord som kaldes for tokens
    lemmas = [lemmatizer.lemmatize(t.lower()) for t in tokens if t.isalpha() and t.lower() not in stop_words]   # Konverterere alle tokens til små bogstaver, isAlpha fjerner tegn og tal t.lower() fjerner stopord som 'the' 'and' osv
    # Lemmas reducerer hvert ord til dets grundform, eks. 'running' bliver til 'run' osv.
    keywords.extend(lemmas) # Lagrer de behandlede nøgleord i keywords listen, bruges senere til analyse af samtalen

def extract_entities(text):
    txt = text.lower()

    # Genre
    genre_aliases = {
        "action": ["action", "fast-paced", "arcade"],
        "strategy": ["strategy", "tactical", "planning"],
        "rpg": ["rpg", "role-playing", "fantasy games"],
        "shooter": ["shooter", "fps", "shooting games", "first person shooter"],
        "moba": ["moba", "league of legends", "dota"],
        "adventure": ["adventure", "story-driven", "exploration"],
        "sports": ["sports", "football games", "fifa", "nba", "racing"]
    }

        # InGamePurchases
    if "i buy skins" in txt or "i buy cosmetics" in txt or "i spend money" in txt:
        user_features['InGamePurchases'] = 1
    elif "i don't buy skins" in txt or "i never spend money" in txt or "no purchases" in txt:
        user_features['InGamePurchases'] = 0

        # PlayerLevel
    lvl_match = re.search(r"(level|lvl)\s*(\d{1,4})", txt)
    if lvl_match:
        user_features['PlayerLevel'] = int(lvl_match.group(2))

        # AvgSession
    m3 = re.search(r"(\d+)\s*(minutes|min|mins)\s*(per session|each session|every time)?", txt)
    if m3:
        user_features['AvgSessionDurationMinutes'] = int(m3.group(1))

        # Achievements
    ach_match = re.search(r"(\d+)\s*(achievements|trophies|badges)", txt)
    if ach_match:
        user_features['AchievementsUnlocked'] = int(ach_match.group(1))

    for genre, aliases in genre_aliases.items():
        for term in aliases:
            if term in txt:
                user_features['GameGenre'] = genre.capitalize()
                break  # Stopper ved første match

    # Difficulty
    for d in difficulties:  # Tjekker ordende fra listen difficulties
        if d in txt:
            user_features['GameDifficulty'] = d.capitalize()

    # Hours played
    m = re.search(r"(about|around|approximately|roughly)?\s*(\d{1,3})\s*(hours?|hrs?|h)\b.*?(week|per week|a week)?",
                  txt)
    if m:                                                   # lidt mere avanceret regex for at være sikker på
        user_features['PlayTimeHours'] = int(m.group(2))    # at den opfylder denne kritiske user feature
                                                            # Kunne godt have taget V3's, men vil gerne have variation
                                                            # mellem de to versioner
    # Sessions per week
    m2 = re.search(r"(\d+)[ ]?(sessions per week|sessions a week|times a week)", txt)
    if m2:
        user_features['SessionsPerWeek'] = int(m2.group(1))
    elif "every day" in txt or "everyday" in txt:
        user_features['SessionsPerWeek'] = 7

    # Gender
    if "i am a woman" in txt or "i'm a woman" in txt or "female" in txt:
        user_features['Gender'] = "Female"
    elif "i am a man" in txt or "i'm a man" in txt or "male" in txt:
        user_features['Gender'] = "Male"

    # Age
    age_match = re.search(r"(\d{1,2})\s?(years? old|yo)?", txt)
    if age_match:
        age = int(age_match.group(1))
        if 5 < age < 100: # Sætter kun alderen hvis et sted mellem (ikke inklusiv) 5 og 100
            user_features['Age'] = age

    # Location (basic)
    countries = ["denmark", "sweden", "norway", "germany", "france", "usa", "canada", "uk"]
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

# --- ML prediction wrappers ---
def predict_preference():
    df = pd.DataFrame([user_features])
    cluster = profile_pipeline.predict(df)[0]
    df["ClusterID"] = cluster  # <- Tilføj ClusterID før RF
    user_features['EngagementLevel'] = rf_pipeline.predict(df)[0]
    return user_features['EngagementLevel']

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


# --- Hybrid chat loop ---
def chat():
    print("🎮 Welcome to Hybrid GamingBot! Type 'exit' to quit.")
    llm_history = [
        {"role": "system", "content": "You are a gaming expert who provides friendly, insightful advice to help players find the right games based on their habits and preferences. Use any analytical results you receive as part of the conversation naturally."}
    ]

    while True:
        user = input("You: ")   # input
        if user.lower() in ('exit', 'quit', 'bye'):
            break

        chat_history.append(user)

        # Preprocess input
        clean_input(user)   # Preprocess
        extract_entities(user)  # Ekstraherer information
        missing = [k for k, v in user_features.items() if v is None and k not in ['EngagementLevel']]   # Laver liste over alle user features hvor værdien er None, undtagen EngagementLevel
        if missing:             # Sat til at printe efter hvert svar
            print("I still need info about:", ", ".join(missing))
                                # Vil stadig bruge modellerne selv med
                                # manglende user features
                                # Så længe playtimehours og gamegenre er opfyldt.
        sentiment_text = sentiment_response(user)   # VADER Score baseret stemningskommentar


        llm_history.append({    # SamtaleLog
            "role": "user",
            "content": f"{user}\n\n(Sentiment analysis: {sentiment_text})"
        })


        ml_output_text = ""
        if 'recommend' in user.lower() or 'profile' in user.lower():    # ML Model keyword trigger
            missing = []
            if not user_features['GameGenre']:
                missing.append("favorite genre")
            if user_features['PlayTimeHours'] is None:
                missing.append("weekly playtime")

            if missing:
                llm_history.append({
                    "role": "user",
                    "content": f"I want recommendations, but I haven’t provided enough information yet. I still need to give: {', '.join(missing)}."
                })
            else:
                # Predict using ML
                cluster = predict_profile()
                pref = predict_preference()
                rec = recommendations(cluster, pref)

                # Append ML-derived context
                ml_output_text = (
                    f"My predicted engagement level is '{pref}', and my gaming cluster is #{cluster}. "
                    f"Suggested recommendation: {rec}"
                )

                llm_history.append({
                    "role": "user",
                    "content": f"Based on my profile, {ml_output_text}"
                })

        # ASSISTANT
        resp = llm.create_chat_completion(  # Her genereres chatbottens svar
            messages=llm_history,
            max_tokens=512,     # Chatbottens MAX svar længde. kan justeres på eget ansvar, anbefales ikke højere 512 tokens = 300 - 400 ord
            temperature=0.7,    # Styrer tilfældigheden af output, dvs. højere = kreative, forudsigelige lavere = faktuelle og forudsigelige
            top_p=0.9           # Model til at vælge mest sandsynlige ord. så 0.9 = 90% sandsynlighed
        )

        assistant = resp['choices'][0]['message']['content']
        print("Assistant:", assistant)  # udskriv svar

        # Gem svar fra bot
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