import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from collections import deque

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Simple sentiment lexicon (no sentiment analysis lib)
positive_words = {"good", "great", "fun", "love", "enjoy", "amazing", "awesome", "like", "giggle", "giggling",
                  "humorous", "hilarious"}
negative_words = {"bad", "boring", "hate", "annoying", "terrible", "dislike", "slow", "lag", "fuck", "asshole",
                  "stupid", "dumbass"}

# Conversation memory (Store past interactions for context)
conversation = deque(maxlen=5)  # Save the last 5 exchanges
sentiment_score = 0

# Define cluster-based personality templates
cluster_profiles = {
    0: "You seem like a strategic player who enjoys deep gameplay and consistency.",
    1: "You might be more of a casual gamer who drops in occasionally.",
    2: "Looks like you care about customization and in-game items!",
    3: "You seem like a competitive player who thrives on leveling up quickly."
}

cluster_recommendations = {  # SPILANBEFALINGER BASERET PÅ PROFIL
    0: ["Civilization VI", "XCOM 2", "Crusader Kings III"],
    1: ["Stardew Valley", "The Sims 4", "Animal Crossing"],
    2: ["Fortnite", "Valorant", "League of Legends (for skins!)"],
    3: ["Call of Duty: Warzone", "Apex Legends", "Rocket League"]
}


# Function to lemmatize the input
def lemmatize_input(text):
    tokens = word_tokenize(text.lower())
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return lemmas


# Function to detect the user's cluster based on input (uses the trained model)
def detect_cluster_with_model(user_lemmas):
    features = {
        "PlayTimeHours": 10,
        "InGamePurchases": 1 if "buy" in user_lemmas or "shop" in user_lemmas else 0,
        "SessionsPerWeek": 5 if "daily" in user_lemmas else 2,
        "AvgSessionDurationMinutes": 60 if "long" in user_lemmas or "hours" in user_lemmas else 20,
        "PlayerLevel": 50 if "level" in user_lemmas else 10,
        "AchievementsUnlocked": 30 if "achievement" in user_lemmas else 5,
        "GameGenre": "Strategy" if "strategy" in user_lemmas else "Action",
        "GameDifficulty": "Hard" if "challenge" in user_lemmas else "Medium",
        "EngagementLevel": "High" if "addicted" in user_lemmas or "love" in user_lemmas else "Medium"
    }

    input_df = pd.DataFrame([features])
    with open("gamer_profile_model.pkl", "rb") as f:
        model = pickle.load(f)
    cluster = model.predict(input_df)[0]
    return cluster


# Function to update the sentiment score based on positive or negative words
def update_sentiment(user_lemmas):
    global sentiment_score
    pos = len(set(user_lemmas) & positive_words)
    neg = len(set(user_lemmas) & negative_words)
    sentiment_score += (pos - neg)


# Function to generate a summary at the end of the conversation
def summarize():
    mood = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
    summary = f"""
--- Conversation Summary ---
You had a {mood} conversation.
Key points you mentioned:
{"\n- ".join([c[0] for c in conversation])}

Thanks for chatting! Based on our discussion, I recommend these games:
"""
    return summary

def generate_wordcloud():
    all_text = " ".join([entry[0] for entry in conversation])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    # Gem som billede
    wordcloud.to_file("wordcloud.png")
    print("WordCloud generated and saved as 'wordcloud.png'.")


# Function to bring the conversation back on topic if off-topic words are detected
def bring_back_on_topic(user_input):
    # Check if the input has words that are clearly off-topic (example: unrelated to gaming)
    off_topic_keywords = ["weather", "politics", "sports", "travel", "food"]
    user_lemmas = lemmatize_input(user_input)
    if any(word in user_lemmas for word in off_topic_keywords):
        return "Let's get back to gaming! What games have you been enjoying lately?"
    return None


# Generate dynamic response based on context and cluster
# Kan måske ikke blive bedre end dette, uden at lave API kald til en anden model
def generate_dynamic_response(user_input):
    # Extract context from the conversation
    last_interaction = conversation[-1][0] if conversation else ""
    if "level" in user_input or "level up" in user_input:
        return "Ah, you're interested in leveling up! What games do you enjoy that really challenge your progression?"
    elif "buy" in user_input or "purchase" in user_input:
        return "Sounds like you're into in-game purchases! Do you prefer games that offer lots of customization?"
    else:
        # Use context from the last interaction to provide dynamic answers
        if "game" in last_interaction or "play" in last_interaction:
            return f"It seems like we were just talking about games! What's your all-time favorite game?"
        else:
            return "Tell me more about your gaming habits!"


# Chatbot function where the conversation takes place
def chatbot():
    print("ChatBot: Hi there! Let's talk about your gaming habits. Type 'bye' to end.")
    while True:
        user_input = input("You: ")

        # Detect off-topic conversation and bring it back on topic
        off_topic_response = bring_back_on_topic(user_input)
        if off_topic_response:
            print("ChatBot:", off_topic_response)
            continue  # Skip processing for this input and ask again

        if user_input.strip().lower() in ["bye", "quit", "exit"]:
            # After conversation ends, detect the cluster based on the final input
            cluster_id = detect_cluster_with_model([lemmatizer.lemmatize(word) for word in word_tokenize(user_input)])
            profile = cluster_profiles.get(cluster_id, "You're a unique gamer!")
            recommendations = cluster_recommendations.get(cluster_id, [])

            summary = summarize()
            print(summary)
            print(f"Based on our conversation, I recommend these games:\n- " + "\n- ".join(recommendations))

            # Generér WordCloud ved afslutning
            generate_wordcloud()

            break  # End the conversation here

        # Continue the conversation without recommending games yet
        lemmas = lemmatize_input(user_input)
        update_sentiment(lemmas)

        # Detect cluster while still in conversation, but don't show recommendations yet
        cluster_id = detect_cluster_with_model(lemmas)
        profile = cluster_profiles.get(cluster_id, "You're a unique gamer!")

        # Generate a dynamic response based on the context
        response = generate_dynamic_response(user_input)

        print("ChatBot:", response)
        conversation.append((user_input, response))  # Add to conversation memory


if __name__ == "__main__":
    chatbot()
