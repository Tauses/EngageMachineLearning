import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

# Initialize tools
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Preprocessing
def preprocess_text(text):
    words = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]

# Sentiment scoring
def get_sentiment(text):
    return sia.polarity_scores(text)

# Generate dynamic reflection
def reflect_user_input(text, sentiment):
    reflections = [
        "That sounds like something that's been weighing on you.",
        "I hear what you're saying — it matters.",
        "Sounds like you're really in the middle of something intense.",
        "That's a valid way to feel about it.",
        "Hmm, that’s a lot to sit with. I'm here."
    ]
    if sentiment['compound'] < -0.4:
        return random.choice(reflections[:3])
    elif sentiment['compound'] > 0.4:
        return "That actually sounds pretty uplifting."
    else:
        return random.choice(reflections)

# Build a dynamic follow-up question
def ask_follow_up(tokens):
    if 'addicted' in tokens or 'addiction' in tokens:
        return "What makes you feel that way about your gaming?"
    elif 'alone' in tokens or 'lonely' in tokens:
        return "Do you often feel this way outside of games too?"
    elif 'multiplayer' in tokens or 'talk' in tokens:
        return "So the social side of gaming really helps, huh?"
    elif 'fun' in tokens or 'enjoy' in tokens:
        return "What do you find most enjoyable when you're playing?"
    elif 'stress' in tokens or 'tired' in tokens:
        return "Has gaming been a way for you to escape that stress?"
    return "Want to explore that thought a little more?"

# Main chatbot function
def chatbot():
    print("🎮 Hey, I'm your gamer psychologist chatbot. What's been going on lately?")

    conversation = []
    sentiment_total = 0
    num_inputs = 0

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("🧠 That was a meaningful conversation. Take care of yourself. Logging off now.")
            break

        # Processing
        tokens = preprocess_text(user_input)
        sentiment = get_sentiment(user_input)
        compound = sentiment['compound']
        sentiment_total += compound
        num_inputs += 1
        conversation.append(user_input)

        # Build a more natural reply
        reflection = reflect_user_input(user_input, sentiment)
        follow_up = ask_follow_up(tokens)

        print(f"Chatbot: {reflection} {follow_up}")

    # Summary
    print("\n📊 Here's a quick look back at our chat...")
    all_text = ' '.join(conversation)
    wordcloud = WordCloud(stopwords=stop_words, background_color='white').generate(all_text)
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Frequent words
    word_freq = Counter(preprocess_text(all_text))
    print("\n🗨️ Most used words:")
    for word, count in word_freq.most_common(10):
        print(f"{word}: {count}")

    # Mood summary
    avg_sentiment = sentiment_total / max(num_inputs, 1)
    if avg_sentiment > 0.3:
        print("\n😄 You sounded generally positive.")
    elif avg_sentiment < -0.3:
        print("\n😔 You were dealing with some heavy thoughts.")
    else:
        print("\n😐 Your mood seemed somewhere in between.")

# Run the chatbot
chatbot()
