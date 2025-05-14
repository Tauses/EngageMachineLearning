import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download necessary NLTK resources
nltk.download('punkt_tab')

nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('punkt')

# Lemmatizer and Sentiment Analyzer
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Stopwords
stop_words = set(stopwords.words('english'))


# Function for preprocessing and lemmatization
def preprocess_text(text):
    words = word_tokenize(text.lower())  # Tokenize and lowercase
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)


# Sentiment Analysis Function
def track_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']
    return compound_score


# Word Frequency Tracker
word_frequencies = Counter()


# Chatbot Conversation
def chatbot():
    print("Hello! I am your gamer psychologist chatbot. How can I help you today?")

    # Initialize conversation variables
    user_sentiment = 0
    conversation = []

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye! Hope to chat again soon. Take care!")
            break

        # Preprocess input text and track sentiment
        processed_input = preprocess_text(user_input)
        sentiment = track_sentiment(user_input)
        user_sentiment += sentiment
        conversation.append(processed_input)

        # Sentiment-based feedback
        if sentiment > 0.1:
            print("Chatbot: You seem to be feeling positive today!")
        elif sentiment < -0.1:
            print("Chatbot: It sounds like you're a bit upset. Want to talk about it?")
        else:
            print("Chatbot: You're feeling neutral, huh? Let's keep going.")

    # After conversation ends, show a summary
    print("\nCreating a summary of our conversation...")
    conversation_text = ' '.join(conversation)

    # Word Frequency
    wordcloud = WordCloud(stopwords=stop_words, background_color='white').generate(conversation_text)

    # Plot wordcloud
    plt.figure(figsize=(8, 8), dpi=80)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Frequency count of words
    word_freq = Counter(conversation_text.split())
    print("Most frequent words in the conversation:")
    print(word_freq.most_common(10))


# Run the chatbot
chatbot()
