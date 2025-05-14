import numpy as np
import pandas as pd
import pickle


# Q-learning Chatbot-klasse
class ChatbotRL:
    def __init__(self, engagement_model_path, engagement_encoders_path, trainprofile_model_path):
        # Indlæs eksisterende modeller
        self.engagement_model = pickle.load(open(engagement_model_path, "rb"))
        self.encoders = pickle.load(open(engagement_encoders_path, "rb"))
        self.trainprofile_model = pickle.load(open(trainprofile_model_path, "rb"))

        # Q-table initialization (for simplicity, using a small range of states and actions)
        self.q_table = np.zeros((10, 5))  # 10 states, 5 actions (just an example)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

    def select_action(self, state):
        # Exploration vs Exploitation
        if np.random.rand() < self.exploration_rate:
            # Exploration: Vælg en tilfældig handling
            action = np.random.choice(range(5))
        else:
            # Exploitation: Vælg den bedste handling ifølge Q-table
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, state, action, reward, next_state):
        # Opdater Q-table med Q-learning formel
        future_reward = np.max(self.q_table[next_state])
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (
                reward + self.discount_factor * future_reward - self.q_table[state, action])

        # Decay exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

    def get_engagement_level(self, user_input):
        features_dict = self.extract_features_from_input(user_input)
        df = pd.DataFrame([features_dict])  # Kolonnenavne matcher træning
        prediction = self.engagement_model.predict(df)
        return prediction[0]

    def get_profile(self, user_input):
        # Trænprofil-modelen bruges til at få spillerens profil
        features = self.extract_features_from_input(user_input)
        profile = self.trainprofile_model.predict([features])
        return profile

    def extract_features_from_input(self, user_input):
        # Dummy: Eksempel på brugerinput konverteret til feature-vektor
        # Sørg for rækkefølgen matcher træningsdata: ["Age", "Gender", "Location", "GameGenre", ...]

        # Brug encodere til at omkode kategoriske værdier
        gender = self.encoders["Gender"].transform(["Male"])[0]
        location = self.encoders["Location"].transform(["Other"])[0]
        genre = self.encoders["GameGenre"].transform(["Strategy"])[0]
        difficulty = self.encoders["GameDifficulty"].transform(["Medium"])[0]

        return {
            "Age": 43,
            "Gender": gender,
            "Location": location,
            "GameGenre": genre,
            "PlayTimeHours": 16.27,
            "InGamePurchases": 0,
            "GameDifficulty": difficulty,
            "SessionsPerWeek": 6,
            "AvgSessionDurationMinutes": 108,
            "PlayerLevel": 79,
            "AchievementsUnlocked": 25
        }

    def chat(self, user_input, user_feedback=None):
        """
        Interager med brugeren og opdater RL agenten baseret på feedback.

        user_input: Brugerens input.
        user_feedback: Brugeren kan give feedback som 1 (positiv) eller -1 (negativ).
        """
        state = 0  # For simplicity, initial state
        action = self.select_action(state)
        response = f"Response {action}"  # Placeholder response

        # Få engagementniveau baseret på brugerens input
        engagement_level = self.get_engagement_level(user_input)

        # Hvis der er feedback, brug den til at opdatere Q-table
        reward = engagement_level  # Belønning er relateret til engagement niveau
        if user_feedback is not None:
            reward = user_feedback  # Brugeren giver en direkte feedback (-1 eller 1)

        next_state = (state + 1) % 10  # For simplicity, næste state
        self.update_q_table(state, action, reward, next_state)

        return response, reward


# Chatbot setup
def setup_chatbot():
    chatbot = ChatbotRL(
        engagement_model_path="engagement_model.pkl",
        engagement_encoders_path="engagement_encoders.pkl",
        trainprofile_model_path="gamer_profile_model.pkl"
    )
    return chatbot


# Brug chatbotten
chatbot = setup_chatbot()

# Simuler en samtale med feedback
user_input = "How do I improve my strategy in the game?"
user_feedback = 1  # 1 betyder positiv feedback fra brugeren

response, reward = chatbot.chat(user_input, user_feedback)

print("Bot response:", response)
print("Reward (Feedback):", reward)

if __name__ == "__main__":
    chatbot = ChatbotRL(
        engagement_model_path="engagement_model.pkl",
        engagement_encoders_path="engagement_encoders.pkl",
        trainprofile_model_path="gamer_profile_model.pkl"
    )

    print("Velkommen til den intelligente chatbot! (skriv 'exit' for at afslutte)\n")

    while True:
        user_input = input("Du: ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Chat afsluttet. På gensyn!")
            break

        response, reward = chatbot.chat(user_input)
        print(f"Bot: {response}")
        print(f"Reward (Feedback): {reward}")
        print("-" * 40)

