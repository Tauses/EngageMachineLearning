import pickle
import torch
import pandas as pd
from huggingface_hub import login

# Log ind på Hugging Face med din token
login("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")  # ← din token her

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. Indlæs engagement- og cluster-modeller
with open("engagement_model.pkl", "rb") as f:
    engagement_model = pickle.load(f)

with open("engagement_encoders.pkl", "rb") as f:
    engagement_encoders = pickle.load(f)

with open("gamer_profile_model.pkl", "rb") as f:
    profile_model = pickle.load(f)
# TEST
# 2. Indlæs Mistral 7B (lokalt)
model_id = "./fine_tuned_mistral"  # ← lokal mappe med din trænede model
print("🔄 Indlæser Mistral-model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, temperature=0.7, top_p=0.9)

# 3. Funktion: forudsig engagementniveau fra input (dummy-konvertering)
def predict_engagement(features: dict):
    df = pd.DataFrame([features])
    for col, encoder in engagement_encoders.items():
        if col in df:
            df[col] = encoder.transform(df[col])
    prediction = engagement_model.predict(df)[0]
    level = [k for k, v in engagement_encoders["EngagementLevel"].classes_.items() if v == prediction]
    return prediction

# 4. Funktion: forudsig gamer-profil (cluster)
def predict_profile(features: dict):
    df = pd.DataFrame([features])
    cluster = profile_model.predict(df)[0]
    return cluster

# 5. Funktion: generér respons med Mistral
def generate_response(user_input: str, engagement_level: str, gamer_profile: int):
    prompt = (
        f"Player is a {engagement_level} engagement gamer, cluster {gamer_profile}. "
        f"{user_input}"
    )
    output = chat_pipeline(prompt)[0]["generated_text"]
    return output[len(prompt):].strip()


# 6. Eksempeldata (erstat med brugerdata i praksis)
default_user_features = {
    "Age": 25,
    "Gender": ["Male"],
    "Location": ["Other"],
    "GameGenre": ["Strategy"],
    "PlayTimeHours": 12.5,
    "InGamePurchases": 2,
    "GameDifficulty": ["Medium"],
    "SessionsPerWeek": 5,
    "AvgSessionDurationMinutes": 90,
    "PlayerLevel": 45,
    "AchievementsUnlocked": 23
}

# 7. Chatloop
print("🧠 Klar til intelligent samtale! (skriv 'exit' for at afslutte)\n")

while True:
    user_input = input("Du: ")
    if user_input.lower() in ["exit", "quit", "stop"]:
        print("👋 Farvel! Tak for samtalen.")
        break

    # Predict engagement og profil
    engagement = predict_engagement(default_user_features)
    profile = predict_profile(default_user_features)

    # Generér svar
    response = generate_response(user_input, engagement, profile)
    print("Bot:", response)
    print("-" * 60)
