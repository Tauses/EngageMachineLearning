import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("online_gaming_behavior_dataset.csv")
df = df.drop(columns=["PlayerID"])

categorical_cols = ["Gender", "Location", "GameGenre", "GameDifficulty", "EngagementLevel"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop(columns=["EngagementLevel"])
y = df["EngagementLevel"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("engagement_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("engagement_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Engagement-model og encodere gemt.")
