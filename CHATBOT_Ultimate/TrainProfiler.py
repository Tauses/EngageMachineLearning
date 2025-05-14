import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import pickle

# Indlæs data
df = pd.read_csv("online_gaming_behavior_dataset.csv")

# Features
features = [
    'PlayTimeHours',
    'InGamePurchases',
    'SessionsPerWeek',
    'AvgSessionDurationMinutes',
    'PlayerLevel',
    'AchievementsUnlocked',
    'GameGenre',
    'GameDifficulty',
    'EngagementLevel'
]
data = df[features]

# Kolonner
numerical_cols = [
    'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek',
    'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked'
]
categorical_cols = ['GameGenre', 'GameDifficulty', 'EngagementLevel']

# Transformere
numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Samlet preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Pipeline med KMeans
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clusterer", KMeans(n_clusters=4, random_state=42))
])

# Træn model
pipeline.fit(data)

# Gem pipeline
with open("gamer_profile_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Modelen er trænet og gemt.")
