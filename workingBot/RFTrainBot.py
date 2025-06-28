import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Indlæs datasættet
df_raw = pd.read_csv("../UsableBots/CHATBOT_Ulti_test_2/online_gaming_behavior_dataset.csv", sep=None, engine='python')

# 2. Indlæs K-Means model og tilføj cluster-ID
with open("gamer_profile_model_k6.pkl", "rb") as f:
    cluster_pipeline = pickle.load(f)

df_features = df_raw.drop(columns=["PlayerID", "Age"], errors="ignore")
cluster_ids = cluster_pipeline.predict(df_features)
df_raw["ClusterID"] = cluster_ids

# 3. Definér input (X) og target (y)
df_clean = df_raw.drop(columns=["PlayerID"])
X = df_clean.drop(columns=["EngagementLevel"])
y = df_clean["EngagementLevel"]

# 4. Kategoriske og numeriske kolonner
categorical_cols = ["Gender", "Location", "GameDifficulty", "GameGenre"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# 5. OneHot + Random Forest pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# 6. Træn modellen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# 7. Evaluer modellen
y_pred = pipeline.predict(X_test)
print("🧠 Model Performance:\n")
print(classification_report(y_test, y_pred))

# 8. Gem modellen
with open("rf_model_with_clusters.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("\n✅ Random Forest model gemt som 'rf_model_with_clusters.pkl'")

# 9. Visualiser feature importance
# Få feature-navne ud fra transformeren
onehot = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
onehot_features = onehot.get_feature_names_out(categorical_cols)
all_features = list(onehot_features) + numerical_cols

importances = pipeline.named_steps["classifier"].feature_importances_

# Sorter og visualiser
sorted_idx = importances.argsort()[::-1]
sorted_features = [all_features[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(10,6))
plt.barh(sorted_features, sorted_importances)
plt.gca().invert_yaxis()
plt.title("🔍 Feature Importances (med ClusterID)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
