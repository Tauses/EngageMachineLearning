# tune_and_label_clusters.py

import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D  # Til 3D plot


class GamerProfiler:
    def __init__(self, n_clusters):
        self.numerical_cols = [
            'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek',
            'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked'
        ]
        self.categorical_cols = ['GameGenre', 'GameDifficulty']
        self.preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), self.numerical_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), self.categorical_cols)
        ])
        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("clusterer", KMeans(n_clusters=n_clusters, random_state=42))
        ])

    def fit(self, df):
        df_clean = df.drop(columns=["PlayerID","Age"], errors="ignore")
        print("Træner på kolonner:", df_clean.columns.tolist())
        self.pipeline.fit(df_clean)

    def predict(self, df):
        df_clean = df.drop(columns=["PlayerID","Age"], errors="ignore")
        return self.pipeline.predict(df_clean)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)


def find_best_k(df, k_min=2, k_max=10):
    df_clean = df.drop(columns=["PlayerID","Age"], errors="ignore")
    transformer = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), [
            'PlayTimeHours','InGamePurchases','SessionsPerWeek',
            'AvgSessionDurationMinutes','PlayerLevel','AchievementsUnlocked'
        ]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), ["GameGenre","GameDifficulty"])
    ])
    X = transformer.fit_transform(df_clean)

    inertias = []
    silhouettes = []
    Ks = list(range(k_min, k_max+1))
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_) if k > 1 else float("nan"))

    # Plot Elbow og Silhouette
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(Ks, inertias, 'o-')
    plt.title("Elbow: inertia vs k")
    plt.xlabel("k")
    plt.ylabel("inertia")

    plt.subplot(1,2,2)
    plt.plot(Ks, silhouettes, 'o-')
    plt.title("Silhouette score vs k")
    plt.xlabel("k")
    plt.ylabel("silhouette")

    plt.tight_layout()
    plt.show()

    return Ks, inertias, silhouettes


def main():
    # 1) Indlæs data
    df = pd.read_csv("../UsableBots/CHATBOT_Ulti_test_2/online_gaming_behavior_dataset.csv")

    # 2) Find anbefalet k
    print("▶️ Tester k fra 2–10 med Elbow og Silhouette …")
    Ks, inertias, silhouettes = find_best_k(df)

    # Manuelt vælger vi f.eks. k=6 ud fra plottene
    best_k = 6
    print(f"\n👉 Vælger k={best_k} for flere nuancer.")

    # 3) Træn endelig model
    profiler = GamerProfiler(n_clusters=best_k)
    profiler.fit(df)
    profiler.save("gamer_profile_model_k6.pkl")
    print("✅ Gemt model med k=6 som gamer_profile_model_k6.pkl")

    # 4) Label clusters
    df_clean = df.drop(columns=["PlayerID","Age"], errors="ignore")
    pipeline = profiler.pipeline
    X_trans = pipeline.named_steps["preprocessor"].transform(df_clean)
    labels = pipeline.named_steps["clusterer"].predict(X_trans)

    df_labeled = df_clean.copy()
    df_labeled["cluster"] = labels

    # 5) Definér navne til de 6 clusters
    cluster_names = {
        0: "Hardcore PC Strategist",
        1: "Casual Mobile Puzzler",
        2: "Competitive Shooter Fan",
        3: "Hardcore RPG Explorer",
        4: "Casual PC Gamer",
        5: "Social Co-op Enthusiast"
    }
    df_labeled["cluster_name"] = df_labeled["cluster"].map(cluster_names)

    def describe_clusters(df_labeled, numerical_cols, categorical_cols):
        cluster_descriptions = {}

        for cluster_id in sorted(df_labeled["cluster"].unique()):
            cluster_df = df_labeled[df_labeled["cluster"] == cluster_id]
            description = []

            # Gennemsnit af numeriske værdier
            for col in numerical_cols:
                avg = cluster_df[col].mean()
                if "Time" in col or "Duration" in col:
                    description.append(f"{int(avg)} min/session")
                elif "Level" in col:
                    description.append(f"Lvl {int(avg)} avg")
                elif "Achievements" in col:
                    description.append(f"{int(avg)} achievements")
                elif "Sessions" in col:
                    description.append(f"{int(avg)} sessions/week")
                elif "Purchases" in col:
                    description.append("often buys items" if avg > 0.5 else "rarely buys items")
                else:
                    description.append(f"{col}: {avg:.1f}")

            # Hyppigste kategori
            for col in categorical_cols:
                if col in cluster_df.columns:
                    top = cluster_df[col].mode()[0]
                    description.append(f"Prefers {top.lower()} games")

            # Samlet
            cluster_descriptions[cluster_id] = " | ".join(description)

        return cluster_descriptions

    df_labeled.to_csv("labeled_profiles_k6.csv", index=False)
    print("✅ Gemt labeled_profiles_k6.csv med cluster-navne")

    # 5b) Beskriv hver cluster med data
    print("\n📊 Cluster-beskrivelser:") # Beskriver hvordan clusters grupperes
    descriptions = describe_clusters(   # Dette er baseret på gennemsnit
        df_labeled,                     # Bruges til at forstå tendenser og typiske træk
        numerical_cols=profiler.numerical_cols,     # Fortæller os 'bare' at det typiske medlem af
        categorical_cols=profiler.categorical_cols  # "Hardcore PC Strategist" klyngen typisk
    )                                               # Spiller lange sessioner
    for cid, desc in descriptions.items():          # Men betyder ikke direkte 10 spil sessioner hver uge = hardcore
        print(f"Cluster {cid}: {cluster_names.get(cid, 'Unknown')} → {desc}")

    # 6) Visualiser med PCA - både 2D og 3D

    # --- 2D PCA ---
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_trans)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels, cmap="tab10", s=50)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.title("k=6 KMeans clusters (2D PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

    # --- 3D PCA ---
    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X_trans)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=labels, cmap='tab10', s=60)

    ax.set_title("k=6 KMeans clusters (3D PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
