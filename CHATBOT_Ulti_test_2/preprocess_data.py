import pandas as pd

# 1. Indlæs den oprindelige CSV-fil
df = pd.read_csv("online_gaming_behavior_dataset.csv")

# 2. Lav en tekstbeskrivelse af hver spiller
def row_to_text(row):
    return (
        f"Player is {row['Age']} years old, {row['Gender']}, from {row['Location']}. "
        f"They play {row['GameGenre']} games, {row['PlayTimeHours']} hours/week, "
        f"difficulty {row['GameDifficulty']}, {row['SessionsPerWeek']} sessions/week, "
        f"{row['AvgSessionDurationMinutes']} minutes/session, level {row['PlayerLevel']}, "
        f"{row['AchievementsUnlocked']} achievements unlocked."
    )

# 3. Lav input/output kolonner
df["formatted_input"] = df.apply(row_to_text, axis=1)
df["formatted_output"] = df["EngagementLevel"].apply(lambda x: f"Engagement level: {x}")

# 4. Gem det som ny fil
df[["formatted_input", "formatted_output"]].to_csv("formatted_data.csv", index=False)

print("✅ Formatteret data gemt som 'formatted_data.csv'")
