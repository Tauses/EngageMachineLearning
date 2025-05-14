import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Sample data
data = {
    'PlayerID': [9000, 9001, 9002],
    'Age': [43, 29, 22],
    'Gender': ['Male', 'Female', 'Female'],
    'Location': ['Other', 'USA', 'USA'],
    'GameGenre': ['Strategy', 'Strategy', 'Sports'],
    'PlayTimeHours': [16.27, 5.53, 8.22],
    'InGamePurchases': [0, 0, 0],
    'GameDifficulty': ['Medium', 'Medium', 'Easy'],
    'SessionsPerWeek': [6, 5, 16],
    'AvgSessionDurationMinutes': [108, 144, 142],
    'PlayerLevel': [79, 11, 35],
    'AchievementsUnlocked': [25, 10, 41],
    'EngagementLevel': ['Medium', 'Medium', 'High']
}

df = pd.DataFrame(data)

# Preprocessing
label_encoder = LabelEncoder()

# Encoding categorical features
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Location'] = label_encoder.fit_transform(df['Location'])
df['GameGenre'] = label_encoder.fit_transform(df['GameGenre'])
df['GameDifficulty'] = label_encoder.fit_transform(df['GameDifficulty'])
df['EngagementLevel'] = label_encoder.fit_transform(df['EngagementLevel'])

# Features and target variable
X = df.drop(columns=['PlayerID', 'EngagementLevel'])
y = df['EngagementLevel']

# Standardize numerical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Model Evaluation Report:\n", classification_report(y_test, y_pred))
