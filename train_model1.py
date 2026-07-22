import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load collected data
df = pd.read_csv("data.csv")

# Features and label
X = df[["left_ratio_x", "right_ratio_x", "left_ratio_y", "right_ratio_y"]]
y = df["label"]

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model
dump(model, "gaze_model.joblib")

print("Model saved as gaze_model.joblib")