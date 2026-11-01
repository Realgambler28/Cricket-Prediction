
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# Load prepared dataset
df = pd.read_csv("training_data.csv")

features = [
    "score",
    "wickets_lost",
    "legal_balls",
    "balls_remaining",
    "target",
    "runs_remaining",
    "current_run_rate",
    "required_run_rate"
]

X = df[features]
y = df["won"]
groups = df["match_id"]

# Split by match to prevent data leakage
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("Log loss:", round(log_loss(y_test, probabilities), 4))
print("Brier score:", round(brier_score_loss(y_test, probabilities), 4))

# Save trained model and feature order
joblib.dump(
    {"model": model, "features": features},
    "cricket_win_model.joblib"
)

print("\nModel saved as cricket_win_model.joblib")
