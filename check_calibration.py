
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier

# Load the training dataset
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

# Use a match-based split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Train a fresh model on the training matches
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate calibration points
actual_frequency, predicted_probability = calibration_curve(
    y_test,
    probabilities,
    n_bins=10,
    strategy="uniform"
)

# Plot calibration curve
plt.figure(figsize=(7, 5))
plt.plot(
    predicted_probability,
    actual_frequency,
    marker="o",
    label="Random Forest"
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

plt.xlabel("Mean predicted win probability")
plt.ylabel("Actual win frequency")
plt.title("Cricket Win Probability Calibration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("calibration_curve.png")
plt.show()

print("Calibration graph saved as calibration_curve.png")
