
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV

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

# Split by match
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Split training matches into model-fitting and calibration subsets
cal_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=43
)

fit_idx, cal_idx = next(
    cal_splitter.split(
        X_train,
        y_train,
        groups=groups.iloc[train_idx]
    )
)

X_fit = X_train.iloc[fit_idx]
y_fit = y_train.iloc[fit_idx]
X_cal = X_train.iloc[cal_idx]
y_cal = y_train.iloc[cal_idx]

# Train the base model
base_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

print("Training base model...")
base_model.fit(X_fit, y_fit)

# Calibrate using held-out matches
print("Calibrating probabilities...")
calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(base_model),
    method="sigmoid"
)

calibrated_model.fit(X_cal, y_cal)

# Evaluate on untouched test matches
probabilities = calibrated_model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("\nCalibrated model results")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Log loss:", round(log_loss(y_test, probabilities), 4))
print("Brier score:", round(brier_score_loss(y_test, probabilities), 4))

# Save the calibrated model
joblib.dump(
    {"model": calibrated_model, "features": features},
    "cricket_win_model_calibrated.joblib"
)

print("\nSaved: cricket_win_model_calibrated.joblib")
