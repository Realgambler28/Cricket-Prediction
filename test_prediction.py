
import joblib
import pandas as pd

# Load the saved model
saved = joblib.load("cricket_win_model.joblib")
model = saved["model"]
features = saved["features"]

# Example: a team chasing 180
match_state = {
    "score": 120,
    "wickets_lost": 3,
    "legal_balls": 90,
    "balls_remaining": 30,
    "target": 180,
    "runs_remaining": 60,
    "current_run_rate": 8.0,
    "required_run_rate": 12.0
}

# Keep feature order consistent with training
input_data = pd.DataFrame([match_state])[features]

# Predict probability
probability = model.predict_proba(input_data)[0][1]

print("Match situation: 120 runs, 3 wickets lost")
print("Target: 180")
print(f"Estimated win probability: {probability:.1%}")
