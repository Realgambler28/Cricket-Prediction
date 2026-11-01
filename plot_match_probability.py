import json
import glob
import os

import joblib
import pandas as pd
import plotly.express as px

# Load the trained model
saved = joblib.load("cricket_win_model.joblib")
model = saved["model"]
features = saved["features"]

# Find an IPL JSON match file
match_files = glob.glob(os.path.join("ipl_json", "*.json"))

if not match_files:
    raise FileNotFoundError("No JSON match files found inside the ipl_json folder.")

# Use the first available match
match_path = match_files[0]

with open(match_path, "r", encoding="utf-8") as file:
    match = json.load(file)

innings = match["innings"]

if len(innings) < 2:
    raise ValueError("This match does not contain two innings.")

# Calculate the first-innings total and target
first_innings_score = sum(
    delivery.get("runs", {}).get("total", 0)
    for over in innings[0]["overs"]
    for delivery in over["deliveries"]
)

target = first_innings_score + 1

# Extract the second innings
second_innings = innings[1]
score = 0
wickets_lost = 0
legal_balls = 0
states = []

for over in second_innings["overs"]:
    for delivery in over["deliveries"]:
        score += delivery.get("runs", {}).get("total", 0)

        for wicket in delivery.get("wickets", []):
            if wicket.get("kind") != "retired hurt":
                wickets_lost += 1

        extras = delivery.get("extras", {})
        if "wides" not in extras and "noballs" not in extras:
            legal_balls += 1

        balls_remaining = max(0, 120 - legal_balls)
        runs_remaining = max(0, target - score)

        current_run_rate = (
            score / (legal_balls / 6)
            if legal_balls > 0 else 0
        )

        required_run_rate = (
            runs_remaining / (balls_remaining / 6)
            if balls_remaining > 0 else 0
        )

        state = {
            "score": score,
            "wickets_lost": wickets_lost,
            "legal_balls": legal_balls,
            "balls_remaining": balls_remaining,
            "target": target,
            "runs_remaining": runs_remaining,
            "current_run_rate": current_run_rate,
            "required_run_rate": required_run_rate,
        }

        states.append(state)

        # Stop once the chase is complete or the innings ends
        if score >= target or wickets_lost >= 10 or legal_balls >= 120:
            break

    if score >= target or wickets_lost >= 10 or legal_balls >= 120:
        break

# Predict the chasing team's win probability at each recorded state
df = pd.DataFrame(states)
df["win_probability"] = model.predict_proba(df[features])[:, 1] * 100
df["over"] = df["legal_balls"] / 6

# Plot the probability curve
fig = px.line(
    df,
    x="over",
    y="win_probability",
    title="IPL Match: Chasing Team's Win Probability",
    labels={
        "over": "Overs Completed",
        "win_probability": "Predicted Win Probability (%)",
    },
    hover_data=["score", "wickets_lost", "runs_remaining"],
)

fig.update_yaxes(range=[0, 100])
fig.update_layout(template="plotly_white")
fig.show()

print("Match file:", os.path.basename(match_path))
print("Target:", target)
print("Final chasing score:", score)
print("Plot generated successfully.")