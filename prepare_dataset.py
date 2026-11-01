
import os
import json
import pandas as pd

DATASET_FOLDER = "ipl_json"
rows = []

for filename in os.listdir(DATASET_FOLDER):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(DATASET_FOLDER, filename)

    with open(path, "r", encoding="utf-8") as file:
        match = json.load(file)

    innings_list = match.get("innings", [])
    outcome = match.get("info", {}).get("outcome", {})
    winner = outcome.get("winner")

    # We need two-innings matches with a known winner
    if len(innings_list) != 2 or not winner:
        continue

    first_innings = innings_list[0]
    second_innings = innings_list[1]

    target = sum(
        delivery["runs"]["total"]
        for over in first_innings["overs"]
        for delivery in over["deliveries"]
    ) + 1

    batting_team = second_innings["team"]
    score = 0
    wickets = 0
    legal_balls = 0
    max_balls = 120

    for over in second_innings["overs"]:
        for delivery in over["deliveries"]:
            score += delivery["runs"]["total"]

            for wicket in delivery.get("wickets", []):
                # Retired hurt and retired out need special treatment.
                if wicket["kind"] not in ["retired hurt"]:
                    wickets += 1

            extras = delivery.get("extras", {})
            if "wides" not in extras and "noballs" not in extras:
                legal_balls += 1

            runs_remaining = max(target - score, 0)
            balls_remaining = max(max_balls - legal_balls, 0)

            # A chasing side has won once it reaches the target.
            won = int(winner == batting_team)

            rows.append({
                "match_id": filename,
                "score": score,
                "wickets_lost": wickets,
                "legal_balls": legal_balls,
                "balls_remaining": balls_remaining,
                "target": target,
                "runs_remaining": runs_remaining,
                "current_run_rate": (
                    score / legal_balls * 6 if legal_balls > 0 else 0
                ),
                "required_run_rate": (
                    runs_remaining / balls_remaining * 6
                    if balls_remaining > 0 else 0
                ),
                "won": won
            })

            # Stop once the chase has ended.
            if score >= target or legal_balls >= max_balls or wickets >= 10:
                break

        if score >= target or legal_balls >= max_balls or wickets >= 10:
            break

df = pd.DataFrame(rows)
df.to_csv("training_data.csv", index=False)

print("Training rows created:", len(df))
print("Unique matches:", df["match_id"].nunique())
print("\nSample rows:")
print(df.head())
print("\nWin label counts:")
print(df["won"].value_counts())
