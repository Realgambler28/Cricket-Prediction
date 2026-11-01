
import json

file_path = "ipl_json/1082591.json"

with open(file_path, "r", encoding="utf-8") as file:
    match = json.load(file)

for innings in match["innings"]:
    for over in innings["overs"]:
        for delivery in over["deliveries"]:
            if "wickets" in delivery:
                print("Batting team:", innings["team"])
                print("Over:", over["over"])
                print("Delivery:", delivery["actual_delivery"])
                print("Wicket details:", delivery["wickets"])
                print("-" * 40)
