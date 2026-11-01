
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "ipl_json"


def load_match(match_path):
    """Load one Cricsheet JSON match."""
    with open(match_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_team_names(match):
    """Return the teams listed in a match."""
    return match.get("info", {}).get("teams", [])


def get_match_date(match):
    """Return the first recorded match date."""
    dates = match.get("meta", {}).get("data_version")
    match_dates = match.get("info", {}).get("dates", [])
    return match_dates[0] if match_dates else "Unknown date"


def get_innings_summary(match):
    """Summarize the runs and wickets in every innings."""
    summaries = []

    for innings_number, innings in enumerate(match.get("innings", []), start=1):
        team = innings.get("team", "Unknown team")
        total_runs = 0
        wickets = 0

        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                total_runs += delivery.get("runs", {}).get("total", 0)

                for wicket in delivery.get("wickets", []):
                    if wicket.get("kind") != "retired hurt":
                        wickets += 1

        summaries.append({
            "innings": innings_number,
            "team": team,
            "runs": total_runs,
            "wickets": wickets,
        })

    return summaries


def get_over_details(match, innings_number, over_number):
    """
    Return the actual runs and delivery details for an over.
    Cricsheet over numbers are zero-based in JSON, so the first over
    is stored as over 0.
    """
    innings_list = match.get("innings", [])

    if innings_number < 1 or innings_number > len(innings_list):
        return None

    innings = innings_list[innings_number - 1]
    overs = innings.get("overs", [])

    if over_number < 1 or over_number > len(overs):
        return None

    over_data = overs[over_number - 1]
    deliveries = over_data.get("deliveries", [])
    

    total_runs = sum(
        delivery.get("runs", {}).get("total", 0)
        for delivery in deliveries
    )

    batter_runs = sum(
        delivery.get("runs", {}).get("batter", 0)
        for delivery in deliveries
    )

    extras = sum(
        delivery.get("runs", {}).get("extras", 0)
        for delivery in deliveries
    )

    wickets = sum(
        1
        for delivery in deliveries
        for wicket in delivery.get("wickets", [])
        if wicket.get("kind") != "retired hurt"
    )

    return {
        "over": over_number,
        "team": innings.get("team", "Unknown team"),
        "innings": innings_number,
        "total_runs": total_runs,
        "batter_runs": batter_runs,
        "extras": extras,
        "wickets": wickets,
        "deliveries": deliveries,
    }


def search_matches(team_name, max_results=20):
    """Find historical matches containing a team name."""
    results = []
    query = team_name.strip().casefold()

    if not query or not DATA_DIR.exists():
        return results

    for path in DATA_DIR.glob("*.json"):
        try:
            match = load_match(path)
            teams = get_team_names(match)

            if any(query in team.casefold() for team in teams):
                results.append({
                    "match_id": path.stem,
                    "file": path,
                    "teams": teams,
                    "date": get_match_date(match),
                    "summaries": get_innings_summary(match),
                })

        except (OSError, json.JSONDecodeError, TypeError):
            continue

    results.sort(key=lambda item: item["date"], reverse=True)
    return results[:max_results]


def get_match_by_id(match_id):
    """Load a match using its JSON filename ID."""
    path = DATA_DIR / f"{match_id}.json"

    if not path.is_file():
        return None

    try:
        return load_match(path)
    except (OSError, json.JSONDecodeError):
        return None
