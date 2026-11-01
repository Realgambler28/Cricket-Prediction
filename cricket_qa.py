import re
from cricket_data import (
    DATA_DIR,
    load_match,
    get_team_names,
    get_match_date,
    get_innings_summary,
    get_over_details,
    search_matches,
)


TEAM_ALIASES = {
    "csk": "Chennai Super Kings",
    "chennai super kings": "Chennai Super Kings",
    "chennai": "Chennai Super Kings",
    "mi": "Mumbai Indians",
    "mumbai indians": "Mumbai Indians",
    "mumbai": "Mumbai Indians",
    "rcb": "Royal Challengers Bangalore",
    "royal challengers bangalore": "Royal Challengers Bangalore",
    "bangalore": "Royal Challengers Bangalore",
    "bengaluru": "Royal Challengers Bangalore",
    "srh": "Sunrisers Hyderabad",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "hyderabad": "Sunrisers Hyderabad",
    "kkr": "Kolkata Knight Riders",
    "kolkata knight riders": "Kolkata Knight Riders",
    "kolkata": "Kolkata Knight Riders",
    "dc": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "delhi": "Delhi Capitals",
    "kxip": "Kings XI Punjab",
    "kings xi punjab": "Kings XI Punjab",
    "pbks": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "punjab": "Punjab Kings",
    "rr": "Rajasthan Royals",
    "rajasthan royals": "Rajasthan Royals",
    "rajasthan": "Rajasthan Royals",
    "gt": "Gujarat Titans",
    "gujarat titans": "Gujarat Titans",
    "gujarat": "Gujarat Titans",
    "lsg": "Lucknow Super Giants",
    "lucknow super giants": "Lucknow Super Giants",
    "lucknow": "Lucknow Super Giants",
}

# Prefer the longest alias at each text position, then return teams in the
# order in which they were mentioned in the question.
ALIASES_BY_LENGTH = sorted(TEAM_ALIASES, key=len, reverse=True)


def find_teams_in_question(question):
    """Return recognized teams in order of appearance in the question."""
    q = question.casefold()
    matches = []

    for alias in ALIASES_BY_LENGTH:
        pattern = r"\b" + re.escape(alias) + r"\b"
        for match in re.finditer(pattern, q):
            matches.append((match.start(), match.end(), TEAM_ALIASES[alias]))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    found = []
    occupied_until = -1
    for start, end, team in matches:
        if start < occupied_until:
            continue
        if team not in found:
            found.append(team)
        occupied_until = end

    return found


def find_target_team(question, teams):
    """
    Try to identify the team whose runs/statistics are requested.
    For 'How many runs did RCB score ... against Gujarat Titans',
    RCB is the target team, not Gujarat Titans.
    """
    q = question.casefold()

    patterns = [
        r"\bdid\s+(.+?)\s+(?:score|make|hit|take|lose|win)\b",
        r"\bhow many runs did\s+(.+?)\s+(?:score|make|hit)\b",
        r"\b(?:score|runs|wickets)\s+of\s+(.+?)(?:\s+in|\s+against|\s+on|[?.!,]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            phrase = match.group(1)
            phrase_teams = find_teams_in_question(phrase)
            if phrase_teams:
                return phrase_teams[0]

    return teams[0] if teams else None


def find_over_number(question):
    """Extract a human-readable over number, e.g. '17th over'."""
    patterns = [
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+over\b",
    r"\bover\s+(\d{1,2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question.casefold())
        if match:
            number = int(match.group(1))
            if 1 <= number <= 20:
                return number
    return None


def find_innings_number(question):
    q = question.casefold()
    if re.search(r"\b(first|1st)\s+innings\b", q):
        return 1
    if re.search(r"\b(second|2nd)\s+innings\b", q):
        return 2
    return None


def find_requested_date(question):
    """Recognize YYYY-MM-DD dates."""
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
    return match.group(1) if match else None


def format_over_facts(match, innings_number, over_number):
    """Format verified facts from one specific match and over."""
    details = get_over_details(match, innings_number, over_number)
    if not details:
        return None

    delivery_lines = []
    for index, delivery in enumerate(details["deliveries"], start=1):
        ball_number = f"{over_number - 1}.{index}"
        runs = delivery.get("runs", {})
        batter_runs = runs.get("batter", 0)
        total_runs = runs.get("total", 0)
        extras = runs.get("extras", 0)
        wickets = delivery.get("wickets", [])

        wicket_text = ", ".join(
            f"{w.get('kind', 'unknown')} ({w.get('player_out', 'unknown')})"
            for w in wickets
        )
        line = (
            f"Ball {ball_number}: total {total_runs} runs, "
            f"batter {batter_runs}, extras {extras}"
        )
        if wicket_text:
            line += f", wicket: {wicket_text}"
        delivery_lines.append(line)

    return (
        f"Team: {details['team']}\n"
        f"Innings: {innings_number}\n"
        f"Over: {over_number}\n"
        f"Runs in over: {details['total_runs']}\n"
        f"Batter runs: {details['batter_runs']}\n"
        f"Extras: {details['extras']}\n"
        f"Wickets in over: {details['wickets']}\n"
        + "\n".join(delivery_lines)
    )


def answer_historical_question(question):
    """
    Retrieve historical IPL facts from the local Cricsheet JSON files.

    Returns:
      {"handled": True, "answer": "..."} for historical-data questions
      {"handled": False, "answer": None} for other questions
    """
    q = question.casefold()
    teams = find_teams_in_question(question)
    target_team = find_target_team(question, teams)
    over_number = find_over_number(question)
    innings_number = find_innings_number(question)
    requested_date = find_requested_date(question)

    historical_keywords = (
        "historical", "previous match", "last match", "latest match",
        "past match", "in the match", "in that match", "scorecard",
        "how many runs", "runs in", "which bowler", "who bowled",
        "who took", "who got out", "wicket", "innings", "over",
        "matches did", "matches played", "show matches",
    )
    looks_historical = (
        any(word in q for word in historical_keywords)
        or over_number is not None
        or "latest matches" in q
    )

    if not looks_historical:
        return {"handled": False, "answer": None}

    if not DATA_DIR.exists():
        return {
            "handled": True,
            "answer": f"Dataset folder not found: {DATA_DIR}",
        }

    # Recent matches for a team.
    recent_phrases = (
        "latest match", "last match", "latest matches",
        "previous match", "show matches", "past matches",
    )
    if teams and any(phrase in q for phrase in recent_phrases):
        team = target_team or teams[0]
        results = search_matches(team, max_results=10)
        if not results:
            return {
                "handled": True,
                "answer": f"No historical matches found for {team}.",
            }

        lines = [f"Historical matches involving {team}:"]
        for item in results:
            summaries = "; ".join(
                f"Innings {s['innings']}: {s['team']} "
                f"{s['runs']}/{s['wickets']}"
                for s in item["summaries"]
            )
            lines.append(
                f"- Match ID {item['match_id']}, date {item['date']}: "
                f"{' vs '.join(item['teams'])}. {summaries}"
            )
        return {"handled": True, "answer": "\n".join(lines)}

    # Historical over question.
    if over_number is not None:
        if not teams:
            return {
                "handled": True,
                "answer": (
                    "Which IPL team are you asking about? For example: "
                    "'How many runs did CSK score in the 17th over?'"
                ),
            }

        team = target_team or teams[0]
        candidates = search_matches(team, max_results=200)

        # If two teams are named, only retain matches containing both.
        if len(teams) > 1:
            candidates = [
                item for item in candidates
                if all(
                    any(
                        candidate_team.casefold() == match_team.casefold()
                        for match_team in item["teams"]
                    )
                    for candidate_team in teams
                )
            ]

        # If a date is supplied, it must match exactly.
        if requested_date:
            candidates = [
                item for item in candidates
                if item["date"] == requested_date
            ]

        matching_overs = []
        for item in candidates:
            match = load_match(item["file"])
            innings_list = match.get("innings", [])

            innings_to_check = (
                [innings_number]
                if innings_number is not None
                else range(1, len(innings_list) + 1)
            )

            for inn_no in innings_to_check:
                if inn_no < 1 or inn_no > len(innings_list):
                    continue

                innings = innings_list[inn_no - 1]
                if innings.get("team") != team:
                    continue

                facts = format_over_facts(match, inn_no, over_number)
                if facts:
                    matching_overs.append({
                        "match_id": item["match_id"],
                        "date": item["date"],
                        "teams": item["teams"],
                        "team": team,
                        "innings": inn_no,
                        "facts": facts,
                    })

        if not matching_overs:
            description = f" for {team}"
            if requested_date:
                description += f" on {requested_date}"
            if len(teams) > 1:
                description += f" against {', '.join(t for t in teams if t != team)}"
            return {
                "handled": True,
                "answer": (
                    f"I couldn't find the {over_number}th over{description} "
                    "in the local dataset. Check the date, team names, and innings."
                ),
            }

        # Avoid choosing arbitrarily if the question still matches multiple games.
        if len(matching_overs) > 1:
            lines = [
                f"I found {len(matching_overs)} matching historical overs "
                f"for {team}. Please specify the opponent, date, or match ID."
            ]
            for item in matching_overs[:10]:
                lines.append(
                    f"- Match ID {item['match_id']}, date {item['date']}, "
                    f"{' vs '.join(item['teams'])}, innings {item['innings']}."
                )
            lines.append(
                "Example: 'How many runs did RCB score in the 17th over "
                "against Gujarat Titans on 2023-05-21?'"
            )
            return {"handled": True, "answer": "\n".join(lines)}

        result = matching_overs[0]
        return {
            "handled": True,
            "answer": (
                f"Verified historical data — Match ID {result['match_id']}, "
                f"date {result['date']}, {' vs '.join(result['teams'])}.\n"
                f"{result['facts']}"
            ),
        }

    # Team scorecard / innings questions.
    if teams and ("scorecard" in q or "innings" in q):
        team = target_team or teams[0]
        results = search_matches(team, max_results=10)
        lines = [
            f"Found historical matches involving {team}. Specify the opponent, "
            "date, or match ID for an exact scorecard."
        ]
        for item in results[:5]:
            summaries = "; ".join(
                f"{s['team']} {s['runs']}/{s['wickets']}"
                for s in item["summaries"]
            )
            lines.append(
                f"- {item['match_id']} ({item['date']}): "
                f"{' vs '.join(item['teams'])}; {summaries}"
            )
        return {"handled": True, "answer": "\n".join(lines)}

    return {
        "handled": True,
        "answer": (
            "I can retrieve historical IPL over details and recent team matches "
            "from the local dataset. Mention a team and, for over statistics, "
            "the over number and preferably the opponent or match date."
        ),
    }
def format_historical_answer(text):
    """Format verified historical cricket facts for readability."""
    if not text:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    match_info = []
    stats = {}
    deliveries = []

    for line in lines:
        if line.startswith("Verified historical data"):
            match_info.append(line)
        elif line.startswith(("Team:", "Innings:", "Over:")):
            key, value = line.split(":", 1)
            stats[key] = value.strip()
        elif line.startswith((
            "Runs in over:", "Batter runs:", "Extras:",
            "Wickets in over:"
        )):
            key, value = line.split(":", 1)
            stats[key] = value.strip()
        elif line.startswith("Ball "):
            deliveries.append(line)

    if "Runs in over" not in stats:
        return text

    result = []

    if match_info:
        result.extend(["### Match Details", *match_info, ""])

    result.append("### Over Summary")
    for key in ("Team", "Innings", "Over", "Runs in over",
                "Batter runs", "Extras", "Wickets in over"):
        if key in stats:
            result.append(f"- **{key}:** {stats[key]}")

    if deliveries:
        result.extend(["", "### Ball-by-Ball Breakdown"])
        for delivery in deliveries:
            result.append(f"- {delivery}")

    return "\n".join(result)