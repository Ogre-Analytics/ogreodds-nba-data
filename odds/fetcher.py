"""
Odds fetcher — pulls live NBA moneyline odds from The Odds API.

Free tier: 500 requests/month.  Sign up at https://the-odds-api.com
Set your key in .env:  ODDS_API_KEY=your_key_here

The Odds API returns odds from multiple bookmakers per game.
We extract the best available line for each side.
"""

import os
import time
import json
import requests
from pathlib import Path
from datetime import date, datetime, timezone
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "basketball_nba"
CACHE_DIR     = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Priority bookmaker list — first match per game is used as "best line"
# Edit this list to prefer bookmakers available in your region
BOOKMAKER_PRIORITY = [
    "draftkings", "fanduel", "betmgm", "caesars",
    "pointsbet_us", "williamhill_us", "betonlineag",
    "mybookieag", "bovada",
]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str, max_age_minutes: float = 30) -> dict | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    age_mins = (time.time() - path.stat().st_mtime) / 60
    if age_mins > max_age_minutes:
        return None
    with open(path) as f:
        return json.load(f)


def _save_cache(key: str, data) -> None:
    with open(_cache_path(key), "w") as f:
        json.dump(data, f)


def fetch_nba_odds(use_cache: bool = True) -> list[dict]:
    """
    Fetch current NBA moneyline (h2h) odds from The Odds API.

    Returns a list of game dicts:
      {
        'id': str,
        'home_team': str,       # full team name e.g. "Detroit Pistons"
        'away_team': str,
        'commence_time': str,   # ISO datetime UTC
        'bookmakers': [
          {
            'key': str,
            'title': str,
            'home_odds': int,   # American format e.g. -110
            'away_odds': int,
          }, ...
        ]
      }
    """
    if not ODDS_API_KEY:
        print("  [odds]  No ODDS_API_KEY set in .env — skipping odds fetch")
        return []

    cache_key = f"odds_nba_{date.today()}"
    if use_cache:
        cached = _load_cache(cache_key, max_age_minutes=20)
        if cached:
            print(f"  [cache] Loaded odds from cache ({len(cached)} games)")
            return cached

    print("  [api]   Fetching live NBA odds from The Odds API ...")
    url = f"{ODDS_API_BASE}/sports/{SPORT}/odds/"
    params = {
        "apiKey":      ODDS_API_KEY,
        "regions":     "us",
        "markets":     "h2h",
        "oddsFormat":  "american",
        "dateFormat":  "iso",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
    except requests.RequestException as e:
        print(f"  [odds]  Request failed: {e}")
        return []

    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")
    print(f"  [odds]  API quota: {remaining} remaining / {used} used this month")

    if resp.status_code != 200:
        print(f"  [odds]  API error {resp.status_code}: {resp.text[:200]}")
        return []

    raw_games = resp.json()
    parsed = []

    for game in raw_games:
        bookmaker_lines = []
        for bm in game.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                home_odds = outcomes.get(game["home_team"])
                away_odds = outcomes.get(game["away_team"])
                if home_odds is not None and away_odds is not None:
                    bookmaker_lines.append({
                        "key":        bm["key"],
                        "title":      bm["title"],
                        "home_odds":  int(home_odds),
                        "away_odds":  int(away_odds),
                    })

        if bookmaker_lines:
            parsed.append({
                "id":             game["id"],
                "home_team":      game["home_team"],
                "away_team":      game["away_team"],
                "commence_time":  game["commence_time"],
                "bookmakers":     bookmaker_lines,
            })

    _save_cache(cache_key, parsed)
    print(f"  [odds]  {len(parsed)} games with odds returned")
    return parsed


# ── Name normalisation ────────────────────────────────────────────────────────

# The Odds API uses full team names; NBA API uses abbreviations.
# This map converts Odds API names -> NBA abbreviation.
ODDS_NAME_TO_ABBREV = {
    "Atlanta Hawks":          "ATL",
    "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN",
    "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI",
    "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL",
    "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET",
    "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU",
    "Indiana Pacers":         "IND",
    "LA Clippers":            "LAC",
    "Los Angeles Clippers":   "LAC",
    "Los Angeles Lakers":     "LAL",
    "Memphis Grizzlies":      "MEM",
    "Miami Heat":             "MIA",
    "Milwaukee Bucks":        "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP",
    "New York Knicks":        "NYK",
    "Oklahoma City Thunder":  "OKC",
    "Orlando Magic":          "ORL",
    "Philadelphia 76ers":     "PHI",
    "Phoenix Suns":           "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings":       "SAC",
    "San Antonio Spurs":      "SAS",
    "Toronto Raptors":        "TOR",
    "Utah Jazz":              "UTA",
    "Washington Wizards":     "WAS",
}


def best_line(game_odds: dict) -> dict:
    """
    From a game's bookmaker list, return the single best line:
    - Tries BOOKMAKER_PRIORITY order first
    - Falls back to first available bookmaker
    """
    bms = game_odds["bookmakers"]
    priority_map = {bm["key"]: bm for bm in bms}
    for key in BOOKMAKER_PRIORITY:
        if key in priority_map:
            return priority_map[key]
    return bms[0]


def match_odds_to_games(today_games: list[dict], odds_list: list[dict]) -> dict:
    """
    Match The Odds API games to scoreboard games by team abbreviation.

    Returns a dict keyed by (home_abbrev, away_abbrev) tuples:
      {
        ('DET', 'OKC'): {
          'home_odds': -115,
          'away_odds': -105,
          'bookmaker': 'DraftKings',
          'all_books': [...],
        }, ...
      }
    """
    matched = {}

    for odds_game in odds_list:
        home_abbrev = ODDS_NAME_TO_ABBREV.get(odds_game["home_team"])
        away_abbrev = ODDS_NAME_TO_ABBREV.get(odds_game["away_team"])
        if not home_abbrev or not away_abbrev:
            continue

        key = (home_abbrev, away_abbrev)
        line = best_line(odds_game)
        matched[key] = {
            "home_odds":  line["home_odds"],
            "away_odds":  line["away_odds"],
            "bookmaker":  line["title"],
            "all_books":  odds_game["bookmakers"],
            "commence_time": odds_game["commence_time"],
        }

    return matched
