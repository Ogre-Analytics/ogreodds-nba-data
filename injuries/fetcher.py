"""
NBA injury report via ESPN's public API (no key required).

Fetches current injury status for all 30 teams in one request.
Results cached for 30 minutes — re-run to get fresh data on game days.
"""

import json
import time
import requests
from pathlib import Path

ESPN_URL   = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
CACHE_FILE = Path(__file__).parent.parent / "data" / "cache" / "injuries_espn.json"
CACHE_TTL  = 30 * 60   # 30 minutes

# ESPN uses slightly different abbreviations from the NBA API for some teams
ESPN_TO_NBA = {
    "GS":   "GSW",
    "NY":   "NYK",
    "SA":   "SAS",
    "NO":   "NOP",
    "UTAH": "UTA",
    "WSH":  "WAS",
}

# Display label per ESPN status string
STATUS_LABEL = {
    "out":          "OUT",
    "day-to-day":   "D2D",
    "doubtful":     "DBT",
    "questionable": "QST",
}


def fetch_nba_injuries(use_cache: bool = True) -> dict:
    """
    Return current injury report keyed by NBA team abbreviation.

    Structure:
      {
        'OKC': [
          {
            'player':  'Shai Gilgeous-Alexander',
            'pos':     'G',
            'status':  'Out',
            'label':   'OUT',    # short display tag
            'comment': 'Abdominal issue, out at least one more week',
          },
          ...
        ],
        ...
      }

    Returns an empty dict if the fetch fails (non-fatal — displayed as a warning).
    """
    if use_cache and CACHE_FILE.exists():
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_TTL:
            print("  [cache] Loaded injury report from cache")
            with open(CACHE_FILE) as f:
                return json.load(f)

    print("  [api]   Fetching injury report from ESPN ...")
    try:
        resp = requests.get(
            ESPN_URL,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        print(f"  [warn]  Injury fetch failed: {exc}")
        return {}

    out: dict = {}
    for team_block in raw.get("injuries", []):
        for inj in team_block.get("injuries", []):
            ath    = inj.get("athlete", {})
            abbr   = ath.get("team", {}).get("abbreviation", "")
            abbr   = ESPN_TO_NBA.get(abbr, abbr)   # normalise to NBA abbrev
            if not abbr:
                continue

            status_raw = inj.get("status", "")
            label      = STATUS_LABEL.get(status_raw.lower(), status_raw.upper()[:3])

            out.setdefault(abbr, []).append({
                "player":  ath.get("displayName", "Unknown"),
                "pos":     ath.get("position", {}).get("abbreviation", ""),
                "status":  status_raw,
                "label":   label,
                "comment": inj.get("shortComment", ""),
            })

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(out, f)

    total = sum(len(v) for v in out.values())
    print(f"  [inj]   {len(out)} teams, {total} players on injury report")
    return out


def summarise_game_injuries(
    home_abbrev: str,
    away_abbrev: str,
    injuries: dict,
    statuses: tuple = ("Out", "Doubtful"),
) -> dict:
    """
    Return injury entries for both teams filtered to the given statuses.

    Default: only 'Out' and 'Doubtful' — Day-To-Day / Questionable are excluded
    because their game-time status is too uncertain to adjust probabilities.

    Returns:
      {
        'home': [...injury dicts...],
        'away': [...injury dicts...],
        'home_out_count': int,   # OUT players only
        'away_out_count': int,
      }
    """
    def _filter(abbrev: str) -> list:
        return [
            inj for inj in injuries.get(abbrev, [])
            if inj["status"] in statuses
        ]

    home_inj = _filter(home_abbrev)
    away_inj = _filter(away_abbrev)

    return {
        "home": home_inj,
        "away": away_inj,
        "home_out_count": sum(1 for i in home_inj if i["status"] == "Out"),
        "away_out_count": sum(1 for i in away_inj if i["status"] == "Out"),
    }
