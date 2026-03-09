"""
Verification script — confirms the NBA API is returning correct 2025-26 data.
Run this first to make sure everything is connected before proceeding.

Usage:
    python verify.py
"""

import sys
import traceback
import pandas as pd
from datetime import date

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "  [PASS]"
FAIL = "  [FAIL]"
INFO = "  [INFO]"

errors = []

def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}" + (f": {detail}" if detail else ""))
        errors.append(label)

# ── 1. imports ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  NBA Betting Tool — API Verification")
print(f"  Season: 2025-26   |   Date: {date.today()}")
print("=" * 60 + "\n")

print("[ 1 ] Importing modules ...")
try:
    from data.fetcher import (
        get_all_teams,
        get_season_games,
        get_team_dashboard,
        get_team_last_n_games,
        get_todays_scoreboard,
        CURRENT_SEASON,
    )
    from features.ratings import (
        build_team_season_stats,
        build_recent_form,
    )
    print(f"{PASS} All modules imported")
except Exception as e:
    print(f"{FAIL} Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 2. Teams ──────────────────────────────────────────────────────────────────

print("\n[ 2 ] Teams ...")
teams = get_all_teams()
check("30 teams returned", len(teams) == 30, f"got {len(teams)}")
check("Expected columns present", all(c in teams.columns for c in ["id", "full_name", "abbreviation"]))
print(f"{INFO} Sample: {', '.join(teams['abbreviation'].head(8).tolist())} ...")

# ── 3. Season game log ────────────────────────────────────────────────────────

print("\n[ 3 ] Season game log (2025-26) ...")
games = get_season_games(season=CURRENT_SEASON)
check("Games DataFrame not empty", not games.empty, "no rows returned")

if not games.empty:
    required_cols = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION",
                     "MATCHUP", "WL", "PTS", "PLUS_MINUS"]
    check("Required columns present",
          all(c in games.columns for c in required_cols),
          f"missing: {[c for c in required_cols if c not in games.columns]}")

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    min_date = games["GAME_DATE"].min().strftime("%Y-%m-%d")
    max_date = games["GAME_DATE"].max().strftime("%Y-%m-%d")
    n_games = games["GAME_ID"].nunique()
    n_rows = len(games)

    check("Season starts in Oct 2025", min_date >= "2025-10-01",
          f"earliest date: {min_date}")
    check("At least 200 unique games played", n_games >= 200,
          f"only {n_games} unique games found (season may be early)")

    print(f"{INFO} Unique games: {n_games}  |  Rows: {n_rows}  |  Range: {min_date} to {max_date}")

    # Sample a few rows
    sample = games[["GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP", "WL", "PTS", "PLUS_MINUS"]].head(5)
    print(f"\n{INFO} Sample rows:")
    print(sample.to_string(index=False))

# ── 4. Team dashboard ─────────────────────────────────────────────────────────

print("\n[ 4 ] Team dashboard stats ...")
dashboard = get_team_dashboard(season=CURRENT_SEASON)
check("Dashboard not empty", not dashboard.empty)
if not dashboard.empty:
    check("30 teams in dashboard", len(dashboard) == 30, f"got {len(dashboard)}")
    check("WIN_PCT column present", "W_PCT" in dashboard.columns)
    top5 = dashboard.sort_values("W_PCT", ascending=False)[["TEAM_NAME", "W_PCT", "PTS", "W", "L"]].head(5)
    print(f"\n{INFO} Top 5 teams by win %:")
    print(top5.to_string(index=False))

# ── 5. Season stats & form ────────────────────────────────────────────────────

print("\n[ 5 ] Derived ratings (from game log) ...")
if not games.empty:
    season_stats = build_team_season_stats(games)
    check("Season stats built", not season_stats.empty)
    check("30 teams in season stats", len(season_stats) == 30, f"got {len(season_stats)}")

    recent_form = build_recent_form(games, n=10)
    check("Recent form built", not recent_form.empty)

    print(f"\n{INFO} Top 5 teams by season win%:")
    cols = ["RANK", "TEAM_ABBREV", "GP", "W", "L", "WIN_PCT", "POINT_DIFF"]
    top5_stats = season_stats[cols].head(5)
    print(top5_stats.to_string(index=False))

    print(f"\n{INFO} Top 5 teams by last-10-game form:")
    form_cols = ["TEAM_ABBREV", "LAST_N_GP", "LAST_N_WIN_PCT", "LAST_N_POINT_DIFF", "STREAK"]
    print(recent_form[form_cols].head(5).to_string(index=False))

# ── 6. Today's scoreboard ─────────────────────────────────────────────────────

print("\n[ 6 ] Today's scoreboard ...")
try:
    board = get_todays_scoreboard()
    header = board["game_header"]
    check("Scoreboard fetched", True)
    if header.empty:
        print(f"{INFO} No games scheduled today ({date.today()}) — check back on a game day")
    else:
        print(f"{INFO} Games today: {len(header)}")
        if "HOME_TEAM_ID" in header.columns and "VISITOR_TEAM_ID" in header.columns:
            # Map team IDs to abbreviations
            team_map = teams.set_index("id")["abbreviation"].to_dict()
            for _, row in header.iterrows():
                home = team_map.get(row.get("HOME_TEAM_ID", 0), "???")
                away = team_map.get(row.get("VISITOR_TEAM_ID", 0), "???")
                status = row.get("GAME_STATUS_TEXT", "")
                print(f"    {away} @ {home}  ({status})")
except Exception as e:
    print(f"{FAIL} Scoreboard error: {e}")
    errors.append("scoreboard")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
if errors:
    print(f"  RESULT: {len(errors)} check(s) FAILED: {errors}")
    print("=" * 60 + "\n")
    sys.exit(1)
else:
    print("  RESULT: All checks PASSED — API connection verified OK")
    print("=" * 60 + "\n")
