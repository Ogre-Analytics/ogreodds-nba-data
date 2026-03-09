"""
NBA Predictions — Public Export
================================
Runs the win probability model and writes predictions.json for the website.
No odds API key required.

Usage:
    python predict.py                      # output predictions.json
    python predict.py --retrain            # force model retraining first
    python predict.py --out custom.json    # custom output path
"""

import json
import math
import argparse
from datetime import date, datetime, timezone, timedelta

import pandas as pd

from data.fetcher import (
    get_all_teams,
    get_season_games,
    get_historical_games,
    get_team_dashboard,
    get_todays_scoreboard,
    get_player_stats,
    CURRENT_SEASON,
)
from injuries.fetcher import fetch_nba_injuries, summarise_game_injuries
from model.predictor import get_or_train_model, predict_game

# ── Timezone config ────────────────────────────────────────────────────────────
# Use UTC offset so we always get the correct US Eastern time regardless of
# whether the host machine observes daylight saving time.
# EDT (Mar 2nd Sun -> Nov 1st Sun) = UTC-4 ; EST (remainder) = UTC-5
def _get_et_offset() -> timezone:
    """Return the current US Eastern timezone offset (auto EDT/EST)."""
    now_utc = datetime.now(timezone.utc)
    # DST in US Eastern: second Sunday of March through first Sunday of November
    year = now_utc.year
    # Second Sunday of March
    march_1  = datetime(year, 3, 1, tzinfo=timezone.utc)
    edt_start = march_1 + timedelta(days=(6 - march_1.weekday()) % 7 + 7)
    edt_start = edt_start.replace(hour=7)   # 2am ET = 7am UTC (during EST)
    # First Sunday of November
    nov_1    = datetime(year, 11, 1, tzinfo=timezone.utc)
    edt_end  = nov_1 + timedelta(days=(6 - nov_1.weekday()) % 7)
    edt_end  = edt_end.replace(hour=6)      # 2am ET = 6am UTC (during EDT)
    if edt_start <= now_utc < edt_end:
        return timezone(timedelta(hours=-4))  # EDT
    return timezone(timedelta(hours=-5))      # EST

_AEDT = timezone(timedelta(hours=11))

# ── Injury adjustment constants (same as main.py) ─────────────────────────────
INJURY_LOGIT_COEF     = 0.025
MAX_INJURY_PROB_SHIFT = 0.20
MIN_MINUTES_TO_COUNT  = 11.0
MIN_GAMES_TO_COUNT    = 10


def get_nba_game_date() -> str:
    _ET = _get_et_offset()
    now_et = datetime.now(_ET)
    if now_et.hour < 5:
        return (now_et.date() - timedelta(days=1)).strftime("%Y-%m-%d")
    return now_et.date().strftime("%Y-%m-%d")


def et_to_aedt(time_str: str) -> str:
    import re
    m = re.match(r'(\d+):(\d+)\s*(am|pm)\s*ET', time_str.strip(), re.IGNORECASE)
    if not m:
        return time_str
    h, mn, ampm = int(m.group(1)), int(m.group(2)), m.group(3).lower()
    if ampm == 'pm' and h != 12:
        h += 12
    elif ampm == 'am' and h == 12:
        h = 0
    # ET to AEDT offset: EDT(UTC-4) -> AEDT(UTC+11) = +15h; EST(UTC-5) -> AEDT = +16h
    et_offset_hours = int(_get_et_offset().utcoffset(None).total_seconds() // 3600)  # -4 or -5
    offset_hours = 11 - et_offset_hours   # AEDT(+11) minus ET offset
    total = (h * 60 + mn + offset_hours * 60) % 1440
    h, mn = total // 60, total % 60
    if h == 0:
        return f"12:{mn:02d} am AEDT"
    elif h < 12:
        return f"{h}:{mn:02d} am AEDT"
    elif h == 12:
        return f"12:{mn:02d} pm AEDT"
    else:
        return f"{h - 12}:{mn:02d} pm AEDT"


def _player_game_score(row) -> float:
    return (
          float(row.get("PTS",  0))
        + 0.4 * float(row.get("FGM",  0))
        - 0.7 * float(row.get("FGA",  0))
        - 0.4 * (float(row.get("FTA", 0)) - float(row.get("FTM", 0)))
        + 0.7 * float(row.get("OREB", 0))
        + 0.3 * float(row.get("DREB", 0))
        + float(row.get("STL",  0))
        + 0.7 * float(row.get("AST",  0))
        + 0.7 * float(row.get("BLK",  0))
        - 0.4 * float(row.get("PF",   0))
        - float(row.get("TOV",  0))
    )


def _calc_injury_impact(abbrev: str, inj_list: list, player_df) -> float:
    if player_df is None or player_df.empty or not inj_list:
        return 0.0
    out_names = [i["player"].lower() for i in inj_list if i["status"] == "Out"]
    if not out_names:
        return 0.0
    team_df = player_df[player_df["TEAM_ABBREVIATION"] == abbrev]
    total = 0.0
    for name in out_names:
        match = team_df[team_df["PLAYER_NAME"].str.lower() == name]
        if match.empty:
            continue
        row    = match.iloc[0]
        min_pg = float(row.get("MIN", 0))
        gp     = int(row.get("GP", 0))
        if min_pg < MIN_MINUTES_TO_COUNT or gp < MIN_GAMES_TO_COUNT:
            continue
        gs = _player_game_score(row)
        if gs <= 0:
            continue
        total += (min_pg / 36.0) * gs
    return round(total, 2)


def _apply_injury_adjustment(base_home_prob, home_abbrev, away_abbrev,
                              home_inj, away_inj, player_df):
    home_impact = _calc_injury_impact(home_abbrev, home_inj, player_df)
    away_impact = _calc_injury_impact(away_abbrev, away_inj, player_df)
    net_impact  = home_impact - away_impact

    if net_impact == 0.0 or not (0.0 < base_home_prob < 1.0):
        return base_home_prob, home_impact, away_impact

    logit_adj = math.log(base_home_prob / (1 - base_home_prob)) - INJURY_LOGIT_COEF * net_impact
    adj = 1.0 / (1.0 + math.exp(-logit_adj))
    adj = max(base_home_prob - MAX_INJURY_PROB_SHIFT,
              min(base_home_prob + MAX_INJURY_PROB_SHIFT, adj))
    return round(adj, 4), home_impact, away_impact


# ── Main ──────────────────────────────────────────────────────────────────────

def run(retrain: bool = False, out_path: str = "predictions.json") -> None:
    today    = get_nba_game_date()
    now_utc  = datetime.now(timezone.utc)
    now_aedt = datetime.now(_AEDT)

    print(f"NBA Predictions  |  {now_aedt.strftime('%a %d %b %Y  %I:%M %p AEDT')}  |  Season {CURRENT_SEASON}")
    print(f"Game date (US ET): {today}")

    # ── Season data ───────────────────────────────────────────────────────────
    print("\nLoading season data ...")
    teams_df      = get_all_teams()
    games_df      = get_season_games(season=CURRENT_SEASON)
    historical_df = get_historical_games()
    get_team_dashboard(season=CURRENT_SEASON)   # warms cache

    team_abbrev_map = teams_df.set_index("id")["abbreviation"].to_dict()
    team_name_map   = teams_df.set_index("id")["full_name"].to_dict()

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model ...")
    pipe = get_or_train_model(games_df, force_retrain=retrain, historical_df=historical_df)

    # ── Today's schedule ──────────────────────────────────────────────────────
    print("Fetching schedule ...")
    model_date = today
    board  = get_todays_scoreboard(today)
    header = board["game_header"].drop_duplicates(subset=["GAME_ID"]).reset_index(drop=True)

    all_final = header.empty or all(
        str(r.get("GAME_STATUS_TEXT", "")).strip() == "Final"
        for _, r in header.iterrows()
    )
    if all_final:
        next_et = (date.fromisoformat(today) + timedelta(days=1)).strftime("%Y-%m-%d")
        next_board  = get_todays_scoreboard(next_et)
        next_header = next_board["game_header"].drop_duplicates(subset=["GAME_ID"]).reset_index(drop=True)
        if not next_header.empty:
            today  = next_et
            header = next_header
        elif header.empty:
            print("No games today or tomorrow.")
            _write_empty(out_path, today, now_utc)
            return

    if header.empty:
        print(f"No games on {today}.")
        _write_empty(out_path, today, now_utc)
        return

    print(f"Found {len(header)} game(s) on {today}")

    # ── Injuries ──────────────────────────────────────────────────────────────
    print("Fetching injury report ...")
    injury_report   = fetch_nba_injuries()
    player_stats_df = get_player_stats(season=CURRENT_SEASON)

    # ── Predictions ───────────────────────────────────────────────────────────
    games_out = []

    for _, row in header.iterrows():
        home_id   = row["HOME_TEAM_ID"]
        away_id   = row["VISITOR_TEAM_ID"]
        tip_et    = str(row.get("GAME_STATUS_TEXT", "")).strip()
        tip_aedt  = et_to_aedt(tip_et)

        home_abbrev = team_abbrev_map.get(home_id, str(home_id))
        away_abbrev = team_abbrev_map.get(away_id, str(away_id))
        home_name   = team_name_map.get(home_id, home_abbrev)
        away_name   = team_name_map.get(away_id, away_abbrev)

        pred = predict_game(
            pipe, games_df,
            home_team_id=home_id,
            away_team_id=away_id,
            game_date=model_date,
        )

        base_prob = pred["home_win_prob"]

        # Apply injury adjustment
        game_inj = summarise_game_injuries(home_abbrev, away_abbrev, injury_report)
        adj_home_prob, home_inj_impact, away_inj_impact = _apply_injury_adjustment(
            base_home_prob=base_prob,
            home_abbrev=home_abbrev,
            away_abbrev=away_abbrev,
            home_inj=game_inj["home"],
            away_inj=game_inj["away"],
            player_df=player_stats_df,
        )
        adj_away_prob = round(1.0 - adj_home_prob, 4)

        # Round to nearest integer % for display
        home_pct = round(adj_home_prob * 100)
        away_pct = 100 - home_pct

        # game_status: 1=pre-game, 2=in-progress, 3=final
        game_status = int(row.get("GAME_STATUS_ID", 1))
        status_label = {1: "scheduled", 2: "live", 3: "final"}.get(game_status, "scheduled")

        games_out.append({
            "home_team":       home_name,
            "away_team":       away_name,
            "home_abbrev":     home_abbrev,
            "away_abbrev":     away_abbrev,
            "home_win_pct":    home_pct,
            "away_win_pct":    away_pct,
            "tip_time_et":     tip_et,
            "tip_time_aedt":   tip_aedt,
            "game_status":     status_label,
            "injury_adjusted": (home_inj_impact > 0 or away_inj_impact > 0),
        })

        # Print in the simple website format
        print(f"  {home_name} ({home_pct}%) vs {away_name} ({away_pct}%)  |  {tip_aedt}")

    # ── Write JSON ────────────────────────────────────────────────────────────
    output = {
        "generated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "game_date":    today,
        "games":        games_out,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(games_out)} prediction(s) to {out_path}")


def _write_empty(out_path: str, game_date: str, now_utc) -> None:
    output = {
        "generated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "game_date":    game_date,
        "games":        [],
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote empty predictions to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Predictions Export")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--out",     default="predictions.json", help="Output JSON file path")
    args = parser.parse_args()
    try:
        run(retrain=args.retrain, out_path=args.out)
    except Exception as exc:
        import traceback
        print(f"[ERROR] predict.py failed: {exc}")
        traceback.print_exc()
        # Write an error state so the website knows something went wrong
        # rather than silently serving stale data
        now_utc = datetime.now(timezone.utc)
        error_payload = {
            "generated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "error": str(exc),
            "games": [],
        }
        try:
            with open(args.out, "w") as f:
                json.dump(error_payload, f, indent=2)
            print(f"Wrote error state to {args.out}")
        except Exception:
            pass
