"""
Team ratings and recent-form features derived from raw game log data.

Key metrics:
  - Season-to-date win%, points for/against
  - Last-N-games form (win%, avg point differential)
  - Home vs away splits
  - Rest days (back-to-back flag)
"""

import pandas as pd
import numpy as np
from datetime import datetime


def build_team_season_stats(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the raw LeagueGameFinder output (one row per team per game),
    compute season-to-date aggregate stats per team.

    Returns a DataFrame indexed by TEAM_ID with columns:
      GP, W, L, WIN_PCT, PTS_FOR, PTS_AGAINST, POINT_DIFF, HOME_WIN_PCT, AWAY_WIN_PCT
    """
    df = games_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WL_BINARY"] = (df["WL"] == "W").astype(int)

    # Determine home/away from MATCHUP: "BOS vs. MIA" = home, "BOS @ MIA" = away
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs\.")

    results = []
    for team_id, group in df.groupby("TEAM_ID"):
        home = group[group["IS_HOME"]]
        away = group[~group["IS_HOME"]]

        results.append({
            "TEAM_ID": team_id,
            "TEAM_ABBREV": group["TEAM_ABBREVIATION"].iloc[0],
            "TEAM_NAME": group["TEAM_NAME"].iloc[0],
            "GP": len(group),
            "W": group["WL_BINARY"].sum(),
            "L": len(group) - group["WL_BINARY"].sum(),
            "WIN_PCT": group["WL_BINARY"].mean(),
            "PTS_FOR": group["PTS"].mean(),
            "PTS_AGAINST": (group["PTS"] - group["PLUS_MINUS"]).mean(),
            "POINT_DIFF": group["PLUS_MINUS"].mean(),
            "HOME_WIN_PCT": home["WL_BINARY"].mean() if len(home) > 0 else np.nan,
            "AWAY_WIN_PCT": away["WL_BINARY"].mean() if len(away) > 0 else np.nan,
            "HOME_GP": len(home),
            "AWAY_GP": len(away),
        })

    out = pd.DataFrame(results).sort_values("WIN_PCT", ascending=False).reset_index(drop=True)
    out["RANK"] = out.index + 1
    return out


def build_recent_form(games_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Compute each team's form over their last N games.

    Returns a DataFrame with columns:
      TEAM_ID, TEAM_ABBREV, LAST_N_GP, LAST_N_WIN_PCT, LAST_N_POINT_DIFF,
      LAST_N_PTS_FOR, LAST_N_PTS_AGAINST, STREAK (current W/L streak)
    """
    df = games_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"], ascending=[True, False])
    df["WL_BINARY"] = (df["WL"] == "W").astype(int)

    results = []
    for team_id, group in df.groupby("TEAM_ID"):
        group = group.sort_values("GAME_DATE", ascending=False)
        last_n = group.head(n)

        # Streak: count consecutive same result from most recent game
        streak_char = group["WL"].iloc[0]
        streak = 0
        for wl in group["WL"]:
            if wl == streak_char:
                streak += 1
            else:
                break
        streak_label = f"{'W' if streak_char == 'W' else 'L'}{streak}"

        results.append({
            "TEAM_ID": team_id,
            "TEAM_ABBREV": group["TEAM_ABBREVIATION"].iloc[0],
            "LAST_N_GP": len(last_n),
            "LAST_N_WIN_PCT": last_n["WL_BINARY"].mean(),
            "LAST_N_POINT_DIFF": last_n["PLUS_MINUS"].mean(),
            "LAST_N_PTS_FOR": last_n["PTS"].mean(),
            "LAST_N_PTS_AGAINST": (last_n["PTS"] - last_n["PLUS_MINUS"]).mean(),
            "STREAK": streak_label,
        })

    return pd.DataFrame(results).sort_values("LAST_N_WIN_PCT", ascending=False).reset_index(drop=True)


def get_rest_days(games_df: pd.DataFrame, team_id: int, game_date: str) -> int:
    """
    Return how many days of rest a team has had before game_date.
    0 = back-to-back, 1 = one day off, etc.
    Returns -1 if no prior game found.
    """
    df = games_df[games_df["TEAM_ID"] == team_id].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    target = pd.to_datetime(game_date)

    prior = df[df["GAME_DATE"] < target].sort_values("GAME_DATE", ascending=False)
    if prior.empty:
        return -1
    last_game = prior.iloc[0]["GAME_DATE"]
    return (target - last_game).days - 1


def build_matchup_features(
    games_df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    game_date: str,
    n_recent: int = 10,
) -> dict:
    """
    Build a feature vector for a single upcoming matchup.
    Combines season stats, recent form, and rest days.
    """
    season_stats = build_team_season_stats(games_df)
    recent_form = build_recent_form(games_df, n=n_recent)

    def _get_season(tid):
        row = season_stats[season_stats["TEAM_ID"] == tid]
        return row.iloc[0] if not row.empty else None

    def _get_form(tid):
        row = recent_form[recent_form["TEAM_ID"] == tid]
        return row.iloc[0] if not row.empty else None

    hs = _get_season(home_team_id)
    as_ = _get_season(away_team_id)
    hf = _get_form(home_team_id)
    af = _get_form(away_team_id)

    if hs is None or as_ is None:
        return {}

    home_rest = get_rest_days(games_df, home_team_id, game_date)
    away_rest = get_rest_days(games_df, away_team_id, game_date)

    return {
        # Season differentials (home minus away)
        "WIN_PCT_DIFF": hs["WIN_PCT"] - as_["WIN_PCT"],
        "POINT_DIFF_DIFF": hs["POINT_DIFF"] - as_["POINT_DIFF"],
        "HOME_WIN_PCT": hs["HOME_WIN_PCT"],
        "AWAY_WIN_PCT_OPP": as_["AWAY_WIN_PCT"],
        # Recent form differentials
        "FORM_WIN_PCT_DIFF": (hf["LAST_N_WIN_PCT"] if hf is not None else 0.5) - (af["LAST_N_WIN_PCT"] if af is not None else 0.5),
        "FORM_POINT_DIFF_DIFF": (hf["LAST_N_POINT_DIFF"] if hf is not None else 0) - (af["LAST_N_POINT_DIFF"] if af is not None else 0),
        # Rest
        "HOME_REST_DAYS": home_rest,
        "AWAY_REST_DAYS": away_rest,
        "REST_ADVANTAGE": home_rest - away_rest,  # positive = home team more rested
        # Raw values for inspection
        "HOME_TEAM_ID": home_team_id,
        "AWAY_TEAM_ID": away_team_id,
        "HOME_SEASON_WIN_PCT": hs["WIN_PCT"],
        "AWAY_SEASON_WIN_PCT": as_["WIN_PCT"],
        "HOME_TEAM_ABBREV": hs["TEAM_ABBREV"],
        "AWAY_TEAM_ABBREV": as_["TEAM_ABBREV"],
    }
