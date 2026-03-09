"""
NBA Betting Tool — Daily Runner
================================
Fetches today's games, predicts win probabilities, pulls live odds,
and outputs a ranked table of bets by expected value.

Usage:
    python main.py            # normal run
    python main.py --retrain  # force model retraining
    python main.py --no-odds  # skip odds fetch (predictions only)
"""

import sys
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
from features.ratings import build_team_season_stats, build_recent_form
from model.predictor import get_or_train_model, predict_game
from odds.fetcher import fetch_nba_odds, match_odds_to_games, ODDS_NAME_TO_ABBREV
from odds.ev import analyse_game, american_to_implied_prob
from tracker import resolve_bets, save_recommendations, print_summary_line, print_report


# ── Config ────────────────────────────────────────────────────────────────────

MIN_EV_THRESHOLD  = 0.05    # only flag bets with EV > 5%  (raised from 2% — high-EV outliers are noise)
MAX_EV_THRESHOLD  = 0.35    # ignore bets with EV > 35%  — these are usually model errors on big dogs
MIN_EDGE_DISPLAY  = -0.10   # show all games with edge > -10% in the full table
KELLY_FRACTION    = 0.5     # half-Kelly staking
INJURY_LOGIT_COEF     = 0.025  # logit-scale sensitivity: per 1-pt of GmSc-based impact score
                               # (GmSc values ~3x larger than raw PM, so coef is ~3x smaller)
MAX_INJURY_PROB_SHIFT = 0.20   # hard cap: no injury adjustment moves a team >20pp
MIN_MINUTES_TO_COUNT  = 11.0   # player must avg >= 11 min/game to register an impact
MIN_GAMES_TO_COUNT    = 10     # player must have played >= 10 games this season
# Market-line blend: blend model probability with de-vigged implied probability from the book.
# 0.0 = pure model; 0.3 = 70% model + 30% market; enable once tracker shows good calibration.
MARKET_BLEND_WEIGHT   = 0.20  # 20% market / 80% model — reduces overconfidence where model disagrees with book

# ── Timezone config ───────────────────────────────────────────────────────────
# US Eastern Standard Time (February is EST = UTC-5; becomes EDT = UTC-4 in March)
_ET   = timezone(timedelta(hours=-5))
# Australian Eastern Daylight Time (Feb is AEDT = UTC+11; DST active Oct-Apr)
_AEDT = timezone(timedelta(hours=11))


def get_nba_game_date() -> str:
    """
    Return the current NBA game date as a YYYY-MM-DD string in US Eastern time.
    Games before 5 am ET are treated as still belonging to the previous night's
    slate (handles the midnight edge-case after late West-Coast tip-offs).
    """
    now_et = datetime.now(_ET)
    if now_et.hour < 5:
        return (now_et.date() - timedelta(days=1)).strftime("%Y-%m-%d")
    return now_et.date().strftime("%Y-%m-%d")


def et_to_aedt(time_str: str) -> str:
    """
    Convert a tip-time string from US Eastern to AEDT for display.

    '7:00 pm ET'  ->  '11:00 am AEDT'   (EST -> AEDT = +16 h in Feb)
    Non-time strings ('Final', 'Halftime', 'Q3 2:45', etc.) pass through unchanged.
    """
    import re
    m = re.match(r'(\d+):(\d+)\s*(am|pm)\s*ET', time_str.strip(), re.IGNORECASE)
    if not m:
        return time_str
    h, mn, ampm = int(m.group(1)), int(m.group(2)), m.group(3).lower()
    if ampm == 'pm' and h != 12:
        h += 12
    elif ampm == 'am' and h == 12:
        h = 0
    total = (h * 60 + mn + 16 * 60) % 1440   # +16 h offset, wrap at 24 h
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
    """
    Hollinger Game Score — a composite per-game impact metric.

    Built entirely from box-score stats already available in the season
    player DataFrame (no extra API calls).

    Unlike PLUS_MINUS, GmSc rewards what the player actually does on the
    court, making it reliable for stars who face heavy defensive coverage
    (e.g. Curry) and for players on weak teams whose PM is dragged negative
    by teammates.

    Formula (Hollinger, 1996):
      GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM)
             + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV

    Illustrative values:
      Steph Curry    ~26 PTS, efficient FG, 5 AST  -> GmSc ~19  (correctly elite)
      Solid starter  ~15 PTS, 3 AST, 5 REB         -> GmSc ~8
      Rotation role  ~8  PTS, 2 AST, 3 REB         -> GmSc ~3
    """
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
    """
    Return a minutes-weighted impact score for OUT players on a team.

    Formula per qualifying player:  (MIN_per_game / 36) * PLUS_MINUS

    Qualifications (must pass all three):
      - Averaging >= MIN_MINUTES_TO_COUNT min/game  (genuine rotation player)
      - Appeared in  >= MIN_GAMES_TO_COUNT games     (valid season sample)
      - Positive PLUS_MINUS                          (net-positive contributor)

    This prevents fringe bench players and DNP-level guys from registering
    as meaningful absences.  The minutes weight means a 35-min star with
    GmSc=19 contributes 17.9, while an 18-min role player with GmSc=5
    contributes 2.5 — proportional to actual share of the game.
    """
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
            continue   # fringe / DNP-level player — no meaningful rotation slot
        gs = _player_game_score(row)
        if gs <= 0:
            continue   # net-negative contribution — absence doesn't hurt
        total += (min_pg / 36.0) * gs
    return round(total, 2)


def _player_stats_tag(name: str, abbrev: str, player_df) -> str:
    """
    Return a compact stats tag for the injury display, showing why a player
    is counted or skipped.  Examples:
      [32min/68gp  +7.4pm  impact=6.6]
      [8min/22gp  -- not counted: < 15 min/g]
      [24min/14gp  -2.1pm  -- not counted: neg pm]
    """
    if player_df is None or player_df.empty:
        return ""
    team_df = player_df[player_df["TEAM_ABBREVIATION"] == abbrev]
    match = team_df[team_df["PLAYER_NAME"].str.lower() == name.lower()]
    if match.empty:
        return "[not in season stats]"
    row    = match.iloc[0]
    min_pg = float(row.get("MIN", 0))
    gp     = int(row.get("GP", 0))
    gs     = _player_game_score(row)
    base   = f"{min_pg:.0f}min/{gp}gp  GmSc={gs:.1f}"
    if min_pg < MIN_MINUTES_TO_COUNT:
        return f"[{base}  -- not counted: < {MIN_MINUTES_TO_COUNT:.0f} min/g]"
    if gp < MIN_GAMES_TO_COUNT:
        return f"[{base}  -- not counted: < {MIN_GAMES_TO_COUNT} games played]"
    if gs <= 0:
        return f"[{base}  -- not counted: neg GmSc]"
    impact = (min_pg / 36.0) * gs
    return f"[{base}  impact={impact:.1f}]"


def _apply_injury_adjustment(
    base_home_prob: float,
    home_abbrev: str,
    away_abbrev: str,
    home_inj: list,
    away_inj: list,
    player_df,
) -> tuple:
    """
    Shift the home win probability on the logit scale based on OUT player impact.

    A home team missing positive-PLUS_MINUS players: logit decreases (worse).
    An away team missing positive-PLUS_MINUS players: logit increases (helps home).

    Returns (adj_home_prob, home_impact, away_impact).
    """
    home_impact = _calc_injury_impact(home_abbrev, home_inj, player_df)
    away_impact = _calc_injury_impact(away_abbrev, away_inj, player_df)
    net_impact  = home_impact - away_impact  # positive = home hurts more

    if net_impact == 0.0 or not (0.0 < base_home_prob < 1.0):
        return base_home_prob, home_impact, away_impact

    logit_adj = math.log(base_home_prob / (1 - base_home_prob)) - INJURY_LOGIT_COEF * net_impact
    adj = 1.0 / (1.0 + math.exp(-logit_adj))
    # Hard cap: prevent injuries from swinging probability more than MAX_INJURY_PROB_SHIFT
    adj = max(base_home_prob - MAX_INJURY_PROB_SHIFT, min(base_home_prob + MAX_INJURY_PROB_SHIFT, adj))
    return round(adj, 4), home_impact, away_impact


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_prob(p: float) -> str:
    return f"{p*100:.1f}%"

def fmt_ev(ev: float) -> str:
    sign = "+" if ev >= 0 else ""
    return f"{sign}{ev*100:.1f}%"

def fmt_odds(o: int | None) -> str:
    if o is None:
        return "  N/A "
    return f"+{o}" if o >= 0 else str(o)

def fmt_kelly(k: float | None) -> str:
    if k is None or k == 0:
        return "  -  "
    return f"{k*100:.1f}%"

def print_divider(char: str = "-", width: int = 90) -> None:
    print(char * width)

def print_header(title: str, width: int = 90) -> None:
    print_divider("=", width)
    print(f"  {title}")
    print_divider("=", width)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(retrain: bool = False, no_odds: bool = False, report: bool = False) -> None:
    if report:
        print_header("NBA BETTING TRACKER  --  Performance Report")
        print_report()
        return

    today      = get_nba_game_date()                       # US ET game date
    now_aedt   = datetime.now(_AEDT)
    aedt_label = now_aedt.strftime("%a %d %b %Y  %I:%M %p AEDT")
    print_header(f"NBA Betting Tool  |  {aedt_label}  |  Season {CURRENT_SEASON}")
    print(f"  NBA game date (US ET): {today}")   # may advance to next day if all games final

    # ── 1. Season data ────────────────────────────────────────────────────────
    print("\n[ 1 ] Loading season data ...")
    teams_df      = get_all_teams()
    games_df      = get_season_games(season=CURRENT_SEASON)
    historical_df = get_historical_games()
    dashboard     = get_team_dashboard(season=CURRENT_SEASON)

    team_id_map    = teams_df.set_index("abbreviation")["id"].to_dict()
    team_abbrev_map = teams_df.set_index("id")["abbreviation"].to_dict()
    team_name_map  = teams_df.set_index("id")["full_name"].to_dict()

    n_games = games_df["GAME_ID"].nunique()
    print(f"  Season games loaded: {n_games}  |  Date range: "
          f"{pd.to_datetime(games_df['GAME_DATE']).min().strftime('%b %d')} - "
          f"{pd.to_datetime(games_df['GAME_DATE']).max().strftime('%b %d, %Y')}")

    season_stats = build_team_season_stats(games_df)
    recent_form  = build_recent_form(games_df, n=10)

    # Resolve any pending bets from previous game days now that results are in
    n_resolved = resolve_bets(games_df)
    if n_resolved:
        print(f"  [tracker] {n_resolved} pending bet(s) resolved from previous game days.")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("\n[ 2 ] Win probability model ...")
    pipe = get_or_train_model(games_df, force_retrain=retrain, historical_df=historical_df)

    # ── 3. Today's games ──────────────────────────────────────────────────────
    print("\n[ 3 ] Today's schedule ...")
    model_date = today   # preserve for model feature lookback regardless of display shift
    board  = get_todays_scoreboard(today)
    header = board["game_header"]

    # NBA API occasionally returns duplicate rows for the same game (rescheduled
    # or makeup games show up twice in ScoreboardV2). Keep the first occurrence.
    before = len(header)
    header = header.drop_duplicates(subset=["GAME_ID"]).reset_index(drop=True)
    if len(header) < before:
        print(f"  [warn] Removed {before - len(header)} duplicate game(s) from scoreboard.")

    # If today's slate is empty or every game is already Final (common in the
    # morning AEDT after all US games finished overnight), look ahead to the
    # next US ET game day so predictions stay useful.
    all_final = header.empty or all(
        str(r.get("GAME_STATUS_TEXT", "")).strip() == "Final"
        for _, r in header.iterrows()
    )
    if all_final:
        next_et = (date.fromisoformat(today) + timedelta(days=1)).strftime("%Y-%m-%d")
        next_board  = get_todays_scoreboard(next_et)
        next_header = next_board["game_header"].drop_duplicates(
            subset=["GAME_ID"]
        ).reset_index(drop=True)

        if not next_header.empty:
            reason = (f"No games today ({today})" if header.empty
                      else f"All {len(header)} game(s) for {today} are complete")
            print(f"  {reason} — showing tomorrow's slate ({next_et}).")
            today  = next_et
            header = next_header
        elif header.empty:
            print(f"  No games today ({today}) or tomorrow ({next_et}). Check back later.")
            return

    if header.empty:
        print(f"  No games scheduled for {today}. Check back on a game day.")
        return

    print(f"  {len(header)} game(s) — {today}\n")

    # ── 4. Live odds ──────────────────────────────────────────────────────────
    odds_matched = {}
    if not no_odds:
        print("[ 4 ] Live odds ...")
        odds_list    = fetch_nba_odds()
        odds_matched = match_odds_to_games(header.to_dict("records"), odds_list)
        if not odds_matched:
            print("  No odds matched (check ODDS_API_KEY in .env)")

    # ── 5. Injury report ──────────────────────────────────────────────────────
    print("\n[ 5 ] Injury report ...")
    injury_report   = fetch_nba_injuries()
    player_stats_df = get_player_stats(season=CURRENT_SEASON)

    # ── 6. Predict & analyse each game ────────────────────────────────────────
    print("\n[ 6 ] Predictions & EV analysis ...\n")

    analyses = []

    for _, row in header.iterrows():
        home_id   = row["HOME_TEAM_ID"]
        away_id   = row["VISITOR_TEAM_ID"]
        tip_text  = et_to_aedt(row.get("GAME_STATUS_TEXT", ""))

        home_abbrev = team_abbrev_map.get(home_id, str(home_id))
        away_abbrev = team_abbrev_map.get(away_id, str(away_id))

        # Win probability — use model_date (original today) as feature cutoff
        # so we never accidentally include future games in rolling stats
        pred = predict_game(
            pipe, games_df,
            home_team_id=home_id,
            away_team_id=away_id,
            game_date=model_date,
        )

        # Odds
        odds_key = (home_abbrev, away_abbrev)
        game_odds = odds_matched.get(odds_key, {})
        home_american = game_odds.get("home_odds")
        away_american = game_odds.get("away_odds")
        bookmaker     = game_odds.get("bookmaker", "")

        # Market-line blend (Tier 3) — optionally weight in de-vigged book implied prob.
        # De-vig: normalize raw implied probs so they sum to 1 (removes the bookmaker margin).
        # Default MARKET_BLEND_WEIGHT = 0.0 means pure model output; raise to 0.2-0.3 once
        # the tracker shows the model is well-calibrated relative to the market.
        base_model_prob = pred["home_win_prob"]
        if MARKET_BLEND_WEIGHT > 0 and home_american is not None and away_american is not None:
            raw_h = american_to_implied_prob(home_american)
            raw_a = american_to_implied_prob(away_american)
            total = raw_h + raw_a
            if total > 0:
                fair_h = raw_h / total  # de-vigged fair implied probability
                base_model_prob = round(
                    (1 - MARKET_BLEND_WEIGHT) * pred["home_win_prob"] + MARKET_BLEND_WEIGHT * fair_h, 4
                )

        # Injury adjustment — shift probabilities for OUT players
        game_inj = summarise_game_injuries(home_abbrev, away_abbrev, injury_report)
        adj_home_prob, home_inj_impact, away_inj_impact = _apply_injury_adjustment(
            base_home_prob=base_model_prob,
            home_abbrev=home_abbrev,
            away_abbrev=away_abbrev,
            home_inj=game_inj["home"],
            away_inj=game_inj["away"],
            player_df=player_stats_df,
        )
        adj_away_prob = round(1.0 - adj_home_prob, 4)

        # Full analysis (uses injury-adjusted probabilities)
        analysis = analyse_game(
            home_abbrev=home_abbrev,
            away_abbrev=away_abbrev,
            home_win_prob=adj_home_prob,
            away_win_prob=adj_away_prob,
            home_american=home_american,
            away_american=away_american,
            tip_time=tip_text,
            kelly_fraction_setting=KELLY_FRACTION,
        )
        analysis["home_rest"]       = pred.get("home_rest_days", -1)
        analysis["away_rest"]       = pred.get("away_rest_days", -1)
        analysis["bookmaker"]       = bookmaker
        analysis["base_home_prob"]  = base_model_prob   # post-blend, pre-injury-adj model output
        analysis["home_inj_impact"] = home_inj_impact
        analysis["away_inj_impact"] = away_inj_impact
        analysis["game_inj"]        = game_inj
        # GAME_STATUS_ID: 1=not started, 2=in progress, 3=final
        analysis["game_status_id"]  = int(row.get("GAME_STATUS_ID", 1))
        analyses.append(analysis)

    # ── 6. Full games table ───────────────────────────────────────────────────
    print_divider("=")
    print(f"  {'MATCHUP':<20} {'TIP (AEDT)':<16}  "
          f"{'HOME WIN%':>9} {'AWAY WIN%':>9}  "
          f"{'ODDS H':>7} {'ODDS A':>7}  "
          f"{'EV H':>7} {'EV A':>7}  "
          f"{'REST':>5}")
    print_divider()

    for a in analyses:
        ev_h = fmt_ev(a["home_ev"]) if a["home_ev"] is not None else "  N/A "
        ev_a = fmt_ev(a["away_ev"]) if a["away_ev"] is not None else "  N/A "
        rest = f"{a['home_rest']}/{a['away_rest']}d" if a["home_rest"] != -1 else "  -- "

        print(f"  {a['matchup']:<20} {a['tip_time']:<16}  "
              f"{fmt_prob(a['model_home_prob']):>9} {fmt_prob(a['model_away_prob']):>9}  "
              f"{fmt_odds(a['home_odds']):>7} {fmt_odds(a['away_odds']):>7}  "
              f"{ev_h:>7} {ev_a:>7}  "
              f"{rest:>5}")

    # ── 7. +EV bets ───────────────────────────────────────────────────────────
    ev_bets = [
        a for a in analyses
        if (a["has_odds"]
            and a["best_bet_ev"] is not None
            and MIN_EV_THRESHOLD <= a["best_bet_ev"] <= MAX_EV_THRESHOLD
            and a.get("game_status_id", 1) == 1)   # pre-game only: skip in-progress / final games
    ]
    ev_bets.sort(key=lambda x: x["best_bet_ev"], reverse=True)

    # One bet per game: if both sides cleared the filter (rare), keep the higher-EV side only.
    _seen_games: set = set()
    _deduped: list = []
    for _a in ev_bets:
        _gk = (_a["home_abbrev"], _a["away_abbrev"])
        if _gk not in _seen_games:
            _seen_games.add(_gk)
            _deduped.append(_a)
    ev_bets = _deduped

    print()
    print_divider("=")
    if not ev_bets:
        if any(a["has_odds"] for a in analyses):
            print("  No +EV bets found today in range "
                  f"{MIN_EV_THRESHOLD*100:.0f}%-{MAX_EV_THRESHOLD*100:.0f}% "
                  "(pre-game only). Sharps have efficient lines tonight.")
        else:
            print("  Odds not available. Add ODDS_API_KEY to .env to see EV analysis.")
            print()
            print("  MODEL PREDICTIONS (no odds):")
            print_divider()
            ranked = sorted(analyses, key=lambda x: abs(x["model_home_prob"] - 0.5), reverse=True)
            for a in ranked:
                fav_side = a["home_abbrev"] if a["model_home_prob"] > 0.5 else a["away_abbrev"]
                fav_prob = max(a["model_home_prob"], a["model_away_prob"])
                dog_side = a["away_abbrev"] if a["model_home_prob"] > 0.5 else a["home_abbrev"]
                dog_prob = min(a["model_home_prob"], a["model_away_prob"])
                print(f"  {a['matchup']:<20} {a['tip_time']:<16}  "
                      f"Fav: {fav_side} {fmt_prob(fav_prob)}   "
                      f"Dog: {dog_side} {fmt_prob(dog_prob)}   "
                      f"Rest: {a['home_rest']}/{a['away_rest']}d")
    else:
        print(f"  +EV ANALYSIS  (EV {MIN_EV_THRESHOLD*100:.0f}%-{MAX_EV_THRESHOLD*100:.0f}%, pre-game only, sorted by EV)")
        print_divider()
        print(f"  {'#':<3}  {'BET ON':<7}  {'EV%':>7}  {'KELLY%':>7}  {'EDGE':>7}  "
              f"{'ODDS':>7}  {'MODEL':>7}  {'IMPLIED':>8}  BOOK")
        print_divider("-", 82)
        for rank, a in enumerate(ev_bets, 1):
            if a["best_bet"] == a["home_abbrev"]:
                bet_odds   = a["home_odds"]
                model_prob = a["model_home_prob"]
                implied    = a["implied_home"]
            else:
                bet_odds   = a["away_odds"]
                model_prob = a["model_away_prob"]
                implied    = a["implied_away"]
            edge      = model_prob - (implied or 0)
            inj_flag  = " *" if (a.get("home_inj_impact", 0) > 0 or a.get("away_inj_impact", 0) > 0) else ""
            bet_label = f"{a['best_bet']}{inj_flag}"
            print(f"  {rank:<3}  {bet_label:<7}  "
                  f"{fmt_ev(a['best_bet_ev']):>7}  "
                  f"{fmt_kelly(a['best_bet_kelly']):>7}  "
                  f"{fmt_ev(edge):>7}  "
                  f"{fmt_odds(bet_odds):>7}  "
                  f"{fmt_prob(model_prob):>7}  "
                  f"{fmt_prob(implied or 0):>8}  "
                  f"{a['bookmaker']}")
        print()
        print("  EV%    = expected profit per $1 wagered  |  Kelly% = fraction of bankroll to stake")
        print("  Edge   = model probability minus book implied  |  * = injury-adjusted")

    # ── 8. Injury report display ──────────────────────────────────────────────
    print()
    print_divider("=")
    print("  INJURY REPORT  (Out + Doubtful only)")
    print_divider()
    any_injuries = False
    for a in analyses:
        inj      = a.get("game_inj", {})
        home_inj = inj.get("home", [])
        away_inj = inj.get("away", [])
        if not home_inj and not away_inj:
            continue
        any_injuries = True
        h_impact  = a.get("home_inj_impact", 0.0)
        aw_impact = a.get("away_inj_impact", 0.0)
        base      = a.get("base_home_prob", a["model_home_prob"])
        adj       = a["model_home_prob"]

        print(f"  {a['matchup']}  ({a['tip_time']})")
        if home_inj:
            print(f"    {a['home_abbrev']} (home)  [weighted impact: {h_impact:.1f}]")
            for i in home_inj:
                tag = _player_stats_tag(i["player"], a["home_abbrev"], player_stats_df)
                print(f"      [{i['label']}] {i['player']} ({i['pos']})  {tag}  {i['comment']}")
        if away_inj:
            print(f"    {a['away_abbrev']} (away)  [weighted impact: {aw_impact:.1f}]")
            for i in away_inj:
                tag = _player_stats_tag(i["player"], a["away_abbrev"], player_stats_df)
                print(f"      [{i['label']}] {i['player']} ({i['pos']})  {tag}  {i['comment']}")
        if h_impact > 0 or aw_impact > 0:
            direction = a["home_abbrev"] if h_impact > aw_impact else a["away_abbrev"]
            print(f"    Prob adj: {a['home_abbrev']} {fmt_prob(base)} -> {fmt_prob(adj)}"
                  f"  (penalises {direction})")
        print()
    if not any_injuries:
        print("  No Out/Doubtful players reported for today's games.")

    # ── 9. League standings snapshot ──────────────────────────────────────────
    print()
    print_divider("=")
    print("  CURRENT STANDINGS  (season to date)")
    print_divider()
    top10 = season_stats[["RANK", "TEAM_ABBREV", "GP", "W", "L", "WIN_PCT",
                           "POINT_DIFF", "HOME_WIN_PCT", "AWAY_WIN_PCT"]].head(10)
    print(top10.to_string(index=False))

    print()
    print_divider("=")
    print("  RECENT FORM  (last 10 games)")
    print_divider()
    form_cols = ["TEAM_ABBREV", "LAST_N_WIN_PCT", "LAST_N_POINT_DIFF", "STREAK"]
    hot  = recent_form.sort_values("LAST_N_WIN_PCT", ascending=False).head(5)
    cold = recent_form.sort_values("LAST_N_WIN_PCT", ascending=True).head(5)
    print("  Hottest:")
    print(hot[form_cols].to_string(index=False))
    print("  Coldest:")
    print(cold[form_cols].to_string(index=False))

    # ── 10. Betting action list ───────────────────────────────────────────────
    print()
    print_divider("*")
    print(f"  WHAT TO BET TODAY  ({aedt_label})")
    print_divider("*")

    if not ev_bets:
        print("  No +EV bets found today. Skip today or wait for line movement.")
    else:
        # Lead with the decision-critical columns (WHO / EV / KELLY), then context
        print(f"  {'#':<3}  {'BET ON':<10}  {'EV%':>7}  {'KELLY%':>7}  |  "
              f"{'OPPONENT':<10}  {'TIP (AEDT)':<16}  {'ODDS':>6}  {'MODEL':>7}  {'IMPLIED':>8}  BOOK")
        print_divider("-", 98)
        for rank, a in enumerate(ev_bets, 1):
            bet_team = a["best_bet"]
            opp_team = a["away_abbrev"] if bet_team == a["home_abbrev"] else a["home_abbrev"]
            is_home  = bet_team == a["home_abbrev"]
            if is_home:
                bet_odds   = a["home_odds"]
                model_prob = a["model_home_prob"]
                implied    = a["implied_home"] or 0
            else:
                bet_odds   = a["away_odds"]
                model_prob = a["model_away_prob"]
                implied    = a["implied_away"] or 0
            venue     = "(H)" if is_home else "(A)"
            inj_flag  = " *" if (a.get("home_inj_impact", 0) > 0 or a.get("away_inj_impact", 0) > 0) else ""
            bet_label = f"{bet_team} {venue}{inj_flag}"
            vs_str    = f"vs {opp_team} ({'A' if is_home else 'H'})"
            print(f"  {rank:<3}  {bet_label:<10}  "
                  f"{fmt_ev(a['best_bet_ev']):>7}  "
                  f"{fmt_kelly(a['best_bet_kelly']):>7}  |  "
                  f"{vs_str:<10}  "
                  f"{a['tip_time']:<16}  "
                  f"{fmt_odds(bet_odds):>6}  "
                  f"{fmt_prob(model_prob):>7}  "
                  f"{fmt_prob(implied):>8}  "
                  f"{a['bookmaker']}")
        print()
        print("  EV%    = expected profit per $1 wagered  (the higher the better)")
        print("  KELLY% = recommended stake as % of bankroll  (Half-Kelly, conservative)")
        print("  MODEL  = our win probability  |  IMPLIED = bookmaker's implied probability")
        if any(a.get("home_inj_impact", 0) > 0 or a.get("away_inj_impact", 0) > 0
               for a in ev_bets):
            print("  *      = probability adjusted for confirmed Out/Doubtful injuries")

    # ── Tracker ───────────────────────────────────────────────────────────────
    n_saved = save_recommendations(analyses, today)
    print()
    print_divider("=")
    print_summary_line()
    if n_saved:
        print(f"  [tracker] {n_saved} bet(s) saved to logs/bet_log.json  "
              f"(auto-resolved tomorrow after results come in)")
    print()
    print(f"  Run complete. Cache expires in ~4h. Re-run for fresh odds.")
    print_divider("=")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Betting Tool")
    parser.add_argument("--retrain",  action="store_true", help="Force model retraining")
    parser.add_argument("--no-odds",  action="store_true", help="Skip odds fetch")
    parser.add_argument("--report",   action="store_true", help="Show bet performance report and exit")
    args = parser.parse_args()
    main(retrain=args.retrain, no_odds=args.no_odds, report=args.report)
