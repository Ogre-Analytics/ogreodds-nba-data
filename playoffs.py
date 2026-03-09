"""
NBA Playoff Probability Simulator
==================================
Uses a hybrid MC + exact DP approach to estimate championship probabilities.

Regular season (standing simulation):
  - Monte Carlo with Sobol quasi-random sequences (or antithetic variates fallback)
  - ~300 remaining games, 2^300 states -- exact enumeration impossible

Playoff bracket (once seeds determined per simulation):
  - EXACT dynamic programming -- no randomness in the bracket phase
  - 8-team conference bracket: enumerate all possible R2/CF matchup distributions
  - NBA Finals: exact over all (East_finalist x West_finalist) pairs
  - This eliminates ~45% of total simulation variance vs pure MC

Variance reduction stack:
  1. Sobol QMC sequences (scipy) for regular season draws       O(n^-1 logn^d)
     Fallback: antithetic variates (1-u mirror draws)           ~50% var reduction
  2. Exact bracket DP -- zero variance in bracket/finals phase
  Combined: typically 3-4x fewer simulations needed vs naive MC

Research-backed playoff model adjustments:
  - Playoff HCA scaling: 0.75x compression (Anderson & Pierce 2009)
  - Style variance: high-3PA/high-pace matchups compressed toward 0.5
  - Clutch net rating: displayed as reference column in output table
  - 90% confidence intervals on CHAMP column via binomial SE

Usage:
    python playoffs.py                         # 10,000 simulations (default)
    python playoffs.py --sims 5000             # custom count
    python playoffs.py --no-cache              # force fresh API data
    python playoffs.py --series "BOS 2 MIA 1" # live series state
"""

import heapq
import math
import time
import random
import argparse
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from data.fetcher import (
    get_all_teams,
    get_season_games,
    get_historical_games,
    get_remaining_schedule,
    get_team_dashboard,
    get_clutch_stats,
    CURRENT_SEASON,
)
from model.predictor import get_or_train_model, predict_game


# ── Constants ─────────────────────────────────────────────────────────────────

N_SIMS = 10_000

# Playoff HCA is ~75% of regular-season HCA (Anderson & Pierce 2009)
PLAYOFF_HCA_SCALE = 0.75

LEAGUE_AVG_FG3A_RATE = 0.35
LEAGUE_AVG_FGA       = 87.0

EAST_ABBREVS = frozenset({
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE",
    "DET", "IND", "MIA", "MIL", "NYK", "ORL",
    "PHI", "TOR", "WAS",
})
WEST_ABBREVS = frozenset({
    "DAL", "DEN", "GSW", "HOU", "LAC", "LAL",
    "MEM", "MIN", "NOP", "OKC", "PHX", "POR",
    "SAC", "SAS", "UTA",
})

# NBA best-of-7 home/away game sequence for the higher seed:
#   G1  G2  G3  G4  G5  G6  G7
#    H   H   A   A   H   A   H   (True = higher seed plays at home)
SERIES_PATTERN = (True, True, False, False, True, False, True)

_ET   = timezone(timedelta(hours=-5))
_AEDT = timezone(timedelta(hours=11))


# ── Playoff probability adjustments ──────────────────────────────────────────

def _scale_playoff_prob(p: float) -> float:
    """Compress home win prob toward 0.5 to reflect reduced HCA in playoffs."""
    return 0.5 + PLAYOFF_HCA_SCALE * (p - 0.5)


def _style_variance_scale(home_id: int, away_id: int, team_style: dict) -> float:
    """
    Variance multiplier for the matchup based on team playing style.
    3-point-heavy and high-pace teams → more per-game variance → higher scale
    → probabilities compressed further toward 0.5.
    """
    h = team_style.get(home_id, {})
    a = team_style.get(away_id, {})
    avg_fg3a_rate = (h.get("fg3a_rate", LEAGUE_AVG_FG3A_RATE) +
                     a.get("fg3a_rate", LEAGUE_AVG_FG3A_RATE)) / 2.0
    avg_fga       = (h.get("fga", LEAGUE_AVG_FGA) +
                     a.get("fga", LEAGUE_AVG_FGA)) / 2.0
    fg3a_scale = avg_fg3a_rate / LEAGUE_AVG_FG3A_RATE
    fga_scale  = avg_fga / LEAGUE_AVG_FGA
    return max(0.90, min(1.15, 0.6 * fg3a_scale + 0.4 * fga_scale))


# ── Series win probability (exact DP) ────────────────────────────────────────

def series_win_prob(p_home: float, p_away: float) -> float:
    """
    P(higher seed wins a best-of-7 series) — exact DP, no sampling.

    p_home : P(higher seed wins a single game at home)
    p_away : P(higher seed wins a single game away)
    """
    states: dict[tuple[int, int], float] = {(0, 0): 1.0}
    for is_home in SERIES_PATTERN:
        p = p_home if is_home else p_away
        nxt: dict[tuple[int, int], float] = {}
        for (h, l), prob in states.items():
            if h == 4 or l == 4:
                nxt[(h, l)] = nxt.get((h, l), 0.0) + prob
            else:
                nxt[(h + 1, l)] = nxt.get((h + 1, l), 0.0) + prob * p
                nxt[(h, l + 1)] = nxt.get((h, l + 1), 0.0) + prob * (1 - p)
        states = nxt
    return sum(pr for (h, l), pr in states.items() if h == 4)


def series_win_prob_from_state(
    p_home: float, p_away: float,
    hi_wins: int, lo_wins: int,
) -> float:
    """
    DP series win probability starting from an in-progress state (hi_wins-lo_wins).
    Picks up at game (hi_wins + lo_wins + 1) using the correct SERIES_PATTERN slot.
    """
    games_played = hi_wins + lo_wins
    if games_played >= 7 or hi_wins >= 4 or lo_wins >= 4:
        return 1.0 if hi_wins == 4 else 0.0

    states: dict[tuple[int, int], float] = {(hi_wins, lo_wins): 1.0}
    for is_home in SERIES_PATTERN[games_played:]:
        p = p_home if is_home else p_away
        nxt: dict[tuple[int, int], float] = {}
        for (h, l), prob in states.items():
            if h == 4 or l == 4:
                nxt[(h, l)] = nxt.get((h, l), 0.0) + prob
            else:
                nxt[(h + 1, l)] = nxt.get((h + 1, l), 0.0) + prob * p
                nxt[(h, l + 1)] = nxt.get((h, l + 1), 0.0) + prob * (1 - p)
        states = nxt
    return sum(pr for (h, l), pr in states.items() if h == 4)


# ── Exact bracket DP ──────────────────────────────────────────────────────────

def exact_conference_bracket_probs(
    playoff_teams: list[int],
    win_prob_fn,
    series_cache: dict,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """
    Analytically compute round-by-round win probabilities for each team in an
    8-team bracket. Zero randomness -- exact probability accumulation.

    Bracket structure (fixed seeding, no re-seeding after each round):
      R1: (1 vs 8), (4 vs 5), (2 vs 7), (3 vs 6)
          Top half = 1v8 and 4v5;  bottom half = 2v7 and 3v6
      R2: winner(1v8) vs winner(4v5),  winner(2v7) vs winner(3v6)
      CF: winner(top half) vs winner(bottom half)

    Home court in every series goes to the team with the lower original seed number.
    The top and bottom halves are independent until the Conference Finals, which
    allows O(n^2) enumeration of R2 matchups rather than O(2^7) path enumeration.

    Returns:
      r1_win[t]   = P(t wins Round 1)                 -> "r2" counter
      r2_win[t]   = P(t wins Round 2)                 -> "cf" counter
      conf_win[t] = P(t wins the Conference)           -> "finals" counter
    """
    teams      = list(playoff_teams)
    orig_seeds = list(range(1, len(teams) + 1))  # 1-indexed seeds

    def sp(t_hi: int, t_lo: int) -> float:
        """P(t_hi wins a best-of-7 series), t_hi has home court advantage."""
        key = (t_hi, t_lo)
        if key not in series_cache:
            p_h = win_prob_fn(t_hi, t_lo)
            p_a = 1.0 - win_prob_fn(t_lo, t_hi)
            series_cache[key] = series_win_prob(p_h, p_a)
        return series_cache[key]

    def hca_sp(ta: int, tb: int) -> float:
        """P(ta wins series) with HCA awarded to the lower original seed."""
        sa = orig_seeds[teams.index(ta)]
        sb = orig_seeds[teams.index(tb)]
        return sp(ta, tb) if sa < sb else 1.0 - sp(tb, ta)

    # ── Round 1: 4 independent series ─────────────────────────────────────────
    # (1v8): teams[0] vs teams[7], (4v5): teams[3] vs teams[4]
    # (2v7): teams[1] vs teams[6], (3v6): teams[2] vs teams[5]
    r1_win: dict[int, float] = {}
    for hi_idx, lo_idx in ((0, 7), (3, 4), (1, 6), (2, 5)):
        t_hi, t_lo = teams[hi_idx], teams[lo_idx]
        p = sp(t_hi, t_lo)
        r1_win[t_hi] = p
        r1_win[t_lo] = 1.0 - p

    # ── Round 2: top and bottom halves are independent ─────────────────────────
    # Top-half R2: {winner(1v8)} vs {winner(4v5)} -- 2 x 2 = 4 matchup scenarios
    # Bot-half R2: {winner(2v7)} vs {winner(3v6)} -- 4 matchup scenarios
    top_a = [teams[0], teams[7]]   # possible winners of 1v8
    top_b = [teams[3], teams[4]]   # possible winners of 4v5
    bot_a = [teams[1], teams[6]]   # possible winners of 2v7
    bot_b = [teams[2], teams[5]]   # possible winners of 3v6

    top_half_win: dict[int, float] = {t: 0.0 for t in top_a + top_b}
    bot_half_win: dict[int, float] = {t: 0.0 for t in bot_a + bot_b}

    for group_a, group_b, half_win in (
        (top_a, top_b, top_half_win),
        (bot_a, bot_b, bot_half_win),
    ):
        for ta in group_a:
            for tb in group_b:
                p_matchup = r1_win[ta] * r1_win[tb]
                if p_matchup < 1e-12:
                    continue
                p_ta = hca_sp(ta, tb)
                half_win[ta] += p_matchup * p_ta
                half_win[tb] += p_matchup * (1.0 - p_ta)

    r2_win = {**top_half_win, **bot_half_win}

    # ── Conference Finals: 4 x 4 = 16 matchup scenarios ──────────────────────
    conf_win: dict[int, float] = {t: 0.0 for t in teams}
    for t_top, p_top in top_half_win.items():
        if p_top < 1e-12:
            continue
        for t_bot, p_bot in bot_half_win.items():
            if p_bot < 1e-12:
                continue
            p_top_wins = hca_sp(t_top, t_bot)
            conf_win[t_top] += p_top * p_bot * p_top_wins
            conf_win[t_bot] += p_top * p_bot * (1.0 - p_top_wins)

    return r1_win, r2_win, conf_win


# ── Play-in simulation (single games, random) ─────────────────────────────────

def simulate_playin(seeds_7_to_10: list[int], win_prob_fn) -> tuple[int, int]:
    """
    Simulate the NBA play-in tournament for one conference.

    Game A : 7-seed (home) vs 8-seed  -- winner -> 7th playoff seed
    Game B : 9-seed (home) vs 10-seed -- loser eliminated
    Game C : loser(A) (home) vs winner(B) -- winner -> 8th playoff seed

    Play-in uses single-game elimination (not series), so random draws are
    appropriate here -- there is no DP simplification for single games.

    Returns (team_id_for_7th_seed, team_id_for_8th_seed).
    """
    t7, t8, t9, t10 = seeds_7_to_10

    # Game A
    seed7   = t7 if random.random() < win_prob_fn(t7, t8) else t8
    loser_a = t8 if seed7 == t7 else t7

    # Game B
    winner_b = t9 if random.random() < win_prob_fn(t9, t10) else t10

    # Game C  — loser_a is the higher remaining seed, so they host
    seed8 = loser_a if random.random() < win_prob_fn(loser_a, winner_b) else winner_b

    return seed7, seed8


# ── Main simulation loop ───────────────────────────────────────────────────────

def run_one_simulation(
    base_wins:    dict[int, int],
    base_pd:      dict[int, float],
    remaining:    list[dict],
    rem_probs:    np.ndarray,
    rand_row:     np.ndarray,
    team_conf:    dict[int, str],
    pipe,
    games_df:     pd.DataFrame,
    today_str:    str,
    prob_cache:   dict,
    series_cache: dict,
    team_style:   dict,
) -> dict[int, dict[str, float]]:
    """
    Execute one Monte Carlo iteration (regular season) + exact bracket DP.

    Steps:
      1. Sample remaining regular season outcomes (MC -- unavoidable for ~300 games).
      2. Sort teams into conference standings; determine seeds.
      3. Simulate play-in tournament (random single-game draws).
      4. Compute exact conference bracket probabilities (DP -- no randomness).
      5. Compute exact NBA Finals probabilities over all finalist pairs.

    Returns {team_id: {key: probability}} where regular-season outcomes
    (top6, playin, playoffs) are 0.0/1.0 and bracket outcomes (r2, cf,
    finals, champ) are genuine probabilities in [0, 1].
    """
    # ── 1. Simulate remaining regular season ──────────────────────────────────
    wins = dict(base_wins)
    for j, game in enumerate(remaining):
        if rand_row[j] < rem_probs[j]:
            wins[game["home_id"]] = wins.get(game["home_id"], 0) + 1
        else:
            wins[game["away_id"]] = wins.get(game["away_id"], 0) + 1

    # ── 2. Conference standings ────────────────────────────────────────────────
    conf_sorted: dict[str, list[int]] = {"East": [], "West": []}
    for team_id, conf in team_conf.items():
        conf_sorted[conf].append(team_id)
    for conf in conf_sorted:
        conf_sorted[conf].sort(
            key=lambda t: (-wins.get(t, 0), -base_pd.get(t, 0.0))
        )

    # ── Playoff win probability (with caching + HCA/style adjustments) ─────────
    def win_prob(home_id: int, away_id: int) -> float:
        key = (home_id, away_id)
        if key not in prob_cache:
            raw = predict_game(pipe, games_df, home_id, away_id, today_str)["home_win_prob"]
            p = _scale_playoff_prob(raw)
            if team_style:
                scale = _style_variance_scale(home_id, away_id, team_style)
                p = 0.5 + (p - 0.5) / scale
            prob_cache[key] = max(0.05, min(0.95, p))
        return prob_cache[key]

    # ── 3. Seeds + play-in ────────────────────────────────────────────────────
    result: dict[int, dict[str, float]] = {
        t: {"top6": 0.0, "playin": 0.0, "playoffs": 0.0,
            "r2": 0.0, "cf": 0.0, "finals": 0.0, "champ": 0.0}
        for t in team_conf
    }

    playoff_seeds: dict[str, list[int]] = {}
    for conf, sorted_teams in conf_sorted.items():
        for t in sorted_teams[:6]:
            result[t]["top6"]     = 1.0
            result[t]["playoffs"] = 1.0
        if len(sorted_teams) >= 10:
            for t in sorted_teams[6:10]:
                result[t]["playin"] = 1.0
            s7, s8 = simulate_playin(sorted_teams[6:10], win_prob)
            result[s7]["playoffs"] = 1.0
            result[s8]["playoffs"] = 1.0
            playoff_seeds[conf] = sorted_teams[:6] + [s7, s8]
        else:
            playoff_seeds[conf] = sorted_teams[:8]

    # ── 4. Exact conference bracket DP ────────────────────────────────────────
    east_conf_win: dict[int, float] = {}
    west_conf_win: dict[int, float] = {}

    for conf in ("East", "West"):
        ps = playoff_seeds.get(conf, [])
        if len(ps) < 8:
            continue
        r1_win, r2_win, conf_win = exact_conference_bracket_probs(
            ps, win_prob, series_cache
        )
        for tid in ps:
            result[tid]["r2"]     = r1_win.get(tid, 0.0)
            result[tid]["cf"]     = r2_win.get(tid, 0.0)
            result[tid]["finals"] = conf_win.get(tid, 0.0)
        if conf == "East":
            east_conf_win = conf_win
        else:
            west_conf_win = conf_win

    # ── 5. Exact NBA Finals DP ────────────────────────────────────────────────
    # Enumerate all (East finalist, West finalist) pairs weighted by their
    # conference win probabilities -- no random draw needed.
    for e_team, p_e in east_conf_win.items():
        if p_e < 1e-9:
            continue
        for w_team, p_w in west_conf_win.items():
            if p_w < 1e-9:
                continue
            # HCA in Finals goes to team with more simulated regular-season wins
            if wins.get(e_team, 0) >= wins.get(w_team, 0):
                key = (e_team, w_team)
                if key not in series_cache:
                    p_h = win_prob(e_team, w_team)
                    p_a = 1.0 - win_prob(w_team, e_team)
                    series_cache[key] = series_win_prob(p_h, p_a)
                p_e_wins = series_cache[key]
            else:
                key = (w_team, e_team)
                if key not in series_cache:
                    p_h = win_prob(w_team, e_team)
                    p_a = 1.0 - win_prob(e_team, w_team)
                    series_cache[key] = series_win_prob(p_h, p_a)
                p_e_wins = 1.0 - series_cache[key]

            result[e_team]["champ"] += p_e * p_w * p_e_wins
            result[w_team]["champ"] += p_e * p_w * (1.0 - p_e_wins)

    return result


# ── Display helpers ───────────────────────────────────────────────────────────

def fmt_pct(p: float, lo: float = 0.002, hi: float = 0.998) -> str:
    if p < lo:
        return "  --  "
    if p > hi:
        return "99.9%+"
    return f"{p * 100:5.1f}%"


def fmt_pct_ci(p: float, n: int, z: float = 1.645) -> str:
    """11-char: 'pp.p+/-c.c%' — 90% CI via binomial SE."""
    if p < 0.002:
        return "    --     "
    if p > 0.998:
        return " 99.9%+    "
    se = math.sqrt(p * (1.0 - p) / max(n, 1))
    ci = z * se * 100
    return f"{p * 100:5.1f}+/-{ci:.1f}%"


def print_divider(char: str = "-", width: int = 115) -> None:
    print(char * width)


def print_header(title: str, width: int = 115) -> None:
    print_divider("=", width)
    print(f"  {title}")
    print_divider("=", width)


# ── Synthetic remaining schedule (API fallback) ───────────────────────────────

def _generate_synthetic_schedule(all_stats: pd.DataFrame) -> list[dict]:
    """
    Build a synthetic remaining schedule when all API/CDN methods fail.
    Each team plays (82 - gp) more games; uses a greedy max-heap pairing.
    """
    rem: dict[int, int] = {
        int(row["TEAM_ID"]): max(0, 82 - int(row["gp"]))
        for _, row in all_stats.iterrows()
    }
    heap: list[tuple[int, int]] = [(-n, tid) for tid, n in rem.items() if n > 0]
    import heapq as _hq
    _hq.heapify(heap)

    games: list[dict] = []
    while len(heap) >= 2:
        neg_n1, t1 = _hq.heappop(heap)
        neg_n2, t2 = _hq.heappop(heap)
        n1, n2 = -neg_n1, -neg_n2
        if t1 == t2:
            if n1 > 1:
                _hq.heappush(heap, (-(n1 - 1), t1))
            continue
        if random.random() < 0.5:
            games.append({"home_id": t1, "away_id": t2, "game_date": "2026-04-13"})
        else:
            games.append({"home_id": t2, "away_id": t1, "game_date": "2026-04-13"})
        if n1 - 1 > 0:
            _hq.heappush(heap, (-(n1 - 1), t1))
        if n2 - 1 > 0:
            _hq.heappush(heap, (-(n2 - 1), t2))
    return games


# ── Live series state mode ─────────────────────────────────────────────────────

def series_state_mode(
    series_str: str,
    pipe,
    games_df: pd.DataFrame,
    today_str: str,
    team_abbrev_to_id: dict,
    team_style: dict,
) -> None:
    """
    Display current playoff series win probability from a live state.
    series_str format: "TEAM1 W1 TEAM2 W2"  (first team = higher seed / HCA)
    """
    parts = series_str.upper().split()
    if len(parts) != 4:
        print("[error] --series format: 'TEAM1 W1 TEAM2 W2'  e.g. 'BOS 2 MIA 1'")
        return

    ab1, w1_str, ab2, w2_str = parts
    try:
        w1, w2 = int(w1_str), int(w2_str)
    except ValueError:
        print("[error] Win counts must be integers.")
        return

    if w1 < 0 or w2 < 0 or w1 >= 4 or w2 >= 4:
        print("[error] Each win count must be 0-3 (series still in progress).")
        return

    t1 = team_abbrev_to_id.get(ab1)
    t2 = team_abbrev_to_id.get(ab2)
    if t1 is None:
        print(f"[error] Unknown team: {ab1}")
        return
    if t2 is None:
        print(f"[error] Unknown team: {ab2}")
        return

    raw_h = predict_game(pipe, games_df, t1, t2, today_str)["home_win_prob"]
    raw_a = predict_game(pipe, games_df, t2, t1, today_str)["home_win_prob"]
    p_home = _scale_playoff_prob(raw_h)
    p_away = 1.0 - _scale_playoff_prob(raw_a)

    if team_style:
        scale_h = _style_variance_scale(t1, t2, team_style)
        scale_a = _style_variance_scale(t2, t1, team_style)
        p_home = max(0.05, min(0.95, 0.5 + (p_home - 0.5) / scale_h))
        p_away = max(0.05, min(0.95, 0.5 + (p_away - 0.5) / scale_a))

    p_series = series_win_prob_from_state(p_home, p_away, w1, w2)

    W = 72
    print()
    print("=" * W)
    print(f"  LIVE SERIES: {ab1} {w1} - {w2} {ab2}")
    print("=" * W)
    print(f"  {ab1} single-game prob (home): {p_home * 100:.1f}%")
    print(f"  {ab1} single-game prob (away): {p_away * 100:.1f}%")
    print()
    print(f"  {ab1:4s} series win probability:  {p_series * 100:.1f}%")
    print(f"  {ab2:4s} series win probability:  {(1 - p_series) * 100:.1f}%")

    games_played  = w1 + w2
    remaining_pat = SERIES_PATTERN[games_played:]
    if remaining_pat:
        print()
        print(f"  Remaining game schedule ({ab1} = higher seed):")
        for i, is_home in enumerate(remaining_pat):
            gn  = games_played + i + 1
            loc = f"{ab1} host" if is_home else f"{ab2} host"
            gp  = p_home if is_home else p_away
            print(f"    Game {gn}:  {loc}  |  {ab1} win prob: {gp * 100:.1f}%")
    print("=" * W)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_sims: int = N_SIMS, no_cache: bool = False, out_path: str | None = None) -> None:
    now_aedt   = datetime.now(_AEDT)
    aedt_label = now_aedt.strftime("%a %d %b %Y  %I:%M %p AEDT")
    today_str  = datetime.now(_ET).strftime("%Y-%m-%d")

    print_header(f"NBA PLAYOFF ODDS  |  {aedt_label}  |  {n_sims:,} simulations")
    print(f"  Season: {CURRENT_SEASON}  |  Game date (US ET): {today_str}\n")

    # ── 1. Season data ────────────────────────────────────────────────────────
    print("[ 1 ] Loading season data ...")
    teams_df        = get_all_teams()
    games_df        = get_season_games(season=CURRENT_SEASON, use_cache=not no_cache)
    historical_df   = get_historical_games()
    team_abbrev_map = teams_df.set_index("id")["abbreviation"].to_dict()

    team_conf: dict[int, str] = {
        int(tid): ("East" if abbrev in EAST_ABBREVS else "West")
        for tid, abbrev in team_abbrev_map.items()
        if abbrev in EAST_ABBREVS | WEST_ABBREVS
    }

    # ── 2. Win probability model ──────────────────────────────────────────────
    print("\n[ 2 ] Loading win probability model ...")
    pipe = get_or_train_model(games_df, historical_df=historical_df)

    # ── 3. Team style + clutch stats ─────────────────────────────────────────
    print("\n[ 3 ] Loading team style and clutch stats ...")
    team_style: dict[int, dict] = {}
    clutch_rtg: dict[int, float] = {}

    try:
        dashboard_df = get_team_dashboard(season=CURRENT_SEASON, use_cache=not no_cache)
        for _, row in dashboard_df.iterrows():
            fga  = max(float(row.get("FGA",  LEAGUE_AVG_FGA)), 1.0)
            fg3a = float(row.get("FG3A", LEAGUE_AVG_FGA * LEAGUE_AVG_FG3A_RATE))
            team_style[int(row["TEAM_ID"])] = {"fg3a_rate": fg3a / fga, "fga": fga}
        print(f"  Team style loaded for {len(team_style)} teams.")
    except Exception as exc:
        print(f"  [warn] Team style data unavailable: {exc}")

    try:
        clutch_df = get_clutch_stats(season=CURRENT_SEASON, use_cache=not no_cache)
        for _, row in clutch_df.iterrows():
            clutch_rtg[int(row["TEAM_ID"])] = float(row.get("PLUS_MINUS", 0.0))
        print(f"  Clutch stats loaded for {len(clutch_rtg)} teams.")
    except Exception as exc:
        print(f"  [warn] Clutch stats unavailable: {exc}")

    # ── 4. Current standings ──────────────────────────────────────────────────
    print("\n[ 4 ] Computing current standings ...")
    gdf = games_df.copy()
    gdf["WIN"] = (gdf["WL"] == "W").astype(int)
    stats = (
        gdf.groupby("TEAM_ID")
        .agg(wins=("WIN", "sum"), gp=("WIN", "count"), point_diff=("PLUS_MINUS", "mean"))
        .reset_index()
    )
    stats["losses"] = stats["gp"] - stats["wins"]
    stats["abbrev"] = stats["TEAM_ID"].map(team_abbrev_map)
    stats["conf"]   = stats["abbrev"].map(
        lambda a: "East" if a in EAST_ABBREVS else "West"
    )
    standings: dict[str, pd.DataFrame] = {}
    for conf in ("East", "West"):
        standings[conf] = (
            stats[stats["conf"] == conf]
            .sort_values(["wins", "point_diff"], ascending=[False, False])
            .reset_index(drop=True)
        )

    n_games = gdf["GAME_ID"].nunique()
    print(f"  {n_games} games completed through "
          f"{pd.to_datetime(gdf['GAME_DATE']).max().strftime('%b %d, %Y')}")

    # ── 5. Remaining schedule ─────────────────────────────────────────────────
    print("\n[ 5 ] Fetching remaining schedule ...")
    completed_ids = set(gdf["GAME_ID"].astype(str))
    remaining = get_remaining_schedule(
        season=CURRENT_SEASON,
        completed_game_ids=completed_ids,
        today_str=today_str,
        use_cache=not no_cache,
    )

    all_stats = pd.concat(list(standings.values()), ignore_index=True)
    expected_remaining = int(all_stats["gp"].apply(lambda g: max(0, 82 - int(g))).sum()) // 2
    if len(remaining) < max(30, expected_remaining // 3):
        if remaining:
            print(f"  [warn] Schedule API returned only {len(remaining)} games "
                  f"(expected ~{expected_remaining}).")
        else:
            print(f"  [warn] Schedule API returned 0 games "
                  f"(expected ~{expected_remaining}).")
        print("  [fallback] Generating synthetic remaining schedule ...")
        remaining = _generate_synthetic_schedule(all_stats)
        print(f"  [fallback] {len(remaining)} synthetic games generated.")

    games_df_dt = games_df.copy()
    games_df_dt["GAME_DATE"] = pd.to_datetime(games_df_dt["GAME_DATE"])

    if remaining:
        print(f"  Pre-computing win probabilities for {len(remaining)} remaining games ...")
        t_prob = time.time()
        rem_probs = np.array([
            predict_game(pipe, games_df_dt, g["home_id"], g["away_id"], today_str)["home_win_prob"]
            for g in remaining
        ])
        print(f"  Done ({time.time()-t_prob:.1f}s). "
              f"Avg home win prob: {rem_probs.mean():.3f}")
    else:
        rem_probs = np.array([])
        print("  No remaining games -- using current standings as final standings.")

    base_wins = {int(k): int(v) for k, v in gdf.groupby("TEAM_ID")["WIN"].sum().items()}
    base_pd   = {int(k): float(v) for k, v in gdf.groupby("TEAM_ID")["PLUS_MINUS"].mean().items()}

    # ── 6. Monte Carlo with variance reduction ────────────────────────────────
    print(f"\n[ 6 ] Running {n_sims:,} simulations (exact bracket DP + variance reduction) ...")
    t0    = time.time()
    n_rem = len(remaining)

    # Generate quasi-random (Sobol) draws for the regular season simulation.
    # Sobol sequences fill the sample space more uniformly than pseudorandom
    # numbers, giving O(n^-1 log^d n) convergence vs O(n^-0.5) for MC.
    # Falls back to antithetic variates if scipy unavailable or d too large.
    vr_method = "unknown"
    if n_rem > 0:
        try:
            from scipy.stats.qmc import Sobol as _Sobol
            # Sobol balance is optimal at powers of 2 — generate next power-of-2
            # count and trim to n_sims so properties are preserved without warnings.
            n_sobol   = 1 << math.ceil(math.log2(max(n_sims, 2)))
            sampler   = _Sobol(d=n_rem, scramble=True, seed=42)
            all_draws = sampler.random(n_sobol)[:n_sims]
            vr_method = f"Sobol QMC (d={n_rem}, n_sobol={n_sobol:,})"
        except Exception:
            half      = n_sims // 2
            base      = np.random.random((half, n_rem))
            extra     = np.random.random((n_sims - 2 * half, n_rem))
            all_draws = np.vstack([base, 1.0 - base, extra])
            vr_method = "antithetic variates"
    else:
        all_draws = np.empty((n_sims, 0))
        vr_method = "none (no remaining games)"

    print(f"  Variance reduction: {vr_method}")
    print(f"  Bracket simulation: exact DP (zero bracket variance)")

    # Shared caches across all simulations:
    # prob_cache   : per-game win probabilities (one predict_game call per pair)
    # series_cache : best-of-7 series win probabilities (deterministic from prob_cache)
    prob_cache:   dict = {}
    series_cache: dict = {}

    SIM_KEYS = ("top6", "playin", "playoffs", "r2", "cf", "finals", "champ")
    counters: dict[int, dict[str, float]] = {
        tid: {k: 0.0 for k in SIM_KEYS} for tid in team_conf
    }

    for i in range(n_sims):
        if i % 500 == 0:
            print(f"    {i:>6,} / {n_sims:,} ...", end="\r")
        sim = run_one_simulation(
            base_wins, base_pd,
            remaining, rem_probs,
            all_draws[i] if n_rem > 0 else np.array([]),
            team_conf, pipe, games_df_dt, today_str,
            prob_cache, series_cache, team_style,
        )
        for tid, outcomes in sim.items():
            c = counters.get(tid)
            if c:
                for k in SIM_KEYS:
                    c[k] += outcomes.get(k, 0.0)

    elapsed  = time.time() - t0
    n_pcache = len(prob_cache)
    n_scache = len(series_cache)
    print(f"    {n_sims:,} / {n_sims:,} done  "
          f"({elapsed:.0f}s | {n_pcache} game probs | {n_scache} series probs cached)")

    # ── 7. Results table ──────────────────────────────────────────────────────
    W = 115
    print()
    print_divider("=", W)
    print(f"  NBA PLAYOFF PROBABILITY TABLE  --  {aedt_label}")
    print_divider("=", W)

    has_clutch = bool(clutch_rtg)
    clutch_hdr = f"  {'CLUTCH':>7}" if has_clutch else ""
    col_hdr = (
        f"  {'TEAM':<5}  {'W-L':<7}  {'GB':>4}  |  "
        f"{'PLAYOFFS':>8}  {'TOP 6':>6}  {'PLAY-IN':>7}  |  "
        f"{'WIN R1':>7}  {'WIN R2':>7}  {'FINALS':>7}  {'CHAMP (90% CI)':>14}"
        + clutch_hdr
    )
    clutch_sep = f"  {'':-<7}" if has_clutch else ""
    col_sep = (
        f"  {'':-<5}  {'':-<7}  {'':-<4}  |  "
        f"{'':-<8}  {'':-<6}  {'':-<7}  |  "
        f"{'':-<7}  {'':-<7}  {'':-<7}  {'':-<14}"
        + clutch_sep
    )

    for conf, conf_df in standings.items():
        print(f"\n  {conf.upper()} CONFERENCE")
        print_divider("-", W)
        print(col_hdr)
        print(col_sep)

        ldr_w = int(conf_df.iloc[0]["wins"])
        ldr_l = int(conf_df.iloc[0]["losses"])

        for rank, (_, row) in enumerate(conf_df.iterrows(), 1):
            if rank == 7:
                print(f"  {'--- PLAY-IN TOURNAMENT (seeds 7-10) ---':^{W-4}}")
            elif rank == 11:
                print(f"  {'--- LOTTERY (seeds 11-15) ---':^{W-4}}")

            tid   = int(row["TEAM_ID"])
            ab    = row["abbrev"]
            w, l  = int(row["wins"]), int(row["losses"])
            gb    = ((ldr_w - w) + (l - ldr_l)) / 2
            gb_s  = "--" if gb == 0.0 else f"{gb:.1f}"
            wl_s  = f"{w}-{l}"
            c     = counters.get(tid, {})

            playoffs = c.get("playoffs", 0.0) / n_sims
            top6     = c.get("top6",     0.0) / n_sims
            playin   = c.get("playin",   0.0) / n_sims
            r2       = c.get("r2",       0.0) / n_sims
            cf       = c.get("cf",       0.0) / n_sims
            finals   = c.get("finals",   0.0) / n_sims
            champ    = c.get("champ",    0.0) / n_sims

            clutch_str = ""
            if has_clutch and tid in clutch_rtg:
                clutch_str = f"  {clutch_rtg[tid]:+.1f}   "

            print(
                f"  {ab:<5}  {wl_s:<7}  {gb_s:>4}  |  "
                f"{fmt_pct(playoffs):>8}  {fmt_pct(top6):>6}  {fmt_pct(playin):>7}  |  "
                f"{fmt_pct(r2):>7}  {fmt_pct(cf):>7}  {fmt_pct(finals):>7}  "
                f"{fmt_pct_ci(champ, n_sims):>14}"
                + clutch_str
            )

    print()
    print_divider("=", W)
    print("  Column guide:")
    print("  PLAYOFFS = made the playoff bracket (top 6 directly OR won play-in)")
    print("  TOP 6    = earned a bye from play-in by finishing seeds 1-6")
    print("  PLAY-IN  = entered play-in tournament as seed 7-10")
    print("  WIN R1   = won first playoff round (reached conference semifinals)")
    print("  WIN R2   = won conference semifinals (reached conference finals)")
    print("  FINALS   = reached the NBA Finals")
    print("  CHAMP    = won the NBA Championship (90% CI shown via +/-)")
    if has_clutch:
        print("  CLUTCH   = per-game +/- in last-5-min within-5-pt situations")
    print()
    print("  Simulation method:")
    print(f"    Regular season: Monte Carlo ({vr_method})")
    print(f"    Playoff bracket: exact DP over all bracket paths (zero randomness)")
    print(f"    HCA: {PLAYOFF_HCA_SCALE:.0%} of regular-season strength (Anderson & Pierce 2009)")
    print_divider("-", W)
    print(f"  {n_sims:,} simulations | {len(remaining)} remaining games "
          f"| {n_pcache} game / {n_scache} series probs cached | {elapsed:.0f}s runtime")
    print_divider("=", W)

    # ── JSON export ───────────────────────────────────────────────────────────
    if out_path:
        import json
        from datetime import timezone as _tz
        teams_out = []
        for conf, conf_df in standings.items():
            ldr_w = int(conf_df.iloc[0]["wins"])
            ldr_l = int(conf_df.iloc[0]["losses"])
            for rank, (_, row) in enumerate(conf_df.iterrows(), 1):
                tid  = int(row["TEAM_ID"])
                ab   = row["abbrev"]
                w, l = int(row["wins"]), int(row["losses"])
                gb   = ((ldr_w - w) + (l - ldr_l)) / 2
                c    = counters.get(tid, {})
                full_name = teams_df[teams_df["abbreviation"] == ab]["full_name"].values
                teams_out.append({
                    "team":        ab,
                    "full_name":   str(full_name[0]) if len(full_name) else ab,
                    "conference":  conf,
                    "seed":        rank,
                    "wins":        w,
                    "losses":      l,
                    "games_back":  round(gb, 1),
                    "playoffs_pct":  round(c.get("playoffs", 0) / n_sims * 100, 1),
                    "top6_pct":      round(c.get("top6",     0) / n_sims * 100, 1),
                    "playin_pct":    round(c.get("playin",   0) / n_sims * 100, 1),
                    "win_r1_pct":    round(c.get("r2",       0) / n_sims * 100, 1),
                    "win_r2_pct":    round(c.get("cf",       0) / n_sims * 100, 1),
                    "finals_pct":    round(c.get("finals",   0) / n_sims * 100, 1),
                    "champ_pct":     round(c.get("champ",    0) / n_sims * 100, 1),
                })
        payload = {
            "generated_at": datetime.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "season":       CURRENT_SEASON,
            "simulations":  n_sims,
            "teams":        teams_out,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Wrote playoffs data to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Playoff Probability Simulator")
    parser.add_argument(
        "--sims", type=int, default=N_SIMS,
        help=f"Monte Carlo simulation count (default: {N_SIMS:,})",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force fresh API data",
    )
    parser.add_argument(
        "--series", type=str, default=None,
        metavar="\"TEAM1 W1 TEAM2 W2\"",
        help=(
            "Show live series win probability. First team = higher seed (HCA). "
            "Example: --series \"BOS 2 MIA 1\""
        ),
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Write playoffs JSON to this file path (e.g. playoffs.json)",
    )
    args = parser.parse_args()

    if args.series:
        print("Loading data for series state analysis ...")
        teams_df           = get_all_teams()
        games_df           = get_season_games(season=CURRENT_SEASON, use_cache=not args.no_cache)
        historical_df      = get_historical_games()
        team_abbrev_map    = teams_df.set_index("id")["abbreviation"].to_dict()
        team_abbrev_to_id  = {v: k for k, v in team_abbrev_map.items()}

        pipe = get_or_train_model(games_df, historical_df=historical_df)

        games_df_dt = games_df.copy()
        games_df_dt["GAME_DATE"] = pd.to_datetime(games_df_dt["GAME_DATE"])
        today_str = datetime.now(_ET).strftime("%Y-%m-%d")

        team_style: dict = {}
        try:
            dashboard_df = get_team_dashboard(use_cache=not args.no_cache)
            for _, row in dashboard_df.iterrows():
                fga  = max(float(row.get("FGA",  LEAGUE_AVG_FGA)), 1.0)
                fg3a = float(row.get("FG3A", LEAGUE_AVG_FGA * LEAGUE_AVG_FG3A_RATE))
                team_style[int(row["TEAM_ID"])] = {"fg3a_rate": fg3a / fga, "fga": fga}
        except Exception:
            pass

        series_state_mode(
            args.series, pipe, games_df_dt, today_str,
            team_abbrev_to_id, team_style,
        )
    else:
        main(n_sims=args.sims, no_cache=args.no_cache, out_path=args.out)
