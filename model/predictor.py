"""
Win probability model for NBA games.

Approach:
  - Builds rolling features for every historical game (no look-ahead bias)
  - Trains XGBRegressor on home score margin (not binary win/loss)
  - Converts predicted margin to win probability via the normal CDF:
      win_prob = Phi(margin / sigma)   [Stern 1994]
  - sigma = std dev of model residuals (unexplained game variance)
  - Caches (model, sigma) to disk; retrain by deleting model/nba_model.pkl

Modelling margin rather than outcomes uses all information in the score
(a 20-pt blowout and a 1-pt win are weighted accordingly) and produces
better-calibrated probabilities without needing Platt scaling.
"""

import math
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

MODEL_PATH = Path(__file__).parent / "nba_model.pkl"


def _norm_cdf(x: float) -> float:
    """Standard normal CDF — no scipy dependency."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _precompute_sos_data(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Pre-compute two O(1) lookups for Strength-of-Schedule calculation.

    Returns:
      wpct_before : {(team_id, game_id): win% entering that game}
      opp_map     : {(team_id, game_id): opponent_team_id in that game}

    Used inside build_training_data() to avoid O(n^2) nested scans.
    """
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WL_BIN"]  = (df["WL"] == "W").astype(int)
    df["IS_HOME"] = df["MATCHUP"].str.contains(r"vs\.")

    # Running win% BEFORE each game (cumsum then shift so current game not included)
    df_s = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    df_s["_g_before"] = df_s.groupby("TEAM_ID").cumcount()          # 0 for first game
    df_s["_w_before"] = df_s.groupby("TEAM_ID")["WL_BIN"].cumsum() - df_s["WL_BIN"]
    df_s["_wpct_bef"]  = (df_s["_w_before"] / df_s["_g_before"].clip(lower=1)).fillna(0.5)
    df_s.loc[df_s["_g_before"] == 0, "_wpct_bef"] = 0.5

    wpct_before = df_s.set_index(["TEAM_ID", "GAME_ID"])["_wpct_bef"].to_dict()

    # Opponent per game
    home_df = df[df["IS_HOME"]][["GAME_ID", "TEAM_ID"]].rename(columns={"TEAM_ID": "HOME_ID"})
    away_df = df[~df["IS_HOME"]][["GAME_ID", "TEAM_ID"]].rename(columns={"TEAM_ID": "AWAY_ID"})
    pairs   = home_df.merge(away_df, on="GAME_ID")
    opp_map = {}
    for _, r in pairs.iterrows():
        opp_map[(r["HOME_ID"], r["GAME_ID"])] = r["AWAY_ID"]
        opp_map[(r["AWAY_ID"], r["GAME_ID"])] = r["HOME_ID"]

    return wpct_before, opp_map


def _compute_sos(
    df: pd.DataFrame,
    team_id: int,
    before_date: pd.Timestamp,
) -> float:
    """
    Average win% of each opponent at the time they were played (no look-ahead bias).
    Used in predict_game() where the pre-computed lookup isn't available.
    Falls back to 0.5 if fewer than 5 prior games exist.
    """
    prior = df[(df["TEAM_ID"] == team_id) & (df["GAME_DATE"] < before_date)]
    if len(prior) < 5:
        return 0.5

    prior_gids = set(prior["GAME_ID"])
    opp_rows   = df[df["GAME_ID"].isin(prior_gids) & (df["TEAM_ID"] != team_id)]
    if opp_rows.empty:
        return 0.5

    sos_vals = []
    for _, orow in opp_rows.iterrows():
        opp_hist = df[(df["TEAM_ID"] == orow["TEAM_ID"]) & (df["GAME_DATE"] < orow["GAME_DATE"])]
        sos_vals.append(float((opp_hist["WL"] == "W").mean()) if len(opp_hist) >= 5 else 0.5)

    return sum(sos_vals) / len(sos_vals)

FEATURE_COLS = [
    # ── Basic (rolling) ───────────────────────────────────────────────────────
    "WIN_PCT_DIFF",        # home season win% minus away season win%
    "PYTH_WIN_PCT_DIFF",   # Pythagorean win% diff (PTS^16.5 / (PTS^16.5 + OPP^16.5)) — luck-filtered quality
    "POINT_DIFF_DIFF",     # home avg point diff minus away avg point diff
    "HOME_WIN_PCT",        # home team's win% in home games specifically
    "AWAY_WIN_PCT_OPP",    # away team's win% in away games specifically
    "FORM_WIN_PCT_DIFF",   # home last-10 win% minus away last-10 win%
    "FORM_DIFF_DIFF",      # home last-10 point diff minus away last-10 point diff
    "REST_ADVANTAGE",      # home rest days minus away rest days
    # ── Strength of Schedule ─────────────────────────────────────────────────
    "HOME_SOS",            # home team's avg opponent win% at time of each matchup
    "AWAY_SOS",            # away team's avg opponent win% at time of each matchup
    # ── Advanced efficiency (computed from game log, no look-ahead bias) ──────
    "TS_PCT_DIFF",         # True Shooting % diff: PTS / (2*(FGA + 0.44*FTA))
    "EFG_PCT_DIFF",        # Effective FG% diff: (FGM + 0.5*FG3M) / FGA
    "TOV_RATE_DIFF",       # Turnover rate diff (higher = more turnovers = worse)
    "OFF_RTG_DIFF",        # Offensive rating proxy diff (pts per 100 poss)
    "DEF_RTG_DIFF",        # Defensive rating proxy diff (pts allowed per 100 poss)
    "NET_RTG_DIFF",        # Net rating diff (off - def per 100 poss); best single team quality signal
    "PACE_DIFF",           # Pace diff (estimated possessions per game)
    # ── Recent form — advanced ────────────────────────────────────────────────
    "FORM_TS_PCT_DIFF",    # Last-N True Shooting % diff
    "FORM_TOV_DIFF",       # Last-N turnover rate diff
    "FORM_OFF_RTG_DIFF",   # Last-N offensive rating diff
    # ── Tier 3: schedule density, venue-split form, H2H ──────────────────────
    "SCHED_DENSITY_DIFF",  # games played in last 7 days (home minus away)
    "HOME_FORM_HOME_WIN",  # home team's recent home-game win% (last N games at home)
    "AWAY_FORM_ROAD_WIN",  # away team's recent road-game win% (last N games away)
    "H2H_HOME_WIN_PCT",    # home team's win% vs this opponent (last 3 seasons)
    # ── Fatigue / back-to-back ────────────────────────────────────────────────
    "HOME_IS_B2B",         # 1 if home team played yesterday (back-to-back)
    "AWAY_IS_B2B",         # 1 if away team played yesterday (back-to-back)
]


# ── Rolling feature builder ───────────────────────────────────────────────────

def _team_rolling_stats(df: pd.DataFrame, team_id: int, before_date: pd.Timestamp, n_form: int = 10) -> dict | None:
    """
    Compute a team's rolling stats from all games strictly before before_date.
    Returns None if fewer than 5 prior games (not enough signal).

    Full-season features use an exponentially weighted mean (EWM, span=30) so
    that recent games are weighted more heavily than early-season games.  A game
    from ~20 games ago has roughly 50% the weight of last night's game.  This
    prevents a good October from masking a terrible February — the root cause of
    tanking/collapsing teams appearing overrated in a simple-mean model.

    Short-window features (last N games) are kept as simple means — they are
    already a narrow recent window and don't need additional weighting.

    Advanced metrics (per game, then EWM-weighted):
      TS%     = PTS / (2*(FGA + 0.44*FTA))
      eFG%    = (FGM + 0.5*FG3M) / FGA
      TOV%    = TOV / (FGA + 0.44*FTA + TOV)
      POSS    = FGA + 0.44*FTA + TOV - OREB  (Hollinger)
      OFF_RTG = PTS / POSS * 100
      DEF_RTG = (PTS - PLUS_MINUS) / POSS * 100
    """
    prior = df[(df["TEAM_ID"] == team_id) & (df["GAME_DATE"] < before_date)].copy()
    if len(prior) < 5:
        return None

    prior = prior.sort_values("GAME_DATE", ascending=False)
    prior["IS_HOME"] = prior["MATCHUP"].str.contains(r"vs\.")
    prior["WL_BIN"]  = (prior["WL"] == "W").astype(int)

    # Recent-window slices (descending order, so .head() = most recent N games)
    recent      = prior.head(n_form)
    recent_home = recent[recent["IS_HOME"]]
    recent_away = recent[~recent["IS_HOME"]]

    # ── Helpers for recent-window advanced stats (simple means, short window) ─
    def _ts(g):
        pts = g["PTS"].sum()
        d   = 2 * (g["FGA"].sum() + 0.44 * g["FTA"].sum())
        return pts / d if d > 0 else 0.46

    def _tov_rate(g):
        tov = g["TOV"].sum()
        d   = g["FGA"].sum() + 0.44 * g["FTA"].sum() + tov
        return tov / d if d > 0 else 0.13

    def _poss_pg(g):
        return (g["FGA"] + 0.44 * g["FTA"] + g["TOV"] - g["OREB"]).mean()

    def _off_rtg(g):
        pos = _poss_pg(g)
        return g["PTS"].mean() / pos * 100 if pos > 0 else 110.0

    # ── EWM-weighted full-season stats ────────────────────────────────────────
    # Ascending order (oldest first) so ewm().mean().iloc[-1] gives highest
    # weight to the most recent game.  span=30 → half-life ≈ 20 games.
    prior_asc = prior.iloc[::-1].copy()

    # Per-game efficiency metrics (vectorized, then EWM-weighted)
    _denom_ts  = 2 * (prior_asc["FGA"] + 0.44 * prior_asc["FTA"])
    _denom_tov = prior_asc["FGA"] + 0.44 * prior_asc["FTA"] + prior_asc["TOV"]
    _poss      = (prior_asc["FGA"] + 0.44 * prior_asc["FTA"]
                  + prior_asc["TOV"] - prior_asc["OREB"])

    prior_asc["_ts_g"]  = np.where(_denom_ts > 0,
                                    prior_asc["PTS"] / _denom_ts, 0.46)
    prior_asc["_efg_g"] = np.where(prior_asc["FGA"] > 0,
                                    (prior_asc["FGM"] + 0.5 * prior_asc["FG3M"])
                                    / prior_asc["FGA"], 0.52)
    prior_asc["_tov_g"] = np.where(_denom_tov > 0,
                                    prior_asc["TOV"] / _denom_tov, 0.13)
    prior_asc["_poss_g"]    = _poss.clip(lower=1)
    prior_asc["_off_rtg_g"] = np.where(_poss > 0,
                                         prior_asc["PTS"] / _poss * 100, 110.0)
    prior_asc["_def_rtg_g"] = np.where(_poss > 0,
                                         (prior_asc["PTS"] - prior_asc["PLUS_MINUS"])
                                         / _poss * 100, 110.0)
    prior_asc["_net_rtg_g"] = prior_asc["_off_rtg_g"] - prior_asc["_def_rtg_g"]

    # Pythagorean win% per game: PTS^16.5 / (PTS^16.5 + OPP_PTS^16.5)
    # Filters out clutch/luck in close games — Morey's basketball exponent = 16.5
    _opp_pts = (prior_asc["PTS"] - prior_asc["PLUS_MINUS"]).clip(lower=1)
    _pts_exp  = prior_asc["PTS"].clip(lower=1) ** 16.5
    _opp_exp  = _opp_pts ** 16.5
    prior_asc["_pyth_g"] = _pts_exp / (_pts_exp + _opp_exp)

    _ewm = lambda s: float(s.ewm(span=30, adjust=True).mean().iloc[-1])

    home_asc = prior_asc[prior_asc["IS_HOME"]]
    away_asc = prior_asc[~prior_asc["IS_HOME"]]

    return {
        # ── Basic (EWM over full season — recency-biased) ─────────────────────
        "win_pct":    _ewm(prior_asc["WL_BIN"]),
        "point_diff": _ewm(prior_asc["PLUS_MINUS"]),
        "home_win_pct": (_ewm(home_asc["WL_BIN"]) if len(home_asc) > 0
                         else _ewm(prior_asc["WL_BIN"])),
        "away_win_pct": (_ewm(away_asc["WL_BIN"]) if len(away_asc) > 0
                         else _ewm(prior_asc["WL_BIN"])),
        "form_win_pct":   (recent["WL"] == "W").mean(),
        "form_diff":      recent["PLUS_MINUS"].mean(),
        "last_game_date": prior.iloc[0]["GAME_DATE"],
        "gp":             len(prior),
        # ── Advanced efficiency (EWM per-game metrics) ────────────────────────
        "ts_pct":    _ewm(prior_asc["_ts_g"]),
        "efg_pct":   _ewm(prior_asc["_efg_g"]),
        "tov_rate":  _ewm(prior_asc["_tov_g"]),
        "off_rtg":      _ewm(prior_asc["_off_rtg_g"]),
        "def_rtg":      _ewm(prior_asc["_def_rtg_g"]),
        "net_rtg":      _ewm(prior_asc["_net_rtg_g"]),
        "pyth_win_pct": _ewm(prior_asc["_pyth_g"]),
        "pace":         _ewm(prior_asc["_poss_g"]),
        # ── Recent form (simple mean of last N — already short window) ────────
        "form_ts_pct":    _ts(recent),
        "form_tov_rate":  _tov_rate(recent),
        "form_off_rtg":   _off_rtg(recent),
        # ── Tier 3 ────────────────────────────────────────────────────────────
        "schedule_density":  int((prior["GAME_DATE"] >= (before_date - pd.Timedelta(days=7))).sum()),
        "form_home_win_pct": (recent_home["WL_BIN"].mean() if len(recent_home) >= 3
                              else _ewm(prior_asc["WL_BIN"])),
        "form_away_win_pct": (recent_away["WL_BIN"].mean() if len(recent_away) >= 3
                              else _ewm(prior_asc["WL_BIN"])),
    }


def _h2h_win_pct(df: pd.DataFrame, home_id: int, away_id: int, before_date: pd.Timestamp) -> float:
    """
    Return the home team's win% in head-to-head matchups vs the away team,
    using only games strictly before before_date and within the last ~3 seasons
    (~1095 days). Falls back to 0.5 if fewer than 2 H2H games found.
    """
    cutoff_back = before_date - pd.Timedelta(days=1095)
    home_ids = set(df[
        (df["TEAM_ID"] == home_id) &
        (df["GAME_DATE"] < before_date) &
        (df["GAME_DATE"] >= cutoff_back)
    ]["GAME_ID"])
    away_ids = set(df[
        (df["TEAM_ID"] == away_id) &
        (df["GAME_DATE"] < before_date) &
        (df["GAME_DATE"] >= cutoff_back)
    ]["GAME_ID"])
    h2h_ids = home_ids & away_ids
    if len(h2h_ids) < 2:
        return 0.5  # insufficient history — neutral prior
    h2h_home = df[(df["TEAM_ID"] == home_id) & (df["GAME_ID"].isin(h2h_ids))]
    wins = (h2h_home["WL"] == "W").sum()
    return round(wins / len(h2h_home), 4)


def build_training_data(games_df: pd.DataFrame, n_form: int = 10) -> pd.DataFrame:
    """
    For every completed game in games_df, build a feature vector
    using only information available BEFORE that game was played.

    Returns a DataFrame with FEATURE_COLS + 'HOME_MARGIN' target column.
    Target is home team PLUS_MINUS (positive = home won by that many points).
    Training on margin instead of binary win/loss uses all score information.
    """
    df = games_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"] = df["MATCHUP"].str.contains(r"vs\.")
    df["WL_BIN"] = (df["WL"] == "W").astype(int)

    home_df = df[df["IS_HOME"]].copy()
    away_df = df[~df["IS_HOME"]].copy()

    # Pair home and away into one row per game
    paired = home_df.merge(
        away_df[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "WL_BIN"]],
        on="GAME_ID",
        suffixes=("_H", "_A"),
    ).rename(columns={
        "TEAM_ID_H":           "HOME_TEAM_ID",
        "TEAM_ID_A":           "AWAY_TEAM_ID",
        "WL_BIN_H":            "HOME_WIN",
        "TEAM_ABBREVIATION_H": "HOME_ABBREV",
        "TEAM_ABBREVIATION_A": "AWAY_ABBREV",
    })
    paired = paired.sort_values("GAME_DATE").reset_index(drop=True)

    # Pre-compute SOS lookups (O(1) dict access in the loop below)
    wpct_before, opp_map = _precompute_sos_data(df)

    def _sos_for(team_id: int, prior_game_ids: list) -> float:
        vals = [
            wpct_before.get((opp_map.get((team_id, gid), -1), gid), 0.5)
            for gid in prior_game_ids
        ]
        return sum(vals) / len(vals) if vals else 0.5

    rows  = []
    total = len(paired)
    print(f"  Building rolling features for {total} games ...")

    for i, game in paired.iterrows():
        if i % 100 == 0:
            print(f"    {i}/{total} ...", end="\r")

        gdate   = game["GAME_DATE"]
        home_id = game["HOME_TEAM_ID"]
        away_id = game["AWAY_TEAM_ID"]

        h = _team_rolling_stats(df, home_id, gdate, n_form)
        a = _team_rolling_stats(df, away_id, gdate, n_form)

        if h is None or a is None:
            continue

        home_rest = (gdate - h["last_game_date"]).days - 1
        away_rest = (gdate - a["last_game_date"]).days - 1

        # SOS: prior game IDs for each team (for O(1) lookup via pre-built dicts)
        home_prior_gids = df[
            (df["TEAM_ID"] == home_id) & (df["GAME_DATE"] < gdate)
        ]["GAME_ID"].tolist()
        away_prior_gids = df[
            (df["TEAM_ID"] == away_id) & (df["GAME_DATE"] < gdate)
        ]["GAME_ID"].tolist()

        rows.append({
            "GAME_ID":      game["GAME_ID"],
            "GAME_DATE":    gdate,
            "HOME_ABBREV":  game["HOME_ABBREV"],
            "AWAY_ABBREV":  game["AWAY_ABBREV"],
            "HOME_TEAM_ID": home_id,
            "AWAY_TEAM_ID": away_id,
            # ── basic features ──────────────────────────────────────────────
            "WIN_PCT_DIFF":       h["win_pct"]         - a["win_pct"],
            "PYTH_WIN_PCT_DIFF":  h["pyth_win_pct"]    - a["pyth_win_pct"],
            "POINT_DIFF_DIFF":    h["point_diff"]       - a["point_diff"],
            "HOME_WIN_PCT":       h["home_win_pct"],
            "AWAY_WIN_PCT_OPP":   a["away_win_pct"],
            "FORM_WIN_PCT_DIFF":  h["form_win_pct"]    - a["form_win_pct"],
            "FORM_DIFF_DIFF":     h["form_diff"]        - a["form_diff"],
            "REST_ADVANTAGE":     home_rest             - away_rest,
            # ── SOS ─────────────────────────────────────────────────────────
            "HOME_SOS":           _sos_for(home_id, home_prior_gids),
            "AWAY_SOS":           _sos_for(away_id, away_prior_gids),
            # ── advanced efficiency features ─────────────────────────────────
            "TS_PCT_DIFF":        h["ts_pct"]           - a["ts_pct"],
            "EFG_PCT_DIFF":       h["efg_pct"]          - a["efg_pct"],
            "TOV_RATE_DIFF":      h["tov_rate"]         - a["tov_rate"],
            "OFF_RTG_DIFF":       h["off_rtg"]          - a["off_rtg"],
            "DEF_RTG_DIFF":       h["def_rtg"]          - a["def_rtg"],
            "NET_RTG_DIFF":       h["net_rtg"]          - a["net_rtg"],
            "PACE_DIFF":          h["pace"]              - a["pace"],
            # ── recent form advanced ─────────────────────────────────────────
            "FORM_TS_PCT_DIFF":   h["form_ts_pct"]      - a["form_ts_pct"],
            "FORM_TOV_DIFF":      h["form_tov_rate"]    - a["form_tov_rate"],
            "FORM_OFF_RTG_DIFF":  h["form_off_rtg"]     - a["form_off_rtg"],
            # ── Tier 3 ───────────────────────────────────────────────────────
            "SCHED_DENSITY_DIFF": h["schedule_density"] - a["schedule_density"],
            "HOME_FORM_HOME_WIN": h["form_home_win_pct"],
            "AWAY_FORM_ROAD_WIN": a["form_away_win_pct"],
            "H2H_HOME_WIN_PCT":   _h2h_win_pct(df, home_id, away_id, gdate),
            # ── Fatigue ──────────────────────────────────────────────────────
            "HOME_IS_B2B":        int(home_rest == 0),
            "AWAY_IS_B2B":        int(away_rest == 0),
            # ── target: home score margin (positive = home won) ───────────────
            "HOME_MARGIN":        float(game.get("PLUS_MINUS", 0)),
        })

    print(f"    {total}/{total} done.      ")
    return pd.DataFrame(rows)


# ── Model training ────────────────────────────────────────────────────────────

def train_model(training_df: pd.DataFrame) -> tuple:
    """
    Fit an XGBRegressor on home score margin, return (model, sigma).

    sigma is the std dev of out-of-sample residuals used to convert
    predicted margin to win probability via the normal CDF.
    Sample weights down-weight pre-COVID seasons (inflated HCA).
    """
    X = training_df[FEATURE_COLS].fillna(0)
    y = training_df["HOME_MARGIN"]        # continuous score margin, not binary

    # Sample weights: down-weight pre-COVID seasons where HCA was inflated.
    # 2020-21 bubble (no fans) and 2021-22 (partial) produced anomalously low HCA;
    # 2019-20 and earlier had higher HCA than current norms (~2.5-3 pts).
    # Giving more weight to 2022+ aligns training with the current HCA regime.
    game_years = pd.to_datetime(training_df["GAME_DATE"]).dt.year
    sample_weights = np.where(
        game_years >= 2022, 2.0,           # current regime — highest weight
        np.where(game_years >= 2021, 1.0,  # transition year
                 0.6)                       # pre-COVID inflated HCA
    )

    base_reg = XGBRegressor(
        n_estimators=500,
        max_depth=4,          # shallow trees — 30 teams, prevent overfitting
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        verbosity=0,
    )

    # 5-fold CV to measure generalisation (MAE on margin, and derived accuracy)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae, cv_acc = [], []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_tr        = sample_weights[train_idx]
        base_reg.fit(X_tr, y_tr, sample_weight=w_tr)
        preds     = base_reg.predict(X_val)
        cv_mae.append(float(np.mean(np.abs(preds - y_val))))
        # Derive accuracy from sign of predicted margin
        cv_acc.append(float(np.mean(np.sign(preds) == np.sign(y_val))))

    home_win_rate = (y > 0).mean()
    print(f"  CV MAE (margin) : {np.mean(cv_mae):.2f} pts +/- {np.std(cv_mae):.2f}")
    print(f"  CV Accuracy     : {np.mean(cv_acc):.3f} +/- {np.std(cv_acc):.3f}")
    print(f"  Baseline (home) : {home_win_rate:.3f}")

    # Fit final model on full dataset with sample weights
    print("  Fitting final model on full training set ...")
    base_reg.fit(X, y, sample_weight=sample_weights)

    # Sigma = std of residuals (unexplained variance per game).
    # Used in the CDF conversion: win_prob = Phi(margin / sigma).
    # A lower sigma means the model explains more variance and gives
    # more confident probability estimates.
    residuals = y.values - base_reg.predict(X)
    sigma     = float(np.std(residuals))
    print(f"  Residual sigma  : {sigma:.2f} pts  "
          f"(NBA game-to-game std ~ 11-12 pts; lower = model explains more)")

    print("\n  Feature importances (XGBoost gain, higher = more predictive):")
    importances = base_reg.feature_importances_
    for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True):
        bar = "#" * min(int(imp * 600), 30)
        print(f"    {feat:<24} {imp:.4f}  {bar}")

    return base_reg, sigma


def save_model(model_tuple: tuple) -> None:
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_tuple, f)
    print(f"  Model saved to {MODEL_PATH}")


def load_model() -> tuple | None:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
        # Validate format — old CalibratedClassifierCV pkl won't be a 2-tuple
        if isinstance(obj, tuple) and len(obj) == 2:
            return obj
        print("  Old model format detected — retraining with new architecture.")
    return None


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_game(
    pipe: tuple,
    games_df: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    game_date: str,
    n_form: int = 10,
) -> dict:
    """
    Predict the win probability for an upcoming game.
    Uses season-to-date stats for both teams as of game_date.

    pipe is a (XGBRegressor, sigma) tuple from train_model().
    Probability derived via: win_prob = Phi(predicted_margin / sigma)

    Returns:
      {
        'home_win_prob': float,      # probability home team wins (0-1)
        'away_win_prob': float,
        'predicted_margin': float,   # raw model output (positive = home favoured)
        'features': dict,            # raw feature values
      }
    """
    model, sigma = pipe

    df = games_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    cutoff = pd.to_datetime(game_date)

    h = _team_rolling_stats(df, home_team_id, cutoff, n_form)
    a = _team_rolling_stats(df, away_team_id, cutoff, n_form)

    if h is None or a is None:
        return {"home_win_prob": 0.57, "away_win_prob": 0.43,
                "predicted_margin": 0.0, "features": {}, "note": "insufficient history"}

    home_rest = (cutoff - h["last_game_date"]).days - 1
    away_rest = (cutoff - a["last_game_date"]).days - 1

    # SOS: use direct function (pre-built lookup not available here)
    home_sos = _compute_sos(df, home_team_id, cutoff)
    away_sos = _compute_sos(df, away_team_id, cutoff)

    features = {
        # ── basic ──────────────────────────────────────────────────────────
        "WIN_PCT_DIFF":       h["win_pct"]          - a["win_pct"],
        "PYTH_WIN_PCT_DIFF":  h["pyth_win_pct"]     - a["pyth_win_pct"],
        "POINT_DIFF_DIFF":    h["point_diff"]        - a["point_diff"],
        "HOME_WIN_PCT":       h["home_win_pct"],
        "AWAY_WIN_PCT_OPP":   a["away_win_pct"],
        "FORM_WIN_PCT_DIFF":  h["form_win_pct"]     - a["form_win_pct"],
        "FORM_DIFF_DIFF":     h["form_diff"]         - a["form_diff"],
        "REST_ADVANTAGE":     home_rest              - away_rest,
        # ── SOS ─────────────────────────────────────────────────────────────
        "HOME_SOS":           home_sos,
        "AWAY_SOS":           away_sos,
        # ── advanced efficiency ─────────────────────────────────────────────
        "TS_PCT_DIFF":        h["ts_pct"]            - a["ts_pct"],
        "EFG_PCT_DIFF":       h["efg_pct"]           - a["efg_pct"],
        "TOV_RATE_DIFF":      h["tov_rate"]          - a["tov_rate"],
        "OFF_RTG_DIFF":       h["off_rtg"]           - a["off_rtg"],
        "DEF_RTG_DIFF":       h["def_rtg"]           - a["def_rtg"],
        "NET_RTG_DIFF":       h["net_rtg"]           - a["net_rtg"],
        "PACE_DIFF":          h["pace"]              - a["pace"],
        # ── recent form advanced ────────────────────────────────────────────
        "FORM_TS_PCT_DIFF":   h["form_ts_pct"]       - a["form_ts_pct"],
        "FORM_TOV_DIFF":      h["form_tov_rate"]     - a["form_tov_rate"],
        "FORM_OFF_RTG_DIFF":  h["form_off_rtg"]      - a["form_off_rtg"],
        # ── Tier 3 ─────────────────────────────────────────────────────────
        "SCHED_DENSITY_DIFF": h["schedule_density"]  - a["schedule_density"],
        "HOME_FORM_HOME_WIN": h["form_home_win_pct"],
        "AWAY_FORM_ROAD_WIN": a["form_away_win_pct"],
        "H2H_HOME_WIN_PCT":   _h2h_win_pct(df, home_team_id, away_team_id, cutoff),
        # ── Fatigue ─────────────────────────────────────────────────────────
        "HOME_IS_B2B":        int(home_rest == 0),
        "AWAY_IS_B2B":        int(away_rest == 0),
    }

    X = pd.DataFrame([features])[FEATURE_COLS].fillna(0)
    raw_margin = float(model.predict(X)[0])

    # Convert margin to probability via standard normal CDF (Stern 1994).
    # Clamp to avoid numerical extremes at > ±4 sigma.
    home_prob = float(np.clip(_norm_cdf(raw_margin / max(sigma, 1.0)), 0.01, 0.99))

    return {
        "home_win_prob":    round(home_prob, 4),
        "away_win_prob":    round(1.0 - home_prob, 4),
        "predicted_margin": round(raw_margin, 2),
        "home_rest_days":   home_rest,
        "away_rest_days":   away_rest,
        "features":         features,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def _game_cache_mtime() -> float:
    """Return the modification time of the season game cache file, or 0 if missing."""
    cache_file = Path(__file__).parent.parent / "data" / "cache" / "season_games_2025_26.json"
    return cache_file.stat().st_mtime if cache_file.exists() else 0.0


def get_or_train_model(
    games_df: pd.DataFrame,
    force_retrain: bool = False,
    historical_df: pd.DataFrame | None = None,
) -> tuple:
    """
    Load cached model if available and up to date, otherwise train a new one.

    Auto-retrains whenever the current season game cache is newer than the
    saved model (i.e. new game results came in since the last training run).

    historical_df: combined game log from prior completed seasons.  When
    provided, it is concatenated with games_df before training to give the
    model 4-5× more examples and significantly improve generalisation.
    Historical seasons are cached with a 1-year TTL so the extra API fetch
    only happens once.
    """
    if not force_retrain and MODEL_PATH.exists():
        model_mtime = MODEL_PATH.stat().st_mtime
        cache_mtime = _game_cache_mtime()
        if cache_mtime <= model_mtime:
            pipe = load_model()
            if pipe is not None:
                print("  Loaded existing model (up to date).")
                return pipe
        else:
            print("  New game results detected — retraining model ...")

    # Combine historical + current season for richer training set
    if historical_df is not None and not historical_df.empty:
        n_hist  = historical_df["GAME_ID"].nunique()
        n_curr  = games_df["GAME_ID"].nunique()
        combined = pd.concat([historical_df, games_df], ignore_index=True)
        print(f"  Training corpus: {n_hist} historical + {n_curr} current = "
              f"{combined['GAME_ID'].nunique()} unique games")
    else:
        combined = games_df
        print("  Training corpus: current season only (no historical data passed)")

    print("  Training new model ...")
    training_df = build_training_data(combined)
    print(f"  Training samples: {len(training_df)}")
    pipe = train_model(training_df)
    save_model(pipe)
    return pipe
