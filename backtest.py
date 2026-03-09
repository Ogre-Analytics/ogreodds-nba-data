"""
Backtest — evaluates model performance against historical 2025-26 results.

Method:
  1. Rebuild rolling features for every completed game (no look-ahead bias)
  2. Chronological split: train on first half, test on second half
     (simulates real-world use — you always predict games you haven't seen)
  3. Report accuracy, calibration, and simulated P&L at various confidence thresholds

P&L simulation:
  - Flat $100 bets at -110 standard line (break-even = 52.38%)
  - Also reports results at higher confidence cutoffs to find optimal threshold

Usage:
    python backtest.py
    python backtest.py --all    # evaluate on all games via 5-fold CV instead
"""

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from data.fetcher import get_season_games, CURRENT_SEASON
from model.predictor import build_training_data, FEATURE_COLS


# ── Config ────────────────────────────────────────────────────────────────────

STAKE        = 100      # dollars per bet
JUICE        = -110     # standard moneyline odds assumed where no historical line exists
JUICE_PAYOUT = 100/110  # profit per $1 wagered at -110


def print_divider(char="-", width=72):
    print(char * width)

def fmt_pct(v):
    return f"{v*100:.1f}%"

def fmt_dollar(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:.2f}"

def fmt_roi(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}{v*100:.1f}%"


# ── P&L simulator ─────────────────────────────────────────────────────────────

def simulate_pnl(df: pd.DataFrame, min_confidence: float = 0.50) -> dict:
    """
    Simulate flat betting: bet $STAKE on whichever team the model favours
    when model confidence >= min_confidence.

    Uses -110 standard juice as a proxy for historical lines.
    Returns dict with summary stats.
    """
    bets = df[df["CONFIDENCE"] >= min_confidence].copy()
    if bets.empty:
        return {"bets": 0, "wins": 0, "pnl": 0, "roi": 0, "win_pct": 0}

    wins  = bets["CORRECT"].sum()
    total = len(bets)
    pnl   = wins * STAKE * JUICE_PAYOUT - (total - wins) * STAKE
    roi   = pnl / (total * STAKE)

    return {
        "bets":     total,
        "wins":     int(wins),
        "win_pct":  wins / total,
        "pnl":      pnl,
        "roi":      roi,
    }


# ── Calibration ───────────────────────────────────────────────────────────────

def calibration_table(probs: np.ndarray, actuals: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """
    Group predictions into probability buckets and compare to actual win rates.
    Well-calibrated model: predicted % ≈ actual win %.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        mean_pred   = probs[mask].mean()
        actual_rate = actuals[mask].mean()
        rows.append({
            "Prob bucket":  f"{lo*100:.0f}-{hi*100:.0f}%",
            "Bets":         int(mask.sum()),
            "Model avg":    f"{mean_pred*100:.1f}%",
            "Actual wins":  f"{actual_rate*100:.1f}%",
            "Gap":          f"{(actual_rate - mean_pred)*100:+.1f}%",
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(use_cv: bool = False) -> None:
    print("=" * 72)
    print("  NBA Model Backtest  |  Season 2025-26")
    print("=" * 72)

    # ── 1. Load data & build rolling features ─────────────────────────────────
    print("\n[ 1 ] Loading season game log ...")
    games_df = get_season_games(season=CURRENT_SEASON)
    n_games  = games_df["GAME_ID"].nunique()
    print(f"  {n_games} unique games loaded")

    print("\n[ 2 ] Building rolling features (no look-ahead) ...")
    train_df = build_training_data(games_df)
    train_df = train_df.sort_values("GAME_DATE").reset_index(drop=True)
    print(f"  {len(train_df)} games with sufficient prior history")

    X = train_df[FEATURE_COLS].fillna(0).values
    y = train_df["HOME_WIN"].values

    # ── 2a. Method: chronological split ──────────────────────────────────────
    if not use_cv:
        split = len(train_df) // 2
        split_date = train_df.iloc[split]["GAME_DATE"].strftime("%Y-%m-%d")
        print(f"\n[ 3 ] Chronological split at game {split} (date: {split_date})")
        print(f"  Train: {split} games   |   Test: {len(train_df)-split} games")

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
        pipe.fit(X[:split], y[:split])

        probs  = pipe.predict_proba(X[split:])[:, 1]
        actual = y[split:]
        results_df = train_df.iloc[split:].copy()
        label = "Chronological test set"

    # ── 2b. Method: 5-fold cross-validation ──────────────────────────────────
    else:
        print("\n[ 3 ] 5-fold cross-validated predictions (all games) ...")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
        probs  = cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 1]
        actual = y
        results_df = train_df.copy()
        label = "5-fold CV (all games)"

    # ── 3. Overall accuracy metrics ───────────────────────────────────────────
    preds   = (probs >= 0.5).astype(int)
    acc     = accuracy_score(actual, preds)
    ll      = log_loss(actual, probs)
    brier   = brier_score_loss(actual, probs)
    home_br = actual.mean()  # baseline: always pick home

    print(f"\n[ 4 ] Model accuracy  ({label})")
    print_divider("=")
    print(f"  Accuracy         : {fmt_pct(acc)}   (baseline home-always: {fmt_pct(home_br)})")
    print(f"  Log-loss         : {ll:.4f}   (lower = better; coin-flip = 0.693)")
    print(f"  Brier score      : {brier:.4f}   (lower = better; perfect = 0)")
    print(f"  Games evaluated  : {len(actual)}")
    print(f"  Home win rate    : {fmt_pct(home_br)}")

    # ── 4. Calibration ────────────────────────────────────────────────────────
    print(f"\n[ 5 ] Calibration  (how accurate are the probability estimates?)")
    print_divider()
    cal_df = calibration_table(probs, actual, n_bins=5)
    print(cal_df.to_string(index=False))
    print("  Note: 'Gap' = actual wins minus model predicted. Positive = model underestimates.")

    # ── 5. P&L simulation ─────────────────────────────────────────────────────
    results_df = results_df.copy()
    results_df["PROB"]       = probs
    results_df["CONFIDENCE"] = np.maximum(probs, 1 - probs)   # distance from 50/50
    results_df["BET_HOME"]   = probs >= 0.5
    results_df["CORRECT"]    = results_df["BET_HOME"] == (actual == 1)

    thresholds = [0.50, 0.55, 0.58, 0.60, 0.62, 0.65]

    print(f"\n[ 6 ] Simulated P&L  (flat ${STAKE} bets at -110 standard line)")
    print_divider("=")
    print(f"  {'Confidence':>12}  {'Bets':>6}  {'Wins':>5}  {'Win%':>7}  {'P&L':>10}  {'ROI':>7}  Break-even=52.4%")
    print_divider()
    for t in thresholds:
        r = simulate_pnl(results_df, min_confidence=t)
        if r["bets"] == 0:
            continue
        marker = " <<" if r["win_pct"] >= 0.524 and r["bets"] >= 20 else ""
        print(f"  {fmt_pct(t):>12}  {r['bets']:>6}  {r['wins']:>5}  "
              f"{fmt_pct(r['win_pct']):>7}  {fmt_dollar(r['pnl']):>10}  "
              f"{fmt_roi(r['roi']):>7}{marker}")

    # ── 6. Monthly breakdown ──────────────────────────────────────────────────
    print(f"\n[ 7 ] Monthly breakdown  (>50% confidence, flat -110 bets)")
    print_divider("=")
    results_df["MONTH"] = pd.to_datetime(results_df["GAME_DATE"]).dt.strftime("%b %Y")
    print(f"  {'Month':>10}  {'Bets':>6}  {'Wins':>5}  {'Win%':>7}  {'P&L':>10}  {'ROI':>7}")
    print_divider()
    for month, group in results_df.groupby("MONTH"):
        r = simulate_pnl(group, min_confidence=0.50)
        if r["bets"] == 0:
            continue
        trend = " (HOT)" if r["roi"] > 0.05 else (" (COLD)" if r["roi"] < -0.05 else "")
        print(f"  {month:>10}  {r['bets']:>6}  {r['wins']:>5}  "
              f"{fmt_pct(r['win_pct']):>7}  {fmt_dollar(r['pnl']):>10}  "
              f"{fmt_roi(r['roi']):>7}{trend}")

    # ── 7. Best and worst calls ───────────────────────────────────────────────
    results_df["PROB_FAV"]      = results_df["CONFIDENCE"]
    results_df["HOME_ABBREV"]   = results_df["HOME_ABBREV"]
    results_df["AWAY_ABBREV"]   = results_df["AWAY_ABBREV"]
    results_df["MATCHUP_STR"]   = results_df["AWAY_ABBREV"] + " @ " + results_df["HOME_ABBREV"]
    results_df["FAV_TEAM"]      = np.where(results_df["BET_HOME"], results_df["HOME_ABBREV"], results_df["AWAY_ABBREV"])
    results_df["RESULT_STR"]    = np.where(results_df["CORRECT"], "WIN", "LOSS")
    results_df["DATE_STR"]      = pd.to_datetime(results_df["GAME_DATE"]).dt.strftime("%b %d")

    # Biggest upsets (model was very confident but wrong)
    upsets = results_df[~results_df["CORRECT"]].nlargest(8, "CONFIDENCE")
    print(f"\n[ 8 ] Biggest upsets  (model was most confident but got it wrong)")
    print_divider("=")
    print(f"  {'Date':>8}  {'Matchup':>14}  {'Backed':>6}  {'Confidence':>11}  Result")
    print_divider()
    for _, row in upsets.iterrows():
        print(f"  {row['DATE_STR']:>8}  {row['MATCHUP_STR']:>14}  "
              f"{row['FAV_TEAM']:>6}  {fmt_pct(row['CONFIDENCE']):>11}  LOSS (upset)")

    # Strongest correct calls
    best = results_df[results_df["CORRECT"]].nlargest(8, "CONFIDENCE")
    print(f"\n[ 9 ] Strongest correct calls")
    print_divider("=")
    print(f"  {'Date':>8}  {'Matchup':>14}  {'Backed':>6}  {'Confidence':>11}  Result")
    print_divider()
    for _, row in best.iterrows():
        print(f"  {row['DATE_STR']:>8}  {row['MATCHUP_STR']:>14}  "
              f"{row['FAV_TEAM']:>6}  {fmt_pct(row['CONFIDENCE']):>11}  WIN")

    # ── 8. Cumulative P&L curve (text) ────────────────────────────────────────
    print(f"\n[ 10 ] Cumulative P&L over time  (>55% confidence, $100/bet)")
    print_divider("=")
    high_conf = results_df[results_df["CONFIDENCE"] >= 0.55].copy()
    if high_conf.empty:
        print("  No bets at this threshold.")
    else:
        high_conf = high_conf.sort_values("GAME_DATE").reset_index(drop=True)
        high_conf["PNL_BET"] = np.where(high_conf["CORRECT"],
                                         STAKE * JUICE_PAYOUT, -STAKE)
        high_conf["CUM_PNL"] = high_conf["PNL_BET"].cumsum()
        high_conf["DATE_STR"] = pd.to_datetime(high_conf["GAME_DATE"]).dt.strftime("%b %d")

        # Print a point every ~20 bets for a text-based curve
        step = max(1, len(high_conf) // 15)
        print(f"  {'Bet#':>5}  {'Date':>8}  {'Cum P&L':>12}  Chart")
        print_divider()
        for i, row in high_conf.iloc[::step].iterrows():
            bet_num = high_conf.index.get_loc(i) + 1
            bar_len = int(row["CUM_PNL"] / 50) if row["CUM_PNL"] != 0 else 0
            bar     = ("#" * abs(bar_len)) if bar_len > 0 else ("." * abs(bar_len))
            prefix  = "  +" if bar_len >= 0 else "  -"
            print(f"  {bet_num:>5}  {row['DATE_STR']:>8}  "
                  f"{fmt_dollar(row['CUM_PNL']):>12}  {prefix}{bar}")
        # Final row
        last = high_conf.iloc[-1]
        total_bets = len(high_conf)
        print(f"  {total_bets:>5}  {'FINAL':>8}  {fmt_dollar(last['CUM_PNL']):>12}")

    # ── Summary ───────────────────────────────────────────────────────────────
    r50  = simulate_pnl(results_df, 0.50)
    r55  = simulate_pnl(results_df, 0.55)
    r60  = simulate_pnl(results_df, 0.60)
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY")
    print_divider()
    print(f"  Overall accuracy  : {fmt_pct(acc)}  ({label})")
    print(f"  All bets (>50%)   : {r50['bets']} bets  {fmt_pct(r50['win_pct'])} win rate  "
          f"{fmt_dollar(r50['pnl'])} P&L  {fmt_roi(r50['roi'])} ROI")
    print(f"  Filter (>55%)     : {r55['bets']} bets  {fmt_pct(r55['win_pct'])} win rate  "
          f"{fmt_dollar(r55['pnl'])} P&L  {fmt_roi(r55['roi'])} ROI")
    print(f"  Filter (>60%)     : {r60['bets']} bets  {fmt_pct(r60['win_pct'])} win rate  "
          f"{fmt_dollar(r60['pnl'])} P&L  {fmt_roi(r60['roi'])} ROI")
    print(f"  Break-even rate   : 52.4% at -110")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Model Backtest")
    parser.add_argument("--all", action="store_true",
                        help="Use 5-fold CV instead of chronological split")
    args = parser.parse_args()
    main(use_cv=args.all)
