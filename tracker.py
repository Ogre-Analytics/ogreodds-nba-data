"""
Bet feedback loop — saves recommendations and tracks P&L over time.

Every time main.py runs:
  1. resolve_bets()          -- looks up yesterday's pending bets and marks W/L
  2. save_recommendations()  -- appends today's +EV picks to the log
  3. print_summary_line()    -- one-line status shown at end of daily output

Standalone report:
  python tracker.py          -- full P&L + calibration breakdown

Log file: logs/bet_log.json  (human-readable JSON, one entry per recommended bet)

Hypothetical bankroll (for P&L $) -- change BANKROLL to match yours:
"""

import json
from pathlib import Path
from datetime import date

import pandas as pd

LOG_FILE = Path(__file__).parent / "logs" / "bet_log.json"
BANKROLL  = 1000.0   # reference bankroll for P&L calculations (change to match yours)


# ── Log I/O ──────────────────────────────────────────────────────────────────

def _load_log() -> list:
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return []


def _save_log(entries: list) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2, default=str)


# ── Save recommendations ──────────────────────────────────────────────────────

def save_recommendations(analyses: list, game_date: str, run_date: str | None = None) -> int:
    """
    Append today's +EV bet recommendations to the log.

    Skips games with no odds or no positive-EV side.
    If a pending entry already exists for this game/side (e.g. re-run same day),
    it is overwritten with the latest odds/analysis so the most recent line is stored.

    Returns the number of NEW entries added (updates don't count).
    """
    if run_date is None:
        run_date = date.today().isoformat()

    entries = _load_log()
    existing_ids = {e["entry_id"]: i for i, e in enumerate(entries)}

    added = 0
    for a in analyses:
        if not a.get("has_odds") or a.get("best_bet") is None:
            continue
        if (a.get("best_bet_ev") or 0) <= 0:
            continue
        # Only save pre-game bets (GAME_STATUS_ID 1 = not started).
        # In-progress games produce false EV signals because live odds shift
        # while our model still uses pre-game stats.
        if a.get("game_status_id", 1) != 1:
            continue

        bet_team = a["best_bet"]
        is_home  = (bet_team == a["home_abbrev"])
        bet_side = "home" if is_home else "away"
        opp_team = a["away_abbrev"] if is_home else a["home_abbrev"]

        model_prob = a["model_home_prob"] if is_home else a["model_away_prob"]
        base_prob  = (
            a.get("base_home_prob", model_prob)
            if is_home
            else round(1.0 - a.get("base_home_prob", 1.0 - model_prob), 4)
        )
        implied = a.get("implied_home" if is_home else "implied_away") or 0.0

        entry_id = f"{game_date}_{a['home_abbrev']}_{a['away_abbrev']}_{bet_side}"

        entry = {
            "entry_id":        entry_id,
            "game_date":       game_date,
            "run_date":        run_date,
            "home_abbrev":     a["home_abbrev"],
            "away_abbrev":     a["away_abbrev"],
            "bet_side":        bet_side,
            "bet_team":        bet_team,
            "opp_team":        opp_team,
            "odds":            a.get("best_bet_odds"),
            "model_prob":      round(model_prob, 4),
            "base_prob":       round(base_prob, 4),
            "implied_prob":    round(implied, 4),
            # model_edge = our probability minus book's implied probability.
            # This is our estimated Closing Line Value (CLV) at time of bet.
            # Positive = we think the book is underpricing this team.
            # Tracking this over time shows whether our edge is genuine.
            "model_edge":      round(model_prob - implied, 4),
            "ev":              round(a.get("best_bet_ev", 0), 4),
            "kelly":           a.get("best_bet_kelly", 0),
            "tip_time":        a.get("tip_time", ""),
            "bookmaker":       a.get("bookmaker", ""),
            "home_inj_impact": a.get("home_inj_impact", 0),
            "away_inj_impact": a.get("away_inj_impact", 0),
            # result fields — filled by resolve_bets()
            "status":          "pending",
            "won":             None,
            "stake":           None,
            "profit_loss":     None,
            "resolved_date":   None,
        }

        # Remove any *other* pending entry for the same matchup (opposite side).
        # Prevents betting both sides if the recommended side flips across re-runs.
        game_prefix = f"{game_date}_{a['home_abbrev']}_{a['away_abbrev']}_"
        before_count = len(entries)
        entries = [
            e for e in entries
            if not (e["entry_id"].startswith(game_prefix)
                    and e["entry_id"] != entry_id
                    and e["status"] == "pending")
        ]
        if len(entries) < before_count:
            # Rebuild index after removal
            existing_ids = {e["entry_id"]: i for i, e in enumerate(entries)}

        if entry_id in existing_ids:
            idx = existing_ids[entry_id]
            if entries[idx]["status"] == "pending":
                entries[idx] = entry   # overwrite with fresh analysis
        else:
            entries.append(entry)
            existing_ids[entry_id] = len(entries) - 1
            added += 1

    _save_log(entries)
    return added


# ── Resolve results ───────────────────────────────────────────────────────────

def resolve_bets(games_df: pd.DataFrame) -> int:
    """
    Match pending bets to actual game results and mark them Won/Lost.

    Uses the season game log (nba_api) — matches on game_date + bet_team abbreviation.
    P&L is calculated using the saved Kelly fraction against BANKROLL.

    Returns the number of bets newly resolved.
    """
    entries = _load_log()
    if not any(e["status"] == "pending" for e in entries):
        return 0

    gdf = games_df.copy()
    gdf["GAME_DATE_STR"] = pd.to_datetime(gdf["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    resolved   = 0
    today_str  = date.today().isoformat()

    for entry in entries:
        if entry["status"] != "pending":
            continue

        rows = gdf[
            (gdf["GAME_DATE_STR"] == entry["game_date"]) &
            (gdf["TEAM_ABBREVIATION"] == entry["bet_team"])
        ]
        if rows.empty:
            continue   # game not in season log yet (today's games, or scraper lag)

        wl = rows.iloc[0]["WL"]
        if wl not in ("W", "L"):
            continue   # game still in progress

        won   = (wl == "W")
        kelly = entry.get("kelly") or 0.0
        stake = round(kelly * BANKROLL, 2)
        odds  = entry.get("odds") or 0

        if odds >= 0:
            win_amount = stake * odds / 100
        else:
            win_amount = stake * 100 / abs(odds)

        profit_loss = round(win_amount if won else -stake, 2)

        entry["status"]        = "resolved"
        entry["won"]           = won
        entry["stake"]         = stake
        entry["profit_loss"]   = profit_loss
        entry["resolved_date"] = today_str
        resolved += 1

    if resolved:
        _save_log(entries)

    return resolved


# ── Report helpers ────────────────────────────────────────────────────────────

def print_summary_line() -> None:
    """One-line tracker status — embedded at the bottom of each daily run."""
    entries  = _load_log()
    resolved = [e for e in entries if e["status"] == "resolved"]
    pending  = [e for e in entries if e["status"] == "pending"]

    if not resolved:
        if pending:
            print(f"  [tracker] {len(pending)} pending bet(s) — results will auto-resolve tomorrow.")
        else:
            print("  [tracker] No bet history yet. Recommendations will be saved after today's run.")
        return

    wins     = sum(1 for e in resolved if e["won"])
    total_pl = sum(e["profit_loss"] for e in resolved)
    total_s  = sum(e.get("stake", 0) for e in resolved)
    roi      = (total_pl / total_s * 100) if total_s > 0 else 0.0

    print(f"  [tracker] Season: {len(resolved)} bets  "
          f"{wins}W-{len(resolved)-wins}L  "
          f"P&L ${total_pl:+.2f}  ROI {roi:+.1f}%  "
          f"| {len(pending)} pending  "
          f"| python tracker.py for full report")


def print_report() -> None:
    """Full P&L, calibration and injury-adjustment audit from the bet log."""
    entries = _load_log()
    if not entries:
        print("  No bet history found.")
        print("  Run main.py on a game day with live odds to start tracking.")
        return

    resolved = [e for e in entries if e["status"] == "resolved"]
    pending  = [e for e in entries if e["status"] == "pending"]

    print(f"  Total bets logged : {len(entries)}  "
          f"({len(resolved)} resolved, {len(pending)} pending)")

    if not resolved:
        print("  No resolved bets yet — check back after game results are in.")
        if pending:
            print(f"\n  Pending ({len(pending)}):")
            for e in sorted(pending, key=lambda x: x["game_date"]):
                odds_str = f"+{e['odds']}" if (e.get('odds') or 0) >= 0 else str(e.get('odds', '?'))
                print(f"    {e['game_date']}  {e['bet_team']} vs {e['opp_team']}"
                      f"  {odds_str}  model {e['model_prob']*100:.1f}%  EV +{e['ev']*100:.1f}%")
        return

    # ── P&L summary ──────────────────────────────────────────────────────────
    wins        = sum(1 for e in resolved if e["won"])
    losses      = len(resolved) - wins
    total_pl    = sum(e["profit_loss"] for e in resolved)
    total_stake = sum(e.get("stake", 0) for e in resolved)
    roi         = (total_pl / total_stake * 100) if total_stake > 0 else 0.0
    hit_rate    = wins / len(resolved) * 100
    avg_ev_pred = sum(e["ev"] for e in resolved) / len(resolved) * 100

    print(f"\n  P&L Summary  (${BANKROLL:.0f} bankroll, Half-Kelly staking)")
    print(f"  {'Bets':<22}: {len(resolved)}")
    print(f"  {'Won / Lost':<22}: {wins}W / {losses}L  ({hit_rate:.1f}% hit rate)")
    print(f"  {'Total P&L':<22}: ${total_pl:+.2f}")
    print(f"  {'Total staked':<22}: ${total_stake:.2f}")
    print(f"  {'ROI per $ staked':<22}: {roi:+.1f}%")
    print(f"  {'Avg predicted EV':<22}: +{avg_ev_pred:.1f}% per bet")

    if total_stake > 0:
        realised_ev = total_pl / total_stake * 100
        delta = realised_ev - avg_ev_pred
        verdict = (
            "OVERPERFORMING" if delta > 5
            else "underperforming (model may be overconfident)" if delta < -10
            else "tracking predicted edge"
        )
        print(f"  {'Realised EV':<22}: {realised_ev:+.1f}% per $ staked")
        print(f"  {'EV delta':<22}: {delta:+.1f}%  ({verdict})")

    # ── Calibration ──────────────────────────────────────────────────────────
    print(f"\n  Calibration  (did X% model predictions actually win X%?)")
    bands = [
        (0.50, 0.55, "50-55%"),
        (0.55, 0.60, "55-60%"),
        (0.60, 0.65, "60-65%"),
        (0.65, 0.70, "65-70%"),
        (0.70, 0.80, "70-80%"),
        (0.80, 1.01, "80%+  "),
    ]
    print(f"  {'Band':<10} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'Model%':>7} {'Delta':>8}")
    print("  " + "-" * 47)
    any_band = False
    for lo, hi, label in bands:
        bets = [e for e in resolved if lo <= e["model_prob"] < hi]
        if not bets:
            continue
        any_band   = True
        bwins      = sum(1 for e in bets if e["won"])
        win_pct    = bwins / len(bets) * 100
        pred_pct   = sum(e["model_prob"] for e in bets) / len(bets) * 100
        delta      = win_pct - pred_pct
        flag       = " !" if abs(delta) > 10 else ""
        print(f"  {label:<10} {len(bets):>5} {bwins:>5} {win_pct:>6.1f}%  {pred_pct:>6.1f}%  {delta:>+7.1f}%{flag}")

    if not any_band:
        print("  (not enough data to populate bands)")
    else:
        print("  ! = delta > 10pp — consider retuning INJURY_LOGIT_COEF or MIN_EV_THRESHOLD")

    # ── Closing Line Value (CLV) analysis ────────────────────────────────────
    # model_edge = model_prob - implied_prob at time of bet.
    # A consistent positive average edge (beating the line) is the best
    # long-term indicator that the model has genuine edge over the market.
    clv_bets = [e for e in resolved if e.get("model_edge") is not None]
    if clv_bets:
        avg_edge   = sum(e["model_edge"] for e in clv_bets) / len(clv_bets) * 100
        pos_edge   = sum(1 for e in clv_bets if e["model_edge"] > 0)
        won_pos    = sum(1 for e in clv_bets if e["model_edge"] > 0 and e["won"])
        won_neg    = sum(1 for e in clv_bets if e.get("model_edge", 0) <= 0 and e["won"])
        neg_edge   = len(clv_bets) - pos_edge
        print(f"\n  Closing Line Value (model edge vs. book at time of bet)")
        print(f"  {'Avg model edge':<28}: {avg_edge:+.1f}%  "
              f"({'beating the line' if avg_edge > 0 else 'behind the line'})")
        print(f"  {'Bets with +edge':<28}: {pos_edge}/{len(clv_bets)}  "
              f"({pos_edge/len(clv_bets)*100:.0f}% of bets)")
        if pos_edge > 0:
            print(f"  {'Win rate on +edge bets':<28}: {won_pos}/{pos_edge}  "
                  f"({won_pos/pos_edge*100:.1f}%)")
        if neg_edge > 0:
            print(f"  {'Win rate on -edge bets':<28}: {won_neg}/{neg_edge}  "
                  f"({won_neg/neg_edge*100:.1f}%)")
        if avg_edge > 2:
            print("  Verdict: Model is consistently ahead of the opening line — real edge detected.")
        elif avg_edge > 0:
            print("  Verdict: Slight positive CLV — accumulate more bets to confirm edge.")
        else:
            print("  Verdict: Negative CLV — model may be reacting to the same info as the book.")

    # ── Injury adjustment audit ───────────────────────────────────────────────
    inj_bets   = [e for e in resolved if (e.get("home_inj_impact") or 0) > 0
                                       or (e.get("away_inj_impact") or 0) > 0]
    plain_bets = [e for e in resolved if e not in inj_bets]

    if inj_bets and plain_bets:
        def _roi(bets):
            pl = sum(b["profit_loss"] for b in bets)
            sk = sum(b.get("stake", 0) for b in bets)
            return (pl / sk * 100) if sk > 0 else 0.0

        print(f"\n  Injury-adjustment audit:")
        print(f"  {'Inj-adjusted bets':<24}: {len(inj_bets):>3}  ROI {_roi(inj_bets):+.1f}%")
        print(f"  {'Non-injury bets':<24}: {len(plain_bets):>3}  ROI {_roi(plain_bets):+.1f}%")
        note = ("injury adjustment helping" if _roi(inj_bets) > _roi(plain_bets)
                else "injury adjustment may be over-firing — consider lowering INJURY_LOGIT_COEF")
        print(f"  Note: {note}")

    # ── Last 10 bets ─────────────────────────────────────────────────────────
    recent = sorted(resolved, key=lambda e: e["game_date"], reverse=True)[:10]
    if recent:
        r_wins  = sum(1 for e in recent if e["won"])
        r_pl    = sum(e["profit_loss"] for e in recent)
        r_stake = sum(e.get("stake", 0) for e in recent)
        r_roi   = (r_pl / r_stake * 100) if r_stake > 0 else 0.0
        print(f"\n  Last {len(recent)} bets: "
              f"{r_wins}W-{len(recent)-r_wins}L  "
              f"P&L ${r_pl:+.2f}  ROI {r_roi:+.1f}%")
        print()
        print(f"  {'Date':<12} {'Bet':<6} {'vs':<6} {'Odds':>6}  {'Model%':>7}  {'Edge%':>6}  {'EV%':>6}  {'Result':>6}  {'P&L':>8}")
        print("  " + "-" * 74)
        for e in recent:
            odds_str  = f"+{e['odds']}" if (e.get('odds') or 0) >= 0 else str(e.get('odds', '?'))
            result    = "WIN " if e["won"] else "LOSS"
            pl_str    = f"${e['profit_loss']:+.2f}"
            edge_pct  = e.get("model_edge", 0) * 100
            print(f"  {e['game_date']:<12} {e['bet_team']:<6} {e['opp_team']:<6} "
                  f"{odds_str:>6}  "
                  f"{e['model_prob']*100:>6.1f}%  "
                  f"{edge_pct:>+5.1f}%  "
                  f"{e['ev']*100:>5.1f}%  "
                  f"{result:>6}  "
                  f"{pl_str:>8}")

    # ── Pending ───────────────────────────────────────────────────────────────
    if pending:
        print(f"\n  Pending ({len(pending)} bets — awaiting results):")
        for e in sorted(pending, key=lambda x: x["game_date"]):
            odds_str = f"+{e['odds']}" if (e.get('odds') or 0) >= 0 else str(e.get('odds', '?'))
            print(f"    {e['game_date']}  {e['bet_team']} vs {e['opp_team']:<4}  "
                  f"{odds_str:>6}  model {e['model_prob']*100:.1f}%  EV +{e['ev']*100:.1f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  NBA BETTING TRACKER  --  Performance Report")
    print("=" * 65)
    print_report()
    print()
    print("=" * 65)
