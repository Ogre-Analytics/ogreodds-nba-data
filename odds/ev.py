"""
Expected Value (EV) and Kelly Criterion calculator.

Core formulas:
  Implied probability  = 1 / decimal_odds
  EV (% of stake)      = model_prob * (decimal_odds - 1) - (1 - model_prob)
  Kelly fraction       = (b * p - q) / b   where b = decimal_odds - 1
  Half-Kelly           = Kelly / 2   (recommended — reduces variance)
"""

from __future__ import annotations


# ── Odds conversions ──────────────────────────────────────────────────────────

def american_to_decimal(american: int) -> float:
    """Convert American moneyline to decimal odds."""
    if american >= 0:
        return 1 + american / 100
    return 1 + 100 / abs(american)


def decimal_to_american(decimal: float) -> int:
    """Convert decimal odds to American moneyline."""
    if decimal >= 2.0:
        return int(round((decimal - 1) * 100))
    return int(round(-100 / (decimal - 1)))


def american_to_implied_prob(american: int) -> float:
    """
    Convert American odds to implied probability (includes bookmaker vig).
    Note: home + away implied probs will sum to > 1.0 due to vig.
    """
    return 1 / american_to_decimal(american)


def remove_vig(home_american: int, away_american: int) -> tuple[float, float]:
    """
    Strip bookmaker margin to get fair implied probabilities that sum to 1.0.
    Uses the standard additive method.
    """
    raw_home = american_to_implied_prob(home_american)
    raw_away = american_to_implied_prob(away_american)
    total = raw_home + raw_away
    return raw_home / total, raw_away / total


# ── EV calculation ────────────────────────────────────────────────────────────

def calculate_ev(model_prob: float, american_odds: int) -> float:
    """
    Expected value as a fraction of stake.

    Example: EV = 0.05 means you expect to profit 5c per $1 wagered on average.
    Positive = profitable bet.
    """
    decimal = american_to_decimal(american_odds)
    return model_prob * (decimal - 1) - (1 - model_prob)


def kelly_fraction(model_prob: float, american_odds: int, fraction: float = 0.5) -> float:
    """
    Kelly Criterion stake as a fraction of bankroll.

    fraction=0.5  → Half-Kelly (recommended: reduces variance significantly)
    fraction=1.0  → Full Kelly (theoretically optimal but high variance)

    Returns 0.0 if the bet has negative EV (don't bet).
    """
    decimal = american_to_decimal(american_odds)
    b = decimal - 1       # net profit per unit staked
    p = model_prob
    q = 1 - p
    full_kelly = (b * p - q) / b
    return max(0.0, round(full_kelly * fraction, 4))


# ── Game analysis ─────────────────────────────────────────────────────────────

def analyse_game(
    home_abbrev: str,
    away_abbrev: str,
    home_win_prob: float,
    away_win_prob: float,
    home_american: int | None,
    away_american: int | None,
    tip_time: str = "",
    kelly_fraction_setting: float = 0.5,
) -> dict:
    """
    Full analysis for a single game: predictions, implied probs, EV, Kelly.

    home/away_american: None if no odds available.

    Returns a dict suitable for display in main.py.
    """
    result = {
        "matchup":        f"{away_abbrev} @ {home_abbrev}",
        "home_abbrev":    home_abbrev,
        "away_abbrev":    away_abbrev,
        "tip_time":       tip_time,
        "model_home_prob": home_win_prob,
        "model_away_prob": away_win_prob,
        "has_odds":       home_american is not None,
        # filled below if odds available
        "home_odds":      home_american,
        "away_odds":      away_american,
        "implied_home":   None,
        "implied_away":   None,
        "fair_home":      None,
        "fair_away":      None,
        "home_ev":        None,
        "away_ev":        None,
        "home_kelly":     None,
        "away_kelly":     None,
        "best_bet":       None,
        "best_bet_ev":    None,
        "best_bet_kelly": None,
        "vig_pct":        None,
    }

    if home_american is None or away_american is None:
        return result

    # Implied probs (with vig)
    raw_home = american_to_implied_prob(home_american)
    raw_away = american_to_implied_prob(away_american)
    vig = (raw_home + raw_away - 1) * 100

    # Fair probs (vig removed)
    fair_home, fair_away = remove_vig(home_american, away_american)

    # EV
    home_ev = calculate_ev(home_win_prob, home_american)
    away_ev = calculate_ev(away_win_prob, away_american)

    # Kelly
    hk = kelly_fraction(home_win_prob, home_american, kelly_fraction_setting)
    ak = kelly_fraction(away_win_prob, away_american, kelly_fraction_setting)

    # Best bet side
    if home_ev >= away_ev and home_ev > 0:
        best_side     = home_abbrev
        best_ev       = home_ev
        best_kelly    = hk
        best_american = home_american
    elif away_ev > 0:
        best_side     = away_abbrev
        best_ev       = away_ev
        best_kelly    = ak
        best_american = away_american
    else:
        best_side     = None
        best_ev       = max(home_ev, away_ev)
        best_kelly    = 0.0
        best_american = None

    result.update({
        "implied_home":   round(raw_home, 4),
        "implied_away":   round(raw_away, 4),
        "fair_home":      round(fair_home, 4),
        "fair_away":      round(fair_away, 4),
        "home_ev":        round(home_ev, 4),
        "away_ev":        round(away_ev, 4),
        "home_kelly":     hk,
        "away_kelly":     ak,
        "best_bet":       best_side,
        "best_bet_ev":    round(best_ev, 4),
        "best_bet_kelly": best_kelly,
        "best_bet_odds":  best_american,
        "vig_pct":        round(vig, 2),
    })
    return result
