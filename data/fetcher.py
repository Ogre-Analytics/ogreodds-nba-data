"""
NBA data fetcher — pulls game logs for the current 2025-26 season.
Caches results locally to avoid hammering the API on every run.
"""

import json
import time
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, date

from nba_api.stats.endpoints import (
    leaguegamefinder,
    leaguedashteamstats,
    leaguedashplayerstats,
    teamgamelog,
    scoreboardv2,
)
from nba_api.stats.static import teams as nba_teams_static

CURRENT_SEASON      = "2025-26"
# Completed seasons used as the fixed historical training corpus.
# These are cached with a 1-year TTL since they will never change.
HISTORICAL_SEASONS  = ["2021-22", "2022-23", "2023-24", "2024-25"]
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# nba_api needs a small delay between requests to avoid rate-limiting
REQUEST_DELAY = 0.6
# Timeout in seconds for each NBA API call. stats.nba.com can be slow from
# non-US locations (e.g. Australia) — 60s gives enough headroom.
NBA_TIMEOUT   = 60
# Number of times to retry a failed NBA API call before giving up.
NBA_RETRIES   = 3

# stats.nba.com blocks GitHub Actions IP ranges entirely.
# When running in CI, accept cached data up to 7 days old so committed
# cache files are always used instead of making blocked API calls.
_IN_CI = os.getenv("GITHUB_ACTIONS") == "true"
_CI_CACHE_TTL_HOURS = 168   # 7 days


def _nba_call(fn, *args, **kwargs):
    """
    Call an nba_api endpoint function with automatic retries.

    fn      : a callable that returns the endpoint object, e.g.
              lambda: leaguegamefinder.LeagueGameFinder(season_nullable=season, ...)
    Retries NBA_RETRIES times with exponential back-off (2s, 4s, 8s).
    Raises the last exception if all attempts fail.
    """
    last_exc = None
    for attempt in range(1, NBA_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < NBA_RETRIES:
                wait = 2 ** attempt
                print(f"  [warn]  NBA API attempt {attempt} failed ({type(exc).__name__}). "
                      f"Retrying in {wait}s ...")
                time.sleep(wait)
    raise last_exc


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str, max_age_hours: float = 6) -> dict | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    # In GitHub Actions, stats.nba.com is blocked — honour committed cache files
    # regardless of age so the pipeline never needs to call stats.nba.com.
    effective_ttl = _CI_CACHE_TTL_HOURS if _IN_CI else max_age_hours
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > effective_ttl:
        return None
    with open(path) as f:
        return json.load(f)


def _save_cache(key: str, data: dict) -> None:
    with open(_cache_path(key), "w") as f:
        json.dump(data, f)


def get_all_teams() -> pd.DataFrame:
    """Return a DataFrame of all 30 NBA teams with id, name, abbreviation."""
    all_teams = nba_teams_static.get_teams()
    return pd.DataFrame(all_teams)[["id", "full_name", "abbreviation", "nickname", "city"]]


def get_season_games(season: str = CURRENT_SEASON, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch every game played in the given season.
    Returns one row per team per game (so each game appears twice).
    """
    cache_key = f"season_games_{season.replace('-', '_')}"
    if use_cache:
        cached = _load_cache(cache_key, max_age_hours=23)
        if cached:
            print(f"  [cache] Loaded season games from cache")
            return pd.DataFrame(cached)

    print(f"  [api]   Fetching season games for {season} ...")
    time.sleep(REQUEST_DELAY)
    finder = _nba_call(lambda: leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",   # 00 = NBA
        season_type_nullable="Regular Season",
        timeout=NBA_TIMEOUT,
    ))
    df = finder.get_data_frames()[0]

    if use_cache and not df.empty:
        _save_cache(cache_key, df.to_dict(orient="records"))

    return df


def get_team_dashboard(season: str = CURRENT_SEASON, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch season-to-date team stats dashboard (offensive/defensive ratings,
    pace, net rating, etc.) for all teams.
    """
    cache_key = f"team_dashboard_{season.replace('-', '_')}"
    if use_cache:
        cached = _load_cache(cache_key, max_age_hours=4)
        if cached:
            print(f"  [cache] Loaded team dashboard from cache")
            return pd.DataFrame(cached)

    print(f"  [api]   Fetching team dashboard for {season} ...")
    time.sleep(REQUEST_DELAY)
    stats = _nba_call(lambda: leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        timeout=NBA_TIMEOUT,
    ))
    df = stats.get_data_frames()[0]

    if use_cache and not df.empty:
        _save_cache(cache_key, df.to_dict(orient="records"))

    return df


def get_team_last_n_games(team_id: int, n: int = 10, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Fetch the last N game log entries for a specific team.
    Useful for recent-form calculations.
    """
    cache_key = f"team_log_{team_id}_{season.replace('-', '_')}"
    cached = _load_cache(cache_key, max_age_hours=4)
    if cached:
        df = pd.DataFrame(cached)
    else:
        time.sleep(REQUEST_DELAY)
        log = _nba_call(lambda: teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star="Regular Season",
            timeout=NBA_TIMEOUT,
        ))
        df = log.get_data_frames()[0]
        if not df.empty:
            _save_cache(cache_key, df.to_dict(orient="records"))

    return df.head(n) if not df.empty else df


def _result_set_to_df(result_sets: list, name: str) -> pd.DataFrame:
    """Extract a named result set from raw ScoreboardV2 dict as a DataFrame."""
    for rs in result_sets:
        if rs["name"] == name:
            headers = rs["headers"]
            rows = rs["rowSet"]
            return pd.DataFrame(rows, columns=headers) if rows else pd.DataFrame(columns=headers)
    return pd.DataFrame()


def _cdn_scoreboard(game_date: str) -> dict:
    """
    Build a scoreboard dict from the NBA CDN schedule JSON.
    Used as a fallback when stats.nba.com is unreachable (e.g. GitHub Actions).
    cdn.nba.com is public and never blocks CI/CD IP ranges.
    """
    import requests as _req
    try:
        resp = _req.get(
            "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        game_dates = resp.json().get("leagueSchedule", {}).get("gameDates", [])
    except Exception as exc:
        print(f"  [cdn]   CDN scoreboard failed: {exc}")
        return {"game_header": pd.DataFrame(), "line_score": pd.DataFrame(),
                "series_standings": pd.DataFrame()}

    rows = []
    for gd in game_dates:
        try:
            gd_str = datetime.strptime(
                gd.get("gameDate", ""), "%m/%d/%Y %H:%M:%S"
            ).strftime("%Y-%m-%d")
        except ValueError:
            continue
        if gd_str != game_date:
            continue
        for game in gd.get("games", []):
            home_id = game.get("homeTeam", {}).get("teamId")
            away_id = game.get("awayTeam", {}).get("teamId")
            if not home_id or not away_id:
                continue
            status_id   = int(game.get("gameStatus", 1))
            status_text = "Final" if status_id == 3 else str(game.get("gameStatusText", "")).strip()
            rows.append({
                "GAME_ID":          game.get("gameId", ""),
                "HOME_TEAM_ID":     int(home_id),
                "VISITOR_TEAM_ID":  int(away_id),
                "GAME_STATUS_TEXT": status_text,
                "GAME_STATUS_ID":   status_id,
            })

    header_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "GAME_STATUS_TEXT", "GAME_STATUS_ID"]
    )
    if rows:
        print(f"  [cdn]   CDN scoreboard: {len(rows)} game(s) for {game_date}")
    return {"game_header": header_df, "line_score": pd.DataFrame(),
            "series_standings": pd.DataFrame()}


def get_todays_scoreboard(game_date: str | None = None) -> dict:
    """
    Fetch games for the given date using ScoreboardV2.

    game_date: 'YYYY-MM-DD' string (US Eastern date). If None, falls back to
               the local system date (legacy behaviour).

    Returns raw scoreboard data as a dict with keys:
      'game_header', 'line_score', 'series_standings'
    """
    if game_date:
        from datetime import datetime as _dt
        query_date_fmt = _dt.strptime(game_date, "%Y-%m-%d").strftime("%m/%d/%Y")
    else:
        game_date      = date.today().strftime("%Y-%m-%d")
        query_date_fmt = date.today().strftime("%m/%d/%Y")

    print(f"  [api]   Fetching scoreboard for {query_date_fmt} ...")
    try:
        time.sleep(REQUEST_DELAY)
        board = _nba_call(lambda: scoreboardv2.ScoreboardV2(
            game_date=query_date_fmt, league_id="00", day_offset=0, timeout=NBA_TIMEOUT,
        ))
        raw = board.get_dict()
        result_sets = raw.get("resultSets", [])
        return {
            "game_header": _result_set_to_df(result_sets, "GameHeader"),
            "line_score":  _result_set_to_df(result_sets, "LineScore"),
            "series_standings": _result_set_to_df(result_sets, "SeriesStandings"),
        }
    except Exception as exc:
        print(f"  [warn]  ScoreboardV2 failed ({type(exc).__name__}), using CDN fallback ...")
        return _cdn_scoreboard(game_date)


def get_player_stats(season: str = CURRENT_SEASON, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch per-game player stats for the season (all players, all teams).
    Used to estimate the impact of injured players on win probability.

    Key columns: PLAYER_NAME, TEAM_ABBREVIATION, PLUS_MINUS, PTS, MIN, GP
    """
    cache_key = f"player_stats_{season.replace('-', '_')}"
    if use_cache:
        cached = _load_cache(cache_key, max_age_hours=4)
        if cached:
            print("  [cache] Loaded player stats from cache")
            return pd.DataFrame(cached)

    print(f"  [api]   Fetching player stats for {season} ...")
    time.sleep(REQUEST_DELAY)
    stats = _nba_call(lambda: leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        timeout=NBA_TIMEOUT,
    ))
    df = stats.get_data_frames()[0]

    if use_cache and not df.empty:
        _save_cache(cache_key, df.to_dict(orient="records"))

    return df


def get_clutch_stats(season: str = CURRENT_SEASON, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch clutch stats (last 5 minutes, score within 5 pts) per team.

    The endpoint returns PLUS_MINUS (per-game point differential in clutch
    situations) which serves as the clutch net rating proxy.
    Key columns: TEAM_ID, TEAM_NAME, GP, W, L, W_PCT, PLUS_MINUS
    """
    from nba_api.stats.endpoints import leaguedashteamclutch
    cache_key = f"clutch_stats_{season.replace('-', '_')}"
    if use_cache:
        cached = _load_cache(cache_key, max_age_hours=4)
        if cached:
            print("  [cache] Loaded clutch stats from cache")
            return pd.DataFrame(cached)

    print(f"  [api]   Fetching clutch stats for {season} ...")
    time.sleep(REQUEST_DELAY)
    stats = _nba_call(lambda: leaguedashteamclutch.LeagueDashTeamClutch(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        clutch_time="Last 5 Minutes",
        point_diff=5,
        timeout=NBA_TIMEOUT,
    ))
    df = stats.get_data_frames()[0]
    if use_cache and not df.empty:
        _save_cache(cache_key, df.to_dict(orient="records"))
    return df


def get_historical_games(
    seasons: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch game logs for multiple completed seasons to use as training data.

    Each season is cached with a 1-year TTL — completed seasons never change
    so the cache is effectively permanent.  On first run this makes 4 API
    calls (~3 seconds); every subsequent run loads from disk instantly.

    Returns a combined DataFrame in the same format as get_season_games().
    """
    if seasons is None:
        seasons = HISTORICAL_SEASONS

    dfs = []
    for season in seasons:
        cache_key = f"season_games_{season.replace('-', '_')}"
        if use_cache:
            cached = _load_cache(cache_key, max_age_hours=8760)  # 1-year TTL
            if cached:
                print(f"  [cache] Loaded {season} from cache ({len(cached)} rows)")
                dfs.append(pd.DataFrame(cached))
                continue

        print(f"  [api]   Fetching {season} season games ...")
        time.sleep(REQUEST_DELAY)
        finder = _nba_call(lambda: leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
            timeout=NBA_TIMEOUT,
        ))
        df = finder.get_data_frames()[0]
        if use_cache and not df.empty:
            _save_cache(cache_key, df.to_dict(orient="records"))
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Historical data: {combined['GAME_ID'].nunique()} unique games across {len(seasons)} seasons")
    return combined


def _parse_schedule_raw(raw: dict) -> pd.DataFrame | None:
    """Extract GameHeader rows from a raw scheduleleaguev2 response dict."""
    rs_list = raw.get("resultSets", [])
    header_rs = next((r for r in rs_list if r["name"] == "GameHeader"), None)
    if header_rs is None or not header_rs.get("rowSet"):
        return None
    return pd.DataFrame(header_rs["rowSet"], columns=header_rs["headers"])


def _fetch_cdn_schedule() -> pd.DataFrame | None:
    """
    Fetch the full season schedule from the NBA public CDN in one HTTP request.

    URL: cdn.nba.com/static/json/staticData/scheduleLeagueV2.json
    Always serves the current season; no authentication or special headers needed.

    Returns a DataFrame with columns:
      GAME_ID, HOME_TEAM_ID, VISITOR_TEAM_ID, GAME_DATE_EST, GAME_STATUS_TEXT
    where GAME_STATUS_TEXT == 'Final' for completed games.
    """
    try:
        import requests as _req
        import datetime as _dt
        time.sleep(REQUEST_DELAY)
        resp = _req.get(
            "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        game_dates = resp.json().get("leagueSchedule", {}).get("gameDates", [])
        rows: list[dict] = []
        for gd in game_dates:
            raw_date = gd.get("gameDate", "")
            try:
                date_str = _dt.datetime.strptime(
                    raw_date, "%m/%d/%Y %H:%M:%S"
                ).strftime("%Y-%m-%d")
            except ValueError:
                continue
            for game in gd.get("games", []):
                home_id = game.get("homeTeam", {}).get("teamId")
                away_id = game.get("awayTeam", {}).get("teamId")
                if not home_id or not away_id:
                    continue
                status = int(game.get("gameStatus", 1))
                rows.append({
                    "GAME_ID":          game.get("gameId", ""),
                    "HOME_TEAM_ID":     int(home_id),
                    "VISITOR_TEAM_ID":  int(away_id),
                    "GAME_DATE_EST":    date_str,
                    "GAME_STATUS_TEXT": "Final" if status == 3 else str(game.get("gameStatusText", "")),
                })
        if rows:
            df = pd.DataFrame(rows)
            print(f"  [cdn]   NBA CDN schedule: {len(df)} total games across the season")
            return df
    except Exception as exc:
        print(f"  [cdn]   NBA CDN unavailable ({type(exc).__name__}): {exc}")
    return None


def _fetch_full_schedule(season: str, today_str: str | None = None) -> pd.DataFrame | None:
    """
    Fetch all regular season game headers (past + future) for schedule filtering.

    Attempt order (fastest/most reliable first):
      1. NBA CDN JSON   — cdn.nba.com, one request, public, no auth, instant
      2. stats.nba.com  — works from US IPs; often times out from non-US locations
      3. ScoreboardV2 day-loop — ~40 nba_api calls, ~25s, always works but slow

    Result is cached by get_remaining_schedule(); this function is called at most
    once per session (or once per 4 hours on re-runs).
    """
    # ── Attempt 1: NBA CDN (one request, public, instant) ────────────────────
    cdn_df = _fetch_cdn_schedule()
    if cdn_df is not None and not cdn_df.empty:
        return cdn_df
    print("  [cdn]   Falling back to stats.nba.com ...")

    # ── Attempt 2: stats.nba.com HTTP (works from US IPs, times out elsewhere) ─
    try:
        import requests as _req
        time.sleep(REQUEST_DELAY)
        resp = _req.get(
            "https://stats.nba.com/stats/scheduleleaguev2",
            params={"LeagueID": "00", "Season": season},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept":             "application/json, text/plain, */*",
                "Accept-Language":    "en-US,en;q=0.9",
                "Accept-Encoding":    "gzip, deflate, br",
                "Referer":            "https://www.nba.com/",
                "x-nba-stats-origin": "stats",
                "x-nba-stats-token":  "true",
                "Connection":         "keep-alive",
                "Host":               "stats.nba.com",
            },
            timeout=20,
        )
        resp.raise_for_status()
        df = _parse_schedule_raw(resp.json())
        if df is not None and not df.empty:
            print(f"  [api]   stats.nba.com schedule: {len(df)} total rows")
            return df
    except Exception as exc:
        print(f"  [api]   stats.nba.com unavailable ({type(exc).__name__}), "
              f"falling back to ScoreboardV2 day-loop ...")

    # ── Attempt 3: ScoreboardV2 day-loop (slow but always works) ─────────────
    # Hits every calendar day from today through end-of-regular-season.
    # ~40 requests at REQUEST_DELAY each = ~25s on first run; cached for 4 hours.
    try:
        from datetime import date as _date, timedelta as _td

        start_str = today_str or date.today().strftime("%Y-%m-%d")
        # Regular season ends in mid-April of the second calendar year
        end_year  = int(season.split("-")[0]) + 1   # 2026 for "2025-26"
        end_str   = f"{end_year}-04-14"

        start = _date.fromisoformat(start_str)
        end   = _date.fromisoformat(end_str)
        n_days = (end - start).days + 1

        print(f"  [api]   ScoreboardV2 day-loop: {n_days} days "
              f"({start_str} -> {end_str}) ...")

        all_rows: list[dict] = []
        current = start
        while current <= end:
            date_str = current.strftime("%m/%d/%Y")
            time.sleep(REQUEST_DELAY)
            try:
                board = scoreboardv2.ScoreboardV2(
                    game_date=date_str, league_id="00", day_offset=0,
                    timeout=NBA_TIMEOUT,
                )
                raw = board.get_dict()
                for rs in raw.get("resultSets", []):
                    if rs["name"] == "GameHeader" and rs.get("rowSet"):
                        hdrs = rs["headers"]
                        for row_data in rs["rowSet"]:
                            all_rows.append(dict(zip(hdrs, row_data)))
                        break
            except Exception:
                pass   # No games scheduled that day, or transient error
            current += _td(days=1)

        if all_rows:
            df = pd.DataFrame(all_rows)
            print(f"  [api]   ScoreboardV2 day-loop: {len(df)} game rows collected")
            return df

        print("  [api]   ScoreboardV2 day-loop returned no rows")

    except Exception as exc:
        print(f"  [api]   ScoreboardV2 day-loop failed: {exc}")

    return None


def get_remaining_schedule(
    season: str = CURRENT_SEASON,
    completed_game_ids: set | None = None,
    today_str: str | None = None,
    use_cache: bool = True,
) -> list[dict]:
    """
    Fetch remaining regular season games as a list of dicts:
      [{"home_id": int, "away_id": int, "game_date": "YYYY-MM-DD"}, ...]

    NBA GAME_ID format: "00" + game_type + season_yr_2digit + sequence
      game_type 2 = regular season  →  2025-26 regular season prefix = "002225"

    Tries nba_api endpoint then direct HTTP. Returns [] on total failure so
    callers can fall back gracefully.
    """
    if today_str is None:
        today_str = date.today().strftime("%Y-%m-%d")
    if completed_game_ids is None:
        completed_game_ids = set()

    cache_key = f"remaining_sched_{season.replace('-', '_')}_{today_str}"
    if use_cache:
        cached = _load_cache(cache_key, max_age_hours=4)
        if cached is not None:
            print(f"  [cache] Remaining schedule: {len(cached)} games")
            return cached

    print(f"  [api]   Fetching remaining schedule for {season} ...")
    df = _fetch_full_schedule(season, today_str=today_str)
    if df is None or df.empty:
        print(f"  [warn]  All schedule fetch attempts failed — returning []")
        return []

    df["GAME_ID"] = df["GAME_ID"].astype(str)

    # NBA GAME_ID: "00" + game_type("2"=reg) + year_2digit + 5-digit sequence
    # e.g. "0022500883" = NBA(00) + regular(2) + 2025(25) + game(00883)
    reg_prefix = "002" + season.split("-")[0][-2:]    # "00225" for 2025-26
    before_filter = len(df)
    df = df[df["GAME_ID"].str.startswith(reg_prefix)].copy()
    print(f"  [api]   Regular season filter ({reg_prefix}*): "
          f"{len(df)} of {before_filter} rows kept")

    # Keep only unplayed games
    status_col = "GAME_STATUS_TEXT" if "GAME_STATUS_TEXT" in df.columns else None
    mask = ~df["GAME_ID"].isin(completed_game_ids)
    if status_col:
        mask = mask & (df[status_col].str.strip() != "Final")
    future = df[mask]

    date_col = "GAME_DATE_EST" if "GAME_DATE_EST" in df.columns else "GAME_DATE"
    result = [
        {
            "home_id":   int(row["HOME_TEAM_ID"]),
            "away_id":   int(row["VISITOR_TEAM_ID"]),
            "game_date": str(row[date_col])[:10],
        }
        for _, row in future.iterrows()
    ]
    if use_cache and result:
        _save_cache(cache_key, result)
    print(f"  [api]   {len(result)} remaining regular season games found")
    return result


if __name__ == "__main__":
    # Quick smoke-test when run directly
    print("=== NBA Data Fetcher — Smoke Test ===\n")

    print("Teams:")
    teams_df = get_all_teams()
    print(teams_df.head(5).to_string(index=False))
    print(f"  Total teams: {len(teams_df)}\n")

    print("Season games (2025-26):")
    games_df = get_season_games()
    print(f"  Rows returned: {len(games_df)}")
    if not games_df.empty:
        print(f"  Columns: {list(games_df.columns)}")
        print(f"  Date range: {games_df['GAME_DATE'].min()} → {games_df['GAME_DATE'].max()}")
    print()
