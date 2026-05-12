"""
extraction.py — Download and load raw StatsBomb event/lineup data.

Idempotent: existing parquet files are skipped, so re-running after a partial
failure resumes where it stopped.
"""

import os
import glob
import traceback
from pathlib import Path

import pandas as pd
import statsbombpy.sb as sb

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'

COMPETITIONS_TO_USE = [
    {'name': 'La Liga 2015/16',        'competition_id': 11, 'season_id': 27},
    {'name': 'Premier League 2015/16', 'competition_id': 2,  'season_id': 27},
    {'name': 'Bundesliga 2015/16',     'competition_id': 9,  'season_id': 27},
    {'name': 'Ligue 1 2015/16',        'competition_id': 7,  'season_id': 27},
    {'name': 'Serie A 2015/16',        'competition_id': 12, 'season_id': 27},
]


def extract_and_save(competitions_list=None, output_dir=DATA_RAW):
    """Download StatsBomb events and lineups, save one parquet per competition.

    For each competition in ``competitions_list``, fetch the match list via
    ``statsbombpy``, then iterate matches to pull events and lineups. Results
    are aggregated by (competition, season) and written as parquet to
    ``output_dir``. Existing files are skipped (no re-download).

    Args:
        competitions_list: List of ``{name, competition_id, season_id}`` dicts.
            Defaults to :data:`COMPETITIONS_TO_USE`.
        output_dir: Destination directory for the parquet files.
    """
    if competitions_list is None:
        competitions_list = COMPETITIONS_TO_USE

    os.makedirs(output_dir, exist_ok=True)

    for comp in competitions_list:
        name = comp['name']
        cid  = comp['competition_id']
        sid  = comp['season_id']

        events_path  = os.path.join(output_dir, f'events_{cid}_{sid}.parquet')
        lineups_path = os.path.join(output_dir, f'lineups_{cid}_{sid}.parquet')

        events_done  = os.path.exists(events_path)
        lineups_done = os.path.exists(lineups_path)

        if events_done and lineups_done:
            print(f'[SKIP] {name} — files already present.')
            continue

        print(f'\n[START] {name} (competition_id={cid}, season_id={sid})')

        try:
            matches = sb.matches(competition_id=cid, season_id=sid)
        except Exception:
            print(f'  [ERROR] Could not load matches for {name}:')
            traceback.print_exc()
            continue

        if matches is None or matches.empty:
            print(f'  [WARN] No matches found for {name}.')
            continue

        match_ids = matches['match_id'].tolist()
        print(f'  {len(match_ids)} matches found.')

        all_events  = []
        all_lineups = []

        for i, mid in enumerate(match_ids, start=1):
            print(f'  [{i}/{len(match_ids)}] match_id={mid}', end=' ')

            if not events_done:
                try:
                    events = sb.events(match_id=mid)
                    events['match_id'] = mid
                    all_events.append(events)
                    print('events OK', end=' ')
                except Exception:
                    print(f'\n    [ERROR] events match_id={mid}:')
                    traceback.print_exc()

            if not lineups_done:
                try:
                    raw_lineups = sb.lineups(match_id=mid)
                    # sb.lineups() returns a dict {team_name: DataFrame}
                    frames = []
                    for team_name, df in raw_lineups.items():
                        df = df.copy()
                        df['team_name'] = team_name
                        df['match_id']  = mid
                        frames.append(df)
                    if frames:
                        all_lineups.append(pd.concat(frames, ignore_index=True))
                    print('lineups OK')
                except Exception:
                    print(f'\n    [ERROR] lineups match_id={mid}:')
                    traceback.print_exc()

        if not events_done and all_events:
            try:
                df_events = pd.concat(all_events, ignore_index=True)
                df_events.to_parquet(events_path, index=False)
                print(f'  [SAVED] {events_path} ({len(df_events)} rows)')
            except Exception:
                print(f'  [ERROR] Saving events for {name}:')
                traceback.print_exc()

        if not lineups_done and all_lineups:
            try:
                df_lineups = pd.concat(all_lineups, ignore_index=True)
                df_lineups.to_parquet(lineups_path, index=False)
                print(f'  [SAVED] {lineups_path} ({len(df_lineups)} rows)')
            except Exception:
                print(f'  [ERROR] Saving lineups for {name}:')
                traceback.print_exc()

    print('\n[DONE] Extraction complete.')


def load_raw_events(raw_dir=DATA_RAW):
    """Load and concatenate all ``events_*.parquet`` files into one DataFrame.

    Args:
        raw_dir: Directory containing the parquet files.

    Returns:
        Combined DataFrame, or empty DataFrame if no file is found.
    """
    pattern = os.path.join(raw_dir, 'events_*.parquet')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f'[WARN] No events file found in {raw_dir}')
        return pd.DataFrame()

    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            print(f'[ERROR] Reading {path}:')
            traceback.print_exc()

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f'[LOAD] {len(files)} events file(s) — {len(df)} rows total.')
    return df


def load_raw_lineups(raw_dir=DATA_RAW):
    """Load and concatenate all ``lineups_*.parquet`` files into one DataFrame.

    Args:
        raw_dir: Directory containing the parquet files.

    Returns:
        Combined DataFrame, or empty DataFrame if no file is found.
    """
    pattern = os.path.join(raw_dir, 'lineups_*.parquet')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f'[WARN] No lineups file found in {raw_dir}')
        return pd.DataFrame()

    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            print(f'[ERROR] Reading {path}:')
            traceback.print_exc()

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f'[LOAD] {len(files)} lineups file(s) — {len(df)} rows total.')
    return df
