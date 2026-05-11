"""
extraction.py — Téléchargement et chargement des données StatsBomb brutes.
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
    """Télécharge les events et lineups StatsBomb et les sauvegarde en parquet.

    Pour chaque compétition de `competitions_list`, charge la liste des matchs
    via statsbombpy, puis pour chaque match récupère les events et les lineups.
    Les données sont agrégées par compétition/saison et écrites en parquet dans
    `output_dir`. Un fichier déjà présent est ignoré (pas de re-téléchargement).

    Args:
        competitions_list (list[dict] | None): Liste de dicts avec les clés
            ``name``, ``competition_id`` et ``season_id``. Si None, utilise
            :data:`COMPETITIONS_TO_USE`.
        output_dir (str): Répertoire de destination des fichiers parquet.
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
            print(f'[SKIP] {name} — fichiers déjà présents.')
            continue

        print(f'\n[START] {name} (competition_id={cid}, season_id={sid})')

        try:
            matches = sb.matches(competition_id=cid, season_id=sid)
        except Exception:
            print(f'  [ERROR] Impossible de charger les matchs pour {name}:')
            traceback.print_exc()
            continue

        if matches is None or matches.empty:
            print(f'  [WARN] Aucun match trouvé pour {name}.')
            continue

        match_ids = matches['match_id'].tolist()
        print(f'  {len(match_ids)} matchs trouvés.')

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
                    # lineups() renvoie un dict {équipe: DataFrame}
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
                print(f'  [SAVED] {events_path} ({len(df_events)} lignes)')
            except Exception:
                print(f'  [ERROR] Sauvegarde events {name}:')
                traceback.print_exc()

        if not lineups_done and all_lineups:
            try:
                df_lineups = pd.concat(all_lineups, ignore_index=True)
                df_lineups.to_parquet(lineups_path, index=False)
                print(f'  [SAVED] {lineups_path} ({len(df_lineups)} lignes)')
            except Exception:
                print(f'  [ERROR] Sauvegarde lineups {name}:')
                traceback.print_exc()

    print('\n[DONE] Extraction terminée.')


def load_raw_events(raw_dir=DATA_RAW):
    """Charge et concatène tous les fichiers events parquet en un seul DataFrame.

    Args:
        raw_dir (str): Répertoire contenant les fichiers ``events_*.parquet``.

    Returns:
        pd.DataFrame: DataFrame consolidé de tous les events. Vide si aucun
        fichier n'est trouvé.
    """
    pattern = os.path.join(raw_dir, 'events_*.parquet')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f'[WARN] Aucun fichier events trouvé dans {raw_dir}')
        return pd.DataFrame()

    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            print(f'[ERROR] Lecture {path}:')
            traceback.print_exc()

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f'[LOAD] {len(files)} fichier(s) events — {len(df)} lignes au total.')
    return df


def load_raw_lineups(raw_dir=DATA_RAW):
    """Charge et concatène tous les fichiers lineups parquet en un seul DataFrame.

    Args:
        raw_dir (str): Répertoire contenant les fichiers ``lineups_*.parquet``.

    Returns:
        pd.DataFrame: DataFrame consolidé de tous les lineups. Vide si aucun
        fichier n'est trouvé.
    """
    pattern = os.path.join(raw_dir, 'lineups_*.parquet')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f'[WARN] Aucun fichier lineups trouvé dans {raw_dir}')
        return pd.DataFrame()

    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            print(f'[ERROR] Lecture {path}:')
            traceback.print_exc()

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f'[LOAD] {len(files)} fichier(s) lineups — {len(df)} lignes au total.')
    return df
