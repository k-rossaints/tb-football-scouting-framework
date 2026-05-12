"""
features.py — Build per-player feature matrix from raw StatsBomb events.

Pipeline:
    1. Load raw event parquets (data/raw/events_*.parquet).
    2. Extract (x, y) coordinates from StatsBomb location arrays.
    3. Compute minutes played per (match, player).
    4. Aggregate ~40 metrics per player over the season, normalised per 90 min.
    5. Filter players with fewer than 450 minutes total.
    6. Save to data/processed/player_features.parquet.

No row-level loops: everything is vectorised through groupby / merge_asof.
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITION_MAP = {
    'Center Back': 'CB',
    'Left Center Back': 'CB', 'Right Center Back': 'CB',
    'Left Back': 'FB', 'Right Back': 'FB',
    'Left Wing Back': 'FB', 'Right Wing Back': 'FB',
    'Center Defensive Midfield': 'MF',
    'Left Defensive Midfield': 'MF', 'Right Defensive Midfield': 'MF',
    'Center Midfield': 'MF',
    'Left Center Midfield': 'MF', 'Right Center Midfield': 'MF',
    'Left Midfield': 'MF', 'Right Midfield': 'MF',
    'Center Attacking Midfield': 'AM',
    'Left Attacking Midfield': 'AM', 'Right Attacking Midfield': 'AM',
    'Left Wing': 'AM', 'Right Wing': 'AM',
    'Center Forward': 'ST',
    'Left Center Forward': 'ST', 'Right Center Forward': 'ST',
    'Secondary Striker': 'ST',
}

MIN_MINUTES = 450        # minimum season minutes for inclusion
PITCH_X = 120.0          # StatsBomb pitch length
PITCH_Y = 80.0           # StatsBomb pitch width
FINAL_THIRD_X = 80.0     # x >= 80 means attacking third
BOX_X_MIN = 102.0        # opponent penalty area
BOX_Y_MIN, BOX_Y_MAX = 18.0, 62.0
PROG_PASS_DELTA = 10.0   # min delta_x to qualify as a progressive pass
PROG_CARRY_DELTA = 5.0   # min delta_x to qualify as a progressive carry
LONG_PASS_LEN = 35.0     # long-pass threshold (yards)
PRESSURE_WINDOW = 5.0    # seconds — pressing-success time window

# Event types that signal possession won / kept by the pressing team
POSSESSION_RECOVERY_TYPES = {
    'Ball Recovery', 'Interception', 'Pass', 'Carry', 'Shot', 'Clearance',
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_concat(raw_dir, prefix):
    """Load and concatenate every ``{prefix}_*.parquet`` file in raw_dir."""
    files = sorted(glob.glob(os.path.join(raw_dir, f'{prefix}_*.parquet')))
    if not files:
        raise FileNotFoundError(f'No {prefix}_*.parquet file in {raw_dir}')
    frames = [pd.read_parquet(p) for p in files]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Vectorised helpers
# ---------------------------------------------------------------------------

def _extract_xy(series):
    """Extract (x, y) arrays from a StatsBomb location column.

    StatsBomb stores each location as a list/ndarray ``[x, y]`` (or
    ``[x, y, z]`` for some shots). Missing values become NaN.

    Args:
        series: Column holding arrays / lists / None.

    Returns:
        Tuple of two float arrays (x, y) of the same length as ``series``.
    """
    mask = series.notna().to_numpy()
    x = np.full(len(series), np.nan, dtype='float64')
    y = np.full(len(series), np.nan, dtype='float64')
    if mask.any():
        # Truncate to first two coords to handle [x, y, z] shot locations.
        vals = np.array(
            [np.asarray(v, dtype='float64')[:2] for v in series[mask]],
            dtype='float64',
        )
        x[mask] = vals[:, 0]
        y[mask] = vals[:, 1]
    return x, y


def _prepare_events(events):
    """Enrich an events DataFrame with derived columns.

    Adds: ``t_sec`` (continuous seconds since kick-off), ``start_x/start_y``,
    ``pass_end_x/pass_end_y``, ``carry_end_x/carry_end_y``, and coerces
    sparse boolean flags (``under_pressure``, ``counterpress``, ...) to
    proper bools so they can be used in vectorised masks.

    Args:
        events: Raw, concatenated event DataFrame.

    Returns:
        Enriched copy, sorted by ``(match_id, period, t_sec)``.
    """
    df = events.copy()

    # Continuous timestamp in seconds (per half — merge_asof groups by period)
    df['t_sec'] = df['minute'].astype('float64') * 60.0 + df['second'].astype('float64')

    # Locations
    sx, sy = _extract_xy(df['location'])
    df['start_x'], df['start_y'] = sx, sy

    pex, pey = _extract_xy(df['pass_end_location'])
    df['pass_end_x'], df['pass_end_y'] = pex, pey

    cex, cey = _extract_xy(df['carry_end_location'])
    df['carry_end_x'], df['carry_end_y'] = cex, cey

    # Coerce sparse booleans (None / NaN -> False)
    for col in ('under_pressure', 'counterpress',
                'pass_shot_assist', 'pass_goal_assist',
                'pass_aerial_won', 'clearance_aerial_won',
                'shot_aerial_won', 'miscontrol_aerial_won',
                'pass_cross', 'pass_switch', 'shot_first_time'):
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    df = df.sort_values(['match_id', 'period', 't_sec'], kind='stable')
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Minutes played
# ---------------------------------------------------------------------------

def compute_minutes(events):
    """Compute minutes played per ``(match_id, player_id)``.

    Definition: difference between the player's first and last event in the
    match. This is exact for starters; substitutes are slightly under-
    estimated on the edges of their window, but the 450-minute season
    filter smooths the effect.

    Args:
        events: Enriched events (must contain ``t_sec``).

    Returns:
        DataFrame with columns ``match_id, player_id, minutes_match``.
    """
    df = events.dropna(subset=['player_id']).copy()
    grp = df.groupby(['match_id', 'player_id'], observed=True)['t_sec']
    minutes = ((grp.max() - grp.min()) / 60.0).rename('minutes_match').reset_index()
    minutes['minutes_match'] = minutes['minutes_match'].clip(lower=0)
    return minutes


# ---------------------------------------------------------------------------
# Pressing success: a Pressure followed by same-team recovery within 5 s
# ---------------------------------------------------------------------------

def _flag_pressure_success(events):
    """Flag each ``Pressure`` as successful when the pressing team touches the
    ball within :data:`PRESSURE_WINDOW` seconds.

    Vectorised through ``merge_asof``: for every Pressure event we search
    forward for the next recovery/possession event of the same team in the
    same half within the time tolerance. A sentinel column (``_rid``) on
    the right side lets us detect whether a match was found.

    Args:
        events: Enriched events.

    Returns:
        Boolean Series aligned on the events index — True iff successful.
    """
    is_press = events['type'].eq('Pressure')
    success = pd.Series(False, index=events.index)

    recov_mask = events['type'].isin(POSSESSION_RECOVERY_TYPES)
    recov = (
        events.loc[recov_mask, ['match_id', 'period', 'team', 't_sec']]
              .assign(_rid=1)
              .sort_values('t_sec', kind='stable')
              .reset_index(drop=True)
    )

    press = events.loc[is_press, ['match_id', 'period', 'team', 't_sec']].copy()
    press = press.sort_values('t_sec', kind='stable')
    press_idx = press.index
    press = press.reset_index(drop=True)

    merged = pd.merge_asof(
        press, recov,
        by=['team', 'match_id', 'period'],
        on='t_sec',
        direction='forward',
        allow_exact_matches=False,
        tolerance=PRESSURE_WINDOW,
    )
    success.loc[press_idx] = merged['_rid'].fillna(0).astype(bool).to_numpy()
    return success


# ---------------------------------------------------------------------------
# xA: link each pass to the xG of the shot it created
# ---------------------------------------------------------------------------

def _compute_xa_per_pass(events):
    """Attach to each pass the xG of the resulting shot (its xA value).

    A pass is a potential assist iff ``pass_assisted_shot_id`` is set; this
    id is joined onto the shot's ``id`` to retrieve the shot's xG.

    Args:
        events: Enriched events.

    Returns:
        Series of xA values (0 for non-assist passes), aligned on the
        events index.
    """
    shots = events.loc[events['type'].eq('Shot'), ['id', 'shot_statsbomb_xg']].rename(
        columns={'id': 'pass_assisted_shot_id', 'shot_statsbomb_xg': '_xa'}
    )
    xa = events[['pass_assisted_shot_id']].merge(
        shots, on='pass_assisted_shot_id', how='left'
    )['_xa']
    return pd.Series(xa.fillna(0.0).to_numpy(), index=events.index)


# ---------------------------------------------------------------------------
# Dominant position per player
# ---------------------------------------------------------------------------

def _dominant_position(events):
    """Pick each player's most-frequent (grouped) position over the season.

    Counts events by ``(player_id, StatsBomb position)``, maps through
    :data:`POSITION_MAP`, then keeps the dominant group. Players with no
    mappable position (goalkeepers, unused substitutes) get NaN.

    Args:
        events: Enriched events.

    Returns:
        DataFrame with columns ``player_id, position_group, position_raw``.
    """
    df = events.dropna(subset=['player_id', 'position']).copy()
    df['position_group'] = df['position'].map(POSITION_MAP)

    counts = (
        df.groupby(['player_id', 'position_group', 'position'], observed=True)
          .size().rename('n').reset_index()
    )
    # Most frequent raw position (informational)
    raw_top = (
        counts.sort_values(['player_id', 'n'], ascending=[True, False])
              .drop_duplicates('player_id')
              [['player_id', 'position']]
              .rename(columns={'position': 'position_raw'})
    )
    # Most frequent grouped position
    group_counts = counts.groupby(['player_id', 'position_group'],
                                  observed=True)['n'].sum().reset_index()
    group_top = (
        group_counts.sort_values(['player_id', 'n'], ascending=[True, False])
                    .drop_duplicates('player_id')
                    [['player_id', 'position_group']]
    )
    return group_top.merge(raw_top, on='player_id', how='left')


# ---------------------------------------------------------------------------
# Per-player feature aggregation
# ---------------------------------------------------------------------------

def _aggregate_player_counts(events):
    """Sum every action indicator per player over the season.

    Builds boolean / integer indicators for each event of interest, then
    sums them per ``player_id``. All metrics are computed vectorially —
    no per-row iteration.

    Args:
        events: Enriched events (output of :func:`_prepare_events`).

    Returns:
        DataFrame indexed by ``player_id`` with raw count columns.
    """
    e = events  # short alias
    is_pass    = e['type'].eq('Pass')
    is_shot    = e['type'].eq('Shot')
    is_carry   = e['type'].eq('Carry')
    is_dribble = e['type'].eq('Dribble')
    is_duel    = e['type'].eq('Duel')
    is_press   = e['type'].eq('Pressure')

    pass_completed = is_pass & e['pass_outcome'].isna()

    # --- Passing
    start_x = e['start_x']
    end_x   = e['pass_end_x']
    long_pass = is_pass & (e['pass_length'] >= LONG_PASS_LEN)
    prog_pass = pass_completed & (end_x > start_x + PROG_PASS_DELTA)
    f3_pass   = pass_completed & (end_x >= FINAL_THIRD_X) & (start_x < FINAL_THIRD_X)
    key_pass  = is_pass & (e['pass_shot_assist'] | e['pass_goal_assist'])
    pass_under_press = is_pass & e['under_pressure']
    pass_comp_under_press = pass_completed & e['under_pressure']
    long_pass_completed = pass_completed & (e['pass_length'] >= LONG_PASS_LEN)

    xa = _compute_xa_per_pass(e)

    # --- Defending
    is_clearance     = e['type'].eq('Clearance')
    is_block         = e['type'].eq('Block')
    is_interception  = e['type'].eq('Interception')
    is_recovery      = e['type'].eq('Ball Recovery')

    # Tackles = Tackle-type duels with a winning outcome
    tackle_attempt = is_duel & e['duel_type'].eq('Tackle')
    tackle_won = tackle_attempt & e['duel_outcome'].isin(
        ['Won', 'Success In Play', 'Success Out']
    )

    # Aerial duels: losses come as Duel/Aerial Lost; wins are flagged on the
    # subsequent event via *_aerial_won booleans.
    aerial_lost = is_duel & e['duel_type'].eq('Aerial Lost')
    aerial_won = (
        e['pass_aerial_won'] | e['clearance_aerial_won']
        | e['shot_aerial_won'] | e['miscontrol_aerial_won']
    )
    aerial_total = aerial_lost | aerial_won

    # Ground duels: tackle attempts + Dribbled Past (defender side)
    is_dribbled_past = e['type'].eq('Dribbled Past')
    ground_total = tackle_attempt | is_dribbled_past
    ground_won = tackle_won  # winning a ground duel == winning the tackle

    # --- Carries
    cdx = e['carry_end_x'] - start_x
    cdy = e['carry_end_y'] - e['start_y']
    carry_dist = np.where(is_carry, np.sqrt(cdx ** 2 + cdy ** 2), 0.0)
    prog_carry = is_carry & (e['carry_end_x'] > start_x + PROG_CARRY_DELTA)
    f3_carry = is_carry & (e['carry_end_x'] >= FINAL_THIRD_X) & (start_x < FINAL_THIRD_X)

    # --- Pressing
    pressure_success = _flag_pressure_success(e) & is_press

    # --- Attacking
    shot_on_target_outcomes = {'Saved', 'Goal', 'Saved to Post', 'Saved Off Target'}
    shot_on_target = is_shot & e['shot_outcome'].isin(shot_on_target_outcomes)
    xg = np.where(is_shot, e['shot_statsbomb_xg'].fillna(0.0), 0.0)

    touches_in_box = (
        e['start_x'].between(BOX_X_MIN, PITCH_X, inclusive='both')
        & e['start_y'].between(BOX_Y_MIN, BOX_Y_MAX, inclusive='both')
        & e['player_id'].notna()
    )

    dribble_complete = is_dribble & e['dribble_outcome'].eq('Complete')

    # --- Extra metrics (crosses, box entries, switches, first-time shots,
    # total defensive actions)
    crosses = is_pass & e['pass_cross']
    switches = is_pass & e['pass_switch']

    in_box_pass_end = (
        (e['pass_end_x'] >= BOX_X_MIN)
        & (e['pass_end_y'] >= BOX_Y_MIN)
        & (e['pass_end_y'] <= BOX_Y_MAX)
    )
    # Entries only: start outside the box, end inside
    passes_into_box = pass_completed & in_box_pass_end & (e['start_x'] < BOX_X_MIN)

    in_box_carry_end = (
        (e['carry_end_x'] >= BOX_X_MIN)
        & (e['carry_end_y'] >= BOX_Y_MIN)
        & (e['carry_end_y'] <= BOX_Y_MAX)
    )
    carries_into_box = is_carry & in_box_carry_end & (e['start_x'] < BOX_X_MIN)

    shots_first_touch = is_shot & e['shot_first_time']

    # defensive_actions = won tackles ∨ interceptions ∨ clearances
    defensive_actions = tackle_won | is_interception | is_clearance

    # --- General
    is_foul_committed = e['type'].eq('Foul Committed')
    is_foul_won       = e['type'].eq('Foul Won')
    is_dispossessed   = e['type'].eq('Dispossessed')
    is_miscontrol     = e['type'].eq('Miscontrol')
    # Touch = any positioned event by the player (proxy for ball touches)
    is_touch = e['player_id'].notna() & e['start_x'].notna()

    work = pd.DataFrame({
        'player_id': e['player_id'],
        # passing
        'passes_attempted':       is_pass.astype('int32'),
        'passes_completed':       pass_completed.astype('int32'),
        'progressive_passes':     prog_pass.astype('int32'),
        'passes_into_final_third': f3_pass.astype('int32'),
        'key_passes':             key_pass.astype('int32'),
        'long_passes':            long_pass.astype('int32'),
        'long_passes_completed':  long_pass_completed.astype('int32'),
        'passes_under_pressure':  pass_under_press.astype('int32'),
        'passes_completed_under_pressure': pass_comp_under_press.astype('int32'),
        'xA':                     xa.astype('float64'),
        # defending
        'tackles':                tackle_won.astype('int32'),
        'interceptions':          is_interception.astype('int32'),
        'clearances':             is_clearance.astype('int32'),
        'blocks':                 is_block.astype('int32'),
        'aerial_duels':           aerial_total.astype('int32'),
        'aerial_duels_won':       aerial_won.astype('int32'),
        'ground_duels':           ground_total.astype('int32'),
        'ground_duels_won':       ground_won.astype('int32'),
        'recoveries':             is_recovery.astype('int32'),
        # progression
        'carries':                is_carry.astype('int32'),
        'progressive_carries':    prog_carry.astype('int32'),
        'carry_distance':         carry_dist.astype('float64'),
        'carries_into_final_third': f3_carry.astype('int32'),
        # pressing
        'pressures':              is_press.astype('int32'),
        'pressures_successful':   pressure_success.astype('int32'),
        # attacking
        'shots':                  is_shot.astype('int32'),
        'shots_on_target':        shot_on_target.astype('int32'),
        'xG':                     xg.astype('float64'),
        'touches_in_box':         touches_in_box.astype('int32'),
        'dribbles_attempted':     is_dribble.astype('int32'),
        'dribbles_completed':     dribble_complete.astype('int32'),
        'shots_first_touch':      shots_first_touch.astype('int32'),
        # extra passing / progression
        'crosses':                crosses.astype('int32'),
        'switches':               switches.astype('int32'),
        'passes_into_penalty_area':  passes_into_box.astype('int32'),
        'carries_into_penalty_area': carries_into_box.astype('int32'),
        # aggregate defensive output
        'defensive_actions':      defensive_actions.astype('int32'),
        # general
        'touches':                is_touch.astype('int32'),
        'fouls_committed':        is_foul_committed.astype('int32'),
        'fouls_won':              is_foul_won.astype('int32'),
        'dispossessed':           is_dispossessed.astype('int32'),
        'miscontrols':            is_miscontrol.astype('int32'),
    })

    # Sum only events with a known player_id
    work = work.dropna(subset=['player_id'])
    return work.groupby('player_id', observed=True).sum(numeric_only=True)


def _player_meta(events):
    """Return the most-frequent name and team per player.

    Args:
        events: Enriched events.

    Returns:
        DataFrame with columns ``player_id, player_name, team``.
    """
    df = events.dropna(subset=['player_id', 'player']).copy()
    name = (
        df.groupby(['player_id', 'player'], observed=True).size()
          .rename('n').reset_index()
          .sort_values(['player_id', 'n'], ascending=[True, False])
          .drop_duplicates('player_id')
          [['player_id', 'player']]
          .rename(columns={'player': 'player_name'})
    )
    team = (
        df.groupby(['player_id', 'team'], observed=True).size()
          .rename('n').reset_index()
          .sort_values(['player_id', 'n'], ascending=[True, False])
          .drop_duplicates('player_id')
          [['player_id', 'team']]
    )
    return name.merge(team, on='player_id', how='left')


# ---------------------------------------------------------------------------
# Rates + per-90 normalisation
# ---------------------------------------------------------------------------

def _finalize(agg, minutes_per_player, meta, positions):
    """Combine raw counts + minutes + meta, then compute rates and per-90.

    Args:
        agg: Raw per-player sums (output of :func:`_aggregate_player_counts`).
        minutes_per_player: Total season minutes per player_id.
        meta: Player name + team per player_id.
        positions: Dominant position per player_id.

    Returns:
        Final, ready-to-write feature table.
    """
    df = agg.copy()
    df['minutes_total'] = minutes_per_player.reindex(df.index).fillna(0.0)

    # --- Rates (these are ratios, NOT per-90 quantities)
    def _safe_div(a, b):
        return np.where(b > 0, a / np.where(b == 0, 1, b), np.nan)

    df['pass_completion_rate'] = _safe_div(df['passes_completed'], df['passes_attempted'])
    df['long_pass_completion_rate'] = _safe_div(df['long_passes_completed'], df['long_passes'])
    df['pass_completion_under_pressure'] = _safe_div(
        df['passes_completed_under_pressure'], df['passes_under_pressure']
    )
    df['aerial_duel_win_rate'] = _safe_div(df['aerial_duels_won'], df['aerial_duels'])
    df['ground_duel_win_rate'] = _safe_div(df['ground_duels_won'], df['ground_duels'])
    df['pressure_success_rate'] = _safe_div(df['pressures_successful'], df['pressures'])
    df['shot_on_target_rate'] = _safe_div(df['shots_on_target'], df['shots'])
    df['xG_per_shot'] = _safe_div(df['xG'], df['shots'])
    df['dribble_success_rate'] = _safe_div(df['dribbles_completed'], df['dribbles_attempted'])

    # --- Per-90 normalisation (count columns only; rates and minutes excluded)
    rate_cols = {
        'pass_completion_rate', 'long_pass_completion_rate',
        'pass_completion_under_pressure', 'aerial_duel_win_rate',
        'ground_duel_win_rate', 'pressure_success_rate',
        'shot_on_target_rate', 'xG_per_shot', 'dribble_success_rate',
        'minutes_total',
    }
    p90_factor = 90.0 / df['minutes_total'].replace(0, np.nan)
    count_cols = [c for c in df.columns if c not in rate_cols]
    p90 = df[count_cols].mul(p90_factor, axis=0)
    p90.columns = [f'{c}_p90' for c in count_cols]

    out = pd.concat([df, p90], axis=1).reset_index()

    # Attach metadata
    out = out.merge(meta, on='player_id', how='left')
    out = out.merge(positions, on='player_id', how='left')

    # Filter: minutes threshold + drop goalkeepers / unmappable positions
    out = out[out['minutes_total'] >= MIN_MINUTES].copy()
    out = out[out['position_group'].notna()].copy()

    # Reorder: metadata first
    front = ['player_id', 'player_name', 'team',
             'position_group', 'position_raw', 'minutes_total']
    front = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest].sort_values(['position_group', 'player_name']).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# Columns actually used downstream (~5x memory reduction at load time)
_USED_COLUMNS = [
    'id', 'match_id', 'period', 'minute', 'second',
    'type', 'team', 'player', 'player_id', 'position',
    'location', 'pass_end_location', 'carry_end_location',
    'duration',
    'pass_outcome', 'pass_length', 'pass_shot_assist', 'pass_goal_assist',
    'pass_assisted_shot_id', 'pass_aerial_won',
    'pass_cross', 'pass_switch',
    'shot_outcome', 'shot_statsbomb_xg', 'shot_first_time',
    'duel_type', 'duel_outcome',
    'dribble_outcome',
    'under_pressure', 'counterpress',
    'clearance_aerial_won', 'shot_aerial_won', 'miscontrol_aerial_won',
]


def _process_file(path):
    """Process one events parquet file in memory isolation.

    Reads a single file with only :data:`_USED_COLUMNS`, prepares the
    events, and returns partial aggregates (minutes per match, counts per
    player, position counts, name/team counts).

    Args:
        path: Path to one ``events_*.parquet``.

    Returns:
        Dict ``{minutes, agg, positions_counts, meta_counts}`` of partial
        DataFrames, ready to be combined across files.
    """
    df = pd.read_parquet(path, columns=_USED_COLUMNS)
    df = _prepare_events(df)

    minutes = compute_minutes(df)
    agg = _aggregate_player_counts(df)

    # Position counts per (player_id, position) — merged across files later
    pos_df = df.dropna(subset=['player_id', 'position'])
    pos_counts = (
        pos_df.groupby(['player_id', 'position'], observed=True)
              .size().rename('n').reset_index()
    )

    # Meta counts per (player_id, player, team)
    meta_df = df.dropna(subset=['player_id', 'player'])
    meta_counts = (
        meta_df.groupby(['player_id', 'player', 'team'], observed=True)
               .size().rename('n').reset_index()
    )

    return {
        'minutes': minutes,
        'agg': agg,
        'positions_counts': pos_counts,
        'meta_counts': meta_counts,
    }


def _reduce_positions(pos_counts):
    """Reduce multi-file position counts to a dominant position per player.

    Args:
        pos_counts: Columns ``player_id, position, n``.

    Returns:
        DataFrame with columns ``player_id, position_group, position_raw``.
    """
    pos_counts = pos_counts.copy()
    pos_counts['position_group'] = pos_counts['position'].map(POSITION_MAP)

    raw_top = (
        pos_counts.groupby(['player_id', 'position'], as_index=False)['n'].sum()
                  .sort_values(['player_id', 'n'], ascending=[True, False])
                  .drop_duplicates('player_id')
                  [['player_id', 'position']]
                  .rename(columns={'position': 'position_raw'})
    )
    group_top = (
        pos_counts.groupby(['player_id', 'position_group'], as_index=False)['n'].sum()
                  .sort_values(['player_id', 'n'], ascending=[True, False])
                  .drop_duplicates('player_id')
                  [['player_id', 'position_group']]
    )
    return group_top.merge(raw_top, on='player_id', how='left')


def _reduce_meta(meta_counts):
    """Reduce multi-file (player_id, player, team) counts to one row per player.

    Args:
        meta_counts: Columns ``player_id, player, team, n``.

    Returns:
        DataFrame with columns ``player_id, player_name, team``.
    """
    name = (
        meta_counts.groupby(['player_id', 'player'], as_index=False)['n'].sum()
                   .sort_values(['player_id', 'n'], ascending=[True, False])
                   .drop_duplicates('player_id')
                   .rename(columns={'player': 'player_name'})[['player_id', 'player_name']]
    )
    team = (
        meta_counts.groupby(['player_id', 'team'], as_index=False)['n'].sum()
                   .sort_values(['player_id', 'n'], ascending=[True, False])
                   .drop_duplicates('player_id')[['player_id', 'team']]
    )
    return name.merge(team, on='player_id', how='left')


def build_features(raw_dir=DATA_RAW, output_dir=DATA_PROCESSED):
    """Build the per-player feature table and save it as parquet.

    To stay within a reasonable memory envelope (~6M events total in 2015/16),
    each ``events_{cid}_{sid}.parquet`` file is processed independently; the
    partial aggregates (minutes, counts, positions, meta) are then summed
    across files.

    Steps:
        1. For each ``events_*.parquet`` file: :func:`_process_file`.
        2. Sum minutes → total minutes per player.
        3. Sum counts → final per-player sums.
        4. Reduce positions and meta to dominant values.
        5. Compute rates + per-90, filter ≥ ``MIN_MINUTES``, save.

    Args:
        raw_dir: Directory containing ``events_*.parquet``.
        output_dir: Destination directory for the processed parquet.

    Returns:
        The final feature DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, 'events_*.parquet')))
    if not files:
        raise FileNotFoundError(f'No events_*.parquet in {raw_dir}')

    all_minutes, all_agg = [], []
    all_pos_counts, all_meta_counts = [], []

    for i, path in enumerate(files, start=1):
        print(f'[{i}/{len(files)}] {os.path.basename(path)} ...', flush=True)
        part = _process_file(path)
        all_minutes.append(part['minutes'])
        all_agg.append(part['agg'])
        all_pos_counts.append(part['positions_counts'])
        all_meta_counts.append(part['meta_counts'])

    print('[reduce] Combining partial aggregates ...', flush=True)
    minutes_total = (
        pd.concat(all_minutes, ignore_index=True)
          .groupby('player_id', observed=True)['minutes_match'].sum()
    )
    agg = pd.concat(all_agg).groupby(level=0, observed=True).sum()
    positions = _reduce_positions(pd.concat(all_pos_counts, ignore_index=True))
    meta = _reduce_meta(pd.concat(all_meta_counts, ignore_index=True))

    print(f'[finalize] Rates + per-90 + filter ≥ {MIN_MINUTES} min ...', flush=True)
    out = _finalize(agg, minutes_total, meta, positions)

    output_path = os.path.join(output_dir, 'player_features.parquet')
    out.to_parquet(output_path, index=False)
    print(f'[DONE] {len(out)} players retained -> {output_path}')
    print(f'       {len(out.columns)} columns.')
    return out


if __name__ == '__main__':
    build_features()
