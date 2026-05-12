"""
matching.py — Score players against tactical role profiles and find similar
players.

Conceptual pipeline:
    1. Each metric of a player is converted to a *percentile rank* within
       the player's ``position_group`` (∈ [0, 100]). Comparisons are
       therefore strictly intra-position: a CB is benchmarked against CBs.
    2. The role-match score is the weighted average of those percentile
       ranks, weighted by the metric weights defined in
       ``config/role_profiles.yaml`` and bounded in [0, 100].
    3. Player-vs-player similarity is the cosine similarity of MinMax-
       normalised metric vectors (also intra-position).

The YAML uses the ``_per90`` suffix convention while the parquet uses
``_p90``; :func:`_resolve_metric` handles the translation and gracefully
drops metrics that don't exist in the feature table.
"""

import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
CONFIG_DIR = PROJECT_ROOT / 'config'


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_CLUSTERED = DATA_PROCESSED / 'player_clustered.parquet'
DEFAULT_PROFILES  = CONFIG_DIR / 'role_profiles.yaml'

# Realistic excellence reference for radar charts and strength/weakness
# tooltips. The score itself is the weighted *percentile rank* average, so a
# 75th-percentile player on every metric scores ~75/100 — perfectly aligned
# with this reference.
IDEAL_PERCENTILE = 75
TOP_STRENGTHS = 3
TOP_WEAKNESSES = 3

# Explicit YAML name -> parquet column aliases (handle minor naming drift)
_ALIASES = {
    'dribbles_success_rate': 'dribble_success_rate',
}

# Metric categories — used by :func:`list_available_metrics` so a user
# building a custom role profile can browse what is available. Membership
# is by metric *stem* (no _p90 suffix); the categoriser checks both forms.
_METRIC_CATEGORIES = {
    'passing': {
        'passes_attempted', 'passes_completed', 'progressive_passes',
        'passes_into_final_third', 'key_passes', 'long_passes',
        'long_passes_completed', 'passes_under_pressure',
        'passes_completed_under_pressure', 'xA', 'crosses', 'switches',
        'passes_into_penalty_area',
    },
    'defending': {
        'tackles', 'interceptions', 'clearances', 'blocks',
        'aerial_duels', 'aerial_duels_won', 'ground_duels',
        'ground_duels_won', 'recoveries', 'defensive_actions',
    },
    'progression': {
        'carries', 'progressive_carries', 'carry_distance',
        'carries_into_final_third', 'carries_into_penalty_area',
    },
    'pressing': {
        'pressures', 'pressures_successful',
    },
    'attacking': {
        'shots', 'shots_on_target', 'xG', 'touches_in_box',
        'dribbles_attempted', 'dribbles_completed', 'shots_first_touch',
    },
    'general': {
        'touches', 'fouls_committed', 'fouls_won', 'dispossessed',
        'miscontrols',
    },
    'rates': {
        'pass_completion_rate', 'long_pass_completion_rate',
        'pass_completion_under_pressure', 'aerial_duel_win_rate',
        'ground_duel_win_rate', 'pressure_success_rate',
        'shot_on_target_rate', 'xG_per_shot', 'dribble_success_rate',
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_accents(s):
    """Remove diacritics from a string (used for accent-insensitive search)."""
    if not isinstance(s, str):
        return ''
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(c))


def _name_match(series, query):
    """Return a boolean mask for rows of ``series`` matching ``query`` in an
    accent- and case-insensitive way."""
    q = _strip_accents(query).lower()
    return series.fillna('').map(lambda s: q in _strip_accents(s).lower())


def _safe_print(s):
    """Print that never raises UnicodeEncodeError on cp1252 consoles."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'))


# ---------------------------------------------------------------------------
# Lazy cache (populated on first use)
# ---------------------------------------------------------------------------

_CACHE = {
    'df': None,
    'profiles': None,
    'norm': {},        # position -> (MinMax-normalised DataFrame, columns)
    'pct': {},         # position -> (percentile rank 0-100 DataFrame, columns)
    'warned': set(),   # role/metric pairs already reported as missing
}


def _load_data(features_path=DEFAULT_CLUSTERED, profiles_path=DEFAULT_PROFILES):
    """Load (and cache) the clustered player parquet and the role YAML.

    Args:
        features_path: Path to the clustered player parquet.
        profiles_path: Path to the role profiles YAML.

    Returns:
        Tuple ``(df, profiles)``.
    """
    if _CACHE['df'] is None:
        _CACHE['df'] = pd.read_parquet(features_path)
    if _CACHE['profiles'] is None:
        with open(profiles_path, 'r', encoding='utf-8') as f:
            _CACHE['profiles'] = yaml.safe_load(f)
    return _CACHE['df'], _CACHE['profiles']


def reset_cache():
    """Empty the cache (call after editing the parquet or YAML in-process)."""
    _CACHE['df'] = None
    _CACHE['profiles'] = None
    _CACHE['norm'] = {}
    _CACHE['pct'] = {}
    _CACHE['warned'] = set()


# ---------------------------------------------------------------------------
# YAML name -> parquet column resolution
# ---------------------------------------------------------------------------

def _resolve_metric(name, df_columns):
    """Translate a YAML metric name to the matching parquet column.

    Resolution order:
        1. The name as-is.
        2. An explicit alias from :data:`_ALIASES`.
        3. Suffix substitution ``_per90 -> _p90``.
        4. Suffix substitution ``_p90 -> _per90`` (reverse).

    Args:
        name: Metric name as written in the YAML.
        df_columns: Available columns in the DataFrame.

    Returns:
        The actual column name, or ``None`` if no match is found.
    """
    cols = set(df_columns)
    if name in cols:
        return name
    if name in _ALIASES and _ALIASES[name] in cols:
        return _ALIASES[name]
    if name.endswith('_per90'):
        cand = name[:-len('_per90')] + '_p90'
        if cand in cols:
            return cand
    if name.endswith('_p90'):
        cand = name[:-len('_p90')] + '_per90'
        if cand in cols:
            return cand
    return None


def _resolved_weights(role_weights, df_columns, role_name='?'):
    """Convert YAML weights into ``{parquet_column: weight}``.

    Missing metrics are dropped (warning emitted once per role/missing-set
    pair) and the remaining weights are **rescaled** to preserve the
    original total — otherwise a role with many absent metrics would be
    artificially penalised.

    Args:
        role_weights: Weights as parsed from the YAML.
        df_columns: Available DataFrame columns.
        role_name: Role name, for warnings.

    Returns:
        Dict ``{parquet_column: rescaled_weight}``.
    """
    resolved = {}
    missing = []
    for k, w in role_weights.items():
        col = _resolve_metric(k, df_columns)
        if col is None:
            missing.append(k)
        else:
            resolved[col] = resolved.get(col, 0.0) + float(w)
    if missing:
        key = (role_name, tuple(sorted(missing)))
        if key not in _CACHE['warned']:
            print(f'[warn] role "{role_name}": {len(missing)} metric(s) '
                  f'not available in the parquet — {missing}')
            _CACHE['warned'].add(key)
    if resolved:
        orig_sum = float(sum(role_weights.values()))
        new_sum = sum(resolved.values())
        if new_sum > 0 and orig_sum > 0:
            factor = orig_sum / new_sum
            resolved = {k: v * factor for k, v in resolved.items()}
    return resolved


# ---------------------------------------------------------------------------
# Intra-position normalisation (MinMax and percentile rank)
# ---------------------------------------------------------------------------

def _feature_columns(df):
    """List the columns used for matching and similarity.

    Args:
        df: Player table.

    Returns:
        List of ``*_p90`` columns plus the numeric rate columns.
    """
    rate_like = (
        'pass_completion_rate', 'aerial_duel_win_rate', 'ground_duel_win_rate',
        'pressure_success_rate', 'shot_on_target_rate', 'xG_per_shot',
        'dribble_success_rate', 'pass_completion_under_pressure',
        'long_pass_completion_rate',
    )
    p90 = [c for c in df.columns if c.endswith('_p90')]
    rates = [c for c in rate_like if c in df.columns]
    return p90 + rates


def _get_normalized(position):
    """Return (and cache) the MinMax-normalised matrix for one position.

    Used by similarity search and radar charts.

    Args:
        position: Position code (CB, FB, MF, AM, ST).

    Returns:
        Tuple ``(DataFrame indexed by player_id, list of columns)``,
        values ∈ [0, 1].
    """
    if position in _CACHE['norm']:
        return _CACHE['norm'][position]

    df, _ = _load_data()
    sub = df[df['position_group'] == position].copy()
    cols = _feature_columns(df)
    X = sub[cols].astype('float64').fillna(0.0).to_numpy()

    Xn = MinMaxScaler().fit_transform(X)
    norm_df = pd.DataFrame(Xn, index=sub['player_id'].to_numpy(), columns=cols)

    _CACHE['norm'][position] = (norm_df, cols)
    return norm_df, cols


def _get_percentile_ranks(position):
    """Return (and cache) the percentile-rank table for one position.

    Used by the matching score. Each value is the player's rank on that
    metric within the position population, expressed in [0, 100]:
        - 100 → best player at the position on that metric
        - 50  → median
        - 0   → worst

    Args:
        position: Position code (CB, FB, MF, AM, ST).

    Returns:
        Tuple ``(DataFrame indexed by player_id, list of columns)``,
        values ∈ [0, 100].
    """
    if position in _CACHE['pct']:
        return _CACHE['pct'][position]

    df, _ = _load_data()
    sub = df[df['position_group'] == position].copy()
    cols = _feature_columns(df)

    raw = sub[cols].astype('float64').fillna(0.0)
    # rank(pct=True) returns percentiles in (0, 1]; ties handled via average.
    pct = raw.rank(pct=True, method='average') * 100.0
    pct.index = sub['player_id'].to_numpy()

    _CACHE['pct'][position] = (pct, cols)
    return pct, cols


def _ideal_vector(position, columns):
    """Reference vector at :data:`IDEAL_PERCENTILE` of each MinMax-normalised
    metric — used as the dashed reference polygon on radar charts.

    Args:
        position: Position code.
        columns: Columns to include.

    Returns:
        Series indexed by column, values ∈ [0, 1].
    """
    norm_df, _ = _get_normalized(position)
    return norm_df[columns].quantile(IDEAL_PERCENTILE / 100.0)


# ---------------------------------------------------------------------------
# Matching score
# ---------------------------------------------------------------------------

def compute_matching_score(player_row, role_weights, all_players_df,
                           role_name='?'):
    """Compute the role-match score of one player as a weighted percentile.

    Algorithm:
        1. For each role metric, take the player's percentile rank within
           the same ``position_group`` (∈ [0, 100]).
        2. Compute the weighted average ``Σ(pct_i × w_i) / Σ(w_i)``.

    Interpretation:
        - 75th percentile on every key metric → ~75/100.
        - 50th percentile (median) on everything → ~50/100.
        - A low percentile on a heavy-weighted metric drags the score down.

    Args:
        player_row: A row from the player DataFrame (must contain
            ``position_group`` and ``player_id``).
        role_weights: Role weights (YAML names accepted).
        all_players_df: Unused (kept for API stability; intra-position
            normalisation goes through the cache).
        role_name: Role name, forwarded to warnings.

    Returns:
        Score ∈ [0, 100].
    """
    position = player_row['position_group']
    pct_df, all_cols = _get_percentile_ranks(position)

    weights = _resolved_weights(role_weights, all_cols, role_name=role_name)
    if not weights:
        return 0.0

    cols = list(weights.keys())
    w = np.array([weights[c] for c in cols], dtype='float64')

    pid = player_row['player_id']
    if pid not in pct_df.index:
        return 0.0

    player_pct = pct_df.loc[pid, cols].to_numpy(dtype='float64')
    wsum = w.sum()
    if wsum <= 0:
        return 0.0
    score = float(np.dot(player_pct, w) / wsum)
    return round(max(0.0, min(100.0, score)), 2)


def _strengths_weaknesses(player_row, role_weights, role_name='?'):
    """Pick the player's top strengths and weaknesses on the role's metrics.

    Consistent with the weighted-percentile scoring:
        - **Strength** = metric where the player's percentile is high,
          weighted by the role weight. Ranked by ``percentile × weight``
          (descending). A minimum 60th-percentile threshold filters out
          mediocre metrics that just happen to have a moderate weight.
        - **Weakness** = metric where the player's percentile is low,
          weighted. Ranked by ``(100 − percentile) × weight`` — i.e. the
          metrics that hurt the score the most. Filtered at ≤ 50th
          percentile.

    Returned percentiles are scaled to [0, 1] (= percentile / 100) so the
    existing visualisation format ``{pv:.2f} vs {iv:.2f}`` reads naturally
    against the reference ``IDEAL_PERCENTILE / 100``.

    Args:
        player_row: A row from the player DataFrame.
        role_weights: Role weights.
        role_name: Role name, for warnings.

    Returns:
        Tuple ``(strengths, weaknesses)`` of lists of
        ``(metric, weighted_score, player_value, ideal_value)`` tuples,
        with values in [0, 1].
    """
    position = player_row['position_group']
    pct_df, all_cols = _get_percentile_ranks(position)
    weights = _resolved_weights(role_weights, all_cols, role_name=role_name)
    if not weights:
        return [], []

    cols = list(weights.keys())
    pid = player_row['player_id']
    if pid not in pct_df.index:
        return [], []

    player_pct = pct_df.loc[pid, cols].to_numpy(dtype='float64')  # 0..100
    w = np.array([weights[c] for c in cols], dtype='float64')

    pv = player_pct / 100.0
    iv = np.full_like(pv, IDEAL_PERCENTILE / 100.0)

    strength_score = pv * w
    s_order = np.argsort(-strength_score)
    strengths = [
        (cols[i], float(strength_score[i]), float(pv[i]), float(iv[i]))
        for i in s_order[:TOP_STRENGTHS] if pv[i] >= 0.60
    ]

    weakness_score = (1.0 - pv) * w
    w_order = np.argsort(-weakness_score)
    weaknesses = [
        (cols[i], float(weakness_score[i]), float(pv[i]), float(iv[i]))
        for i in w_order[:TOP_WEAKNESSES] if pv[i] <= 0.50
    ]
    return strengths, weaknesses


# ---------------------------------------------------------------------------
# Public API: rank_players_by_role
# ---------------------------------------------------------------------------

def rank_players_by_role(role_name, position, min_minutes=450, top=20,
                         verbose=True):
    """Rank every player of a position by their role-match score.

    Args:
        role_name: Role key in the YAML (e.g. ``deep_lying_playmaker``).
        position: Position code (CB, FB, MF, AM, ST).
        min_minutes: Minimum minutes-played threshold.
        top: How many players to detail in the printed output.
        verbose: If True, also print the strengths/weaknesses of the top
            ``top`` players.

    Returns:
        DataFrame with columns ``player_name, team, minutes_total, score,
        cluster, role_label``, sorted by ``score`` descending.
    """
    df, profiles = _load_data()
    if position not in profiles:
        raise KeyError(f'Position "{position}" missing from the YAML. '
                       f'Available: {list(profiles)}')
    if role_name not in profiles[position]:
        raise KeyError(f'Role "{role_name}" missing for {position}. '
                       f'Available: {list(profiles[position])}')

    role_weights = profiles[position][role_name]
    pool = df[(df['position_group'] == position)
              & (df['minutes_total'] >= min_minutes)].copy()

    pool['score'] = pool.apply(
        lambda r: compute_matching_score(r, role_weights, df, role_name=role_name),
        axis=1,
    )
    out = (pool[['player_name', 'team', 'minutes_total', 'score',
                 'cluster', 'role_label']]
           .sort_values('score', ascending=False)
           .reset_index(drop=True))

    if verbose:
        print(f'\n=== {position} / {role_name} — top {top} players ===')
        for i, row in out.head(top).iterrows():
            try:
                line = (f'{i+1:2d}. {row["player_name"]:<35s} '
                        f'{row["team"]:<25s} '
                        f'{row["minutes_total"]:6.0f}min  '
                        f'score={row["score"]:.2f}  [{row["role_label"]}]')
            except (TypeError, ValueError):
                line = f'{i+1:2d}. score={row["score"]:.2f}'
            _safe_print(line)

            player_row = pool[pool['player_name'] == row['player_name']].iloc[0]
            strengths, weaknesses = _strengths_weaknesses(
                player_row, role_weights, role_name=role_name)
            if strengths:
                _safe_print('     [+] ' + ' | '.join(
                    f'{m} ({pv:.2f} vs {iv:.2f})'
                    for m, _, pv, iv in strengths))
            if weaknesses:
                _safe_print('     [-] ' + ' | '.join(
                    f'{m} ({pv:.2f} vs {iv:.2f})'
                    for m, _, pv, iv in weaknesses))

    return out


# ---------------------------------------------------------------------------
# Public API: find_similar_players
# ---------------------------------------------------------------------------

def find_similar_players(player_name, position=None, top_n=10):
    """Find the players whose statistical signature most resembles the target.

    Similarity is the cosine similarity computed on the **full** intra-
    position MinMax-normalised metric vector — no role weighting is applied
    here. This is a role-agnostic, data-driven neighbourhood search.

    Args:
        player_name: Name (or fragment) of the target player. Search is
            accent- and case-insensitive.
        position: If given, restrict the target to that position group
            (useful for disambiguating homonyms). Otherwise inferred.
        top_n: Number of neighbours to return.

    Returns:
        DataFrame with columns ``player_name, team, position_group,
        minutes_total, similarity, role_label``, sorted by similarity
        descending. ``similarity`` is reported as a 0-100 percentage.
    """
    df, _ = _load_data()
    mask = _name_match(df['player_name'], player_name)
    if position is not None:
        mask &= df['position_group'] == position
    candidates = df[mask]
    if candidates.empty:
        raise KeyError(f'No player matches "{player_name}"'
                       + (f' (position={position})' if position else ''))
    if len(candidates) > 1:
        # Disambiguate homonyms by picking the one with the most minutes
        target = candidates.sort_values('minutes_total', ascending=False).iloc[0]
        _safe_print(f'[info] multiple matches ({len(candidates)}), '
                    f'selected "{target["player_name"]}" ({target["team"]})')
    else:
        target = candidates.iloc[0]

    pos = target['position_group']
    norm_df, cols = _get_normalized(pos)

    target_vec = norm_df.loc[target['player_id'], cols].to_numpy(dtype='float64')
    target_norm = np.linalg.norm(target_vec)
    if target_norm == 0:
        return pd.DataFrame()

    # Vectorised cosine: M @ target / (||M|| * ||target||)
    M = norm_df[cols].to_numpy(dtype='float64')
    norms = np.linalg.norm(M, axis=1)
    norms[norms == 0] = 1.0
    sims = (M @ target_vec) / (norms * target_norm)

    sim_series = pd.Series(sims, index=norm_df.index, name='similarity')
    pos_df = df[df['position_group'] == pos].set_index('player_id')
    out = (pos_df.join(sim_series)
                 .reset_index()
                 [['player_name', 'team', 'position_group',
                   'minutes_total', 'similarity', 'role_label']]
                 .sort_values('similarity', ascending=False))

    # Drop the target itself and scale to 0-100
    out = out[out['player_name'] != target['player_name']].head(top_n).reset_index(drop=True)
    out['similarity'] = (out['similarity'] * 100).round(2)
    return out


# ---------------------------------------------------------------------------
# Public API: profile_player
# ---------------------------------------------------------------------------

def profile_player(player_name, position=None, verbose=True):
    """Score the player against every role defined for their position.

    The role with the highest score is declared the player's *natural role*.

    Args:
        player_name: Name (or fragment) of the target player.
        position: Force the position if the name is ambiguous.
        verbose: If True, also print the ranking and highlight the
            natural role.

    Returns:
        DataFrame with columns ``role_name, score, is_natural`` sorted by
        score descending. ``is_natural`` is True for the highest score.
    """
    df, profiles = _load_data()
    mask = _name_match(df['player_name'], player_name)
    if position is not None:
        mask &= df['position_group'] == position
    matches = df[mask]
    if matches.empty:
        raise KeyError(f'No player matches "{player_name}"')
    target = matches.sort_values('minutes_total', ascending=False).iloc[0]

    pos = target['position_group']
    role_defs = profiles.get(pos, {})

    rows = [
        {'role_name': name,
         'score': compute_matching_score(target, w, df, role_name=name)}
        for name, w in role_defs.items()
    ]
    out = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    out['is_natural'] = False
    if not out.empty:
        out.loc[0, 'is_natural'] = True

    if verbose:
        _safe_print(f'\n=== Profile of "{target["player_name"]}" '
                    f'({target["team"]}, {pos}, {target["minutes_total"]:.0f} min) ===')
        for _, r in out.iterrows():
            marker = '  <-- natural role' if r['is_natural'] else ''
            print(f'  {r["role_name"]:<28s}  {r["score"]:6.2f}{marker}')
    return out


# ---------------------------------------------------------------------------
# Public API: compare_players
# ---------------------------------------------------------------------------

def compare_players(player1_name, player2_name, position, metrics=None,
                    top_variance=10):
    """Side-by-side percentile comparison of two players on key metrics.

    Both players are looked up (accent-insensitive) within the supplied
    ``position`` group; their percentile ranks within that group are then
    placed in a tidy DataFrame.

    If ``metrics`` is None, the comparison is built on the ``top_variance``
    most discriminative metrics for the position — defined here as the
    metrics whose MinMax-normalised values have the highest variance across
    the position population. (Note that percentile ranks themselves cannot
    be used to pick discriminative metrics — by construction their
    distribution is uniform and their variance is constant.)

    Args:
        player1_name: Name (fragment) of the first player.
        player2_name: Name (fragment) of the second player.
        position: Position code (CB, FB, MF, AM, ST). Both players must
            belong to this group; otherwise a ValueError is raised.
        metrics: Optional list of metric names. Names accept both
            ``_p90`` and ``_per90`` suffixes; rate columns have no suffix.
            Unknown names are warned and dropped.
        top_variance: How many metrics to keep when ``metrics`` is None
            (default 10).

    Returns:
        DataFrame indexed by metric name with columns:
        ``{player1_name}_pct``, ``{player2_name}_pct``, ``diff`` (player1
        minus player2). Sorted by absolute diff descending so the metrics
        that *most* distinguish the two players appear first.
    """
    df, _ = _load_data()

    # Locate both players within the requested position
    def _locate(name):
        mask = _name_match(df['player_name'], name)
        mask &= df['position_group'] == position
        cand = df[mask]
        if cand.empty:
            raise KeyError(f'No player matches "{name}" at position {position}.')
        return cand.sort_values('minutes_total', ascending=False).iloc[0]

    p1 = _locate(player1_name)
    p2 = _locate(player2_name)
    if p1['player_id'] == p2['player_id']:
        raise ValueError('player1 and player2 resolved to the same player.')

    pct_df, all_cols = _get_percentile_ranks(position)

    # Resolve / pick the comparison metrics
    if metrics is None:
        norm_df, _ = _get_normalized(position)
        variances = norm_df[all_cols].var().sort_values(ascending=False)
        chosen = list(variances.head(top_variance).index)
    else:
        chosen = []
        missing = []
        for m in metrics:
            col = _resolve_metric(m, all_cols)
            if col is None:
                missing.append(m)
            else:
                chosen.append(col)
        if missing:
            print(f'[warn] compare_players: dropped unknown metric(s) — {missing}')
        if not chosen:
            raise ValueError('None of the supplied metrics could be resolved.')

    p1_pct = pct_df.loc[p1['player_id'], chosen].astype('float64')
    p2_pct = pct_df.loc[p2['player_id'], chosen].astype('float64')

    short1 = p1['player_name'].split()[-1]
    short2 = p2['player_name'].split()[-1]
    c1 = f'{short1}_pct'
    c2 = f'{short2}_pct'

    out = pd.DataFrame({
        c1: p1_pct.round(1),
        c2: p2_pct.round(1),
    })
    out['diff'] = (out[c1] - out[c2]).round(1)
    out = out.reindex(out['diff'].abs().sort_values(ascending=False).index)
    out.index.name = 'metric'
    return out


# ---------------------------------------------------------------------------
# Public API: list_available_metrics
# ---------------------------------------------------------------------------

def list_available_metrics(position):
    """List every metric usable in a custom role profile, grouped by category.

    This is the discovery helper for :func:`custom_role_search`: it tells a
    user *which* metric names they can put into their ``custom_weights``
    dictionary. Only metrics that actually exist for this position group
    in the clustered parquet are returned.

    Args:
        position: Position code (CB, FB, MF, AM, ST).

    Returns:
        Dict ``{category: sorted list of metric names}`` with categories
        ``passing, defending, progression, pressing, attacking, general,
        rates``. Per-90 metrics are returned with the ``_p90`` suffix used
        in the parquet; rates have no suffix.
    """
    pct_df, cols = _get_percentile_ranks(position)
    available = set(cols)

    grouped = {cat: [] for cat in _METRIC_CATEGORIES}
    for col in available:
        stem = col[:-len('_p90')] if col.endswith('_p90') else col
        for cat, members in _METRIC_CATEGORIES.items():
            if stem in members:
                grouped[cat].append(col)
                break

    return {cat: sorted(metrics) for cat, metrics in grouped.items() if metrics}


# ---------------------------------------------------------------------------
# Public API: custom_role_search
# ---------------------------------------------------------------------------

def _format_metric_tuples(items):
    """Format a list of ``(metric, _, pv, _)`` tuples as a compact string."""
    return ' | '.join(f'{m} ({pv * 100:.0f}p)' for m, _, pv, _ in items)


def custom_role_search(position, custom_weights, min_minutes=450, top=20,
                       verbose=True):
    """Rank every player of a position against a user-defined role profile.

    Unlike :func:`rank_players_by_role`, the weights are supplied directly
    by the caller — there is no YAML lookup. This is the framework's
    "open-ended scouting" entry point: define any tactical profile you
    care about and surface the players that best fit it.

    The score is the same weighted-percentile-rank average as
    :func:`compute_matching_score`, so a player at the 75th percentile on
    every requested metric scores ~75 / 100.

    Each returned row also includes inline strengths and weaknesses, so
    the analyst can see *why* a player is ranked where they are without
    a follow-up call.

    Args:
        position: Position code (CB, FB, MF, AM, ST).
        custom_weights: Dict ``{metric_name: weight}``. Names accept
            both ``_p90`` and ``_per90`` suffixes (the resolver handles
            translation). Unknown metrics are reported and discarded; the
            remaining weights are rescaled to preserve the original sum.
        min_minutes: Minimum minutes-played threshold.
        top: How many players to detail in the printed output.
        verbose: If True, print the top-N to stdout.

    Returns:
        DataFrame sorted by score descending, with columns:
        ``player_name, team, minutes_total, score, cluster, role_label,
        strengths, weaknesses``. Strengths/weaknesses are compact strings
        like ``"progressive_passes_p90 (94p) | xA_p90 (87p)"`` where the
        number is the player's percentile rank for that metric.
    """
    if not isinstance(custom_weights, dict) or not custom_weights:
        raise ValueError('custom_weights must be a non-empty dict '
                         '{metric_name: weight}.')

    df, _ = _load_data()
    if position not in {'CB', 'FB', 'MF', 'AM', 'ST'}:
        raise ValueError(f'position must be one of CB/FB/MF/AM/ST, '
                         f'got {position!r}.')

    # Validate weights against the available columns of this position
    _, all_cols = _get_percentile_ranks(position)
    resolved = _resolved_weights(custom_weights, all_cols,
                                 role_name='custom')
    if not resolved:
        raise ValueError('None of the supplied metrics could be resolved '
                         'against the dataset. Use list_available_metrics('
                         f'{position!r}) to inspect available names.')

    pool = df[(df['position_group'] == position)
              & (df['minutes_total'] >= min_minutes)].copy()

    pool['score'] = pool.apply(
        lambda r: compute_matching_score(r, custom_weights, df,
                                         role_name='custom'),
        axis=1,
    )

    # Compute strengths/weaknesses per row
    strengths_list, weaknesses_list = [], []
    for _, row in pool.iterrows():
        s, w = _strengths_weaknesses(row, custom_weights, role_name='custom')
        strengths_list.append(_format_metric_tuples(s) if s else '—')
        weaknesses_list.append(_format_metric_tuples(w) if w else '—')
    pool['strengths'] = strengths_list
    pool['weaknesses'] = weaknesses_list

    out = (pool[['player_name', 'team', 'minutes_total', 'score',
                 'cluster', 'role_label', 'strengths', 'weaknesses']]
           .sort_values('score', ascending=False)
           .reset_index(drop=True))

    if verbose:
        sorted_weights = dict(sorted(resolved.items(), key=lambda kv: -kv[1]))
        print(f'\n=== {position} / custom profile — top {top} players ===')
        print(f'    Weights: {sorted_weights}')
        for i, row in out.head(top).iterrows():
            try:
                _safe_print(
                    f'{i+1:2d}. {row["player_name"]:<35s} '
                    f'{row["team"]:<25s} '
                    f'{row["minutes_total"]:6.0f}min  '
                    f'score={row["score"]:5.2f}  [{row["role_label"]}]'
                )
            except (TypeError, ValueError):
                _safe_print(f'{i+1:2d}. score={row["score"]:5.2f}')
            if row['strengths'] != '—':
                _safe_print(f'     [+] {row["strengths"]}')
            if row['weaknesses'] != '—':
                _safe_print(f'     [-] {row["weaknesses"]}')

    return out


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _demo():
    """Quick smoke demo: one ranking, one similarity search, two profiles."""
    rank_players_by_role('deep_lying_playmaker', 'MF', top=10)
    print()
    _safe_print('=== Players similar to Verratti ===')
    _safe_print(find_similar_players('Verratti', top_n=10).to_string(index=False))
    print()
    profile_player('Verratti')
    print()
    profile_player('Suarez', position='ST')


if __name__ == '__main__':
    _demo()
