"""
clustering.py — Unsupervised clustering per position group.

For each group (CB, FB, MF, AM, ST), an independent pipeline is fitted:
    1. Build a feature matrix (only ``*_p90`` columns + rate columns —
       raw counts are excluded so a player's minutes total does not bias
       the clustering).
    2. Standardise (StandardScaler).
    3. Reduce dimensionality with PCA, keeping 80% of the variance.
    4. Cluster with KMeans, k chosen by the elbow method (k tested 2..8).

Outputs:
    - models/{scaler,pca,kmeans}_{POS}.pkl (one of each per position)
    - data/processed/player_clustered.parquet (with cluster, pca_1, pca_2,
      role_label columns)
    - Console report: characteristic metrics and representative players
      for each cluster.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITION_GROUPS = ['CB', 'FB', 'MF', 'AM', 'ST']

RATE_COLUMNS = [
    'pass_completion_rate',
    'aerial_duel_win_rate',
    'ground_duel_win_rate',
    'pressure_success_rate',
    'shot_on_target_rate',
    'xG_per_shot',
    'dribble_success_rate',
    'pass_completion_under_pressure',
    'long_pass_completion_rate',
]

PCA_VARIANCE_TARGET = 0.80
K_MIN, K_MAX = 2, 8
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_feature_columns(df):
    """Return the columns used for clustering.

    Keeps only ``*_p90`` columns plus the rate columns in :data:`RATE_COLUMNS`.
    Raw counts are excluded so the clustering does not reflect playing-time
    differences between regulars and rotation players.

    Args:
        df: Player feature table.

    Returns:
        List of column names to feed into the pipeline.
    """
    p90_cols = [c for c in df.columns if c.endswith('_p90')]
    rate_cols = [c for c in RATE_COLUMNS if c in df.columns]
    return p90_cols + rate_cols


# ---------------------------------------------------------------------------
# Elbow method
# ---------------------------------------------------------------------------

def find_elbow_k(inertias, k_values):
    """Pick the optimal k via the kneedle / distance-to-chord method.

    For each point (k, inertia), compute its perpendicular distance to the
    line joining the first and last point of the curve. The elbow is the
    point that maximises this distance — where the curve deviates the
    most from the straight line, i.e. where it starts plateauing.

    Args:
        inertias: KMeans inertia for each tested k.
        k_values: Matching k values.

    Returns:
        The optimal k.
    """
    pts = np.column_stack([k_values, inertias]).astype(float)
    p0, p1 = pts[0], pts[-1]
    line_vec = p1 - p0
    line_norm = np.linalg.norm(line_vec)
    if line_norm == 0:
        return int(k_values[0])
    # Perpendicular distance from each point to the chord (2D cross product)
    rel = pts - p0
    cross = np.abs(rel[:, 0] * line_vec[1] - rel[:, 1] * line_vec[0])
    distances = cross / line_norm
    return int(k_values[int(np.argmax(distances))])


# ---------------------------------------------------------------------------
# Per-position pipeline
# ---------------------------------------------------------------------------

def _fit_position_group(df_group, feature_cols):
    """Fit scaler + PCA + KMeans for a single position group.

    Args:
        df_group: Subset of players for this position.
        feature_cols: Columns to use.

    Returns:
        Dict with the fitted artefacts and intermediate arrays.
    """
    # Impute NaNs (e.g. xG_per_shot for a player with no shots) with 0
    X = df_group[feature_cols].astype('float64').fillna(0.0).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_VARIANCE_TARGET, svd_solver='full',
              random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # Search for the optimal k
    k_values = list(range(K_MIN, K_MAX + 1))
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_pca)
        inertias.append(km.inertia_)
    best_k = find_elbow_k(inertias, k_values)

    # Final model with more inits for stability
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(X_pca)

    return {
        'scaler': scaler,
        'pca': pca,
        'kmeans': kmeans,
        'k': best_k,
        'inertias': inertias,
        'k_values': k_values,
        'X_scaled': X_scaled,
        'X_pca': X_pca,
        'labels': labels,
        'feature_cols': feature_cols,
    }


def _save_pickle(obj, path):
    """Serialise ``obj`` to ``path`` using the highest pickle protocol."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Cluster description
# ---------------------------------------------------------------------------

def _describe_cluster(fit, df_group, cluster_id, top_n_metrics=3, top_n_players=5):
    """Identify the characteristic metrics and the representative players.

    Characteristic metrics: since the input space is standardised
    (``mean=0, std=1`` globally), the cluster mean on each feature is
    directly a z-score relative to the position. The features are ranked
    by absolute z-score.

    Representative players: Euclidean distance to the cluster centroid in
    PCA space — the closest players are returned.

    Args:
        fit: Output of :func:`_fit_position_group`.
        df_group: Players in this group (same order as ``fit['labels']``).
        cluster_id: Cluster index.
        top_n_metrics: How many top metrics to return.
        top_n_players: How many top players to return.

    Returns:
        Dict ``{metrics, players, size}``.
    """
    mask = fit['labels'] == cluster_id
    Xs = fit['X_scaled'][mask]
    cluster_means = Xs.mean(axis=0)

    feature_cols = fit['feature_cols']
    order = np.argsort(-np.abs(cluster_means))[:top_n_metrics]
    metrics = [(feature_cols[i], float(cluster_means[i])) for i in order]

    centroid = fit['kmeans'].cluster_centers_[cluster_id]
    Xp = fit['X_pca'][mask]
    dists = np.linalg.norm(Xp - centroid, axis=1)
    sub = df_group.loc[mask].copy()
    sub['_dist'] = dists
    sub = sub.nsmallest(top_n_players, '_dist')
    players = list(zip(
        sub['player_name'].tolist(),
        sub['team'].tolist(),
        sub['_dist'].tolist(),
    ))

    return {'metrics': metrics, 'players': players, 'size': int(mask.sum())}


def _print_cluster_report(pos, fit, df_group):
    """Print the cluster summary for one position group."""
    print(f'\n=== {pos} — {len(df_group)} players, k={fit["k"]} '
          f'(inertias {[round(x, 1) for x in fit["inertias"]]}) ===')
    print(f'    PCA: {fit["pca"].n_components_} components, '
          f'explained variance = {fit["pca"].explained_variance_ratio_.sum():.2%}')

    for cid in range(fit['k']):
        desc = _describe_cluster(fit, df_group, cid)
        print(f'\n  -- Cluster {cid} ({desc["size"]} players) --')
        print('     Characteristic metrics:')
        for name, z in desc['metrics']:
            print(f'       {z:+.2f} sd   {name}')
        print('     Representative players:')
        for name, team, dist in desc['players']:
            try:
                print(f'       - {name} ({team})  d={dist:.2f}')
            except UnicodeEncodeError:
                safe_name = name.encode('ascii', 'replace').decode('ascii')
                safe_team = str(team).encode('ascii', 'replace').decode('ascii')
                print(f'       - {safe_name} ({safe_team})  d={dist:.2f}')


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_clustering(features_path=DATA_PROCESSED / 'player_features.parquet',
                   output_dir=PROJECT_ROOT):
    """Run the full clustering pipeline, position by position.

    For each group ∈ {CB, FB, MF, AM, ST}:
        1. Filter the sub-population.
        2. Select ``*_p90`` and rate columns.
        3. Standardise → PCA (80% variance) → KMeans (elbow-chosen k ∈ [2, 8]).
        4. Save scaler / pca / kmeans to ``{output_dir}/models/``.
        5. Print characteristic metrics and representative players.

    The full table is then augmented with ``cluster``, ``pca_1``, ``pca_2``,
    ``role_label`` (= ``"{POS}-{cluster}"``) and written to
    ``{output_dir}/data/processed/player_clustered.parquet``.

    Args:
        features_path: Path to the player feature parquet.
        output_dir: Project root (must contain ``models/`` and
            ``data/processed/``).

    Returns:
        The augmented DataFrame written to disk.
    """
    models_dir = os.path.join(output_dir, 'models')
    processed_dir = os.path.join(output_dir, 'data', 'processed')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print(f'[load] {features_path}')
    df = pd.read_parquet(features_path)
    feature_cols = select_feature_columns(df)
    n_p90 = sum(c.endswith('_p90') for c in feature_cols)
    print(f'[features] {len(feature_cols)} columns used '
          f'({n_p90} per-90 + {len(feature_cols) - n_p90} rates)')

    df = df.copy()
    df['cluster'] = -1
    df['pca_1'] = np.nan
    df['pca_2'] = np.nan
    df['role_label'] = pd.NA

    for pos in POSITION_GROUPS:
        sub = df[df['position_group'] == pos]
        if len(sub) < K_MAX + 1:
            print(f'[skip] {pos}: only {len(sub)} players — too few to test '
                  f'k up to {K_MAX}.')
            continue

        fit = _fit_position_group(sub, feature_cols)

        _save_pickle(fit['scaler'], os.path.join(models_dir, f'scaler_{pos}.pkl'))
        _save_pickle(fit['pca'],    os.path.join(models_dir, f'pca_{pos}.pkl'))
        _save_pickle(fit['kmeans'], os.path.join(models_dir, f'kmeans_{pos}.pkl'))

        df.loc[sub.index, 'cluster'] = fit['labels']
        df.loc[sub.index, 'pca_1'] = fit['X_pca'][:, 0]
        if fit['X_pca'].shape[1] >= 2:
            df.loc[sub.index, 'pca_2'] = fit['X_pca'][:, 1]
        df.loc[sub.index, 'role_label'] = [f'{pos}-{c}' for c in fit['labels']]

        _print_cluster_report(pos, fit, sub)

    out_path = os.path.join(processed_dir, 'player_clustered.parquet')
    df.to_parquet(out_path, index=False)
    print(f'\n[DONE] -> {out_path}  ({len(df)} players, '
          f'{df["cluster"].ne(-1).sum()} clustered)')
    return df


if __name__ == '__main__':
    run_clustering()
