"""
clustering.py — Clustering non supervisé par groupe de position.

Pour chaque groupe (CB, FB, MF, AM, ST), on construit indépendamment :
    1. Une matrice de features (uniquement _p90 + taux, pas de totaux bruts).
    2. Une standardisation (StandardScaler).
    3. Une réduction PCA gardant 80% de la variance expliquée.
    4. Un KMeans dont le k est choisi par méthode du coude (k de 2 à 8).

Sortie :
    - models/scaler_{POS}.pkl, pca_{POS}.pkl, kmeans_{POS}.pkl
    - data/processed/player_clustered.parquet (avec colonnes cluster, pca_1, pca_2)
    - Affichage console : top métriques + top 5 joueurs par cluster.
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
# Constantes
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
# Sélection de features
# ---------------------------------------------------------------------------

def select_feature_columns(df):
    """Détermine les colonnes utilisées pour le clustering.

    Retient uniquement les colonnes ``*_p90`` et les colonnes de taux listées
    dans :data:`RATE_COLUMNS`. Exclut les totaux bruts (counts non normalisés)
    pour éviter que la durée de jeu d'un joueur ne biaise le clustering.

    Args:
        df (pd.DataFrame): la table de features joueurs.

    Returns:
        list[str]: noms des colonnes à utiliser.
    """
    p90_cols = [c for c in df.columns if c.endswith('_p90')]
    rate_cols = [c for c in RATE_COLUMNS if c in df.columns]
    return p90_cols + rate_cols


# ---------------------------------------------------------------------------
# Méthode du coude
# ---------------------------------------------------------------------------

def find_elbow_k(inertias, k_values):
    """Trouve le k optimal par méthode du coude (kneedle / distance à la corde).

    Pour chaque point (k, inertia), on calcule sa distance perpendiculaire à
    la droite reliant le premier et le dernier point de la courbe. Le coude
    est le point qui maximise cette distance — c'est-à-dire l'endroit où la
    courbe s'écarte le plus de la ligne droite, donc où elle "plafonne".

    Args:
        inertias (list[float]): inerties KMeans pour chaque k testé.
        k_values (list[int]): valeurs de k correspondantes.

    Returns:
        int: la valeur de k optimale.
    """
    pts = np.column_stack([k_values, inertias]).astype(float)
    # Vecteur ligne entre premier et dernier point
    p0, p1 = pts[0], pts[-1]
    line_vec = p1 - p0
    line_norm = np.linalg.norm(line_vec)
    if line_norm == 0:
        return int(k_values[0])
    # Distance perpendiculaire de chaque point à la corde
    rel = pts - p0
    # Produit vectoriel 2D = norm(rel x line)
    cross = np.abs(rel[:, 0] * line_vec[1] - rel[:, 1] * line_vec[0])
    distances = cross / line_norm
    best_idx = int(np.argmax(distances))
    return int(k_values[best_idx])


# ---------------------------------------------------------------------------
# Pipeline par groupe de position
# ---------------------------------------------------------------------------

def _fit_position_group(df_group, feature_cols):
    """Entraîne scaler + PCA + KMeans pour un groupe de position donné.

    Args:
        df_group (pd.DataFrame): sous-ensemble des joueurs d'un groupe.
        feature_cols (list[str]): colonnes à utiliser.

    Returns:
        dict: ``{scaler, pca, kmeans, k, inertias, X_scaled, X_pca, labels,
        feature_cols}``.
    """
    # Imputation : taux NaN (ex. xG_per_shot sans tir) → 0, p90 NaN → 0
    X = df_group[feature_cols].astype('float64').fillna(0.0).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_VARIANCE_TARGET, svd_solver='full',
              random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # Recherche du k optimal
    k_values = list(range(K_MIN, K_MAX + 1))
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_pca)
        inertias.append(km.inertia_)
    best_k = find_elbow_k(inertias, k_values)

    # Modèle final
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
    """Sérialise ``obj`` dans ``path`` via pickle (protocole le plus récent).

    Args:
        obj: objet picklable.
        path (str): chemin de destination.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Description des clusters
# ---------------------------------------------------------------------------

def _describe_cluster(fit, df_group, cluster_id, top_n_metrics=3, top_n_players=5):
    """Identifie les métriques caractéristiques et les joueurs représentatifs.

    Métriques caractéristiques : on calcule la moyenne z-scorée du cluster sur
    chaque feature (l'espace standardisé a déjà ``mean=0, std=1`` au global),
    puis on trie par valeur absolue décroissante.

    Joueurs représentatifs : distance euclidienne au centroïde dans l'espace
    PCA — les 5 plus proches.

    Args:
        fit (dict): sortie de :func:`_fit_position_group`.
        df_group (pd.DataFrame): joueurs du groupe (même ordre que fit).
        cluster_id (int): identifiant du cluster.
        top_n_metrics (int): nombre de métriques à retourner.
        top_n_players (int): nombre de joueurs à retourner.

    Returns:
        dict: ``{metrics: [(name, z), ...], players: [(name, team, dist), ...]}``
    """
    mask = fit['labels'] == cluster_id
    Xs = fit['X_scaled'][mask]  # déjà standardisé sur tout le groupe
    cluster_means = Xs.mean(axis=0)

    feature_cols = fit['feature_cols']
    order = np.argsort(-np.abs(cluster_means))[:top_n_metrics]
    metrics = [(feature_cols[i], float(cluster_means[i])) for i in order]

    # Distance au centroïde en espace PCA
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
    """Affiche le rapport texte des clusters d'un groupe de position.

    Args:
        pos (str): code de groupe (ex. 'CB').
        fit (dict): sortie de :func:`_fit_position_group`.
        df_group (pd.DataFrame): joueurs du groupe.
    """
    print(f'\n=== {pos} — {len(df_group)} joueurs, k={fit["k"]} '
          f'(inertias {[round(x, 1) for x in fit["inertias"]]}) ===')
    print(f'    PCA : {fit["pca"].n_components_} composantes, '
          f'variance expliquée = {fit["pca"].explained_variance_ratio_.sum():.2%}')

    for cid in range(fit['k']):
        desc = _describe_cluster(fit, df_group, cid)
        print(f'\n  -- Cluster {cid} ({desc["size"]} joueurs) --')
        print('     Métriques caractéristiques :')
        for name, z in desc['metrics']:
            sign = '+' if z >= 0 else ''
            print(f'       {sign}{z:+.2f} sd   {name}')
        print('     Joueurs représentatifs :')
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
    """Lance le clustering complet position par position.

    Étapes pour chaque groupe ∈ {CB, FB, MF, AM, ST} :
        1. Filtre la sous-population.
        2. Sélectionne les colonnes _p90 et de taux.
        3. Standardise → PCA 80% var → KMeans (k optimal par coude, k∈[2,8]).
        4. Sauvegarde scaler/pca/kmeans dans ``{output_dir}/models/``.
        5. Affiche métriques caractéristiques et joueurs représentatifs.

    À la fin, la table complète est augmentée des colonnes ``cluster``,
    ``pca_1``, ``pca_2`` (et ``role_label`` = "{POS}-{cluster}") et écrite
    dans ``{output_dir}/data/processed/player_clustered.parquet``.

    Args:
        features_path (str): chemin du parquet de features joueurs.
        output_dir (str): racine du projet (contient ``models/`` et
            ``data/processed/``).

    Returns:
        pd.DataFrame: la table augmentée écrite sur disque.
    """
    models_dir = os.path.join(output_dir, 'models')
    processed_dir = os.path.join(output_dir, 'data', 'processed')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print(f'[load] {features_path}')
    df = pd.read_parquet(features_path)
    feature_cols = select_feature_columns(df)
    print(f'[features] {len(feature_cols)} colonnes utilisées '
          f'({sum(c.endswith("_p90") for c in feature_cols)} per90 '
          f'+ {len(feature_cols) - sum(c.endswith("_p90") for c in feature_cols)} taux)')

    # Colonnes à remplir dans le df final
    df = df.copy()
    df['cluster'] = -1
    df['pca_1'] = np.nan
    df['pca_2'] = np.nan
    df['role_label'] = pd.NA

    for pos in POSITION_GROUPS:
        sub = df[df['position_group'] == pos]
        if len(sub) < K_MAX + 1:
            print(f'[skip] {pos} : seulement {len(sub)} joueurs, '
                  f'pas assez pour tester k jusqu\'à {K_MAX}.')
            continue

        fit = _fit_position_group(sub, feature_cols)

        # Sauvegarde des artefacts
        _save_pickle(fit['scaler'], os.path.join(models_dir, f'scaler_{pos}.pkl'))
        _save_pickle(fit['pca'],    os.path.join(models_dir, f'pca_{pos}.pkl'))
        _save_pickle(fit['kmeans'], os.path.join(models_dir, f'kmeans_{pos}.pkl'))

        # Remontée dans le df principal
        df.loc[sub.index, 'cluster'] = fit['labels']
        df.loc[sub.index, 'pca_1'] = fit['X_pca'][:, 0]
        if fit['X_pca'].shape[1] >= 2:
            df.loc[sub.index, 'pca_2'] = fit['X_pca'][:, 1]
        df.loc[sub.index, 'role_label'] = [f'{pos}-{c}' for c in fit['labels']]

        _print_cluster_report(pos, fit, sub)

    # Sauvegarde finale
    out_path = os.path.join(processed_dir, 'player_clustered.parquet')
    df.to_parquet(out_path, index=False)
    print(f'\n[DONE] -> {out_path}  ({len(df)} joueurs, '
          f'{df["cluster"].ne(-1).sum()} clustérisés)')
    return df


if __name__ == '__main__':
    run_clustering()
