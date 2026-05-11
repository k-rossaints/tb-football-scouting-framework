"""
matching.py — Scoring de joueurs contre des profils tactiques et recherche
de similarité.

Pipeline conceptuel :
    1. Chaque métrique du joueur est normalisée [0, 1] par MinMax **dans son
       groupe de position** (la référence est la population du même poste).
    2. Le "joueur idéal" pour un rôle est défini comme un vecteur dont chaque
       coordonnée vaut le 90e centile de la métrique correspondante (intra-
       position) — toujours normalisé en [0, 1].
    3. Le score d'adéquation = cosine similarity entre le vecteur du joueur
       et celui du rôle idéal, chaque dimension pondérée par le poids du
       rôle. Sortie : 0–100.

Le YAML de profils utilise par convention les suffixes ``_per90``, alors que
le parquet utilise ``_p90`` : un résolveur de nom (:func:`_resolve_metric`)
gère la traduction et la robustesse aux métriques non disponibles.
"""

import os
import unicodedata

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler


def _strip_accents(s):
    """Retire les accents/diacritiques d'une chaîne (utile pour la recherche)."""
    if not isinstance(s, str):
        return ''
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(c))


def _name_match(series, query):
    """Renvoie un masque booléen pour les lignes de ``series`` matchant
    ``query`` de manière insensible aux accents et à la casse."""
    q = _strip_accents(query).lower()
    return series.fillna('').map(lambda s: q in _strip_accents(s).lower())


# ---------------------------------------------------------------------------
# Chemins par défaut
# ---------------------------------------------------------------------------

ROOT = 'C:/tb-scouting'
DEFAULT_CLUSTERED = os.path.join(ROOT, 'data', 'processed', 'player_clustered.parquet')
DEFAULT_PROFILES  = os.path.join(ROOT, 'config', 'role_profiles.yaml')

IDEAL_PERCENTILE = 75        # le rôle "idéal" = 75e centile (top quartile)
                             # (réf. réaliste : un excellent joueur typique,
                             # pas le sommet utopique du 90e centile)
TOP_STRENGTHS = 3
TOP_WEAKNESSES = 3

# Alias explicites YAML -> colonne parquet
_ALIASES = {
    'dribbles_success_rate': 'dribble_success_rate',
}


# ---------------------------------------------------------------------------
# Caches lazy (chargés à la première utilisation)
# ---------------------------------------------------------------------------

_CACHE = {
    'df': None,
    'profiles': None,
    'norm': {},        # dict[position_group -> (DataFrame MinMax-normalisé, columns)]
    'pct': {},         # dict[position_group -> (DataFrame percentile rank 0-100, columns)]
    'warned': set(),   # métriques YAML déjà signalées comme absentes
}


def _load_data(features_path=DEFAULT_CLUSTERED, profiles_path=DEFAULT_PROFILES):
    """Charge (avec cache) le parquet clustérisé et le YAML des profils.

    Args:
        features_path (str): chemin du parquet joueurs (clusterisé).
        profiles_path (str): chemin du YAML des profils tactiques.

    Returns:
        tuple[pd.DataFrame, dict]: ``(df, profiles)``.
    """
    if _CACHE['df'] is None:
        _CACHE['df'] = pd.read_parquet(features_path)
    if _CACHE['profiles'] is None:
        with open(profiles_path, 'r', encoding='utf-8') as f:
            _CACHE['profiles'] = yaml.safe_load(f)
    return _CACHE['df'], _CACHE['profiles']


def reset_cache():
    """Vide le cache (à appeler si parquet ou YAML changent en cours d'exécution)."""
    _CACHE['df'] = None
    _CACHE['profiles'] = None
    _CACHE['norm'] = {}
    _CACHE['pct'] = {}
    _CACHE['warned'] = set()


# ---------------------------------------------------------------------------
# Résolution des noms de métriques YAML -> parquet
# ---------------------------------------------------------------------------

def _resolve_metric(name, df_columns):
    """Traduit un nom de métrique YAML vers la colonne correspondante du parquet.

    Tente successivement :
        1. Le nom tel quel.
        2. Un alias explicite (:data:`_ALIASES`).
        3. La substitution suffixe ``_per90 -> _p90``.
        4. La substitution suffixe ``_p90 -> _per90`` (inverse).

    Args:
        name (str): nom de la métrique dans le YAML.
        df_columns (Iterable[str]): colonnes disponibles dans le DataFrame.

    Returns:
        str | None: nom de colonne réel, ou ``None`` si introuvable.
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
    """Convertit les poids du YAML en {colonne parquet: poids}.

    Les métriques absentes du parquet sont écartées (warning émis une fois),
    et les poids restants sont **renormalisés** pour conserver la somme
    d'origine — sans cela, un rôle avec beaucoup de métriques manquantes
    serait artificiellement pénalisé.

    Args:
        role_weights (dict[str, float]): poids tels que lus dans le YAML.
        df_columns (Iterable[str]): colonnes disponibles.
        role_name (str): nom du rôle, utilisé pour les warnings.

    Returns:
        dict[str, float]: ``{colonne_parquet: poids_renormalisé}``.
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
            print(f'[warn] role "{role_name}" : {len(missing)} métrique(s) '
                  f'non disponible(s) dans le parquet — {missing}')
            _CACHE['warned'].add(key)
    if resolved:
        orig_sum = float(sum(role_weights.values()))
        new_sum = sum(resolved.values())
        if new_sum > 0 and orig_sum > 0:
            factor = orig_sum / new_sum
            resolved = {k: v * factor for k, v in resolved.items()}
    return resolved


# ---------------------------------------------------------------------------
# Normalisation MinMax intra-position
# ---------------------------------------------------------------------------

def _feature_columns(df):
    """Renvoie la liste des colonnes utilisées pour matching et similarité.

    Args:
        df (pd.DataFrame): table joueurs.

    Returns:
        list[str]: colonnes ``*_p90`` + colonnes de taux numériques.
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
    """Retourne (avec cache) la matrice MinMax-normalisée d'un groupe.

    Args:
        position (str): code de position (CB, FB, MF, AM, ST).

    Returns:
        tuple[pd.DataFrame, list[str]]: DataFrame normalisé indexé par
        ``player_id`` (valeurs ∈ [0, 1]) et liste des colonnes utilisées.
    """
    if position in _CACHE['norm']:
        return _CACHE['norm'][position]

    df, _ = _load_data()
    sub = df[df['position_group'] == position].copy()
    cols = _feature_columns(df)
    X = sub[cols].astype('float64').fillna(0.0).to_numpy()

    scaler = MinMaxScaler()
    Xn = scaler.fit_transform(X)
    norm_df = pd.DataFrame(Xn, index=sub['player_id'].to_numpy(), columns=cols)

    _CACHE['norm'][position] = (norm_df, cols)
    return norm_df, cols


def _get_percentile_ranks(position):
    """Retourne (avec cache) la table de percentile rank d'un groupe.

    Pour chaque métrique, chaque joueur reçoit son rang en percentile parmi
    les joueurs du même ``position_group``, exprimé dans [0, 100] :
        - 100 → meilleur joueur du poste sur cette métrique
        - 50  → médiane
        - 0   → pire joueur du poste

    Args:
        position (str): code de position (CB, FB, MF, AM, ST).

    Returns:
        tuple[pd.DataFrame, list[str]]: DataFrame indexé par ``player_id``
        (valeurs ∈ [0, 100]) et liste des colonnes utilisées.
    """
    if position in _CACHE['pct']:
        return _CACHE['pct'][position]

    df, _ = _load_data()
    sub = df[df['position_group'] == position].copy()
    cols = _feature_columns(df)

    raw = sub[cols].astype('float64').fillna(0.0)
    # rank(pct=True) → centiles ∈ (0, 1] ; method='average' gère les ex æquo
    pct = raw.rank(pct=True, method='average') * 100.0
    pct.index = sub['player_id'].to_numpy()

    _CACHE['pct'][position] = (pct, cols)
    return pct, cols


def _ideal_vector(position, columns):
    """Vecteur des valeurs idéales (90e centile, normalisées) pour un poste.

    Args:
        position (str): code de position.
        columns (list[str]): colonnes à inclure.

    Returns:
        pd.Series: index = colonnes, valeurs ∈ [0, 1].
    """
    norm_df, _ = _get_normalized(position)
    return norm_df[columns].quantile(IDEAL_PERCENTILE / 100.0)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine(u, v):
    """Cosine similarity entre deux vecteurs.

    Args:
        u (np.ndarray): vecteur 1.
        v (np.ndarray): vecteur 2.

    Returns:
        float: similarité ∈ [-1, 1] (typiquement [0, 1] ici car valeurs ≥ 0).
    """
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


# ---------------------------------------------------------------------------
# Score de matching
# ---------------------------------------------------------------------------

def compute_matching_score(player_row, role_weights, all_players_df,
                           role_name='?'):
    """Calcule le score d'adéquation d'un joueur à un rôle (percentile pondéré).

    Méthode (réécrite pour discriminer correctement) :
        1. Pour chaque métrique du rôle, on prend le **percentile rank** du
           joueur parmi tous les joueurs du même ``position_group`` (∈ [0, 100]).
        2. On calcule la moyenne pondérée des percentile ranks par les poids
           du rôle : ``score = Σ(pct_i × w_i) / Σ(w_i)``.

    Sémantique :
        - Joueur au 90e centile sur **toutes** les métriques clés → ~90/100.
        - Joueur au 50e centile (médian) sur toutes → ~50/100.
        - Faiblesses sur une métrique de poids élevé → tire le score vers le bas.

    Le score est désormais **borné par les percentiles réels** plutôt que par
    la direction d'un vecteur (problème de la cosine sur valeurs positives).

    Args:
        player_row (pd.Series): ligne d'un joueur (doit contenir
            ``position_group`` et ``player_id``).
        role_weights (dict[str, float]): poids du rôle (noms YAML acceptés).
        all_players_df (pd.DataFrame): inutilisé (gardé pour rétro-compat de
            l'API ; la normalisation passe par le cache intra-position).
        role_name (str): nom du rôle, propagé aux warnings.

    Returns:
        float: score ∈ [0, 100].
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
    """Identifie les 3 forces et 3 faiblesses du joueur sur les métriques du rôle.

    Cohérent avec le scoring percentile pondéré :
        - **Force** = métrique où le joueur est haut en percentile, pondérée
          par le poids du rôle. Concrètement on classe par
          ``percentile_rank × weight`` décroissant.
        - **Faiblesse** = métrique où le joueur est bas en percentile,
          pondérée par le poids. On classe par
          ``(100 − percentile_rank) × weight`` décroissant — autrement dit
          les métriques qui *pénalisent* le plus le score final.

    Les valeurs retournées sont exprimées en fraction [0, 1] (= percentile/100)
    pour rester compatibles avec l'affichage existant de
    :mod:`visualisation` (format ``{pv:.2f} vs {iv:.2f}``). L'idéal vaut
    par convention 1.0 (= 100e centile).

    Args:
        player_row (pd.Series): ligne d'un joueur.
        role_weights (dict[str, float]): poids du rôle.
        role_name (str): nom du rôle (warnings).

    Returns:
        tuple[list, list]: ``(strengths, weaknesses)`` ; chacun est une
        liste de tuples ``(metric, weighted_score, player_value, ideal_value)``
        où ``player_value`` et ``ideal_value`` sont dans [0, 1].
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

    pv = player_pct / 100.0     # 0..1 pour affichage
    # Idéal réaliste = 75e centile (cohérent avec IDEAL_PERCENTILE)
    iv = np.full_like(pv, IDEAL_PERCENTILE / 100.0)

    # Forces : metriques où le joueur tire le score VERS LE HAUT
    strength_score = pv * w
    s_order = np.argsort(-strength_score)
    strengths = [
        (cols[i], float(strength_score[i]), float(pv[i]), float(iv[i]))
        for i in s_order[:TOP_STRENGTHS] if pv[i] >= 0.60  # min 60e centile
    ]

    # Faiblesses : metriques où le joueur tire le score VERS LE BAS
    weakness_score = (1.0 - pv) * w
    w_order = np.argsort(-weakness_score)
    weaknesses = [
        (cols[i], float(weakness_score[i]), float(pv[i]), float(iv[i]))
        for i in w_order[:TOP_WEAKNESSES] if pv[i] <= 0.50  # max 50e centile
    ]
    return strengths, weaknesses


# ---------------------------------------------------------------------------
# API publique : rank_players_by_role
# ---------------------------------------------------------------------------

def rank_players_by_role(role_name, position, min_minutes=450, top=20,
                         verbose=True):
    """Classe tous les joueurs d'un poste selon leur adéquation à un rôle.

    Args:
        role_name (str): clé du rôle dans le YAML (ex. ``deep_lying_playmaker``).
        position (str): code de position (CB, FB, MF, AM, ST).
        min_minutes (int): seuil minimal de minutes jouées.
        top (int): nombre de joueurs détaillés à l'écran.
        verbose (bool): si True, affiche aussi forces/faiblesses des top.

    Returns:
        pd.DataFrame: colonnes ``player_name, team, minutes_total, score,
        cluster, role_label``, trié par ``score`` décroissant.
    """
    df, profiles = _load_data()
    if position not in profiles:
        raise KeyError(f'Position "{position}" absente du YAML. '
                       f'Disponibles : {list(profiles)}')
    if role_name not in profiles[position]:
        raise KeyError(f'Rôle "{role_name}" absent pour {position}. '
                       f'Disponibles : {list(profiles[position])}')

    role_weights = profiles[position][role_name]
    pool = df[(df['position_group'] == position)
              & (df['minutes_total'] >= min_minutes)].copy()

    scores = pool.apply(
        lambda r: compute_matching_score(r, role_weights, df, role_name=role_name),
        axis=1,
    )
    pool['score'] = scores
    out = (pool[['player_name', 'team', 'minutes_total', 'score',
                 'cluster', 'role_label']]
           .sort_values('score', ascending=False)
           .reset_index(drop=True))

    if verbose:
        print(f'\n=== {position} / {role_name} — top {top} joueurs ===')
        for i, row in out.head(top).iterrows():
            try:
                line = (f'{i+1:2d}. {row["player_name"]:<35s} '
                        f'{row["team"]:<25s} '
                        f'{row["minutes_total"]:6.0f}min  '
                        f'score={row["score"]:.2f}  [{row["role_label"]}]')
            except (TypeError, ValueError):
                line = f'{i+1:2d}. score={row["score"]:.2f}'
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode('ascii', 'replace').decode('ascii'))

            # Forces / faiblesses
            player_row = pool[pool['player_name'] == row['player_name']].iloc[0]
            strengths, weaknesses = _strengths_weaknesses(
                player_row, role_weights, role_name=role_name)
            if strengths:
                msg = '     [+] ' + ' | '.join(
                    f'{m} ({pv:.2f} vs {iv:.2f})'
                    for m, _, pv, iv in strengths)
                _safe_print(msg)
            if weaknesses:
                msg = '     [-] ' + ' | '.join(
                    f'{m} ({pv:.2f} vs {iv:.2f})'
                    for m, _, pv, iv in weaknesses)
                _safe_print(msg)

    return out


def _safe_print(s):
    """Print qui ne lève pas d'UnicodeEncodeError sur les consoles cp1252."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'))


# ---------------------------------------------------------------------------
# API publique : find_similar_players
# ---------------------------------------------------------------------------

def find_similar_players(player_name, position=None, top_n=10):
    """Trouve les joueurs les plus similaires à un joueur donné.

    La similarité est la cosine similarity calculée sur **toutes** les
    métriques (``*_p90`` + taux), normalisées MinMax intra-poste — donc
    indépendante de tout profil tactique.

    Args:
        player_name (str): nom (ou fragment) du joueur cible.
        position (str | None): si fourni, force la position du joueur cible
            (utile pour les homonymes). Sinon déduit du DataFrame.
        top_n (int): nombre de joueurs à retourner.

    Returns:
        pd.DataFrame: colonnes ``player_name, team, position_group,
        minutes_total, similarity, role_label`` triées par
        ``similarity`` décroissante.
    """
    df, _ = _load_data()
    mask = _name_match(df['player_name'], player_name)
    if position is not None:
        mask &= df['position_group'] == position
    candidates = df[mask]
    if candidates.empty:
        raise KeyError(f'Aucun joueur ne correspond à "{player_name}"'
                       + (f' (position={position})' if position else ''))
    if len(candidates) > 1:
        # On prend le plus joueur ayant le plus de minutes pour lever l'ambiguïté
        target = candidates.sort_values('minutes_total', ascending=False).iloc[0]
        _safe_print(f'[info] plusieurs joueurs trouvés ({len(candidates)}), '
                    f'sélection de "{target["player_name"]}" ({target["team"]})')
    else:
        target = candidates.iloc[0]

    pos = target['position_group']
    norm_df, cols = _get_normalized(pos)

    target_vec = norm_df.loc[target['player_id'], cols].to_numpy(dtype='float64')
    target_norm = np.linalg.norm(target_vec)
    if target_norm == 0:
        return pd.DataFrame()

    # Cosine vectorisée
    M = norm_df[cols].to_numpy(dtype='float64')
    norms = np.linalg.norm(M, axis=1)
    norms[norms == 0] = 1.0
    sims = (M @ target_vec) / (norms * target_norm)

    sim_series = pd.Series(sims, index=norm_df.index, name='similarity')
    # Joindre infos joueur
    pos_df = df[df['position_group'] == pos].set_index('player_id')
    out = (pos_df.join(sim_series)
                 .reset_index()
                 [['player_name', 'team', 'position_group',
                   'minutes_total', 'similarity', 'role_label']]
                 .sort_values('similarity', ascending=False))

    # Exclure le joueur cible lui-même
    out = out[out['player_name'] != target['player_name']].head(top_n).reset_index(drop=True)
    out['similarity'] = (out['similarity'] * 100).round(2)
    return out


# ---------------------------------------------------------------------------
# API publique : profile_player
# ---------------------------------------------------------------------------

def profile_player(player_name, position=None, verbose=True):
    """Calcule le score du joueur pour tous les rôles de son poste.

    Args:
        player_name (str): nom (ou fragment) du joueur cible.
        position (str | None): force le poste si ambigu.
        verbose (bool): si True, imprime le tableau et le rôle naturel.

    Returns:
        pd.DataFrame: colonnes ``role_name, score, is_natural`` triées par
        score décroissant. ``is_natural`` est True pour le score max.
    """
    df, profiles = _load_data()
    mask = _name_match(df['player_name'], player_name)
    if position is not None:
        mask &= df['position_group'] == position
    matches = df[mask]
    if matches.empty:
        raise KeyError(f'Aucun joueur ne correspond à "{player_name}"')
    target = matches.sort_values('minutes_total', ascending=False).iloc[0]

    pos = target['position_group']
    role_defs = profiles.get(pos, {})

    rows = []
    for role_name, weights in role_defs.items():
        score = compute_matching_score(target, weights, df, role_name=role_name)
        rows.append({'role_name': role_name, 'score': score})
    out = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    out['is_natural'] = False
    if not out.empty:
        out.loc[0, 'is_natural'] = True

    if verbose:
        _safe_print(f'\n=== Profil de "{target["player_name"]}" '
                    f'({target["team"]}, {pos}, {target["minutes_total"]:.0f} min) ===')
        for _, r in out.iterrows():
            marker = '  <-- rôle naturel' if r['is_natural'] else ''
            print(f'  {r["role_name"]:<28s}  {r["score"]:6.2f}{marker}')
    return out


# ---------------------------------------------------------------------------
# Démo CLI
# ---------------------------------------------------------------------------

def _demo():
    """Petite démo console : un ranking, une similarité, un profil."""
    rank_players_by_role('deep_lying_playmaker', 'MF', top=10)
    print()
    _safe_print('=== Similaires à Verratti ===')
    _safe_print(find_similar_players('Verratti', top_n=10).to_string(index=False))
    print()
    profile_player('Verratti')
    print()
    profile_player('Suarez', position='ST')


if __name__ == '__main__':
    _demo()
