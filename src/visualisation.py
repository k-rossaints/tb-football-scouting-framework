"""
visualisation.py — Visualisations de scouting : radar, ranking, PCA, card.

Quatre fonctions principales :
    * :func:`plot_radar_chart` — radar joueur vs profil idéal.
    * :func:`plot_ranking_table` — tableau top-N pour un rôle.
    * :func:`plot_pca_scatter` — scatter PCA 2D par cluster.
    * :func:`plot_player_card` — fiche A4 paysage (radar + texte).

Toutes les sorties sont exportables en PNG 300 DPI dans le dossier
``results/`` à la racine du projet.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Radar

from src import matching as _m


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

ROOT = 'C:/tb-scouting'
RESULTS_DIR = os.path.join(ROOT, 'results')

COLOR_PLAYER = '#1f77b4'   # bleu
COLOR_IDEAL = '#d62728'    # rouge
COLOR_BG = '#f5f5f5'
COLOR_GRID = '#cccccc'

CLUSTER_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                   '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

DPI = 300


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _ensure_results_dir():
    """Crée ``results/`` à la racine si absent."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _resolve_path(save_path):
    """Renvoie un chemin absolu pour la sauvegarde.

    Si ``save_path`` est relatif ou un simple nom de fichier, le préfixe avec
    :data:`RESULTS_DIR`. Si absolu, on respecte le chemin fourni.

    Args:
        save_path (str | None): chemin proposé.

    Returns:
        str | None: chemin absolu prêt à l'usage, ou ``None``.
    """
    if save_path is None:
        return None
    _ensure_results_dir()
    if os.path.isabs(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path
    return os.path.join(RESULTS_DIR, save_path)


def _find_player(player_name, position=None):
    """Recherche un joueur dans le parquet clustérisé (insensible aux accents).

    Args:
        player_name (str): nom (fragment accepté).
        position (str | None): force le poste si plusieurs matchs.

    Returns:
        pd.Series: la ligne du joueur sélectionné.

    Raises:
        KeyError: si aucun joueur ne correspond.
    """
    df, _ = _m._load_data()
    mask = _m._name_match(df['player_name'], player_name)
    if position is not None:
        mask &= df['position_group'] == position
    cand = df[mask]
    if cand.empty:
        raise KeyError(f'Aucun joueur ne correspond à "{player_name}"')
    return cand.sort_values('minutes_total', ascending=False).iloc[0]


def _role_normalized_values(player_row, role_weights, role_name='?'):
    """Renvoie (metric_names, player_values, ideal_values, weights).

    Les métriques sont les colonnes parquet résolues depuis le YAML ; les
    valeurs sont MinMax-normalisées intra-position (∈ [0, 1]).

    Args:
        player_row (pd.Series): ligne joueur.
        role_weights (dict[str, float]): poids YAML.
        role_name (str): nom de rôle (warnings).

    Returns:
        tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        (cols, player_vec, ideal_vec, weight_vec).
    """
    position = player_row['position_group']
    norm_df, all_cols = _m._get_normalized(position)
    weights = _m._resolved_weights(role_weights, all_cols, role_name=role_name)
    cols = list(weights.keys())
    pid = player_row['player_id']
    player_vec = norm_df.loc[pid, cols].to_numpy(dtype='float64')
    ideal_vec = _m._ideal_vector(position, cols).to_numpy(dtype='float64')
    w = np.array([weights[c] for c in cols], dtype='float64')
    return cols, player_vec, ideal_vec, w


def _pretty_metric(name):
    """Transforme un nom de colonne brut en label lisible.

    Exemples :
        ``progressive_passes_p90`` -> ``Prog. Passes /90``
        ``pass_completion_rate``   -> ``Pass Compl. %``
    """
    s = name
    s = s.replace('_p90', ' /90')
    s = s.replace('_rate', ' %')
    s = s.replace('_', ' ')
    s = s.replace('xg', 'xG').replace('xa', 'xA')
    s = s.title()
    s = s.replace('Xg', 'xG').replace('Xa', 'xA')
    s = s.replace('Progressive', 'Prog.').replace('Completion', 'Compl.')
    s = s.replace('Penalty Area', 'Box').replace('Final Third', 'Final 3rd')
    return s


# ---------------------------------------------------------------------------
# 1) Radar chart
# ---------------------------------------------------------------------------

def plot_radar_chart(player_name, role_name, save_path=None, position=None):
    """Construit un radar comparant un joueur au profil idéal d'un rôle.

    Le radar utilise les métriques du rôle (yaml), normalisées MinMax intra-
    position dans [0, 1]. Deux polygones :
        - **Joueur** (bleu plein).
        - **Idéal 90e centile** (rouge, pointillé).

    Le titre affiche le score de matching renvoyé par
    :func:`matching.compute_matching_score`.

    Args:
        player_name (str): nom (fragment) du joueur.
        role_name (str): clé du rôle dans ``role_profiles.yaml``.
        save_path (str | None): si fourni, exporte en PNG 300 DPI dans
            ``results/`` (relatif) ou à l'emplacement spécifié (absolu).
        position (str | None): force le poste si plusieurs joueurs matchent.

    Returns:
        matplotlib.figure.Figure: la figure produite.
    """
    df, profiles = _m._load_data()
    target = _find_player(player_name, position=position)
    pos = target['position_group']
    if pos not in profiles or role_name not in profiles[pos]:
        raise KeyError(f'Rôle "{role_name}" indisponible pour {pos}.')

    role_weights = profiles[pos][role_name]
    cols, player_vec, ideal_vec, _ = _role_normalized_values(
        target, role_weights, role_name=role_name)
    score = _m.compute_matching_score(target, role_weights, df, role_name=role_name)

    params = [_pretty_metric(c) for c in cols]
    radar = Radar(
        params=params,
        min_range=[0.0] * len(cols),
        max_range=[1.0] * len(cols),
        num_rings=4,
        ring_width=1.0,
        center_circle_radius=1.0,
    )

    fig, ax = plt.subplots(figsize=(9, 9), dpi=DPI / 3)
    fig.patch.set_facecolor(COLOR_BG)
    radar.setup_axis(ax=ax, facecolor='white')
    radar.draw_circles(ax=ax, facecolor='#ebebeb', edgecolor=COLOR_GRID)

    # Idéal d'abord (en arrière-plan)
    radar.draw_radar_solid(
        ideal_vec, ax=ax,
        kwargs={'facecolor': COLOR_IDEAL, 'alpha': 0.20,
                'edgecolor': COLOR_IDEAL, 'linewidth': 2.0,
                'linestyle': '--'},
    )
    radar.draw_radar_solid(
        player_vec, ax=ax,
        kwargs={'facecolor': COLOR_PLAYER, 'alpha': 0.45,
                'edgecolor': COLOR_PLAYER, 'linewidth': 2.5},
    )
    radar.draw_range_labels(ax=ax, fontsize=8, color='#555555')
    radar.draw_param_labels(ax=ax, fontsize=10, color='#222222')

    legend_handles = [
        mpatches.Patch(color=COLOR_PLAYER, alpha=0.6,
                       label=str(target['player_name'])),
        mpatches.Patch(color=COLOR_IDEAL, alpha=0.4,
                       label=f'Idéal {role_name} (90e centile {pos})'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              bbox_to_anchor=(1.30, 1.05), frameon=False, fontsize=10)

    title = (f'{target["player_name"]} — {role_name}\n'
             f'Score de matching : {score:.1f}/100  '
             f'({target["team"]}, {target["minutes_total"]:.0f} min, '
             f'cluster {target["role_label"]})')
    ax.set_title(title, fontsize=13, pad=20, fontweight='bold')

    path = _resolve_path(save_path)
    if path:
        fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'[saved] {path}')
    return fig


# ---------------------------------------------------------------------------
# 2) Ranking table
# ---------------------------------------------------------------------------

def plot_ranking_table(role_name, position, top_n=10, save_path=None,
                       min_minutes=450):
    """Tableau matplotlib des top-N joueurs pour un rôle, gradient vert.

    Args:
        role_name (str): clé du rôle dans le YAML.
        position (str): code de position (CB, FB, MF, AM, ST).
        top_n (int): nombre de joueurs à afficher.
        save_path (str | None): chemin PNG (relatif → ``results/``).
        min_minutes (int): filtre minutes jouées.

    Returns:
        matplotlib.figure.Figure: la figure produite.
    """
    ranking = _m.rank_players_by_role(role_name, position,
                                       min_minutes=min_minutes,
                                       top=top_n, verbose=False)
    top = ranking.head(top_n).reset_index(drop=True)

    headers = ['#', 'Joueur', 'Équipe', 'Min.', 'Score', 'Cluster']
    rows = []
    for i, r in top.iterrows():
        rows.append([
            str(i + 1),
            str(r['player_name']),
            str(r['team']),
            f'{r["minutes_total"]:.0f}',
            f'{r["score"]:.1f}',
            str(r['role_label']),
        ])

    # Gradient vert basé sur le score (normalisé sur la plage du top)
    cmap = LinearSegmentedColormap.from_list(
        'score_green', ['#ffffff', '#a8e0a0', '#2ca02c'])
    smin, smax = top['score'].min(), top['score'].max()
    span = max(smax - smin, 1e-9)
    score_colors = [cmap(0.15 + 0.85 * (s - smin) / span)
                    for s in top['score']]

    fig, ax = plt.subplots(figsize=(12, 0.55 * (top_n + 1) + 1.2),
                           dpi=DPI / 3)
    fig.patch.set_facecolor('white')
    ax.set_axis_off()

    table = ax.table(
        cellText=rows, colLabels=headers, loc='center', cellLoc='center',
        colWidths=[0.05, 0.30, 0.22, 0.10, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # En-tête
    for j, _ in enumerate(headers):
        cell = table[0, j]
        cell.set_facecolor('#2f4858')
        cell.set_text_props(color='white', fontweight='bold')

    # Lignes : alterner blanc / gris très clair + colorer la colonne Score
    score_col_idx = headers.index('Score')
    for i in range(len(rows)):
        bg = '#fafafa' if i % 2 else 'white'
        for j in range(len(headers)):
            cell = table[i + 1, j]
            cell.set_edgecolor('#dddddd')
            if j == score_col_idx:
                cell.set_facecolor(score_colors[i])
                cell.set_text_props(fontweight='bold')
            else:
                cell.set_facecolor(bg)
        # Aligner joueur/équipe à gauche pour la lisibilité
        table[i + 1, 1].set_text_props(ha='left')
        table[i + 1, 2].set_text_props(ha='left')
        table[i + 1, 1].PAD = 0.02
        table[i + 1, 2].PAD = 0.02

    ax.set_title(f'Top {top_n} — {role_name} ({position})',
                 fontsize=15, fontweight='bold', pad=14)

    path = _resolve_path(save_path)
    if path:
        fig.savefig(path, dpi=DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'[saved] {path}')
    return fig


# ---------------------------------------------------------------------------
# 3) PCA scatter
# ---------------------------------------------------------------------------

def plot_pca_scatter(position, highlight_player=None, save_path=None):
    """Scatter 2D des joueurs d'un poste dans l'espace PCA, coloré par cluster.

    Args:
        position (str): code de position (CB, FB, MF, AM, ST).
        highlight_player (str | None): si fourni, ajoute une étoile dorée
            sur ce joueur (recherche insensible aux accents).
        save_path (str | None): chemin PNG.

    Returns:
        matplotlib.figure.Figure: la figure produite.
    """
    df, _ = _m._load_data()
    sub = df[df['position_group'] == position].copy()
    if sub.empty:
        raise KeyError(f'Aucun joueur pour la position {position}.')

    fig, ax = plt.subplots(figsize=(11, 8), dpi=DPI / 3)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fbfbfb')

    clusters = sorted(sub['cluster'].unique())
    for c in clusters:
        mask = sub['cluster'] == c
        ax.scatter(
            sub.loc[mask, 'pca_1'], sub.loc[mask, 'pca_2'],
            s=42, alpha=0.65,
            color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
            edgecolors='white', linewidths=0.6,
            label=f'Cluster {c} ({mask.sum()} joueurs)',
        )

    # Centroïdes
    cent = sub.groupby('cluster')[['pca_1', 'pca_2']].mean()
    for c, row in cent.iterrows():
        ax.scatter(row['pca_1'], row['pca_2'], marker='X', s=220,
                   color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax.text(row['pca_1'], row['pca_2'] + 0.25, f'C{c}',
                ha='center', fontsize=10, fontweight='bold', zorder=6)

    # Highlight
    if highlight_player is not None:
        try:
            target = _find_player(highlight_player, position=position)
            ax.scatter(target['pca_1'], target['pca_2'],
                       marker='*', s=520, color='#ffd700',
                       edgecolors='black', linewidths=1.4, zorder=10,
                       label=f'★ {target["player_name"]}')
            ax.annotate(
                str(target['player_name']),
                (target['pca_1'], target['pca_2']),
                xytext=(12, 12), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', edgecolor='#ffd700',
                          alpha=0.9),
                arrowprops=dict(arrowstyle='-', color='#888888', lw=0.8),
                zorder=11,
            )
        except KeyError as e:
            print(f'[warn] {e}')

    ax.axhline(0, color='#dddddd', lw=0.8, zorder=0)
    ax.axvline(0, color='#dddddd', lw=0.8, zorder=0)
    ax.set_xlabel('PCA component 1', fontsize=11)
    ax.set_ylabel('PCA component 2', fontsize=11)
    ax.set_title(f'Espace PCA des {position} ({len(sub)} joueurs)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='best', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)

    path = _resolve_path(save_path)
    if path:
        fig.savefig(path, dpi=DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'[saved] {path}')
    return fig


# ---------------------------------------------------------------------------
# 4) Player card (radar + infos textuelles)
# ---------------------------------------------------------------------------

def plot_player_card(player_name, role_name, save_path=None, position=None):
    """Fiche A4 paysage : radar à gauche, infos texte à droite.

    Le panneau de droite affiche :
        - Identité (nom, équipe, position, minutes).
        - Score de matching pour le rôle demandé.
        - Rôle naturel (meilleur score parmi tous les rôles du poste).
        - Top 3 forces + top 3 faiblesses face au profil idéal.
        - Cluster d'appartenance + 3 voisins les plus similaires.

    Args:
        player_name (str): nom (fragment) du joueur.
        role_name (str): rôle ciblé.
        save_path (str | None): chemin PNG.
        position (str | None): force le poste en cas d'homonymie.

    Returns:
        matplotlib.figure.Figure: la figure produite.
    """
    df, profiles = _m._load_data()
    target = _find_player(player_name, position=position)
    pos = target['position_group']
    if role_name not in profiles.get(pos, {}):
        raise KeyError(f'Rôle "{role_name}" indisponible pour {pos}.')

    role_weights = profiles[pos][role_name]
    cols, player_vec, ideal_vec, _ = _role_normalized_values(
        target, role_weights, role_name=role_name)
    score = _m.compute_matching_score(target, role_weights, df, role_name=role_name)

    strengths, weaknesses = _m._strengths_weaknesses(
        target, role_weights, role_name=role_name)

    # Profil sur tous les rôles
    natural_df = _m.profile_player(target['player_name'],
                                    position=pos, verbose=False)
    natural_role = natural_df.iloc[0]['role_name']
    natural_score = natural_df.iloc[0]['score']

    # Joueurs similaires (top 3 hors lui-même)
    sims = _m.find_similar_players(target['player_name'],
                                    position=pos, top_n=3)

    # ----- Figure A4 paysage : 11.69 x 8.27 pouces (297 x 210 mm)
    fig = plt.figure(figsize=(11.69, 8.27), dpi=DPI / 3)
    fig.patch.set_facecolor(COLOR_BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10,
                          left=0.04, right=0.97, top=0.92, bottom=0.05)

    # ---- Panneau gauche : radar
    ax_radar = fig.add_subplot(gs[0, 0])
    params = [_pretty_metric(c) for c in cols]
    radar = Radar(
        params=params,
        min_range=[0.0] * len(cols),
        max_range=[1.0] * len(cols),
        num_rings=4, ring_width=1.0, center_circle_radius=1.0,
    )
    radar.setup_axis(ax=ax_radar, facecolor='white')
    radar.draw_circles(ax=ax_radar, facecolor='#ebebeb', edgecolor=COLOR_GRID)
    radar.draw_radar_solid(
        ideal_vec, ax=ax_radar,
        kwargs={'facecolor': COLOR_IDEAL, 'alpha': 0.18,
                'edgecolor': COLOR_IDEAL, 'linewidth': 2.0,
                'linestyle': '--'},
    )
    radar.draw_radar_solid(
        player_vec, ax=ax_radar,
        kwargs={'facecolor': COLOR_PLAYER, 'alpha': 0.45,
                'edgecolor': COLOR_PLAYER, 'linewidth': 2.4},
    )
    radar.draw_range_labels(ax=ax_radar, fontsize=7, color='#555555')
    radar.draw_param_labels(ax=ax_radar, fontsize=9, color='#222222')

    # ---- Panneau droit : texte
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.set_axis_off()
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    def _line(y, text, **kw):
        ax_text.text(0.02, y, text, transform=ax_text.transAxes,
                     va='top', **kw)

    # Identité
    _line(0.98, str(target['player_name']),
          fontsize=20, fontweight='bold', color='#1a1a1a')
    _line(0.91, f'{target["team"]}  ·  {pos}  ·  '
                f'{target["minutes_total"]:.0f} min',
          fontsize=12, color='#666666')

    # Bandeau score
    ax_text.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.78), 0.96, 0.09,
        boxstyle='round,pad=0.01,rounding_size=0.01',
        transform=ax_text.transAxes,
        facecolor=COLOR_PLAYER, edgecolor='none', alpha=0.92))
    ax_text.text(0.04, 0.825,
                 f'Score "{role_name}"',
                 transform=ax_text.transAxes, va='center',
                 fontsize=12, color='white')
    ax_text.text(0.96, 0.825, f'{score:.1f}/100',
                 transform=ax_text.transAxes, va='center', ha='right',
                 fontsize=22, fontweight='bold', color='white')

    # Rôle naturel
    natural_marker = '★ ' if natural_role == role_name else ''
    _line(0.71, f'Rôle naturel : {natural_marker}{natural_role}  '
                f'({natural_score:.1f}/100)',
          fontsize=11, color='#1a1a1a', fontweight='bold')
    _line(0.665,
          f'Cluster data-driven : {target["role_label"]}',
          fontsize=10, color='#444444')

    # Forces
    _line(0.61, 'Top forces vs profil idéal', fontsize=11,
          fontweight='bold', color='#2ca02c')
    for k, (m, _, pv, iv) in enumerate(strengths[:3]):
        _line(0.575 - k * 0.04,
              f'  + {_pretty_metric(m)}  ({pv:.2f} vs {iv:.2f})',
              fontsize=10, color='#2ca02c')
    if not strengths:
        _line(0.575,
              '  (aucune métrique au-dessus du profil idéal)',
              fontsize=10, color='#888888', style='italic')

    # Faiblesses
    _line(0.43, 'Top faiblesses vs profil idéal', fontsize=11,
          fontweight='bold', color='#d62728')
    for k, (m, _, pv, iv) in enumerate(weaknesses[:3]):
        _line(0.395 - k * 0.04,
              f'  − {_pretty_metric(m)}  ({pv:.2f} vs {iv:.2f})',
              fontsize=10, color='#d62728')
    if not weaknesses:
        _line(0.395, '  (aucune faiblesse marquée)',
              fontsize=10, color='#888888', style='italic')

    # Joueurs similaires
    _line(0.25, 'Joueurs les plus similaires', fontsize=11,
          fontweight='bold', color='#1a1a1a')
    for k, row in sims.iterrows():
        _line(0.215 - k * 0.04,
              f'  · {row["player_name"]} ({row["team"]})  '
              f'sim={row["similarity"]:.1f}',
              fontsize=10, color='#333333')

    # Pied de page
    _line(0.04,
          'tb-scouting · radar normalisé MinMax intra-position · '
          'idéal = 90e centile du poste',
          fontsize=8, color='#888888', style='italic')

    fig.suptitle(f'Fiche joueur — adéquation au rôle "{role_name}"',
                 fontsize=14, fontweight='bold', y=0.97)

    path = _resolve_path(save_path)
    if path:
        fig.savefig(path, dpi=DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'[saved] {path}')
    return fig


# ---------------------------------------------------------------------------
# Démo CLI
# ---------------------------------------------------------------------------

def _demo():
    """Génère un exemple de chaque visualisation dans ``results/``."""
    plot_radar_chart('Verratti', 'deep_lying_playmaker',
                     save_path='radar_verratti_dlp.png')
    plot_ranking_table('deep_lying_playmaker', 'MF', top_n=10,
                       save_path='ranking_mf_dlp.png')
    plot_pca_scatter('MF', highlight_player='Verratti',
                     save_path='pca_mf_verratti.png')
    plot_player_card('Verratti', 'deep_lying_playmaker',
                     save_path='card_verratti_dlp.png')
    plot_player_card('Suarez', 'poacher', position='ST',
                     save_path='card_suarez_poacher.png')
    plt.close('all')


if __name__ == '__main__':
    _demo()
