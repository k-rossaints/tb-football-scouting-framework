"""
visualisation.py — Scouting visualisations: radar, ranking table, PCA scatter
and player card.

Four public functions:
    * :func:`plot_radar_chart` — player vs role-reference radar.
    * :func:`plot_ranking_table` — top-N ranking table for a role.
    * :func:`plot_pca_scatter` — 2D PCA scatter coloured by cluster.
    * :func:`plot_player_card` — A4-landscape sheet (radar + textual panel).

All outputs are exportable as 300-DPI PNGs into the project's ``results/``
directory.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Radar

from src import matching as _m

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLOR_PLAYER = '#1f77b4'   # blue
COLOR_IDEAL  = '#d62728'   # red
COLOR_BG     = '#f5f5f5'
COLOR_GRID   = '#cccccc'

CLUSTER_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                   '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

DPI = 300


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_results_dir():
    """Create ``results/`` at the project root if it does not exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _resolve_path(save_path):
    """Resolve a user-supplied save path into an absolute target.

    Relative paths and bare filenames are prefixed with :data:`RESULTS_DIR`;
    absolute paths are used as-is.

    Args:
        save_path: User-supplied path (or ``None``).

    Returns:
        Absolute path string ready to use, or ``None`` if input was None.
    """
    if save_path is None:
        return None
    _ensure_results_dir()
    if os.path.isabs(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path
    return os.path.join(RESULTS_DIR, save_path)


def _find_player(player_name, position=None):
    """Find a player in the clustered parquet (accent-insensitive search).

    Args:
        player_name: Name (fragment accepted).
        position: If supplied, restrict the search to that position group.

    Returns:
        The selected player's row.

    Raises:
        KeyError: If no player matches.
    """
    df, _ = _m._load_data()
    mask = _m._name_match(df['player_name'], player_name)
    if position is not None:
        mask &= df['position_group'] == position
    cand = df[mask]
    if cand.empty:
        raise KeyError(f'No player matches "{player_name}"')
    return cand.sort_values('minutes_total', ascending=False).iloc[0]


def _role_normalized_values(player_row, role_weights, role_name='?'):
    """Return aligned ``(columns, player_vec, ideal_vec, weight_vec)``.

    Columns are the parquet columns resolved from the YAML metric names;
    player and reference vectors are MinMax-normalised intra-position
    (∈ [0, 1]) so they can both be plotted on the same radar.

    Args:
        player_row: Player row.
        role_weights: YAML role weights.
        role_name: Role name (for warnings).

    Returns:
        Tuple ``(cols, player_vec, ideal_vec, weight_vec)``.
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
    """Turn a raw column name into a human-readable label.

    Examples:
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
    """Plot a player's metric profile against a role-reference polygon.

    The radar uses the role's metrics (from the YAML), MinMax-normalised
    intra-position in [0, 1]. Two polygons are drawn:
        - **Player** — solid blue.
        - **Reference (top-quartile)** — dashed red, at the
          :data:`matching.IDEAL_PERCENTILE`-th percentile of the position.

    The title shows the role-match score from
    :func:`matching.compute_matching_score`.

    Args:
        player_name: Name (fragment) of the target player.
        role_name: Role key in ``role_profiles.yaml``.
        save_path: If supplied, export as a 300-DPI PNG. Relative paths
            resolve under ``results/``; absolute paths are used as-is.
        position: Force the position group if the name is ambiguous.

    Returns:
        The matplotlib Figure object.
    """
    df, profiles = _m._load_data()
    target = _find_player(player_name, position=position)
    pos = target['position_group']
    if pos not in profiles or role_name not in profiles[pos]:
        raise KeyError(f'Role "{role_name}" not available for {pos}.')

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

    # Reference polygon first so the player overlay sits on top
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
                       label=f'{role_name} reference '
                             f'({_m.IDEAL_PERCENTILE}th pct of {pos})'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              bbox_to_anchor=(1.30, 1.05), frameon=False, fontsize=10)

    title = (f'{target["player_name"]} — {role_name}\n'
             f'Match score: {score:.1f}/100  '
             f'({target["team"]}, {target["minutes_total"]:.0f} min, '
             f'cluster {target["role_label"]})')
    ax.set_title(title, fontsize=13, pad=20, fontweight='bold')

    path = _resolve_path(save_path)
    if path:
        fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'[saved] {path}')
    return fig


# ---------------------------------------------------------------------------
# 1bis) Custom radar — same as plot_radar_chart but with caller-supplied weights
# ---------------------------------------------------------------------------

def plot_custom_radar(player_name, custom_weights, position=None,
                      save_path=None, profile_label='Custom Profile'):
    """Plot a player's profile against a user-defined set of weighted metrics.

    Identical to :func:`plot_radar_chart` but bypasses the YAML lookup:
    the radar's axes are exactly the metrics supplied in ``custom_weights``
    (resolved through :func:`matching._resolved_weights`), and the dashed
    reference polygon is the :data:`matching.IDEAL_PERCENTILE`-th
    percentile of the position on those same metrics.

    The title says "Custom Profile" by default so the figure is unambiguous
    when filed next to predefined-role radars.

    Args:
        player_name: Name (fragment) of the target player.
        custom_weights: Dict ``{metric_name: weight}``. Same naming
            convention as :func:`matching.custom_role_search`.
        position: Force the position group if the name is ambiguous.
        save_path: PNG path (relative paths resolve under ``results/``).
        profile_label: Title label for the profile (default
            ``'Custom Profile'``).

    Returns:
        The matplotlib Figure object.
    """
    if not isinstance(custom_weights, dict) or not custom_weights:
        raise ValueError('custom_weights must be a non-empty dict '
                         '{metric_name: weight}.')

    df, _ = _m._load_data()
    target = _find_player(player_name, position=position)
    pos = target['position_group']

    # Resolve weights and build player/reference vectors on the same axes
    norm_df, all_cols = _m._get_normalized(pos)
    resolved = _m._resolved_weights(custom_weights, all_cols,
                                    role_name='custom')
    if not resolved:
        raise ValueError('None of the supplied metrics could be resolved '
                         'against the dataset.')

    cols = list(resolved.keys())
    player_vec = norm_df.loc[target['player_id'], cols].to_numpy(dtype='float64')
    ideal_vec = _m._ideal_vector(pos, cols).to_numpy(dtype='float64')
    score = _m.compute_matching_score(target, custom_weights, df,
                                      role_name='custom')

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
                       label=f'Reference '
                             f'({_m.IDEAL_PERCENTILE}th pct of {pos})'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              bbox_to_anchor=(1.30, 1.05), frameon=False, fontsize=10)

    title = (f'{target["player_name"]} — {profile_label}\n'
             f'Match score: {score:.1f}/100  '
             f'({target["team"]}, {target["minutes_total"]:.0f} min, '
             f'cluster {target["role_label"]})')
    ax.set_title(title, fontsize=13, pad=20, fontweight='bold')

    path = _resolve_path(save_path)
    if path:
        fig.savefig(path, dpi=DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'[saved] {path}')
    return fig


# ---------------------------------------------------------------------------
# 2) Ranking table
# ---------------------------------------------------------------------------

def plot_ranking_table(role_name, position, top_n=10, save_path=None,
                       min_minutes=450):
    """Render a top-N ranking table for a role, with a green score gradient.

    Args:
        role_name: Role key in the YAML.
        position: Position code (CB, FB, MF, AM, ST).
        top_n: How many players to display.
        save_path: PNG path (relative paths resolve under ``results/``).
        min_minutes: Minimum minutes-played filter.

    Returns:
        The matplotlib Figure object.
    """
    ranking = _m.rank_players_by_role(role_name, position,
                                      min_minutes=min_minutes,
                                      top=top_n, verbose=False)
    top = ranking.head(top_n).reset_index(drop=True)

    headers = ['#', 'Player', 'Team', 'Min.', 'Score', 'Cluster']
    rows = [
        [str(i + 1), str(r['player_name']), str(r['team']),
         f'{r["minutes_total"]:.0f}', f'{r["score"]:.1f}', str(r['role_label'])]
        for i, r in top.iterrows()
    ]

    # Score gradient (white -> light green -> strong green) over the top range
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

    # Header row
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor('#2f4858')
        cell.set_text_props(color='white', fontweight='bold')

    # Body: alternating row background + green gradient on the Score column
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
        # Left-align player and team for readability
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
    """Scatter the players of a position in 2D PCA space, coloured by cluster.

    Cluster centroids are drawn as labelled black-edged X markers. If a
    ``highlight_player`` is given, a gold star and a callout label are
    added at their PCA coordinates.

    Args:
        position: Position code (CB, FB, MF, AM, ST).
        highlight_player: If given, add a gold star on this player
            (accent-insensitive search).
        save_path: PNG path.

    Returns:
        The matplotlib Figure object.
    """
    df, _ = _m._load_data()
    sub = df[df['position_group'] == position].copy()
    if sub.empty:
        raise KeyError(f'No players for position {position}.')

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
            label=f'Cluster {c} ({mask.sum()} players)',
        )

    # Cluster centroids
    cent = sub.groupby('cluster')[['pca_1', 'pca_2']].mean()
    for c, row in cent.iterrows():
        ax.scatter(row['pca_1'], row['pca_2'], marker='X', s=220,
                   color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax.text(row['pca_1'], row['pca_2'] + 0.25, f'C{c}',
                ha='center', fontsize=10, fontweight='bold', zorder=6)

    # Highlight a specific player
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
    ax.set_title(f'PCA space — {position} ({len(sub)} players)',
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
# 4) Player card (radar + textual panel)
# ---------------------------------------------------------------------------

def plot_player_card(player_name, role_name, save_path=None, position=None):
    """A4-landscape player card: radar on the left, textual panel on the right.

    The right-hand panel shows:
        - Identity (name, team, position, minutes).
        - Role-match score for the requested role.
        - Natural role (highest-scoring role across the position).
        - Top 3 strengths and top 3 weaknesses vs the reference profile.
        - Data-driven cluster + 3 closest stylistic neighbours.

    Args:
        player_name: Name (fragment) of the target player.
        role_name: Role to score against.
        save_path: PNG path.
        position: Force the position if the name is ambiguous.

    Returns:
        The matplotlib Figure object.
    """
    df, profiles = _m._load_data()
    target = _find_player(player_name, position=position)
    pos = target['position_group']
    if role_name not in profiles.get(pos, {}):
        raise KeyError(f'Role "{role_name}" not available for {pos}.')

    role_weights = profiles[pos][role_name]
    cols, player_vec, ideal_vec, _ = _role_normalized_values(
        target, role_weights, role_name=role_name)
    score = _m.compute_matching_score(target, role_weights, df, role_name=role_name)

    strengths, weaknesses = _m._strengths_weaknesses(
        target, role_weights, role_name=role_name)

    # Cross-role profile
    natural_df = _m.profile_player(target['player_name'],
                                   position=pos, verbose=False)
    natural_role = natural_df.iloc[0]['role_name']
    natural_score = natural_df.iloc[0]['score']

    # Closest stylistic neighbours (top 3, excluding the player itself)
    sims = _m.find_similar_players(target['player_name'],
                                   position=pos, top_n=3)

    # ----- A4 landscape figure: 11.69 x 8.27 inches (297 x 210 mm)
    fig = plt.figure(figsize=(11.69, 8.27), dpi=DPI / 3)
    fig.patch.set_facecolor(COLOR_BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10,
                          left=0.04, right=0.97, top=0.92, bottom=0.05)

    # ---- Left panel: radar
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

    # ---- Right panel: text
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.set_axis_off()
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    def _line(y, text, **kw):
        ax_text.text(0.02, y, text, transform=ax_text.transAxes,
                     va='top', **kw)

    # Identity
    _line(0.98, str(target['player_name']),
          fontsize=20, fontweight='bold', color='#1a1a1a')
    _line(0.91, f'{target["team"]}  ·  {pos}  ·  '
                f'{target["minutes_total"]:.0f} min',
          fontsize=12, color='#666666')

    # Score banner
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

    # Natural role
    natural_marker = '★ ' if natural_role == role_name else ''
    _line(0.71, f'Natural role: {natural_marker}{natural_role}  '
                f'({natural_score:.1f}/100)',
          fontsize=11, color='#1a1a1a', fontweight='bold')
    _line(0.665,
          f'Data-driven cluster: {target["role_label"]}',
          fontsize=10, color='#444444')

    # Strengths
    _line(0.61, 'Top strengths vs reference', fontsize=11,
          fontweight='bold', color='#2ca02c')
    for k, (m, _, pv, iv) in enumerate(strengths[:3]):
        _line(0.575 - k * 0.04,
              f'  + {_pretty_metric(m)}  ({pv:.2f} vs {iv:.2f})',
              fontsize=10, color='#2ca02c')
    if not strengths:
        _line(0.575,
              '  (no metric above the reference profile)',
              fontsize=10, color='#888888', style='italic')

    # Weaknesses
    _line(0.43, 'Top weaknesses vs reference', fontsize=11,
          fontweight='bold', color='#d62728')
    for k, (m, _, pv, iv) in enumerate(weaknesses[:3]):
        _line(0.395 - k * 0.04,
              f'  − {_pretty_metric(m)}  ({pv:.2f} vs {iv:.2f})',
              fontsize=10, color='#d62728')
    if not weaknesses:
        _line(0.395, '  (no marked weakness)',
              fontsize=10, color='#888888', style='italic')

    # Similar players
    _line(0.25, 'Most similar players', fontsize=11,
          fontweight='bold', color='#1a1a1a')
    for k, row in sims.iterrows():
        _line(0.215 - k * 0.04,
              f'  · {row["player_name"]} ({row["team"]})  '
              f'sim={row["similarity"]:.1f}',
              fontsize=10, color='#333333')

    # Footer
    _line(0.04,
          f'tb-scouting · MinMax-normalised intra-position radar · '
          f'reference = {_m.IDEAL_PERCENTILE}th percentile of the position',
          fontsize=8, color='#888888', style='italic')

    fig.suptitle(f'Player card — fit to role "{role_name}"',
                 fontsize=14, fontweight='bold', y=0.97)

    path = _resolve_path(save_path)
    if path:
        fig.savefig(path, dpi=DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'[saved] {path}')
    return fig


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _demo():
    """Generate one example of each visualisation in ``results/``."""
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
