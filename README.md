# tb-football-scouting-framework

**Bachelor Thesis — HEG Geneva 2026**
**Author:** André Dos Santos
**Supervisor:** Dr. Grigorios Anagnostopoulos

---

## Overview

A data-driven football scouting framework that bridges the gap between tactical needs
expressed by professional club staff and the analytical capabilities offered by open
football data.

The framework allows a recruiter to define tactical role profiles (e.g., "Ball-Playing
Centre-Back", "Pressing Forward") and returns a ranked list of players with a 0-100
compatibility score, radar charts, and strength/weakness summaries.

## Pipeline

```
StatsBomb Open Data
        ↓
  Data Extraction       → src/extraction.py
        ↓
  Feature Engineering   → src/features.py     (~30 metrics per player, per 90 min)
        ↓
  Normalisation + PCA   → src/clustering.py   (dimensionality reduction)
        ↓
  K-Means Clustering    → src/clustering.py   (role profile discovery)
        ↓
  Tactical Matching     → src/matching.py     (cosine similarity, 0-100 score)
        ↓
  Visualisation         → src/visualisation.py (radar charts, rankings)
```

## Data Sources

- [StatsBomb Open Data](https://github.com/statsbomb/open-data) — high-resolution
  match event data (passes, shots, duels, pressures, carries) via `statsbombpy`
- [Understat](https://understat.com) — xG and shot-level data via `soccerdata`

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/tb-football-scouting-framework.git
cd tb-football-scouting-framework

python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Usage

```python
# Quick example
from src.matching import rank_players_by_role

# Find the best ball-playing centre-backs
results = rank_players_by_role(
    role="ball_playing_cb",
    position="CB",
    min_minutes=450
)
print(results.head(10))
```

See `notebooks/04_demo.ipynb` for full use case demonstrations.

## Project Structure

```
tb-football-scouting-framework/
├── data/
│   ├── raw/                  ← StatsBomb raw events (auto-generated)
│   └── processed/            ← Cleaned player feature matrices
├── notebooks/
│   ├── 00_check_data.ipynb   ← Verify StatsBomb availability
│   ├── 01_exploration.ipynb  ← Data exploration
│   ├── 02_features.ipynb     ← Feature engineering development
│   ├── 03_clustering.ipynb   ← PCA + K-Means development
│   └── 04_demo.ipynb         ← Use case demonstrations (Chapter 6)
├── src/
│   ├── __init__.py
│   ├── extraction.py         ← Data extraction pipeline
│   ├── features.py           ← Metric calculation per player per 90
│   ├── clustering.py         ← Normalisation, PCA, K-Means
│   ├── matching.py           ← Cosine similarity, scoring, ranking
│   └── visualisation.py      ← Radar charts, tables, scatter plots
├── config/
│   └── role_profiles.yaml    ← Tactical role definitions (weights)
├── results/                  ← Output figures and tables
├── requirements.txt
└── README.md
```

## Role Catalogue

Roles are inspired by Football Manager 2026 archetypes, translated into
quantifiable StatsBomb metrics. Each role is defined as a weighted combination
of metrics in `config/role_profiles.yaml`.

| Position | Roles |
|----------|-------|
| CB | Ball-Playing Defender, No-Nonsense CB, Aerial Dominant CB |
| FB | Complete Wing-Back, Inverted Wing-Back, Defensive Full-Back |
| MF | Deep-Lying Playmaker, Ball-Winning Midfielder, Box-to-Box |
| AM | Advanced Playmaker, Inside Forward, Pressing Winger |
| ST | Advanced Forward, Pressing Forward, Poacher |

## License

MIT License — open source for academic use.
