# RailwayNetwork (SIH_Railways)

Railway simulation and RL control project using:
- SimPy for discrete-event railway operations
- Graph-based railway topology with IR-style signal placement
- PPO + LGNN (Liquid Graph Neural Network) for learning signal-control policies
- Visualization and training dashboards for analysis

## Folder Context (Important)

- On GitHub, this repository now has a top-level folder: `SIH_RAILWAYS/`.
- All project code (`graph/`, `simulation/`, `models/`, etc.) is inside `SIH_RAILWAYS/`.
- On your local machine, the repository folder may be named `SIH_Railways`, and the project code is in the nested `SIH_RAILWAYS/` folder.

Local example:

```text
/home/cyborg-prithwish/PycharmProjects/SIH_Railways/SIH_RAILWAYS
```

All commands in this README assume you are inside that `SIH_RAILWAYS/` folder.

## What This Project Can Do

- Build/prepare railway graphs from raw geospatial data.
- Add Indian Railway style signals to graph edges.
- Run simulation in visual mode (Pygame) or headless mode.
- Export simulation episodes/events for analysis.
- Train an RL policy (PPO + LGNN) against the simulation environment.
- Generate learning curves and diagnostic training dashboards.

## Repository Layout

Top-level repository:

```text
RailwayNetwork/
  SIH_RAILWAYS/
```

Project folder contents:

```text
SIH_Railways/
  data/
    raw/                  # Optional raw input data (your own sources)
    processed/            # Simulation-ready graph files
    episodes/             # Headless simulation exports (events/snapshots/metrics)
  dataset/                # Training metrics and dashboard outputs
  graph/                  # Graph generation + signal placement scripts
  simulation/             # Core and enhanced SimPy simulators
  models/                 # LGNN and PPO model modules
  training/               # RL training entrypoint
  visualization/          # Renderer and training plot generation
  tests/                  # Basic tests
  requirements.txt
```

## Prerequisites

- Python 3.10.x (recommended: 3.10.19+)
- Conda environment (recommended)
- Linux/macOS/Windows with OpenGL support for visual mode

## Environment Setup

If you already have `tf-torch` or `tf-torch-gpu`, activate it.

```bash
conda activate tf-torch
# or
conda activate tf-torch-gpu
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start (Minimum Working Flow)

From project root (`SIH_RAILWAYS` project folder):

```bash
cd /home/cyborg-prithwish/PycharmProjects/SIH_Railways/SIH_RAILWAYS
```

If you freshly clone this repo, use:

```bash
git clone https://github.com/InnovationStation-IR/RailwayNetwork.git
cd RailwayNetwork/SIH_RAILWAYS
```

1) Ensure base graph exists:

```bash
ls data/processed/godot_station_graph.json
```

2) Generate signal-enriched graph:

```bash
python -m graph.add_signals \
  --input data/processed/godot_station_graph.json \
  --output data/processed/godot_station_graph_with_signals.json
```

3) Run a short headless simulation:

```bash
python -m simulation.core simulate \
  --datafile data/processed/godot_station_graph_with_signals.json \
  --duration 120 \
  --num-trains 20
```

## Simulation Modes

### 1) Core Simulation (`simulation/core.py`)

### Visual mode (Pygame)

```bash
python -m simulation.core visualize \
  --datafile data/processed/godot_station_graph_with_signals.json \
  --initial-trains 20 \
  --seed 42
```

Use this when you want to watch train movement, signals, and block occupancy.

### Headless mode (faster, export-ready)

```bash
python -m simulation.core simulate \
  --datafile data/processed/godot_station_graph_with_signals.json \
  --duration 600 \
  --num-trains 60 \
  --snapshot-interval 1.0 \
  --episode-name ep_test_001
```

Outputs are exported to `data/episodes/`.

### 2) Enhanced Simulation (`simulation/enhanced.py`)

This includes richer metrics and more execution options.

### Enhanced visual

```bash
python -m simulation.enhanced visualize \
  --datafile data/processed/godot_station_graph.json \
  --initial-trains 20 \
  --enable-disruptions \
  --enable-collisions \
  --debug
```

### Enhanced headless

```bash
python -m simulation.enhanced simulate \
  --datafile data/processed/godot_station_graph.json \
  --duration 900 \
  --num-trains 80 \
  --enable-disruptions \
  --export-format jsonl
```

### Batch simulation

```bash
python -m simulation.enhanced batch \
  --datafile data/processed/godot_station_graph.json \
  --episodes 10 \
  --duration 600 \
  --num-trains 80 \
  --parallel \
  --max-workers 4
```

Use batch mode for statistical comparison and stress testing.

## RL Training (PPO + LGNN)

Training entrypoint:

```bash
python -m training.train_rl \
  --datafile data/processed/godot_station_graph_with_signals.json \
  --timesteps 25000 \
  --n-steps 512 \
  --trains 30 \
  --decision-interval 30 \
  --lr 3e-4
```

Generated outputs:

- `models/trained_ppo.pt` (saved model checkpoint)
- `dataset/training_metrics.json` (episode/update-level metrics)
- `dataset/training_dashboard.png` (auto-generated dashboard when possible)

## Generate Training Dashboard Manually

```bash
python -m visualization.training_plots \
  --metrics dataset/training_metrics.json \
  --out dataset/training_dashboard.png
```

## How To Tell If Model Is Working

Check these indicators in `dataset/training_dashboard.png`:

- Episode reward trend should generally improve.
- Entropy should reduce gradually (less random policy over time).
- KL and clip fraction should stay stable (no extreme spikes every update).
- LGNN gradient norms should be non-zero (`grad_norm_lgnn`) showing LGNN is being trained.
- Episode length and completion metrics should trend in a better direction over long runs.

## Using Your Own Data

You have three practical options.

### Option A: Replace processed graph directly (fastest)

If you already have a graph JSON, place it as:

- `data/processed/godot_station_graph.json`

Then regenerate signals:

```bash
python -m graph.add_signals \
  --input data/processed/godot_station_graph.json \
  --output data/processed/godot_station_graph_with_signals.json
```

### Option B: Build from station JSON (`graph/build_graph.py`)

Input expected by script:

- `data/raw/west_bengal_stations.json`

Then run:

```bash
python graph/build_graph.py
```

This writes nodes/edges GraphML/GEXF/CSV to `data/processed/`.

### Option C: Build from GeoJSON and stations CSV (`graph/project_graph.py`)

Default inputs:

- `data/raw/export (2).geojson`
- `data/processed/stations_only_nodes.csv`

Then run:

```bash
python graph/project_graph.py
```

This produces `godot_station_signal_graph.json` (copy/rename to `data/processed/godot_station_graph.json` if you want to use it with this pipeline), then run `graph.add_signals`.

## Minimum Graph JSON Format (for custom data)

Your `godot_station_graph.json` should have:

```json
{
  "nodes": [
    {"id": "A", "label": "Station A", "pos_x": 100.0, "pos_y": 200.0}
  ],
  "edges": [
    {"source": "A", "target": "B"}
  ]
}
```

Then `graph.add_signals` will enrich each edge with signal definitions and distances.

## Output Files You Should Expect

After simulation/training, common outputs include:

- `data/episodes/*_snapshots.jsonl`
- `data/episodes/*_events.jsonl`
- `data/episodes/*_metrics.json` (enhanced mode)
- `dataset/training_metrics.json`
- `dataset/training_dashboard.png`
- `models/trained_ppo.pt`

## Reproducibility Tips

- Pass `--seed` for deterministic sampling where supported.
- Keep Python version at 3.10.x for parity with this codebase.
- Use one dedicated conda env for all runs to avoid package mismatch.

## Troubleshooting

### `Pygame not available`

Install or reinstall pygame:

```bash
pip install pygame
```

### `File not found: godot_station_graph_with_signals.json`

Run signal generation step first:

```bash
python -m graph.add_signals --input data/processed/godot_station_graph.json --output data/processed/godot_station_graph_with_signals.json
```

### RL training starts but no dashboard image

Generate manually:

```bash
python -m visualization.training_plots --metrics dataset/training_metrics.json --out dataset/training_dashboard.png
```

### Slow training or memory pressure

- Reduce `--timesteps`, `--trains`, and `--n-steps`.
- Use headless simulation for data generation.
- Avoid running visual mode during long training.

## Suggested Workflow For New Users

1. Install dependencies in a Python 3.10 env.
2. Prepare or replace `data/processed/godot_station_graph.json`.
3. Run `graph.add_signals`.
4. Run a short headless simulation to validate data and pipeline.
5. Train PPO+LGNN on short timesteps first.
6. Inspect dashboard and metrics.
7. Scale up training and batch simulation for experiments.

## License and Notes

This repository is intended for railway simulation/RL experimentation. Validate all assumptions before any real-world use.
