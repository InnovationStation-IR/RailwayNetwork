# RailwayNetwork

AI-assisted railway traffic control for Indian Railways. The project simulates a railway network with Indian Railway style signalling and trains a reinforcement learning agent to control signals, with the aim of reducing delays and keeping traffic flowing safely through disruptions.

> **Note:** The `main` branch is a landing page. The project code lives on the branches listed below.

## Branches

| Branch | What's there |
| --- | --- |
| [`ai`](https://github.com/InnovationStation-IR/RailwayNetwork/tree/ai) | Simulation, graph pipeline and RL training (the main codebase, inside `SIH_RAILWAYS/`) |
| [`backend`](https://github.com/InnovationStation-IR/RailwayNetwork/tree/backend) | Backend service (early stage) |
| [`frontend`](https://github.com/InnovationStation-IR/RailwayNetwork/tree/frontend) | Web interface (early stage) |

## What it does

- **Railway graph** built from real station and track geodata (West Bengal network), with signals placed on each edge following Indian Railway spacing rules.
- **Signalling model** implementing 4-aspect MACLS colour light signals (Red, Yellow, Double Yellow, Green), absolute block sections, interlocking and SPAD (signal passed at danger) detection.
- **Discrete-event simulation** in SimPy, with a Pygame visual mode, a fast headless mode and parallel batch runs that export events, snapshots and metrics.
- **Disruptions** such as track faults, signal failures, overhead wire breaks, weather, animal intrusion, landslides and equipment failures, scheduled at random and applied to blocks and signals.
- **Learning controller** using PPO with a Liquid Graph Neural Network (GAT + GCN message passing, integrated with RK4) so the policy reasons over the network's structure, plus dashboards to track training.

## Tech stack

Python 3.10 · SimPy · NetworkX · PyTorch · PyTorch Geometric · Gymnasium · Pygame · Matplotlib

## Getting started

```bash
git clone -b ai https://github.com/InnovationStation-IR/RailwayNetwork.git
cd RailwayNetwork/SIH_RAILWAYS
pip install -r requirements.txt

# Add signals to the base graph
python -m graph.add_signals \
  --input data/processed/godot_station_graph.json \
  --output data/processed/godot_station_graph_with_signals.json

# Run a short headless simulation
python -m simulation.core simulate \
  --datafile data/processed/godot_station_graph_with_signals.json \
  --duration 120 --num-trains 20

# Train the PPO + LGNN agent
python -m training.train_rl \
  --datafile data/processed/godot_station_graph_with_signals.json \
  --timesteps 25000
```

The full guide (visual mode, batch runs, using your own data, troubleshooting) is in the [`ai` branch README](https://github.com/InnovationStation-IR/RailwayNetwork/tree/ai).

## Team

### Web
- [Prativa](https://github.com/Prativa705)
- [Rahul Singh](https://github.com/amrahulsingh)
- [Subhayan Shaha](https://github.com/Itssubhayan)

### AI-ML
- [Rupam Neogi](https://github.com/plant7-arch)
- [Prithwish Ghosh](https://github.com/cogni-prith)

### Design
- Nilanjana Tripathy

## Acknowledgements

Based on the original work of **Innovation Station**.

This repository is for railway simulation and RL research. Validate all assumptions before any real-world use.
