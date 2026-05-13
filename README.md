# ✈️ SkySlot-MARL: Schiphol Air Cargo Optimization

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Environment%20Complete-green.svg)]()
[![Engine: SimPy](https://img.shields.io/badge/Engine-SimPy-blue.svg)]()
[![Interface: PettingZoo](https://img.shields.io/badge/Interface-PettingZoo-orange.svg)]()

A high-fidelity **Multi-Agent Reinforcement Learning (MARL)** environment designed to optimize truck slot bookings and landside congestion at Amsterdam Airport Schiphol. This project bridges discrete-event simulation (DES) with modern deep RL frameworks.

---

## 🏗️ Project Architecture

This project implements a **Centralized Training, Decentralized Execution (CTDE)** paradigm. It models the complex interactions between independent Ground Handling Agents (GHAs), a central Transporter agent, and an optional Airport Orchestrator.

---

### 🧩 Core Modules
| Module | Responsibility |
| :--- | :--- |
| **`dtp_platform.py`** | The "Digital Twin" platform rules; manages slot publishing, booking, and validation. |
| **`objects.py`** | Physical entities: `Trucks`, `GHATerminal` (Docks/Queues), and `TP3Buffer` (Parking). |
| **`demand.py`** | Stochastic arrival engine; manages the lifecycle and journey of every truck. |
| **`infrastructure.py`** | The sensor layer; tracks ANPR events and timestamps for KPI calculation. |
| **`kpi_tracker.py`** | Translates raw events into rewards (WPR, Turnaround Time, Utilization). |

---

### Layers
The project is organized logically into 3 "Layers" so that every layer acts as a base for the next one:
- **Platform Layer (the foundations):** The first layer encorces the DTP (Digital Truck Slot Planning) rules through pure Python logic.
- **Simulation Layer (the logistics):** The second layer leverages `SimPy` to build a simulation environment that allows to iterate through episodes, in order to give to the MARL engine something to learn from.
- **MARL Layer (the brain):** The last layer is the Reinforcement Learning engine, which leverages `BenchMARL`, a state-of-the-art Python library coming from the Meta labs, in order to learn smart policies from the SimPy simulations.

---

## 📂 Project Structure

```text
.
├── config/             # YAML/Python global parameters
├── env/                # Core Simulation Logic
│   ├── schiphol_env.py
│   ├── objects.py
│   ├── dtp_platform.py
│   └── ...            
├── testing/            # Sanity checks and execution scripts
├── marl/               # Reinforcement Learning layer
└── requirements.txt    # Dependency manifest
```

---

## 🚀 Current Milestone: [1/3] Environment Engineering
At the moment the `Platform Layer` and the `Simulation Layer` are ultimated. The project is currently at the integration phase, where the SimPy-based logistics engine needs to be successfully exposed to the Multi-Agent BenchMARL environment.

- [x] DTP Rules: Implementation of the slot booking "Freeze-time" and "Lead-time" constraints.

- [x] Core Simulation: Discrete event logic for cargo handling and truck movement.

- [x] PettingZoo Wrapper: Full implementation of observation_space, action_space, and action_masking.

- [ ] BenchMARL Adapter: Full translation of PettingZoo objects into BenchMARL tasks.

- [ ] Phase 2 (Next): Training MAPPO/IPPO agents using the BenchMARL framework.

- [ ] Phase 3: Multi-Scenario Benchmarking (Scenario M vs. Scenario MO).

---

## 🛠️ Installation & Usage
This project is implemented on ubuntu 24.04 using `UV` as a package manager, for any implementation with on different platforms please consult official docs and make sure to check out the `pyproject.toml` file for library dependencies.

Installation steps:
  1. Clone the repository
```Bash
git clone https://github.com/SnappierCape/ds-marl-air-cargo.git
cd ds-marl-air-cargo
```

  2. Setup Environment
```Bash
uv lock
uv sync
```

  3. Run a Sanity Check to verify that the PettingZoo wrapper is correctly communicating with the SimPy engine:
```Bash
uv run ./testing/full_episode.py --steps=3000
```

---

## 📊 Logic & Flow
The simulation uses a 1-minute time step resolution. Trucks are generated based on historical demand profiles, redirected to the TP3 buffer if slots aren't ready, and processed at GHAs using a stochastic service-time model.

Key Feature: The Transporter agent must learn to book slots that minimize Wait-to-Process Ratio (WPR) while GHA agents learn to publish slots that maximize Dock Utilization. The Orchestrator is a neutral entity with *almost* unlimited powers that aims at optimizing the whole system.

---

## 📜 Acknowledgments
Developed for researchers and logistics engineers interested in the intersection of Operations Research and AMachine Learning.

Special thanks to the Schiphol Landside Cargo community for the operational insights.