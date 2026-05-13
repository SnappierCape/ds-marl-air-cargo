# ✈️ Multi-Agent Reinforcement Learning in Schiphol Airport Landside Cargo Logistics

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Environment%20Complete-green.svg)]()
[![Engine: SimPy](https://img.shields.io/badge/Simulation-SimPy-blue.svg)]()
[![Interface: PettingZoo](https://img.shields.io/badge/Interface-PettingZoo-orange.svg)]()
[![Engine: BenchMARL](https://img.shields.io/badge/Engine-BenchMARL-red.svg)]()

A high-fidelity **Multi-Agent Reinforcement Learning (MARL)** environment designed to optimize truck slot bookings and landside congestion at Amsterdam Airport Schiphol Cargo Hub. This project bridges discrete-event simulation (DES) with modern deep MARL frameworks.

---

## 📦 Schiphol Cargo Hub Background

The Schiphol Cargo Hub is a primary European logistical gateway with high-density freight traffic managed by several independent Ground Handling Agents (GHAs). Landside operations currently rely on a decentralized model where GHAs manage their own warehouse and dock infrastructures. A primary operational challenge is the absence of a unified Truck Slot Booking System to synchronize inbound truck traffic with available dock capacity.

During peak periods, such as the concentrated arrival patterns on Friday afternoons, high logistics volume often exceeds immediate infrastructure capacity. Without a centralized layer to align slot availability with truck arrivals, the system can experience a temporal mismatch. This leads to uneven dock utilization and increased dwell times for transport vehicles.

This project uses a discrete-event simulation to study whether Multi-Agent Reinforcement Learning (MARL) can address this coordination gap. By modeling the hub as a multi-agent environment, the research evaluates if autonomous agents can learn to synchronize scheduling and resource allocation in a decentralized way. The objective is to determine if MARL can effectively transform independent decision-making into a cohesive logistical flow to improve the overall efficiency of the Schiphol cargo ecosystem.

---

### 📖 DTP (Digital Truck Slot Planning) Rules

This project operates under the conditions that the DTP (Digital Truck Slot Planning) system is already implemented at Schiphol. For this reason, it is based on a set of `synthetic` rules that should replicate the actual goal of the project. In particular:
  1. Every truck must have a confirmed DTP booking to pass the Main Gate. Enforced by ANPR at the Main Gate. No exceptions.
  2. Slots are 45-minutes time windows. One slot = one license plate = one time frame = one ground handler. One truck per slot.
  3. All GHAs publish their slots on the shared DTP platform. Visible to all participants in real time. This is the only way to publish a slot.
  4. DTP platform is operated by a neutral third party (Schiphol/ACN).
  5. The Orchestrator has full authority to cancel or modify any booking unilaterally as long as the truck isn't already docked, without requiring Transporter or GHA acceptance. It can NOT remove a published slot from the DTP platform.
  6. Minimum booking lead time is double the slot duration. After that slots are frozen: no new bookings, no cancellations. this is called the "frozen window".
  7. GHAs may publish slots up to 72h before the start of the slot and before the frozen window starts.
  8. Slots are divided into a "priority window" (first 10m) and a "release window" (m11 to slot end), the dock is held still until minute 10. Then the slot is available for the next trucks at the GHA queue or the trucks sitting at TP3. The DTP platform or the Orchestrator may release them. If the original truck shows up between minute 11 and slot end:
     - If the dock is still free → the original truck is admitted directly. No rebooking required. A small late-penalty is logged against the transporter account for RL feedback purposes but the truck proceeds.
     - If the dock is not free → the original truck is redirected to TP3 as a standby truck. It does not need to rebook a new slot. Its existing booking remains valid and it re-enters the queue when the Orchestrator or its own timing releases it.
  9. A truck that shows up after its own slot has expired is recorded as "no show" and the penalty is logged for RL feedback purposes. The truck is redirected at tp3. It can book another slot or the Orchestrator can send it to a GHA.
  10. The transporter can cancel a booking outside of the frozen window, only the Orchestrator can cancel a booking inside the froozen window.
  11. GHAs can not remove a published slot.
  12. For each GHA, the total amount of docks is split equally among export and import. GHAs can not have an odd number of docks.

For more information visit the official [Schiphol SCMP Web Page](https://www.schiphol.nl/nl/cargo/smart-cargo-mainport-program).

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

## ⚙️ Technology Stack
The project is implemented in `Python 3.12`. The following external libraries are used:

| Library        | Scope                                               |
|----------------|-----------------------------------------------------|
| **Numpy**      | High-performance vectorial math                     |
| **SimPy**      | Simulations                                         |
| **Gymnasium**  | Agents' action spaces                               |
| **PettingZoo** | SimPy wrapper for bridging simulations with RL      |
| **TorchRL**    | MARL Engine                                         |
| **BenchMARL**  | TorchRL wrapper for ease of use and reproducibility |
| **Hydra**      | Command line utility                                |

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