# ✈️ Multi-Agent Reinforcement Learning in Schiphol Airport Landside Cargo Logistics

[![Project Status: Complete](https://img.shields.io/badge/Project%20Status-Complete-green.svg)]()
[![Engine: SimPy](https://img.shields.io/badge/Simulation-SimPy-blue.svg)]()
[![Interface: PettingZoo](https://img.shields.io/badge/Interface-PettingZoo-orange.svg)]()
[![RL Engine: BenchMARL](https://img.shields.io/badge/RL%20Engine-BenchMARL-red.svg)]()
[![Results Tracking: WandB](https://img.shields.io/badge/Results%20Tracking-WandB-yellow.svg)]()

> A high-fidelity **Multi-Agent Reinforcement Learning (MARL)** environment designed to optimize truck slot bookings and landside congestion at Amsterdam Airport Schiphol Cargo Hub. This project bridges discrete-event simulation (DES) with modern deep MARL frameworks leveraging the **Multi Agent Proximal Policy Optimization (MAPPO)** algorithm.

---

## 📦 Schiphol Cargo Hub Background

The Schiphol Cargo Hub is a primary European logistical gateway with high-density freight traffic managed by several independent Ground Handling Agents (GHAs). Landside operations currently rely on a decentralized model where GHAs manage their own warehouse and dock infrastructures. A primary operational challenge is the absence of a unified Truck Slot Booking System to synchronize inbound truck traffic with available dock capacity.

During peak periods, such as the concentrated arrival patterns on Friday afternoons, high logistics volume often exceeds immediate infrastructure capacity. Without a centralized layer to align slot availability with truck arrivals, the system can experience a temporal mismatch. This leads to uneven dock utilization and increased dwell times for transport vehicles.

This project uses a discrete-event simulation to study whether Multi-Agent Reinforcement Learning (MARL) can address this coordination gap. By modeling the hub as a multi-agent environment, the research evaluates if autonomous agents can learn to synchronize scheduling and resource allocation in a decentralized way. The objective is to determine if MARL can effectively transform independent decision-making into a cohesive logistical flow to improve the overall efficiency of the Schiphol cargo ecosystem.

---

### 📖 DTP (Digital Truck Slot Planning) Rules

This project operates under the conditions that the DTP (Digital Truck Slot Planning) system is already implemented at Schiphol. The DTP project is a real project present in the official Schiphol Roadmap, but currently it is not implemented yet. For this reason, it is based on a set of `synthetic` rules that should replicate the actual goal of the project in the future. In particular:
  1. Every truck must have a confirmed DTP booking to pass the Main Gate. Enforced by Automatic Number Plate Recognition (ANPR) at the Main Gate. No exceptions.
  2. Slots are 45-minutes time windows. One slot = one license plate = one time frame = one ground handler. One truck per slot.
  3. All GHAs publish their slots on the shared DTP platform. Visible to all participants in real time. This is the only way to publish a slot.
  4. DTP platform is operated by a neutral third party (Schiphol/ACN).
  5. The Orchestrator has full authority to cancel or modify any booking unilaterally as long as the truck isn't already docked, without requiring Transporter or GHA acceptance. It can NOT remove a published slot from the DTP platform.
  6. Minimum booking lead time is double the slot duration. After that slots are frozen: no new bookings, no cancellations. this is called the "frozen window".
  7. GHAs may publish slots up to 72h before the start of the slot and before the frozen window starts.
  8. Slots are divided into a "priority window" (first 10m) and a "release window" (m11 to slot end), the dock is held still until minute 10. Then the slot is available for the next trucks at the GHA queue or the trucks sitting at TP3. The DTP platform or the Orchestrator may release them. If the original truck shows up between minute 11 and slot end:
     - If the dock is still free → the original truck is admitted directly. No rebooking required. A late-penalty is logged against the transporter account for RL feedback purposes but the truck proceeds.
     - If the dock is not free → the original truck is redirected to TP3 as a standby truck. It does not need to rebook a new slot. Its existing booking remains valid and it re-enters the queue when the Orchestrator or its own timing releases it.
  9. A truck that shows up after its own slot has expired is recorded as "no show" and the penalty is logged for RL feedback purposes. The truck is redirected at tp3. It can book another slot or the Orchestrator can send it to a GHA.
  10. The transporter can cancel a booking outside of the frozen window, only the Orchestrator can cancel a booking inside the froozen window.
  11. GHAs can not retire a published slot.
  12. GHAs have dedicated slots for import and export flows, a dock can not change its flow type.

The simulation environment ensures these rules are enforced through plain python logic.

For more information about the DTP project visit the official [Schiphol SCMP Web Page](https://www.schiphol.nl/nl/cargo/smart-cargo-mainport-program).

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
| **`road.py`** | Calculates intra-airport travel times between each node based on real-world distances. |
| **`service_time.py`** | Applies log-normal random noise to the dock service time. |
| **`schiphol_env.py`** | PettingZoo env wrapper to bridge with multiple MARL engines. |
| **`train.py`** | Main training script. |
| **`benchmarl_task`** | Bridges the PettingZoo environment and the BenchMARL task. |

---

### 🪜 Layered Structure

The project is logically organized into 3 "Layers" so that every layer acts as a base for the next one, there is no clear layer separation among modules, but the logical layers are spread across the modules:

- **Platform Layer (the foundations):** The first layer enforces the DTP (Digital Truck Slot Planning) rules through pure Python logic, handles slot publication and slot booking, and gives dock utilization info.
- **Simulation Layer (the logistics):** The second layer leverages `SimPy` to build a simulation environment that allows to iterate layer 1 through tens of thousands of episodes, in order to give to the MARL engine something to learn from.
- **MARL Layer (the brain):** The last layer is the Reinforcement Learning engine, which leverages `BenchMARL`, a state-of-the-art Python library coming from the Meta labs, in order to allow the agents to learn smart policies from the SimPy simulations.

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
├── marl/               # Reinforcement Learning adapters
└── scripts/            # Training script
```

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
| **WandB**      | Results summary and comparison                      |

---

## ⚙️ Technical Deep Dive & Module Interactions

This section outlines the low-level mechanical workflows of the project, detailing how continuous discrete-event simulation (DES) converges with a step-based Markov Decision Process (MDP) for Multi-Agent Reinforcement Learning.

---

### 1. The Core Simulation & RL Step Loop

A primary challenge in this architecture is bridging **SimPy** (which evaluates time continuously based on event-driven intervals) with **PettingZoo** (which expects discrete steps). 

To synchronize them, the environment uses a "Time-Warping Window" macro-step pattern inside `schiphol_env.py`:

```
[RL Step Start] ──> Apply Agent Actions ──> Advance SimPy Environment ──> Flush Sensor Logs ──> Compute Rewards ──> [RL Step End]
```

When `env.step(actions)` is called, the execution cascade follows these precise stages:
1. **Action Resolution:** All actions provided by active agents are translated from flat discrete integers into direct structural mutations on the `DTPPlatform` and the `DemandGenerator`.
2. **Continuous Simulation Execution:** The environment yields control back to SimPy by executing `self.sim.run(until=self.sim.now + self.step_min)`. SimPy fires and processes all queued internal micro-events (trucks traveling down roads, entering queues, docking, or completing service cargo handling) that occur within that $\Delta t$ block.
3. **Sensor Log Flushing:** Once SimPy pauses at the new time boundary, the `InfrastructureLayer` flushes its write-only step event buffer.
4. **Reward & Observation Synthesis:** The `KPITracker` consumes these telemetry records, recalculates operational state spaces, and emits step-wise rewards alongside valid action masks for the next RL cycle.

---

### 2. Module Communication Lifecycle

The modules are structurally decoupled, interacting through an explicit hierarchical cascade rather than direct circular bindings. The diagram below illustrates how data and control propagate across a single step boundary:

```
              ┌────────────────────────────────────────┐
              │           schiphol_env.py              │
              │  (Coordinates Step & Drives SimPy)     │
              └───────┬────────────────────────┬───────┘
                      │                        │
    [1] Mutate State  │                        │ [4] Fetch Logs & Collect Obs
                      ▼                        ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│         demand.py               │    │       infrastructure.py         │
│ (Spawns & Tracks Truck Status)  │    │  (Sensor Layer / ANPR Logs)     │
└───────────────┬─────────────────┘    └─────────────────▲───────────────┘
                │                                        │
      [2] Query │ Register / Update                      │ [3] Emit Timestamp
                ▼                                        │     Events
┌─────────────────────────────────┐    ┌─────────────────┴───────────────┐
│        dtp_platform.py          │    │           objects.py            │
│ (Universal DTP Booking Ledger)  │    │  (Truck, GHA Terminal Entities) │
└─────────────────────────────────┘    └─────────────────────────────────┘
```

| Inter-Module Transaction | Type | Pipeline Description |
| :--- | :--- | :--- |
| `schiphol_env` $\rightarrow$ `demand` / `dtp_platform` | **Control** | Translates agent actions into explicit bookings, modifications, or truck dispatches. |
| `demand` $\rightarrow$ `dtp_platform` | **State** | Verifies slot window availability, confirms booking conditions, and increments occupancy counts. |
| `objects` $\rightarrow$ `infrastructure` | **Telemetry** | Physical entities (`Truck` instances traveling or docking via SimPy processes) trigger write-only milestone events containing exact system timestamps. |
| `infrastructure` $\rightarrow$ `kpi_tracker` | **Data** | The environment flushes the raw log arrays to compute key metrics such as Turnaround Time (TAT) and Dock Utilization. |

---

### 3. State Management & Consistency Maintenance

The codebase maintains data consistency using a **Universal Central Ledger + Entity Cache** design to ensure fast, safe multi-agent execution:

* **The Source of Truth (`DTPPlatform`):** Houses the global registry (`self.registry`) tracking published slots and current structural allocations across all handlers. This ledger maps which truck ID is committed to which slot window.
* **The Entity Local Cache (`objects.Truck`):** Each truck carries an isolated `booked_slots` dictionary. 

**Why this matters for development:** To optimize action-masking execution speeds, agent decision masks (`_get_mask`) evaluate the `Truck` local cache to check for active bookings. If you write an action that mutates the central ledger (`dtp_platform.py`), **you must explicitly update the corresponding truck object's local dictionary** within the same step. Failure to update this cache will result in state desynchronization, leading to duplicated booking allocations or unmanifested booking validation exceptions.

---

### 4. Parallel Action Execution & Race Condition Protection

Under the PettingZoo `ParallelEnv` interface, all agents select and submit actions simultaneously at the start of a step. Because actions are resolved within a sequential Python loop inside the environment, a severe multi-agent race condition can emerge:

```
[Step Start] ──> Transporter reads Cache (No Bookings) ──> Legal Action: Book GHA_1 ──> Orchestrator reads Cache (No Bookings) ──> Legal Action: Book GHA_1

============================= Loop Resolution Sequence =============================

Transporter action executes  ──> DTP Registry Books GHA_1 ──> Truck Cache Updates

Orchestrator action executes ──> Re-executes Booking command ──> CRASH / DUPLICATION
```

To shield the state engine from these simultaneous step collisions, the environment implements strict **Mid-Step Guard Clauses** within `_apply_action`:

```python
# Guard example inside schiphol_env.py
if gha in truck.booked_slots:
    return  # Silent mitigation against simultaneous same-step collisions
```

Action Prioritization: Within the step resolution loop, the Transporter and Ground Handling Agent actions are evaluated first, while the Airport Orchestrator is resolved last.

Guard Mitigation: The moment an action loop attempts to apply a command, it runs a final sanity check against the local cache. If another agent already filled that specific allocation earlier in the exact same execution loop, the system executes a graceful return, neutralizing the conflicting command before it corrupts the central registry.

---

## 🏅 Reward Formulation & Incentive Engineering

To support the **Centralized Training, Decentralized Execution (CTDE)** framework, the environment uses a hybrid reward structure. Because different actors in the Schiphol landside cargo ecosystem have conflicting operational priorities (e.g., a GHA wants to optimize its own docks, while the Transporter wants to minimize total fleet transit delay), the environment implements an $\alpha$-blended mixed incentive model.

---

### 1. Private vs. Global Rewards

Every agent $i$ receives a synthesized step-wise reward $R_i$ composed of two separate vector components:

#### A. Private Rewards ($R_{\text{private}, i}$)
These track local, agent-centric operational metrics. They penalize or reward behaviors directly controlled by or affecting that specific entity.
* **Transporter Agent:** Penalized for structural lateness, missed slot windows ("no-shows"), and excessive truck dwell or waiting time at the TP3 holding buffer.
* **GHA Agents (dnata, klm, wfs, swiss):** Penalized for idling docks when a queue is present, and penalized for excessive local physical queue congestion.

#### B. Global Reward ($R_{\text{global}}$)
This is a shared macro-signal distributed identically to all active agents. It tracks system-wide coordination, ecosystem welfare, and absolute hub stability. In this case it is based on:
- **Net Turnaround Time per Parcel (NTTP):** The average time spent at the airport balanced by the amount of parcels in the truck.
- **Wait-to-Process Ratio (WPR):** The fraction of time that the trucks spends idling.

---

### 2. The $\alpha$-Blending Mechanics

To balance competitive individual performance with holistic network cooperation, the environment computes the final scalar reward for agent $i$ at each step using a linear combination parameter, $\alpha$:

$$R_i = \alpha \cdot R_{\text{private}, i} + (1 - \alpha) \cdot R_{\text{global}}$$

Where $\alpha \in [0, 1]$ represents the **Selfishness Coefficient**. 

```
┌────────────────────────┐
│  Private Reward (R_i)  │───┐
└────────────────────────┘   │    (x α)
                             ├───────────> [ (+) Combined Agent Reward ]
┌────────────────────────┐   │  (x 1-α)
│   Global Reward (R_g)  │───┘
└────────────────────────┘
```

The system behaves under three distinct operational paradigms depending on your configuration file (`config/params.yaml`):

| Configuration Mode | Value of $\alpha$ | Behavioral Outcome | MARL Dynamic |
| :--- | :--- | :--- | :--- |
| **Fully Cooperative** | $\alpha = 0.0$ | All agents optimize strictly for total hub throughput and global traffic reduction. Agents will actively sacrifice individual efficiency if it prevents downstream system gridlock. | Pure Coordination (Social Welfare) |
| **Purely Competitive** | $\alpha = 1.0$ | Agents act as completely selfish utility-maximizers. GHAs ignore global congestion to force dock utilization spikes; the Transporter overbooks slots to minimize its individual delay metrics. | Decentralized Non-Cooperative Game |
| **Mixed-Incentive** | $0.0 < \alpha < 1.0$ | **(Default)** Agents seek local operational excellence while maintaining systemic boundaries. A GHA optimizes local docks but avoids greedily scheduling actions that degrade global hub traffic conditions. | General-Sum Stochastic Game |

---

## 🚀 Current Milestone: [4/5] Environment Engineering
At the moment the `Platform Layer` and the `Simulation Layer` are ultimated. The project is currently at the integration phase, where the SimPy-based logistics engine needs to be successfully exposed to the Multi-Agent BenchMARL environment.

- [x] DTP Rules: Implementation of the slot booking "Freeze-time" and "Lead-time" constraints.

- [x] Core Simulation: Discrete event logic for cargo handling and truck movement.

- [x] PettingZoo Wrapper: Full implementation of observation_space, action_space, and action_masking.

- [x] BenchMARL Adapter: Full translation of PettingZoo objects into BenchMARL tasks.

- [x] Phase 2 (Next): Training MAPPO agents using the BenchMARL framework.

- [ ] Phase 3: Multi-Scenario Benchmarking (Scenario M vs. Scenario MO).

---

## 🛠️ Installation & Usage
This project is implemented on ubuntu 24.04 using `UV` as a package manager, for any implementation with on different platforms please consult official docs and make sure to check out the `pyproject.toml` file for library dependencies.

Installation steps:

  1. Clone the GitHub repository
```Bash
git clone https://github.com/SnappierCape/ds-marl-air-cargo.git
cd /your-path/ds-marl-air-cargo
git fetch origin
git pull origin main
```

  2. Setup UV Environment
```Bash
uv lock
uv sync
```

  3. Run a Sanity Check to verify that the PettingZoo wrapper is correctly communicating with the SimPy engine:
```Bash
uv run ./scripts/simulation.py --steps=2000 --orchestrator
```

  4. Run the MARL training loop:
```Bash
uv run ./scripts/train.py
```

---

## 📜 Acknowledgments

Developed for researchers and logistics engineers interested in the intersection of Operations Research and Machine Learning.

Special thanks to the Schiphol Landside Cargo community for the operational insights.