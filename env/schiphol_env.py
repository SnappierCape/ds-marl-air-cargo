# =============================================================================
# PETTINGZOO ENVIRONMENT WRAPPER
# =============================================================================
# DESCRIPTION:
#     Bridges the SimPy simulation and the BenchMARL training framework.
#     Implements the PettingZoo ParallelEnv API.
#
# RESPONSIBILITIES:
#     reset()  → instantiate fresh simulation objects, return first observations
#     step()   → apply actions, advance SimPy, return (obs, rewards, dones, infos)
# =============================================================================
from typing import Dict, List, Tuple

import simpy
import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv

from env.objects import GHATerminal, TP3Buffer
from env.dtp_platform import DTPPlatform
from env.infrastructure import InfrastructureLayer
from env.service_time import ServiceTimeModel
from env.road import RoadNetwork
from env.demand import DemandGenerator
from env.kpi_tracker import KPITracker

from config.config import load_params
params = load_params()

# ── Fixed constants ──────────────────────────────────────────────────────────
N_SLOT_ACTIONS = 20    # max bookable slots visible to Transporter at any step
N_TP3_ACTIONS = 10    # max trucks the Orchestrator can release per step
N_PENDING_TRUCKS = 10    # max trucks the Transporter can process per step
GHA_IDS = list(params["ghas"].keys())
N_GHAS = len(GHA_IDS)
N_BOOK_ACTIONS = N_PENDING_TRUCKS * N_GHAS
N_DISPATCH_ACTIONS = N_PENDING_TRUCKS
TRANSPORTER_ACTION_DIM = 1 + N_BOOK_ACTIONS + N_DISPATCH_ACTIONS

# =============================================================================
# MAIN CLASS
# =============================================================================
class SchipholCargoEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv: all agents act simultaneously every step.
    One step = step_min minutes of simulated time (from params).
    """
    metadata = {"name": "schiphol_cargo_v0"}

    def __init__(self, with_orchestrator: bool = False):
        super().__init__()
        self.with_orchestrator = with_orchestrator
        self.step_min = params["marl"]["step_min"]
        self.alpha = params["marl"]["alpha"]

        self.possible_agents = ["transporter"] + [f"{g}" for g in GHA_IDS]
        if with_orchestrator:
            self.possible_agents.append("orchestrator")

    # ─────────────────────────────────────────────────────────────────────────
    # SPACE DECLARATIONS
    # ─────────────────────────────────────────────────────────────────────────
    def observation_space(self, agent: str) -> gym.Space:
        return gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_dim(agent),), dtype=np.float32
        )

    def action_space(self, agent: str) -> gym.Space:
        return gym.spaces.Discrete(self._action_dim(agent))

    # ─────────────────────────────────────────────────────────────────────────
    # RESET — called at the start of every episode
    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None) -> Tuple[Dict]:
        self.sim = simpy.Environment()
        self.infra = InfrastructureLayer()
        self.svc_tm = ServiceTimeModel(params)
        self.road = RoadNetwork(params["road"])
        self.dtp = DTPPlatform(self.sim)
        self.tp3 = TP3Buffer(self.sim, self.infra)
        self.kpi = KPITracker()
        self.terminals: Dict[str, GHATerminal] = {
            gha: GHATerminal(self.sim, gha, self.svc_tm, self.infra)
            for gha in GHA_IDS
        }

        # Pre-publish slots for the first 72h so trucks can book on arrival
        self._prepopulate_slots()

        # Start the demand generator — runs as a background SimPy process
        self.demand = DemandGenerator(
            self.sim, self.dtp, self.terminals,
            self.tp3, self.infra, self.road
        )
        self.sim.process(self.demand.run())

        # Reset agent list (PettingZoo convention)
        self.agents = self.possible_agents[:]

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {"action_mask": self._get_mask(a)} for a in self.agents}
        return obs, infos

    # ─────────────────────────────────────────────────────────────────────────
    # MARL STEP
    # ─────────────────────────────────────────────────────────────────────────
    def step(self, actions: Dict[str, int]) -> Tuple[Dict]:
        # 1. Apply each agent's action to the DTP platform
        for agent, action in actions.items():
            self._apply_action(agent, action)

        # 2. Advance SimPy by one step
        self.sim.run(until=self.sim.now + self.step_min)

        # 3. Ingest new sensor events into KPI tracker
        new_events = self.infra.flush_step_buffer()
        self.kpi.ingest(new_events)

        # 4. Snapshot dock utilization for load-balancing reward
        self.kpi.snapshot_utilization(self.terminals)

        # 5. Compute global reward (shared component for all agents)
        r_global = self.kpi.global_reward()

        # 6. Collect outputs
        obs = {a: self._get_obs(a) for a in self.agents}
        rewards = {a: self._get_reward(a, r_global) for a in self.agents}
        term = {a: False for a in self.agents}
        trunc = {a: False for a in self.agents}
        infos = {a: {"action_mask": self._get_mask(a)} for a in self.agents}

        # PettingZoo convention: empty agents list signals episode end.
        # Episode is controlled externally by env.run(until=T) in train.py.
        return obs, rewards, term, trunc, infos

    # ─────────────────────────────────────────────────────────────────────────
    # ACTIONS
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_action(self, agent: str, action: int) -> None:
        """Translate integer action into a DTP method call."""
        if action == 0:
            return    # no_op — valid for every agent

        # ── Transporter ──────────────────────────────────────────────────────
        if agent == "transporter":
            if 1 <= action <= N_BOOK_ACTIONS:
                # Decode: which truck, which GHA
                idx = action - 1
                truck_idx = idx // N_GHAS
                gha_idx = idx %  N_GHAS
                gha = GHA_IDS[gha_idx]
                pending = self.demand.pending_trucks
                if truck_idx < len(pending):
                    truck = pending[truck_idx]
                    self.demand.book_one_slot(truck.truck_id, gha, truck.flow_type)

            elif N_BOOK_ACTIONS + 1 <= action <= N_BOOK_ACTIONS + N_DISPATCH_ACTIONS:
                # Decode: which truck to dispatch
                truck_idx = action - N_BOOK_ACTIONS - 1
                pending = self.demand.pending_trucks
                if truck_idx < len(pending):
                    truck = pending[truck_idx]
                    self.demand.dispatch_truck(truck.truck_id)

        # ── GHA ──────────────────────────────────────────────────────────────
        elif agent in GHA_IDS:
            # Action 1: publish next slot window
            # Action 2: publish the slot window after that
            next_windows = self._next_publishable_windows()
            idx = action - 1
            if idx < len(next_windows):
                for flow_type in ("import", "export"):
                    self.dtp.publish_slot(agent, next_windows[idx], flow_type)

        # ── Orchestrator ─────────────────────────────────────────────────────
        elif agent == "orchestrator":
            # Actions encode (truck_index, gha_index) pairs
            # action = truck_i * N_GHAS + gha_j + 1
            action_idx = action - 1
            truck_idx = action_idx // N_GHAS
            gha_idx = action_idx %  N_GHAS
            gha = GHA_IDS[gha_idx]

            parked = self.tp3.get_parked_trucks()
            if truck_idx < len(parked):
                truck = parked[truck_idx]
                self.tp3.release(truck.truck_id)
                # If no booking exists for this GHA, book the next available
                if gha not in truck.booked_slots:
                    slots = self.dtp.get_available_slots(gha, truck.flow_type, horizon=120)
                    if slots:
                        self.dtp.orch_book_slot(gha, slots[0], truck.truck_id, truck.flow_type)
                        truck.booked_slots[gha] = slots[0]

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVATION SPACE
    # ─────────────────────────────────────────────────────────────────────────
    def _get_obs(self, agent: str) -> np.ndarray:
        """Build the observation vector for one agent."""
        obs = np.zeros(self._obs_dim(agent), dtype=np.float32)
        tod = (self.sim.now % 1440) / 1440    # time of day normalised

        # ── Transporter ──────────────────────────────────────────────────────
        if agent == "transporter":
            i = 0
            obs[i] = self.tp3.occupancy_ratio(); i += 1
            obs[i] = min(self.tp3.n_overflow() / 20, 1.0); i += 1
            obs[i] = (np.sin(2 * np.pi * tod) + 1) / 2; i += 1
            obs[i] = (np.cos(2 * np.pi * tod) + 1) / 2; i += 1
            
            # Available slot count per GHA (normalised by total docks)
            for gha in GHA_IDS:
                n_slots_exp = len(self.dtp.get_available_slots(gha, "export", horizon=120))
                obs[i] = min(n_slots_exp / params["ghas"][gha]["export"], 1.0)
                i += 1
                
                n_slots_imp = len(self.dtp.get_available_slots(gha, "import", horizon=120))
                obs[i] = min(n_slots_imp / params["ghas"][gha]["import"], 1.0)
                i += 1
                
            # Export and import occupancy per GHA
            for gha in GHA_IDS:
                obs[i] = self.terminals[gha].exp_occupancy(); i += 1
                obs[i] = self.terminals[gha].imp_occupancy(); i += 1

            # Per-pending-truck features — gives agent context about its fleet
            for t_idx in range(N_PENDING_TRUCKS):
                pending = self.demand.pending_trucks
                if t_idx < len(pending):
                    truck = pending[t_idx]
                    n_needed = len(truck.manifest)
                    n_booked = len(truck.booked_slots)
                    obs[i] = 1.0 if truck.flow_type == "export" else 0.0; i += 1
                    obs[i] = n_booked / max(n_needed, 1); i += 1
                    obs[i] = min(n_needed / 4, 1.0); i += 1
                    obs[i] = min(self.sim.now / 1440, 1.0); i += 1
                else:
                    i += 4    # pad with zeros

        # ── GHA ───────────────────────────────────────────────────────────────
        elif agent in GHA_IDS:
            t = self.terminals[agent]
            i = 0
            obs[i] = t.exp_occupancy(); i += 1
            obs[i] = t.imp_occupancy(); i += 1
            obs[i] = t.exp_queue_norm(); i += 1
            obs[i] = t.imp_queue_norm(); i += 1
            obs[i] = t.upcoming_bookings_norm(self.dtp, horizon=45); i += 1
            obs[i] = t.upcoming_bookings_norm(self.dtp, horizon=90); i += 1
            obs[i] = (np.sin(2 * np.pi * tod) + 1) / 2; i += 1
            obs[i] = (np.cos(2 * np.pi * tod) + 1) / 2; i += 1
            obs[i] = self.tp3.occupancy_ratio(); i += 1
            # Other GHAs' occupancies (context for load balancing)
            for other in GHA_IDS:
                if other != agent:
                    obs[i] = self.terminals[other].exp_occupancy(); i += 1
                    obs[i] = self.terminals[other].imp_occupancy(); i += 1

        # ── Orchestrator ──────────────────────────────────────────────────────
        elif agent == "orchestrator":
            # Full global state: concatenation of all GHA obs + TP3 + time
            i = 0
            obs[i] = self.tp3.occupancy_ratio(); i += 1
            obs[i] = min(self.tp3.n_overflow() / 20, 1.0); i += 1
            obs[i] = (np.sin(2 * np.pi * tod) + 1) / 2; i += 1
            obs[i] = (np.cos(2 * np.pi * tod) + 1) / 2; i += 1
            for gha in GHA_IDS:
                t = self.terminals[gha]
                obs[i] = t.exp_occupancy(); i += 1
                obs[i] = t.imp_occupancy(); i += 1
                obs[i] = t.exp_queue_norm(); i += 1
                obs[i] = t.imp_queue_norm(); i += 1
                obs[i] = t.upcoming_bookings_norm(self.dtp, horizon=45); i += 1

        return obs

    # ─────────────────────────────────────────────────────────────────────────
    # MIX REWARDS
    # ─────────────────────────────────────────────────────────────────────────
    def _get_reward(self, agent: str, r_global: float) -> float:
        """Mix private and global reward by alpha."""
        if agent == "orchestrator":
            return r_global    # Orchestrator has no private incentive

        if agent == "transporter":
            r_private = self.kpi.transporter_reward(self.dtp, self.demand)
        else:
            r_private = self.kpi.gha_reward(agent, self.terminals[agent])

        return (1 - self.alpha) * r_private + self.alpha * r_global

    # ─────────────────────────────────────────────────────────────────────────
    # ILLEGAL ACTION MASKING
    # ─────────────────────────────────────────────────────────────────────────
    def _get_mask(self, agent: str) -> np.ndarray:
        """1 = valid action, 0 = masked. Action 0 (no_op) is always valid."""
        dim = self._action_dim(agent)
        mask = np.zeros(dim, dtype=np.int8)
        mask[0] = 1

        if agent == "transporter":
            pending = self.demand.pending_trucks

            # Book actions: valid if truck needs this GHA and has no booking there yet
            for t_idx, truck in enumerate(pending[:N_PENDING_TRUCKS]):
                needed = {s["gha"] for s in truck.stops_remaining}
                for g_idx, gha in enumerate(GHA_IDS):
                    if gha in needed and gha not in truck.booked_slots:
                        if self.dtp.get_available_slots(gha, truck.flow_type, horizon=480):
                            action = t_idx * N_GHAS + g_idx + 1
                            if action < dim:
                                mask[action] = 1

            # Dispatch actions: valid only if ALL stops for this truck are booked
            for t_idx, truck in enumerate(pending[:N_PENDING_TRUCKS]):
                needed = {s["gha"] for s in truck.stops_remaining}
                booked = set(truck.booked_slots.keys())
                if needed.issubset(booked):
                    action = N_BOOK_ACTIONS + t_idx + 1
                    if action < dim:
                        mask[action] = 1

        elif agent in GHA_IDS:
            windows = self._next_publishable_windows()
            for i, _ in enumerate(windows):
                if i + 1 < dim:
                    mask[i + 1] = 1

        elif agent == "orchestrator":
            parked = self.tp3.get_parked_trucks()
            for t_idx, truck in enumerate(parked[:N_TP3_ACTIONS]):
                for g_idx, gha in enumerate(GHA_IDS):
                    # condition 1: truck actually needs to visit this GHA
                    needs_gha = any(s["gha"] == gha for s in truck.stops_remaining)
                    if not needs_gha:
                        continue
                    # condition 2: at least one slot of the truck's flow_type exists
                    has_slots = bool(
                        self.dtp.get_available_slots(gha, truck.flow_type, horizon=120)
                    )
                    if not has_slots:
                        continue
                    action = t_idx * N_GHAS + g_idx + 1
                    if action < dim:
                        mask[action] = 1

        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # SPACE DIMANSION HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _obs_dim(self, agent: str) -> int:
        if agent == "transporter":
            # 4 (tp3+time) + 2*N_GHAS (slot counts imp+exp) + 2*N_GHAS (occupancies imp+exp) + N_PENDING_TRUCKS * 4 features
            return 4 + 2 * N_GHAS + 2 * N_GHAS + 4 * N_PENDING_TRUCKS
        elif agent in GHA_IDS:
            # 9 own + 2*(N_GHAS-1) others
            return 9 + 2 * (N_GHAS - 1)
        elif agent == "orchestrator":
            # 4 system + 5*N_GHAS per GHA
            return 4 + 5 * N_GHAS
        raise ValueError(f'Agent {agent} is unknown, please input a known agent')

    def _action_dim(self, agent: str) -> int:
        if agent == "transporter":
            return TRANSPORTER_ACTION_DIM
        elif agent in GHA_IDS:
            return 3    # no_op, publish_next, publish_one_after
        elif agent == "orchestrator":
            return N_TP3_ACTIONS * N_GHAS + 1
        raise ValueError(f'Agent {agent} is unknown, please input a known agent')

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _prepopulate_slots(self) -> None:
        """
        Publish slots for all GHAs for the next 72h at episode start.
        One slot per dock per 45-min window.
        """
        now = self.sim.now
        freeze = params["dtp_rules"]["freeze_time"]
        slot_dur = params["dtp_rules"]["slot_duration"]

        for gha in GHA_IDS:
            for flow_type in ("import", "export"):
                n_docks = params["ghas"][gha][flow_type]
                for i in range(n_docks):
                    t = now + freeze + 1 + (i * (slot_dur / 10))
                    self.dtp.publish_slot(gha, t, flow_type)

    def _next_publishable_windows(self) -> List[int]:
        """Next two slot windows a GHA can publish right now."""
        now = int(self.sim.now)
        slot_dur = params["dtp_rules"]["slot_duration"]
        freeze = params["dtp_rules"]["freeze_time"]
        start = now + freeze + 1
        # Round up to next clean slot boundary
        start = start + (slot_dur - start % slot_dur) % slot_dur
        return [start, start + slot_dur]