# =============================================================================
# KPI TRACKER MODULE
# =============================================================================
# DESCRIPTION:
#     Computes KPIs from the stream of SensorEvents produced by
#     InfrastructureLayer. Operates incrementally: ingests events step
#     by step rather than recomputing from scratch each time.
#
# TRACKED KPIs:
#     WPR      — Wait-to-Process Ratio: total wait / total service time
#                Perfect system = 0. Measures queue inefficiency.
#     NTTP     — Normalized Turnaround Time per Parcel: (gate_out - gate_in)
#                / n_parcels. Measures end-to-end delivery speed.
#     Peak WPR — WPR computed only during the peak window.
#
# HOW IT WORKS:
#     ingest(events) is called every MARL step with the events from that step.
#     It builds per-truck state dicts from GATE_IN to GATE_OUT, computing
#     wait and service times from the dock timestamps.
#     Reward helpers are called by schiphol_env.py to compute step rewards.
#     summary() is called at episode end for W&B logging.
# =============================================================================
from typing import Dict, List

from env.infrastructure import CheckpointID, SensorEvent
from env.dtp_platform import DTPPlatform
from env.demand import DemandGenerator

from config.config import load_params
params = load_params()

# =============================================================================
# MAIN CLASS
# =============================================================================
class KPITracker:
    def __init__(self):
        p = params["demand"]
        self._peak_start = p["peak_window"][0]
        self._peak_end = p["peak_window"][1]
        self._ghas = params["ghas"].keys()
        self.w = params["marl"]["reward_weights"]

        # Per-truck working state — built incrementally as events arrive
        # {truck_id: {"gate_in": t, "n_parcels": n, "gha_in": {gha: t}, "dock_start": {gha: t}, "dock_end": {gha: t}}}
        self._truck: Dict[str, Dict] = {}

        # Episode-level accumulators
        self._total_wait: float = 0.0
        self._total_service: float = 0.0
        self._peak_wait: float = 0.0
        self._peak_service: float = 0.0
        self._nttp_sum: float = 0.0
        self._n_completed: int = 0
        
        # Per-flow accumulators
        self._exp_wait: float = 0.0
        self._imp_wait: float = 0.0
        self._exp_service: float = 0.0
        self._imp_service: float = 0.0
        self._exp_completed: float = 0.0
        self._imp_completed: float = 0.0
        self._exp_nttp_sum: float = 0.0
        self._imp_nttp_sum: float = 0.0

        # Per-GHA dock utilization snapshots — appended each step
        # {gha: {"export": [ratio, ...], "import": [ratio, ...]}}
        self._util: Dict[str, Dict[str, List[float]]] = {
            gha: {"export": [], "import": []}
            for gha in self._ghas
        }
        
        self._prev_proc: Dict[str, int] = {gha: 0 for gha in self._ghas}
        self._prev_total_wait: float = 0.0
        self._prev_no_shows: int = 0
        self._prev_late: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # EVENT INGESTION — called every MARL step by schiphol_env.py
    # ─────────────────────────────────────────────────────────────────────────
    def ingest(self, events: List[SensorEvent]) -> None:
        """Process a batch of events from the current step. Updates per-truck state and running accumulators."""
        for e in events:
            tid = e.truck_id

            if e.checkpoint == CheckpointID.GATE_IN:
                # Start tracking this truck
                self._truck[tid] = {
                    "gate_in": e.sim_time,
                    "n_parcels": e.n_parcels or 0,
                    "flow_type": e.flow_type or "unknown",
                    "gha_in": {},
                    "dock_start": {},
                    "dock_end": {},
                }

            elif e.checkpoint == CheckpointID.GHA_IN:
                state = self._truck.get(tid)
                if state and e.gha_id:
                    state["gha_in"][e.gha_id] = e.sim_time

            elif e.checkpoint == CheckpointID.DOCK_START:
                state = self._truck.get(tid)
                if state and e.gha_id:
                    state["dock_start"][e.gha_id] = e.sim_time
                    # Wait time = from GHA entrance to dock start
                    gha_in = state["gha_in"].get(e.gha_id, e.sim_time)
                    wait = e.sim_time - gha_in
                    self._total_wait += wait
                    if self._peak_start <= e.sim_time <= self._peak_end:
                        self._peak_wait += wait
                    
                    state = self._truck.get(tid)
                    if state:
                        if state["flow_type"] == "export":
                            self._exp_wait += wait
                        elif state["flow_type"] == "import":
                            self._imp_wait += wait
                        else:
                            raise ValueError(f'"{state["flow_type"]}" is not a supported flow type.')

            elif e.checkpoint == CheckpointID.DOCK_END:
                state = self._truck.get(tid)
                if state and e.gha_id:
                    state["dock_end"][e.gha_id] = e.sim_time
                    # Service time = dock_end - dock_start
                    dock_start = state["dock_start"].get(e.gha_id, e.sim_time)
                    service = e.sim_time - dock_start
                    self._total_service += service
                    if self._peak_start <= e.sim_time <= self._peak_end:
                        self._peak_service += service

            elif e.checkpoint == CheckpointID.GATE_OUT:
                state = self._truck.get(tid)
                if state and state["n_parcels"] > 0:
                    turnaround = e.sim_time - state["gate_in"]
                    nttp_contrib = turnaround / state["n_parcels"]
                    self._nttp_sum  += nttp_contrib
                    self._n_completed += 1
                    
                    if state["flow_type"] == "export":
                        self._exp_nttp_sum += nttp_contrib
                        self._exp_completed += 1
                        
                    elif state["flow_type"] == "import":
                        self._imp_nttp_sum += nttp_contrib
                        self._imp_completed += 1
                # Remove from working state — truck is done
                self._truck.pop(tid, None)

    def snapshot_utilization(self, terminals: Dict) -> None:    # NOTE: terminal should be a GHATerminal??
        """
        Record current dock occupancy for all GHAs.
        Called once per MARL step by schiphol_env.py.
        Used to compute utilization std for load-balancing reward.
        """
        for gha, terminal in terminals.items():
            self._util[gha]["export"].append(terminal.exp_occupancy())
            self._util[gha]["import"].append(terminal.imp_occupancy())

    # ─────────────────────────────────────────────────────────────────────────
    # KPI PROPERTIES — called by schiphol_env.py for rewards and logging
    # ─────────────────────────────────────────────────────────────────────────
    def wpr(self) -> float:
        return 0.0 if self._total_service == 0 else self._total_wait / self._total_service

    def peak_wpr(self) -> float:
        return 0.0 if self._peak_service == 0 else self._peak_wait / self._peak_service

    def nttp(self) -> float:
        return 0.0 if self._n_completed == 0 else self._nttp_sum / self._n_completed
    
    def exp_wpr(self) -> float:
        return (0.0 if self._exp_service == 0 else self._exp_wait / self._exp_service)
    
    def imp_wpr(self) -> float:
        return (0.0 if self._imp_service == 0 else self._imp_wait / self._imp_service)
    
    def exp_nttp(self) -> float:
        return (0.0 if self._exp_completed == 0 else self._exp_nttp_sum / self._exp_completed)
    
    def imp_nttp(self) -> float:
        return (0.0 if self._imp_completed == 0 else self._imp_nttp_sum / self._imp_completed)
    
    def flow_type_wpr_gap(self) -> float:
        return abs(self.exp_wpr() - self.imp_wpr())

    def utilization_std(self) -> float:
        import numpy as np
        means = []
        for _, flows in self._util.items():
            all_snaps = flows["export"] + flows["import"]
            if all_snaps:
                means.append(sum(all_snaps) / len(all_snaps))
        return float(np.std(means)) if len(means) > 1 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # REWARD HELPERS — called by schiphol_env.py every step
    # ─────────────────────────────────────────────────────────────────────────
    def global_reward(self) -> float:
        w = self.w
        return -(w["wpr_global"] * self.wpr() + w["util_std"] * self.utilization_std())

    def transporter_reward(self, dtp: DTPPlatform, demand: DemandGenerator) -> float:
        w = self.w
        
        # Calculate current step wait
        delta_wait = self._total_wait - self._prev_total_wait
        
        # No shows and late arrivals
        current_no_shows = sum(dtp.no_shows.values())
        delta_no_shows = current_no_shows - self._prev_no_shows
        current_late = sum(dtp.late_arrivals.values())
        delta_late = current_late - self._prev_late
        
        # Update trackers for the next step
        self._prev_total_wait = self._total_wait
        self._prev_no_shows = current_no_shows
        self._prev_late = current_late
        
        return -(
            w["wait_per_min"] * delta_wait +
            w["no_show"] * delta_no_shows +
            w["missed_slot"] * delta_late +
            w["pending_trucks"] * len(demand.pending_trucks)
        )

    def gha_reward(self, gha: str, terminal) -> float:    # NOTE: terminal should be a GHATerminal??
        w = self.w
        
        util = (terminal.exp_occupancy() + terminal.imp_occupancy()) / 2
        
        q = terminal.exp_queue_norm() + terminal.imp_queue_norm()
        
        total_proc = (
            terminal.stats["export"]["processed"] +
            terminal.stats["import"]["processed"]
        )
        delta_proc = total_proc - self._prev_proc[gha]
        self._prev_proc[gha] = total_proc
        
        return (
            w["dock_util"] * util +
            w["parcel_on_time"] * delta_proc -
            w["queue_per_step"] * q
        )

    # ─────────────────────────────────────────────────────────────────────────
    # EPISODE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    def summary(self) -> Dict:
        return {
            "wpr": self.wpr(),
            "peak_wpr": self.peak_wpr(),
            "exp_wpr": self.exp_wpr(),
            "imp_wpr": self.imp_wpr(),
            "flow_type_wpr_gap": self.flow_type_wpr_gap(),
            "nttp": self.nttp(),
            "exp_nttp": self.exp_nttp(),
            "imp_nttp": self.imp_nttp(),
            "util_std": self.utilization_std(),
            "n_completed": self._n_completed,
            "exp_completed": self._exp_completed,
            "imp_completed": self._imp_completed,
            "global_reward": self.global_reward(),
        }