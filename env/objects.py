# =============================================================================
# SIMULATION OBJECTS MODULE
# =============================================================================
# DESCRIPTION:
#     Defines the three core SimPy entities: Truck, GHATerminal, TP3Buffer.
#     Owns all physical logistics that takes simulated time.
#     Knows nothing about MARL, rewards, or routing policies.
# =============================================================================
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import simpy

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
import config.config
from env.infrastructure import InfrastructureLayer
from env.dtp_platform import DTPPlatform
from env.service_time import ServiceTimeModel

params = config.load_params()

# =============================================================================
# TRUCK
# =============================================================================
@dataclass
class Truck:
    # Immutable identity
    truck_id: str
    flow_type: str
    origin_type: str
    manifest: List[Dict]    # [{"gha": "dnata", "parcels": 5}, ...]

    # Mutable journey state
    status: str = "in_transit"
    current_node: str = "origin"
    booked_slots: Dict = field(default_factory=dict)    # {gha: slot_start}
    timestamps: Dict = field(default_factory=dict)    # {event_name: sim_time}
    stops_remaining: List[Dict]= field(default_factory=list)

    # Status constants
    STATUS_IN_TRANSIT = "in_transit"
    STATUS_AT_TP3 = "at_tp3"
    STATUS_QUEUED = "queued"
    STATUS_DOCKED = "docked"
    STATUS_DEPARTED = "departed"

    def __post_init__(self):
        self.stops_remaining = list(self.manifest)

    def total_parcels(self) -> int:
        return sum(s["parcels"] for s in self.manifest)

    def parcels_for(self, gha: str) -> int:
        for stop in self.manifest:
            if stop["gha"] == gha:
                return stop["parcels"]
        return 0

    def next_slot(self) -> Optional[int]:
        """Earliest booked slot among remaining stops."""
        remaining = {s["gha"] for s in self.stops_remaining}
        active = {g: t for g, t in self.booked_slots.items() if g in remaining}
        return min(active.values()) if active else None

    def next_stop(self) -> Optional[Dict]:
        return self.stops_remaining[0] if self.stops_remaining else None

    def complete_stop(self, gha: str):
        self.stops_remaining = [s for s in self.stops_remaining if s["gha"] != gha]

# =============================================================================
# GHA TERMINAL
# =============================================================================
class GHATerminal:
    def __init__(
        self,
        env: simpy.Environment,
        gha: str,
        svc_tm: ServiceTimeModel,
        infra: InfrastructureLayer,
        cfg: Dict = params,
    ):
        self.env = env
        self.gha = gha
        self.svc_tm = svc_tm
        self.infra = infra

        dock_cfg = cfg["ghas"][gha]
        self.n_exp = dock_cfg["export"]
        self.n_imp = dock_cfg["import"]

        self.docks_exp = simpy.Resource(env, capacity=self.n_exp)
        self.docks_imp = simpy.Resource(env, capacity=self.n_imp)

        self.queue_exp: List[Truck] = []
        self.queue_imp: List[Truck] = []

        # KPI accumulators — keys match flow_type strings exactly
        self.stats = {
            "export": {"processed": 0, "tot_wait": 0.0, "tot_serv": 0.0},
            "import": {"processed": 0, "tot_wait": 0.0, "tot_serv": 0.0},
        }

    # ── Routing helpers ──────────────────────────────────────────────────────
    def _dock_pool(self, flow_type: str) -> simpy.Resource:
        return self.docks_exp if flow_type == "export" else self.docks_imp

    def _queue(self, flow_type: str) -> List[Truck]:
        return self.queue_exp if flow_type == "export" else self.queue_imp

    # ── Core service process ─────────────────────────────────────────────────
    def process_truck(self, truck: Truck, dtp: DTPPlatform):
        """
        SimPy generator: truck arrives → phase check → queue → dock → depart.
        Returns (via truck.status = AT_TP3) when truck must be redirected.
        """
        arrival_time = self.env.now
        pool = self._dock_pool(truck.flow_type)
        queue = self._queue(truck.flow_type)
        slot_start = truck.booked_slots.get(self.gha)

        # ANPR fires at GHA entrance
        self.infra.gha_in(arrival_time, truck, self.gha)

        phase = dtp.get_slot_phase(
            slot_start,
            arrival_time,
            dock_is_free=pool.count < pool.capacity
        )

        # ── Phase routing ────────────────────────────────────────────────────
        if phase == "early":
            # Wait until slot window opens, then proceed as priority
            if slot_start is not None:
                yield self.env.timeout(max(0.0, slot_start - self.env.now))

        elif phase == "release":
            dtp.record_late(truck.truck_id)
            # Dock is free — truck proceeds, small penalty logged

        elif phase in ("release_dock_taken", "no_show"):
            if phase == "no_show":
                dtp.record_no_show(self.gha, slot_start, truck.truck_id)
            else:
                dtp.record_late(truck.truck_id)
            truck.status = Truck.STATUS_AT_TP3
            return    # caller (demand.py) handles TP3 redirect

        # ── Queue and dock ───────────────────────────────────────────────────
        truck.status = Truck.STATUS_QUEUED
        queue.append(truck)
        queue_start = self.env.now    # measure wait from here, not from arrival

        with pool.request() as req:
            yield req    # FIFO — SimPy guarantees priority order naturally

            queue_time = self.env.now - queue_start
            service_time = self.svc_tm.sample(truck.flow_type)

            queue.remove(truck)
            truck.status = Truck.STATUS_DOCKED
            dock_id = pool.count    # proxy; replace when real mapping available

            self.infra.dock_start(self.env.now, truck, self.gha, dock_id)
            if slot_start is not None:
                dtp.mark_docked(self.gha, slot_start, truck.truck_id)

            yield self.env.timeout(service_time)

            self.infra.dock_end(self.env.now, truck, self.gha, dock_id)
            if slot_start is not None:
                dtp.mark_closed(self.gha, slot_start, truck.truck_id)

            truck.complete_stop(self.gha)
            truck.status = Truck.STATUS_IN_TRANSIT

            self.stats[truck.flow_type]["processed"] += 1
            self.stats[truck.flow_type]["tot_wait"]  += queue_time
            self.stats[truck.flow_type]["tot_serv"]  += service_time

    # ── Release window watcher ───────────────────────────────────────────────
    def release_window_watcher(self, slot_start: int, dtp: DTPPlatform, tp3: "TP3Buffer"):
        """
        SimPy process: one instance per published slot.
        Fires at minute 10 of the slot. If the booked truck hasn't appeared,
        signals TP3 that a standby truck may fill the dock.
        """
        yield self.env.timeout(
            max(0.0, slot_start + dtp.priority_window - self.env.now)
        )
        if dtp.release_to_standby(self.gha, slot_start):
            tp3.signal_standby_opportunity(self.gha, slot_start, self.env.now)

    # ── Observation helpers (called by schiphol_env.py) ──────────────────────
    def exp_occupancy(self) -> float:
        return self.docks_exp.count / self.n_exp if self.n_exp > 0 else 0.0

    def imp_occupancy(self) -> float:
        return self.docks_imp.count / self.n_imp if self.n_imp > 0 else 0.0

    def exp_queue_norm(self) -> float:
        max_q = params["gha"][self.gha]["export"]
        return min(len(self.queue_exp) / max_q, 1.0)

    def imp_queue_norm(self) -> float:
        max_q = params["gha"][self.gha]["import"]
        return min(len(self.queue_imp) / max_q, 1.0)

    def upcoming_bookings_norm(self, dtp: DTPPlatform, horizon: int) -> float:
        """Fraction of total docks committed in the next `horizon` minutes."""
        now  = self.env.now
        total = self.n_exp + self.n_imp
        committed = sum(
            1
            for slot_start, entries in dtp.registry.get(self.gha, {}).items()
            if 0 <= slot_start - now <= horizon
            for entry in entries
            if entry["phase"] in ("booked", "docked")
        )
        return min(committed / total, 1.0) if total > 0 else 0.0


# =============================================================================
# TP3 BUFFER
# =============================================================================
class TP3Buffer:
    CAPACITY = params["tp3"]["capacity"]

    def __init__(self, env: simpy.Environment, infra: InfrastructureLayer):
        self.env = env
        self.infra = infra

        self.slots = simpy.Resource(env, capacity=self.CAPACITY)
        self._parked: List[tuple] = []    # [(Truck, simpy_request), ...]
        self.queue_overflow: List[Truck] = []    # trucks waiting because TP3 is full
        self.standby_opportunities: List[Dict] = []

    # ── Entry ─────────────────────────────────────────────────────────────────
    def enter(self, truck: Truck):
        """SimPy generator. If TP3 is full, truck waits in overflow queue."""
        req = self.slots.request()
        result = yield req | self.env.timeout(0)

        if req not in result:
            # TP3 full — wait on approach road
            self.queue_overflow.append(truck)
            yield req
            self.queue_overflow.remove(truck)

        self._parked.append((truck, req))
        truck.status = Truck.STATUS_AT_TP3
        self.infra.tp3_in(self.env.now, truck)

    # ── Release ───────────────────────────────────────────────────────────────
    def release(self, truck_id: str) -> Optional[Truck]:
        """Targeted release by truck_id. Called by Orchestrator or Transporter."""
        for i, (truck, req) in enumerate(self._parked):
            if truck.truck_id == truck_id:
                self._parked.pop(i)
                self.slots.release(req)
                self.infra.tp3_out(self.env.now, truck)
                return truck
        return None

    def release_next(self, gha: str) -> Optional[Truck]:
        """FCFS release: first parked truck with a booking for gha."""
        for i, (truck, req) in enumerate(self._parked):
            if gha in truck.booked_slots:
                self._parked.pop(i)
                self.slots.release(req)
                self.infra.tp3_out(self.env.now, truck)
                return truck
        return None

    # ── Standby signalling ────────────────────────────────────────────────────
    def signal_standby_opportunity(self, gha: str, slot_start: int, signal_time: float) -> List[Dict]:
        """Called by GHATerminal.release_window_watcher when a slot enters release window."""
        self.standby_opportunities.append({
            "gha": gha, "slot_start": slot_start,
            "signal_time": signal_time, "consumed": False
        })

    def get_pending_signals(self) -> List[Dict]:
        return [s for s in self.standby_opportunities if not s["consumed"]]

    # ── Observation helpers ───────────────────────────────────────────────────
    def occupancy_ratio(self) -> float:
        return self.slots.count / self.CAPACITY

    def n_parked(self) -> int:
        return self.slots.count

    def n_overflow(self) -> int:
        return len(self.queue_overflow)

    def parked_by_flow_type(self, flow_type: str) -> int:
        return sum(1 for truck, _ in self._parked if truck.flow_type == flow_type)

    def get_parked_trucks(self) -> List[Truck]:
        return [truck for truck, _ in self._parked]