# =============================================================================
# SIMULATION MODULE
# =============================================================================
# DESCRIPTION:
#     Core SimPy objects: Truck dataclass, GHATerminal (with import/export
#     dock split), and TP3Buffer (140-slot constrained buffer).
#     This module owns all physical logistics logic.
#     It knows nothing about MARL, rewards, or policies.
# =============================================================================
import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .infrastructure import InfrastructureLayer
    from .dtp_platform import DTPPlatform
    from .service_time import ServiceTimeModel


# =============================================================================
# TRUCK
# =============================================================================
@dataclass
class Truck:
    """
    Represents a single truck in the simulation.
    Carries a manifest of stops (one per GHA it needs to visit).
    Stateless from the MARL perspective — pure simulation data.
    """
    truck_id:       str
    flow_type:      str                     # "export" | "import"
    origin_type:    str                     # "rfs" | "random_nl" | "forwarder_hub" | "tp3"
    manifest:       List[Dict]              # [{"gha": "dnata", "n_parcels": 8}, ...]
    departure_time: float                   # sim minutes from origin

    # Mutable state — updated as the truck moves through the system
    status:         str = "in_transit"      # see STATUS_* constants below
    current_node:   str = "origin"
    booked_slots:   Dict = field(default_factory=dict)   # {gha_id: slot_start}
    timestamps:     Dict = field(default_factory=dict)   # all ANPR/sensor events
    stops_remaining: List[Dict] = field(default_factory=list)  # copy of manifest, consumed

    # Status constants (used for observation vector encoding)
    STATUS_IN_TRANSIT  = "in_transit"
    STATUS_AT_TP3      = "at_tp3"
    STATUS_QUEUED      = "queued_at_gha"
    STATUS_DOCKED      = "docked"
    STATUS_DEPARTED    = "departed"

    def __post_init__(self):
        # stops_remaining is a working copy consumed as deliveries complete
        self.stops_remaining = list(self.manifest)

    def total_parcels(self) -> int:
        return sum(s["n_parcels"] for s in self.manifest)

    def parcels_for(self, gha_id: str) -> int:
        for stop in self.manifest:
            if stop["gha"] == gha_id:
                return stop["n_parcels"]
        return 0

    def next_slot_window(self) -> Optional[float]:
        """Returns the earliest booked slot across all remaining GHA stops."""
        if not self.booked_slots:
            return None
        remaining_ghas = {s["gha"] for s in self.stops_remaining}
        relevant = {g: t for g, t in self.booked_slots.items() if g in remaining_ghas}
        return min(relevant.values()) if relevant else None

    def next_stop(self) -> Optional[Dict]:
        """Returns the next unvisited GHA stop."""
        return self.stops_remaining[0] if self.stops_remaining else None

    def complete_stop(self, gha_id: str):
        """Mark a GHA stop as completed."""
        self.stops_remaining = [s for s in self.stops_remaining if s["gha"] != gha_id]


# =============================================================================
# GHA TERMINAL
# =============================================================================
class GHATerminal:
    """
    Models one Ground Handling Agent as a pair of SimPy Resources.
    Export docks and import docks are independent — a truck's flow_type
    determines which pool it enters.

    The terminal knows nothing about agent policies. It provides:
      - SimPy processes for truck service
      - Observation helper methods for the MARL layer
    """

    def __init__(self, env: simpy.Environment, gha_id: str,
                 cfg: Dict, svc: "ServiceTimeModel",
                 infra: "InfrastructureLayer"):
        self.env    = env
        self.gha_id = gha_id
        self.svc    = svc
        self.infra  = infra

        # Dock counts from config
        dock_cfg = cfg["gha_docks"][gha_id]
        self.n_export = dock_cfg["export"]
        self.n_import = dock_cfg["import"]

        # Two independent SimPy Resources — core of the import/export split
        self.docks_export = simpy.Resource(env, capacity=self.n_export)
        self.docks_import = simpy.Resource(env, capacity=self.n_import)

        # Waiting queues (trucks that have arrived but dock is busy)
        self.queue_export: List[Truck] = []
        self.queue_import: List[Truck] = []

        # Accumulators for KPI computation (separated by flow)
        self.stats = {
            "export": {"processed": 0, "total_wait": 0.0, "total_service": 0.0},
            "import": {"processed": 0, "total_wait": 0.0, "total_service": 0.0},
        }

    # ── Resource routing ──────────────────────────────────────────────────────

    def _pool(self, flow_type: str) -> simpy.Resource:
        return self.docks_export if flow_type == "export" else self.docks_import

    def _queue(self, flow_type: str) -> List[Truck]:
        return self.queue_export if flow_type == "export" else self.queue_import

    # ── Core SimPy process ────────────────────────────────────────────────────

    def process_truck(self, truck: Truck, dtp: "DTPPlatform"):
        """
        SimPy generator: truck arrives at GHA → waits for dock → served → departs.
        Implements R8 slot phase logic via DTPPlatform.
        """
        arrival_time = self.env.now
        flow         = truck.flow_type
        pool         = self._pool(flow)
        queue        = self._queue(flow)

        # ANPR camera at GHA entrance fires
        self.infra.gha_in(arrival_time, truck, self.gha_id)

        # Check slot phase (R8)
        slot_start = truck.booked_slots.get(self.gha_id)
        is_dock_free = pool.count < pool.capacity
        phase = dtp.get_slot_phase(self.gha_id, slot_start, arrival_time, is_dock_free)

        if phase == "early":
            # Should have been held at TP3 — this is a routing error, log and wait
            # Wait until the priority window opens (-15 min before slot)
            if slot_start is not None:
                wait = max(0, (slot_start - 15) - self.env.now)
                yield self.env.timeout(wait)

        elif phase == "late_dock_taken":
            # R8: arrived in release window, dock already taken by standby truck
            # Truck goes back to TP3 with original booking still valid
            truck.status = Truck.STATUS_AT_TP3
            return  # caller (demand generator / truck journey) handles TP3 redirect

        elif phase == "no_show":
            # Truck showed up after the window — record and proceed anyway
            # (they still need to deliver their cargo — just penalized)
            dtp.record_no_show(truck.truck_id, self.gha_id, slot_start)

        # Truck joins the queue for a dock
        truck.status = Truck.STATUS_QUEUED
        queue.append(truck)

        # Priority window check: if in priority window, we do NOT allow
        # standby trucks to jump ahead. The pool.request() handles this
        # naturally via FIFO ordering in SimPy.

        with pool.request() as req:
            yield req   # blocks until a dock of the correct type is free

            wait_time = self.env.now - arrival_time
            if truck in queue:
                queue.remove(truck)

            truck.status = Truck.STATUS_DOCKED

            # Dock sensor fires: truck backs in
            dock_id = pool.count  # proxy — replace with real mapping when available
            self.infra.dock_start(self.env.now, truck, self.gha_id, dock_id)

            # Mark slot as in use (prevents standby trucks from taking it)
            if slot_start is not None:
                dtp.mark_docking(self.gha_id, slot_start)

            # Sample service time (config-driven, decoupled from this method)
            service_time = self.svc.sample(flow)
            yield self.env.timeout(service_time)

            # Dock sensor fires: truck pulls out
            # SimPy releases the resource automatically at end of 'with' block
            self.infra.dock_end(self.env.now, truck, self.gha_id, dock_id)

            if slot_start is not None:
                dtp.mark_closed(self.gha_id, slot_start)

            truck.complete_stop(self.gha_id)
            truck.status = Truck.STATUS_IN_TRANSIT

            # Update stats
            s = self.stats[flow]
            s["processed"]    += 1
            s["total_wait"]    += wait_time
            s["total_service"] += service_time

    # ── Release window trigger ────────────────────────────────────────────────

    def release_window_monitor(self, slot_start: float, dtp: "DTPPlatform",
                                tp3: "TP3Buffer"):
        """
        SimPy process: waits until the priority window expires (minute 10),
        then checks if the slot should be released to a standby truck.
        Run one instance per published slot.
        """
        # Wait until minute 10 of the slot window
        yield self.env.timeout(max(0, slot_start + dtp.PRIORITY_WINDOW - self.env.now))

        if dtp.should_release_to_standby(self.gha_id, slot_start):
            # Notify TP3 that a standby truck may be released for this GHA
            # The Orchestrator (MO) or auto-release logic (M) acts on this signal
            tp3.signal_standby_opportunity(self.gha_id, slot_start, self.env.now)

    # ── Observation helpers (used by MARL layer) ──────────────────────────────

    def export_occupancy(self) -> float:
        """Fraction of export docks in use [0, 1]."""
        return self.docks_export.count / self.n_export if self.n_export > 0 else 0.0

    def import_occupancy(self) -> float:
        """Fraction of import docks in use [0, 1]."""
        return self.docks_import.count / self.n_import if self.n_import > 0 else 0.0

    def export_queue_norm(self, max_q: int = 20) -> float:
        return min(len(self.queue_export) / max_q, 1.0)

    def import_queue_norm(self, max_q: int = 20) -> float:
        return min(len(self.queue_import) / max_q, 1.0)

    def upcoming_bookings_norm(self, dtp: "DTPPlatform",
                               flow: str, horizon: float, max_b: int = 10) -> float:
        """Count of confirmed bookings of given flow type within horizon minutes."""
        now = self.env.now
        count = sum(
            1 for t, s in dtp.registry[self.gha_id].items()
            if s["truck_id"] is not None
            and 0 <= t - now <= horizon
            # We don't track flow_type on the slot directly yet — placeholder
            # When real data available, filter by truck's flow_type
        )
        return min(count / max_b, 1.0)


# =============================================================================
# TP3 BUFFER
# =============================================================================
class TP3Buffer:
    """
    Models the TP3 parking lot as a constrained SimPy Resource (140 slots).
    Trucks that cannot enter join an overflow queue (approach road).

    TP3 itself is passive — it does not decide who to release.
    The Orchestrator (Scenario MO) or the Transporter (Scenario M)
    calls release() to dispatch a truck toward a GHA.
    """
    CAPACITY = 140

    def __init__(self, env: simpy.Environment, infra: "InfrastructureLayer"):
        self.env   = env
        self.infra = infra

        # The 140-slot parking resource
        self.slots = simpy.Resource(env, capacity=self.CAPACITY)

        # Active entries: list of (truck, simpy_request)
        self._parked: List[tuple] = []

        # Trucks that couldn't enter (approach road overflow)
        self.overflow_queue: List[Truck] = []

        # Standby opportunity signals: [(gha_id, slot_start, signal_time)]
        # Consumed by Orchestrator or auto-release logic
        self.standby_signals: List[Dict] = []

    # ── Entry ─────────────────────────────────────────────────────────────────

    def enter(self, truck: Truck):
        """
        SimPy generator: truck requests a TP3 parking slot.
        If full, truck waits in overflow queue (on approach road N0→N1).
        """
        req = self.slots.request()

        # Try to get a slot immediately
        result = yield req | self.env.timeout(0)

        if req in result:
            # Got a slot immediately
            self._parked.append((truck, req))
            truck.status = Truck.STATUS_AT_TP3
            self.infra.tp3_in(self.env.now, truck)
        else:
            # TP3 full — join overflow queue
            self.overflow_queue.append(truck)
            # Wait indefinitely for a slot to open
            yield req
            if truck in self.overflow_queue:
                self.overflow_queue.remove(truck)
            self._parked.append((truck, req))
            truck.status = Truck.STATUS_AT_TP3
            self.infra.tp3_in(self.env.now, truck)

    # ── Release ───────────────────────────────────────────────────────────────

    def release(self, truck_id: str) -> Optional[Truck]:
        """
        Release a specific truck from TP3.
        Called by the Orchestrator (MO) or Transporter (M).
        Returns the Truck object if found, else None.
        """
        for i, (truck, req) in enumerate(self._parked):
            if truck.truck_id == truck_id:
                self._parked.pop(i)
                self.slots.release(req)
                self.infra.tp3_out(self.env.now, truck)
                return truck
        return None

    def release_next_for_gha(self, gha_id: str) -> Optional[Truck]:
        """
        Release the first truck in TP3 that has a booking for gha_id.
        Used by auto-release logic in Scenario M.
        """
        for i, (truck, req) in enumerate(self._parked):
            if gha_id in truck.booked_slots:
                self._parked.pop(i)
                self.slots.release(req)
                self.infra.tp3_out(self.env.now, truck)
                return truck
        return None

    # ── Standby signals ───────────────────────────────────────────────────────

    def signal_standby_opportunity(self, gha_id: str,
                                    slot_start: float, signal_time: float):
        """
        Called by GHATerminal.release_window_monitor() when a slot
        enters its release window with no booked truck present.
        """
        self.standby_signals.append({
            "gha_id":     gha_id,
            "slot_start": slot_start,
            "signal_time": signal_time,
            "consumed":   False
        })

    def get_pending_signals(self) -> List[Dict]:
        """Returns unconsumed standby signals. Used by Orchestrator observation."""
        return [s for s in self.standby_signals if not s["consumed"]]

    # ── Observation helpers ───────────────────────────────────────────────────

    def occupancy_ratio(self) -> float:
        """Core state variable: TP3 fullness [0, 1]."""
        return self.slots.count / self.CAPACITY

    def n_parked(self) -> int:
        return self.slots.count

    def n_overflow(self) -> int:
        return len(self.overflow_queue)

    def parked_by_flow(self, flow_type: str) -> int:
        """Count of parked trucks by flow type."""
        return sum(1 for t, _ in self._parked if t.flow_type == flow_type)

    def get_parked_trucks(self) -> List[Truck]:
        """Returns list of all currently parked trucks (for Orchestrator obs)."""
        return [t for t, _ in self._parked]