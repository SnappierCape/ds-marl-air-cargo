# =============================================================================
# DEMAND GENERATOR MODULE
# =============================================================================
# DESCRIPTION:
#     Two responsibilities:
#       1. run(): infinite arrival loop, spawns one SimPy process per truck
#       2. _truck_journey(): drives one truck through its complete journey
#
# HOW IT USES OTHER MODULES:
#     dtp_platform   → book and cancel slots (_book_slots)
#     objects        → process_truck and tp3.enter are yielded (blocking calls)
#     infrastructure → gate_in / gate_out ANPR events
#     road           → travel time samples at every leg
# =============================================================================
from typing import Dict, List, Optional

import simpy
import numpy as np

from env.objects import Truck, GHATerminal, TP3Buffer
from env.dtp_platform import DTPPlatform
from env.infrastructure import InfrastructureLayer
from env.road import RoadNetwork

from config.config import load_params
params = load_params()

# =============================================================================
# MAIN CLASS
# =============================================================================
class DemandGenerator:
    """
    Generates truck arrivals and runs each truck's journey through the system.
    Instantiated once per episode in schiphol_env.reset().
    """
    GHA_IDS = list(params["ghas"].keys())

    def __init__(
        self,
        env: simpy.Environment,
        dtp: DTPPlatform,
        terminals: Dict[str, GHATerminal],
        tp3: TP3Buffer,
        infra: InfrastructureLayer,
        road: RoadNetwork,
    ):
        self.env = env
        self.dtp = dtp
        self.terminals = terminals
        self.tp3 = tp3
        self.infra = infra
        self.road = road

        d = params["demand"]
        self._arrival_rate = d["arrival_rate"]
        self._peak_mult = d["peak_multiplier"]
        self._peak_window = d["peak_window"]
        self._ramp_dur = d["ramp_dur"]
        self._flow_split = d["flow_split"]
        self._orig_split = d["origin_split"]
        self._multi_stop = d["multi_stop_probs"]
        self._max_tp3_wait = d["max_tp3_wait"]
        self._parcels_min = d["parcels_min"]
        self._parcels_max = d["parcels_max"]
        self._travel_time = d["origin_travel_time"]

        self._truck_counter = 0
        self.pending_trucks: List[Truck] = []
        self._dispatch_events: Dict[str, simpy.Event] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # ARRIVAL LOOP
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        """
        Infinite SimPy generator. Samples inter-arrival times and spawns
        one independent truck journey process per accepted arrival.
        Uses thinning: sample at max rate, accept with probability
        actual_rate / max_rate.
        """
        max_rate = self._arrival_rate * self._peak_mult

        while True:
            # Sample next potential arrival from the max-rate Poisson
            yield self.env.timeout(np.random.exponential(1.0 / max_rate))

            # Accept or reject based on current actual rate
            if np.random.random() < self._rate_at(self.env.now) / max_rate:
                truck = self._create_truck()
                
                # Add to pending list so Transporter can see it
                self.pending_trucks.append(truck)
                
                # Create the dispatch event
                dispatch_event = self.env.event()
                self._dispatch_events[truck.truck_id] = dispatch_event

                # Start journey
                self.env.process(self._truck_journey(truck, dispatch_event))

    def _rate_at(self, t: float) -> float:
        """
        Arrival rate (trucks/min) at simulation time t.
        Periodic over 24h. Ramps up before peak, flat during peak, ramps down.
        """
        t = t % 1440
        peak_start = self._peak_window[0]
        peak_end = self._peak_window[1]
        ramp = self._ramp_dur

        if t < peak_start - ramp or t > peak_end + ramp:
            return self._arrival_rate

        elif peak_start - ramp <= t < peak_start:
            frac = (t - (peak_start - ramp)) / ramp
            return self._arrival_rate * (1 + frac * (self._peak_mult - 1))

        elif peak_start <= t <= peak_end:
            return self._arrival_rate * self._peak_mult

        else:
            frac = (t - peak_end) / ramp
            return self._arrival_rate * (self._peak_mult - frac * (self._peak_mult - 1))

    # ─────────────────────────────────────────────────────────────────────────
    # TRUCK CREATION
    # ─────────────────────────────────────────────────────────────────────────
    def _create_truck(self) -> Truck:
        """Samples truck attributes from config distributions."""
        self._truck_counter += 1

        flow_type = np.random.choice(
            list(self._flow_split.keys()),
            p=list(self._flow_split.values())
        )
        origin_type = np.random.choice(
            list(self._orig_split.keys()),
            p=list(self._orig_split.values())
        )
        n_stops = np.random.choice([1, 2, 3, 4], p=self._multi_stop)
        n_stops = min(n_stops, len(self.GHA_IDS))
        ghas = np.random.choice(self.GHA_IDS, size=n_stops, replace=False)

        manifest = [
            {"gha": gha, "parcels": int(np.random.randint(
                self._parcels_min, self._parcels_max
            ))}
            for gha in ghas
        ]

        return Truck(
            truck_id=f"TRK-{self._truck_counter:05d}",
            flow_type=flow_type,
            origin_type=origin_type,
            manifest=manifest,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TRANSPORTER AGENT INTERFACE
    # ─────────────────────────────────────────────────────────────────────────
    def book_one_slot(self, truck_id: str, gha: str, flow_type: str) -> bool:
        """Called by the Transporter agent action handler in schiphol_env.py."""
        truck = self._get_pending_truck(truck_id)
        if truck is None:
            return False

        # Truck must not already have a booking at this GHA
        if gha in truck.booked_slots:
            return False

        # Truck must actually need this GHA
        if not any(s["gha"] == gha for s in truck.stops_remaining):
            return False

        # Earliest feasible start: after all existing bookings end + buffer
        buffer = self._intra_airport_buffer()
        earliest = self.env.now + self.dtp.freeze_time
        if truck.booked_slots:
            latest_booked_end = max(truck.booked_slots.values()) + self.dtp.slot_duration
            earliest = max(earliest, latest_booked_end + buffer)

        available = self.dtp.get_available_slots(gha, horizon=480, flow_type=truck.flow_type)
        feasible  = [s for s in available if s >= earliest]

        if not feasible:
            return False

        return self.dtp.book_slot(gha, feasible[0], truck_id, truck.flow_type) and \
            self._record_booking(truck, gha, feasible[0])

    def dispatch_truck(self, truck_id: str) -> bool:
        """
        Called by the Transporter agent when it decides the truck is ready
        to depart. All required GHA stops must be booked first.
        """
        truck = self._get_pending_truck(truck_id)
        if truck is None:
            return False

        # Enforce: all stops must be booked before dispatch
        needed_ghas = {s["gha"] for s in truck.manifest}
        booked_ghas = set(truck.booked_slots.keys())
        if not needed_ghas.issubset(booked_ghas):
            return False

        # Remove from pending list
        self.pending_trucks = [t for t in self.pending_trucks if t.truck_id != truck_id]

        # Fire the dispatch gate — the frozen journey process wakes up
        event = self._dispatch_events.pop(truck_id, None)
        if event:
            event.succeed()

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # TRUCK JOURNEY LOGIC
    # ─────────────────────────────────────────────────────────────────────────
    def _truck_journey(self, truck: Truck, dispatch_event: simpy.Event):
        """
        SimPy generator: complete lifecycle of one truck.

        Stage 1 — Book slots (before dispatching the truck)
        Stage 2 — Wait until departure time, then travel to gate
        Stage 3 — Gate ANPR
        Stage 4 — Route to first GHA directly or via TP3 if too early
        Stage 5 — Visit each GHA stop in slot-time order
        Stage 6 — Exit gate
        """
        # ── Stage 1: Wait for Transporter to book ────────────────────────────
        yield dispatch_event

        # ── Stage 2: Wait for departure time, then travel to gate ─────────────
        first_gha = min(truck.booked_slots, key=truck.booked_slots.get)
        first_slot = truck.booked_slots[first_gha]
        travel_to_gate = self._origin_to_gate(truck.origin_type)
        gate_to_gha_time = self.road.time_from_to("gate", first_gha)
        depart_at = first_slot - travel_to_gate - gate_to_gha_time
        yield self.env.timeout(max(0.0, depart_at - self.env.now))
        yield self.env.timeout(travel_to_gate)

        # ── Stage 3: Gate ANPR ────────────────────────────────────────────────
        self.infra.gate_in(self.env.now, truck)

        # ── Stage 4: Route to first GHA or TP3 ───────────────────────────────
        # Sort stops by slot time — visit in chronological order
        ordered_stops = sorted(
            truck.stops_remaining,
            key=lambda s: truck.booked_slots[s["gha"]]
        )

        first_gha = ordered_stops[0]["gha"]
        first_slot_start = truck.booked_slots[first_gha]
        eta_to_first_gha = self.road.time_from_to("gate", first_gha)

        if self.env.now + eta_to_first_gha < first_slot_start:
            # Truck arrives too early — hold in TP3
            yield self.env.timeout(self.road.time_from_to("gate", "tp3"))
            yield self.env.process(self.tp3.enter(truck))

            # Wait in TP3 until departure time for first GHA
            tp3_to_gha_time = self.road.time_from_to("tp3", first_gha)
            depart_tp3_at = first_slot_start - tp3_to_gha_time
            yield self.env.timeout(max(0.0, depart_tp3_at - self.env.now))

            self.tp3.release(truck.truck_id)
            yield self.env.timeout(tp3_to_gha_time)
            prev_location = "tp3"
        else:
            # Truck arrives on time — go directly
            yield self.env.timeout(eta_to_first_gha)
            prev_location = "gate"

        # ── Stage 5: Visit each GHA stop ──────────────────────────────────────
        for i, stop in enumerate(ordered_stops):
            gha = stop["gha"]

            # Intra-airport travel for stops after the first
            if i > 0:
                yield self.env.timeout(
                    self.road.time_from_to(prev_location, gha)
                )

            yield self.env.process(
                self.terminals[gha].process_truck(truck, self.dtp)
            )

            if truck.status == Truck.STATUS_AT_TP3:
                yield self.env.process(
                    self._handle_tp3_redirect(truck, gha)
                )

            prev_location = gha

        # ── Stage 6: Exit ─────────────────────────────────────────────────────
        truck.status = Truck.STATUS_DEPARTED
        self.infra.gate_out(self.env.now, truck)

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER FOR TP3 REDIRECTION
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_tp3_redirect(self, truck: Truck, gha: str):
        """SimPy generator: called when process_truck returns STATUS_AT_TP3."""
        yield self.env.timeout(self.road.time_from_to(gha, "tp3"))
        yield self.env.process(self.tp3.enter(truck))

        waited = 0
        while waited < self._max_tp3_wait:
            yield self.env.timeout(1)
            waited += 1

            for signal in self.tp3.get_pending_signals():
                if signal["gha"] == gha:
                    signal["consumed"] = True
                    self.tp3.release(truck.truck_id)
                    yield self.env.timeout(self.road.time_from_to("tp3", gha))
                    yield self.env.process(self.terminals[gha].process_truck(truck, self.dtp))
                    return

            # Check if it is time to depart for the booked slot
            booking = self.dtp.get_booking(gha, truck.truck_id)
            if booking is not None:
                tp3_to_gha = self.road.time_from_to("tp3", gha)
                if self.env.now >= booking - tp3_to_gha:
                    self.tp3.release(truck.truck_id)
                    yield self.env.timeout(tp3_to_gha)
                    yield self.env.process(self.terminals[gha].process_truck(truck, self.dtp))
                    return

        # Timeout — give up on this stop
        self.tp3.release(truck.truck_id)
        truck.complete_stop(gha)

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER FOR TP3 REDIRECTION
    # ─────────────────────────────────────────────────────────────────────────
    def _origin_to_gate(self, origin_type: str) -> float:
        lo, hi = self._travel_time[origin_type]
        return float(np.random.uniform(lo, hi))

    def _intra_airport_buffer(self) -> float:
        return max(params["road"]["segments"].values())
    
    def _get_pending_truck(self, truck_id: str) -> Optional[Truck]:
        for truck in self.pending_trucks:
            if truck.truck_id == truck_id:
                return truck
        return None

    def _record_booking(self, truck: Truck, gha: str, slot_start: int) -> bool:
        truck.booked_slots[gha] = slot_start
        return True