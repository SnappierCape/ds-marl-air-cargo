# =============================================================================
# DEMAND GENERATOR MODULE
# =============================================================================
# DESCRIPTION:
#     Generates truck arrivals and orchestrates each truck's full journey
#     through the Schiphol cargo area.
#
# TWO RESPONSIBILITIES:
#     1. Arrival loop (run):
#        A single SimPy process that runs for the whole episode. Samples
#        inter-arrival times from a time-varying Poisson process and spawns
#        one independent SimPy process per truck via env.process().
#        Trucks never block each other — each lives in its own process.
#
#     2. Truck journey (_truck_journey):
#        One SimPy generator per truck. Moves the truck through every
#        physical stage: origin → gate → TP3 or GHA → service → exit.
#        Each yield consumes simulated time (travel or service).
#
# BASELINE BOOKING POLICY:
#     _book_slots() implements the FCFS Transporter heuristic — earliest
#     available slot per GHA stop. This will be replaced by the learned
#     MARL Transporter policy. Everything else in this module stays fixed.
#
# SLOT SEQUENCING:
#     Multi-stop trucks visit GHAs in chronological slot order.
#     _book_slots() enforces non-overlapping windows so a truck is never
#     booked at two GHAs at the same time.
#
# DEPENDENCIES:
#     objects.py       → Truck, GHATerminal, TP3Buffer
#     dtp_platform.py  → DTPPlatform (slot booking and phase logic)
#     infrastructure.py → InfrastructureLayer (ANPR events)
#     road.py          → RoadNetwork (stochastic travel times)
# =============================================================================
import sys
import os
from typing import Dict

import simpy
import numpy as np

# Setting base path for local imports
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
import config.config

from env.objects import Truck, GHATerminal, TP3Buffer
from env.dtp_platform import DTPPlatform
from env.infrastructure import InfrastructureLayer
from env.road import RoadNetwork

# =============================================================================
# PARAMETERS IMPORT
# =============================================================================
params = config.load_params()

# =============================================================================
# DEMAND GENERATOR
# =============================================================================
class DemandGenerator:
    """
    Generates truck arrivals and runs each truck's journey through the system.

    Instantiated once per episode inside SchipholCargoEnv.reset().
    Call env.process(generator.run()) to start the arrival loop.
    """

    # GHA names in the order trucks visit them when sorting by slot time.
    # Must match keys in params["gha"] and dtp.registry.
    GHA_IDS = ["dnata", "klm", "swissport", "menzies_wfs"]

    def __init__(
        self,
        env:       simpy.Environment,
        dtp:       DTPPlatform,
        terminals: Dict[str, GHATerminal],
        tp3:       TP3Buffer,
        infra:     InfrastructureLayer,
        road:      RoadNetwork,
    ):
        self.env       = env
        self.dtp       = dtp
        self.terminals = terminals   # {"dnata": GHATerminal, ...}
        self.tp3       = tp3
        self.infra     = infra
        self.road      = road

        # Pull demand parameters from config
        self._base_rate  = params["demand"]["arrival_rate"] / 60.0
        self._peak_mult  = params["demand"]["peak_multiplier"]
        self._peak_win   = params["demand"]["peak_window"]          # [900, 1050]
        self._flow_split = params["demand"]["flow_split"]           # {"export": 0.6, "import": 0.4}
        self._orig_split = params["demand"]["origin_split"]         # {"rfs": 0.25, ...}
        self._multi_stop = params["demand"].get("multi_stop_probs", [0.60, 0.30, 0.10])

        # Counter for generating unique truck IDs
        self._truck_counter = 0

    # =========================================================================
    # MAIN ARRIVAL LOOP
    # =========================================================================
    def run(self):
        max_rate = self._base_rate * self._peak_mult

        while True:
            inter_arrival = np.random.exponential(1.0 / max_rate)
            yield self.env.timeout(inter_arrival)

            actual_rate = self._rate_at(self.env.now)
            if np.random.random() < actual_rate / max_rate:
                truck = self._create_truck()
                self.env.process(self._truck_journey(truck))

    # =========================================================================
    # TIME-VARYING RATE FUNCTION
    # =========================================================================
    def _rate_at(self, t: float) -> float:
        """
        Returns the truck arrival rate (trucks/minute) at simulation time t.
        Time is mapped to minutes within the current 24-hour cycle (t mod 1440)
        so the peak pattern repeats every day regardless of episode length.
        """
        t_mod      = t % 1440          # position within the current day (0–1439 min)
        peak_start = self._peak_win[0] % 1440   # e.g. 900 → 15:00
        peak_end   = self._peak_win[1] % 1440   # e.g. 1050 → 17:30
        ramp_dur   = 60.0

        if t_mod < peak_start - ramp_dur or t_mod > peak_end + ramp_dur:
            return self._base_rate

        elif peak_start - ramp_dur <= t_mod < peak_start:
            frac = (t_mod - (peak_start - ramp_dur)) / ramp_dur
            return self._base_rate * (1 + frac * (self._peak_mult - 1))

        elif peak_start <= t_mod <= peak_end:
            return self._base_rate * self._peak_mult

        else:
            frac = (t_mod - peak_end) / ramp_dur
            return self._base_rate * (self._peak_mult - frac * (self._peak_mult - 1))

    # =========================================================================
    # TRUCK CREATION
    # =========================================================================
    def _create_truck(self) -> Truck:
        """
        Instantiates one Truck with sampled attributes.
        All probabilities come from config — no hardcoding here.
        """
        self._truck_counter += 1
        truck_id = f"TRK-{self._truck_counter:05d}"

        # Sample flow type: "export" or "import"
        flow_type = np.random.choice(
            list(self._flow_split.keys()),
            p=list(self._flow_split.values())
        )

        # Sample origin type: "rfs", "random_nl", "forwarder_hub", "tp3"
        origin_type = np.random.choice(
            list(self._orig_split.keys()),
            p=list(self._orig_split.values())
        )

        # Sample number of GHA stops: 1, 2, or 3 stops
        # _multi_stop = [P(1 stop), P(2 stops), P(3 stops)]
        n_stops = np.random.choice([1, 2, 3], p=self._multi_stop)

        # Randomly pick n_stops distinct GHAs (no duplicate stops per trip)
        ghas_for_trip = np.random.choice(self.GHA_IDS, size=n_stops, replace=False)

        # Build manifest: each stop gets a random parcel count
        manifest = [
            {
                "gha":     gha,
                "parcels": int(np.random.randint(
                    params["demand"]["parcels_min"],   # e.g. 3
                    params["demand"]["parcels_max"]    # e.g. 25
                ))
            }
            for gha in ghas_for_trip
        ]

        return Truck(
            truck_id=truck_id,
            flow_type=flow_type,
            origin_type=origin_type,
            manifest=manifest,
            # departure_time not in original Truck dataclass — add if needed for NTTP
        )

    # =========================================================================
    # BASELINE BOOKING POLICY (FCFS TRANSPORTER)
    # =========================================================================
    def _book_slots(self, truck: Truck) -> bool:
        """
        Books one DTP slot per GHA stop in the truck's manifest.
        This is the FCFS baseline policy: take the earliest available slot.

        SLOT SEQUENCING RULE:
          A truck cannot be at two GHAs at the same time.
          After booking a slot at GHA A (e.g. 09:00–09:45), the next GHA's
          slot must start AFTER the previous slot ends plus travel time.
          earliest_next = previous_slot_end + intra_airport_travel (~5 min)

        Returns True if all stops were successfully booked, False if any failed.
        The truck will not enter the gate unless all bookings are secured (R1).
        """
        # Track the earliest time the truck can start its next booking window
        earliest_next_slot = self.env.now + self.dtp.freeze_time  # must be outside frozen window

        booked_count = 0

        for stop in truck.manifest:
            gha = stop["gha"]

            # Get slots available from earliest_next_slot onwards
            available = sorted(self.dtp.get_available_slots(gha, horizon=480))

            # Filter to slots that start at or after earliest_next_slot
            feasible = [s for s in available if s >= earliest_next_slot]

            if not feasible:
                # No feasible slot found for this stop — booking fails entirely
                # Undo any bookings made so far to keep the registry clean
                for booked_gha, booked_start in truck.booked_slots.items():
                    self.dtp.cancel_book(booked_gha, booked_start, truck.truck_id)
                truck.booked_slots.clear()
                return False

            # Take the earliest feasible slot
            chosen_slot = feasible[0]
            success = self.dtp.book_slot(gha, chosen_slot, truck.truck_id)

            if not success:
                # Race condition: slot was taken between get_available_slots and book_slot
                for booked_gha, booked_start in truck.booked_slots.items():
                    self.dtp.cancel_book(booked_gha, booked_start, truck.truck_id)
                truck.booked_slots.clear()
                return False

            truck.booked_slots[gha] = chosen_slot
            booked_count += 1

            # Next booking must start after this slot ends + intra-airport travel buffer
            # 5 minutes is the approximate maximum intra-airport travel time
            earliest_next_slot = chosen_slot + self.dtp.slot_duration + 5

        return booked_count == len(truck.manifest)

    # =========================================================================
    # TRUCK JOURNEY
    # =========================================================================
    def _truck_journey(self, truck: Truck):
        """
        SimPy generator: complete lifecycle of one truck from origin to departure.

        STAGES:
          1. Travel from origin to Main Gate (stochastic, origin-dependent)
          2. Book DTP slots (FCFS baseline policy)
          3. Gate ANPR check (R1: booking required)
          4. Route to TP3 (if early for first slot) or directly to first GHA
          5. For each GHA stop in slot-time order:
               a. Travel to GHA (from gate or from TP3 or from previous GHA)
               b. GHATerminal.process_truck() → dock → service → depart
               c. Handle TP3 redirect if process_truck returned STATUS_AT_TP3
          6. Exit gate (ANPR gate_out event)
        """
        # ── Stage 1: Travel from origin to Main Gate ──────────────────────────
        travel_to_gate = self._origin_to_gate(truck.origin_type)
        yield self.env.timeout(travel_to_gate)

        # ── Stage 2: Book DTP slots (before the gate check) ───────────────────
        # In the real DTP system, transporters book before dispatching the truck.
        # We model it here for simplicity; the timing is close enough since
        # origin-to-gate travel is typically >> freeze_time.
        all_booked = self._book_slots(truck)

        # ── Stage 3: Gate ANPR check (R1) ─────────────────────────────────────
        if not all_booked:
            # Truck has no valid booking → denied entry, journey ends here.
            # No ANPR event fired — truck never entered the perimeter.
            return

        # All bookings confirmed → ANPR fires, truck enters perimeter
        self.infra.gate_in(self.env.now, truck)

        # ── Stage 4: Route to TP3 or first GHA ───────────────────────────────
        # Sort remaining stops by booked slot time — visit in chronological order
        ordered_stops = sorted(
            truck.stops_remaining,
            key=lambda s: truck.booked_slots[s["gha"]]
        )

        first_gha      = ordered_stops[0]["gha"]
        first_slot_start = truck.booked_slots[first_gha]

        # Compute ETA to first GHA if going directly from gate
        eta_to_first_gha = self.road.time_gate_to_gha(first_gha)
        arrival_if_direct = self.env.now + eta_to_first_gha

        if arrival_if_direct < first_slot_start:
            # Truck would arrive too early for its first slot.
            # Send to TP3 to wait — R8a (early trucks held at TP3).
            yield self.env.timeout(self.road.time_gate_to_tp3())
            yield self.env.process(self.tp3.enter(truck))

            # Wait in TP3 until it's time to leave for the first GHA.
            # We want to arrive at the GHA just as the priority window opens (t=0).
            tp3_to_gha_time = self.road.time_tp3_to_gha(first_gha)
            depart_tp3_at   = first_slot_start - tp3_to_gha_time

            hold_time = max(0.0, depart_tp3_at - self.env.now)
            if hold_time > 0:
                yield self.env.timeout(hold_time)

            # Release from TP3 (FCFS — in Scenario MO, Orchestrator overrides this)
            self.tp3.release(truck.truck_id)

            # Travel from TP3 to first GHA
            yield self.env.timeout(tp3_to_gha_time)
            from_tp3 = True

        else:
            # Truck will arrive on time or slightly late — go directly to GHA
            yield self.env.timeout(eta_to_first_gha)
            from_tp3 = False

        # ── Stage 5: Visit each GHA stop ─────────────────────────────────────
        # Re-sort after potential TP3 wait — manifest order may differ from slot order
        for stop in ordered_stops:
            gha = stop["gha"]

            if not from_tp3 and stop != ordered_stops[0]:
                # Intra-airport travel between consecutive GHA stops.
                # All GHAs share the internal cargo road — travel is short.
                yield self.env.timeout(self._intra_airport_travel(gha))

            from_tp3 = False  # only the first leg from TP3 uses tp3 travel time

            # Hand off to GHATerminal — this yields internally for queue + service
            terminal = self.terminals[gha]
            yield self.env.process(terminal.process_truck(truck, self.dtp))

            # Check if GHATerminal sent the truck back to TP3
            # (phase was "release_dock_taken" or "no_show")
            if truck.status == Truck.STATUS_AT_TP3:
                # Truck needs to reach TP3, park, wait for release, then retry this GHA
                yield self.env.process(self._handle_tp3_redirect(truck, gha))

            # After successful service, complete_stop() has been called inside
            # process_truck(). stops_remaining is already updated.

        # ── Stage 6: Exit gate ────────────────────────────────────────────────
        truck.status = Truck.STATUS_DEPARTED
        self.infra.gate_out(self.env.now, truck)

    # =========================================================================
    # TP3 REDIRECT HANDLER
    # =========================================================================
    def _handle_tp3_redirect(self, truck: Truck, gha: str):
        """
        SimPy generator: handles the case where GHATerminal returned STATUS_AT_TP3.
        Truck travels to TP3, parks, waits for a standby signal or its next available
        slot, then travels back to the GHA for another attempt.

        This is the FCFS fallback. In Scenario MO, the Orchestrator calls
        tp3.release() directly with a targeted truck_id, bypassing this timer.
        """
        # Travel from GHA to TP3 (reverse of the tp3→gha leg)
        yield self.env.timeout(self.road.time_tp3_to_gha(gha))  # approx same distance
        yield self.env.process(self.tp3.enter(truck))

        # Wait for a standby opportunity signal for this GHA.
        # Poll every minute — in Scenario MO the Orchestrator releases directly.
        MAX_WAIT = 90  # cap wait at 90 minutes (2 full slot windows)
        waited   = 0

        while waited < MAX_WAIT:
            yield self.env.timeout(1)  # check every simulated minute
            waited += 1

            # Check if a standby opportunity exists for this GHA
            pending = self.tp3.get_pending_signals()
            for signal in pending:
                if signal["gha"] == gha:
                    # Signal found — release truck and head to GHA
                    self.tp3.release(truck.truck_id)
                    signal["consumed"] = True
                    yield self.env.timeout(self.road.time_tp3_to_gha(gha))
                    # Retry the GHA visit
                    terminal = self.terminals[gha]
                    yield self.env.process(terminal.process_truck(truck, self.dtp))
                    return

            # No signal yet — also check if truck has a booking in the next slot window
            # (the original booking is still valid unless it became a no_show)
            booking = self.dtp.get_booking(gha, truck.truck_id)
            if booking is not None:
                # Travel time to arrive just as next slot opens
                tp3_to_gha = self.road.time_tp3_to_gha(gha)
                depart_at  = booking - tp3_to_gha
                if self.env.now >= depart_at:
                    self.tp3.release(truck.truck_id)
                    yield self.env.timeout(tp3_to_gha)
                    terminal = self.terminals[gha]
                    yield self.env.process(terminal.process_truck(truck, self.dtp))
                    return

        # Timeout — truck gives up on this stop after MAX_WAIT minutes
        # Release from TP3 and skip this GHA stop
        self.tp3.release(truck.truck_id)
        truck.complete_stop(gha)  # mark as skipped (no penalty beyond existing no_show)

    # =========================================================================
    # TRAVEL TIME HELPERS
    # =========================================================================
    def _origin_to_gate(self, origin_type: str) -> float:
        """
        Sample travel time from origin to Main Gate.
        These are placeholder ranges — replace with calibrated distributions
        when real eLink data is available.

        origin_type    range (minutes)    comment
        ──────────────────────────────────────────
        rfs            15–90              flight-linked, tighter schedule
        random_nl      20–180             general NL locations, high variance
        forwarder_hub  20–120             logistics centres, moderate variance
        tp3            ~0                 truck is already inside the perimeter
        """
        ranges = {
            "rfs":           (15,  90),
            "random_nl":     (20, 180),
            "forwarder_hub": (20, 120),
            "tp3":           (0,    2),   # already inside perimeter
        }
        lo, hi = ranges.get(origin_type, (20, 120))
        return float(np.random.uniform(lo, hi))

    def _intra_airport_travel(self, next_gha: str) -> float:
        """
        Travel time between consecutive GHA stops within the airport perimeter.
        All GHAs are on the same internal road — roughly N2 junction to each GHA.
        Uses RoadNetwork for consistency with the rest of the travel model.
        Approximation: use gate-to-GHA minus the shared N0→N2 segment.
        """
        # All intra-airport legs pass through N2 junction, so we use the
        # N2→GHA segment only (skip the N0→N2 leg already driven)
        node_map = {
            "dnata":       "N3",
            "menzies_wfs": "N4",
            "swissport":   "N5",
            "klm":         "N6",
        }
        target_node = node_map[next_gha]
        base        = self.road.segments[f"N2_{target_node}"]
        return self.road._apply_stochastic_noise(base)