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
#     objects.py           → Truck, GHATerminal, TP3Buffer
#     dtp_platform.py      → DTPPlatform (slot booking and phase logic)
#     infrastructure.py    → InfrastructureLayer (ANPR events)
#     road.py              → RoadNetwork (stochastic travel times)
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
    ghas = list(params["gha"].keys())
    
    def __init__(
        self,
        env: simpy.Environment,
        dtp: DTPPlatform,
        terminals: Dict[str, GHATerminal],
        tp3: TP3Buffer,
        infra: InfrastructureLayer,
        road: RoadNetwork
    ):
        self.env = env
        self.dtp = dtp
        self.terminals = terminals
        self.tp3 = tp3
        self.infra = infra
        self.road = road

        # ── Demand parameters ────────────────────────────────────────────────
        self._arrival_rate = params["demand"]["arrival_rate"]
        self._peak_mult = params["demand"]["peak_multiplier"]
        self._peak_window = params["demand"]["peak_window"]
        self._flow_split = params["demand"]["flow_split"]
        self._orig_split = params["demand"]["origin_split"]
        self._multi_stop = params["demand"]["multi_stop_probs"]
        
        self._truck_counter = 0
        
    # ─────────────────────────────────────────────────────────────────────────
    # Main arrival logic
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        max_rate = self._arrival_rate * self._peak_mult

        while True:
            inter_arrival = np.random.exponential(1, max_rate)
            yield self.env.timeout(inter_arrival)
            
            actual_rate = self._rate_at(self.env.now)
            if np.random.random() < actual_rate / max_rate:
                truck = self._create_truck()
                self.env.process(self._truck_journey(truck))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main arrival logic
    # ─────────────────────────────────────────────────────────────────────────
    def _rate_at(self, t: int) -> float:
        """Returns the correct arrival rate at the time 't' of the day."""
        t = t % 1440
        peak_start = self._peak_window[0] % 1440
        peak_end = self._peak_window[1] % 1440
        ramp_dur = params["demand"]["ramp_dur"]
        
        if t < peak_start - ramp_dur or t > peak_end + ramp_dur:
            return self._arrival_rate
        
        elif peak_start - ramp_dur <= t < peak_start:
            frac = (t - (peak_start - ramp_dur)) / ramp_dur
            return self._arrival_rate * (1 + frac * (self._peak_mult - 1))
        
        elif peak_start <= t <= peak_end:
            return self._arrival_rate * self._peak_mult
        
        else:
            frac = (t - peak_end) / ramp_dur
            return self._arrival_rate * (self._peak_mult - frac * (self._peak_mult - 1))
        
    # ─────────────────────────────────────────────────────────────────────────
    # Truck creation
    # ─────────────────────────────────────────────────────────────────────────
    def _create_truck(self) -> Truck:
        """Creates a truck with incremental license plate. All probs come from params."""
        self._truck_counter += 1
        truck_id = f"TRK-{self._truck_counter:05d}"
        
        flow_type = np.random.choice(list(self._flow_split.keys()), p=list(self._flow_split.values()))
        origin_type = np.random.choice(list(self._orig_split.keys()), p=list(self._orig_split.values()))
        n_stops = np.random.choice([1, 2, 3, 4], p=self._multi_stop)
        ghas = np.random.choice(self.ghas, size=n_stops, replace=False)
        
        manifest = [
            {
                "gha": gha,
                "parcels": int(np.random.randint(
                    params["demand"]["parcels_min"],
                    params["demand"]["parcels_max"]
                ))
            }
            for gha in ghas
        ]
        
        return Truck(
            truck_id=truck_id,
            flow_type=flow_type,
            origin_type=origin_type,
            manifest=manifest
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Slot booking
    # ─────────────────────────────────────────────────────────────────────────
    def _book_slots(self, truck: Truck) -> bool:
        """FCFS logic, will be replaced by MARL engine."""
        # Earliest possible arrival at first GHA given origin travel time
        max_travel = max(self._origin_to_gate("nl"))    # conservative
        earliest_next_slot = self.env.now + max_travel + self.dtp.freeze_time
        booked_count = 0
        
        for stop in truck.manifest:
            gha = stop["gha"]
            
            available_slots = sorted(self.dtp.get_available_slots(gha, horizon=480))
            feasible = [s for s in available_slots if s >= earliest_next_slot]
            
            if not feasible:
                for booked_gha, booked_start in truck.booked_slots.items():
                    self.dtp.cancel_book(booked_gha, booked_start, truck.truck_id)
                truck.booked_slots.clear()
                return False
            
            chosen_slot = feasible[0]
            success = self.dtp.book_slot(gha, chosen_slot, truck.truck_id)
            
            if not success:
                for booked_gha, booked_start in truck.booked_slots.items():
                    self.dtp.cancel_book(booked_gha, booked_start, truck.truck_id)
                truck.booked_slots.clear()
                return False
            
            truck.booked_slots[gha] = chosen_slot
            booked_count += 1
            
            earliest_next_slot = chosen_slot + self.dtp.slot_duration + 5    # hardcoded
        
        return booked_count == len(truck.manifest)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Truck journey
    # ─────────────────────────────────────────────────────────────────────────
    def _truck_journey(self, truck: Truck):
        """
        SimPy generator: complete lifecycle of one truck from origin to departure.

        STAGES:
          1. Book DTP slots (FCFS baseline policy)
          2. Travel from origin to Main Gate
          3. Gate ANPR check
          4. Route to TP3 (if early for first slot) or directly to first GHA
          5. For each GHA stop in slot-time order:
               a. Travel to GHA (from gate or from TP3 or from previous GHA)
               b. GHATerminal.process_truck() → dock → service → depart
               c. Handle TP3 redirect if process_truck returned STATUS_AT_TP3
          6. Exit gate (ANPR gate_out event)
        """
        # ── Stage 1: Book DTP Slots ──────────────────────────────────────────
        all_booked = self._book_slots(truck)
        if not all_booked:
            return
        
        # ── Stage 2: Travel to Main Gate ─────────────────────────────────────
        first_slot = min(truck.booked_slots.values())
        travel_to_gate = self._origin_to_gate(truck.origin_type)
        depart_at = first_slot - travel_to_gate - self.road.time_gate_to_gha(
            min(truck.booked_slots, key=truck.booked_slots.get)
        )
        
        wait = max(0.0, depart_at - self.env.now)
        yield self.env.timeout(wait)
        yield self.env.timeout(travel_to_gate)
        
        # ── Stage 3: Main Gate ANPR Check ────────────────────────────────────
        if not all_booked:
            return
        self.infra.gate_in(self.env.now, truck)
        
        # ── Stage 4: travel to first gha or tp3 ──────────────────────────────
        # Sort remaining slots
        ordered_stops = sorted(
            truck.stops_remaining,
            key=lambda s: truck.booked_slots[s["gha"]]
        )
        first_gha = ordered_stops[0]["gha"]
        first_slot_start = truck.booked_slots[first_gha]
        
        # Compute ETA to first gha
        eta_first_gha = self.road.time_gate_to_gha(first_gha)
        arrival_if_direct = self.env.now + eta_first_gha
        
        if arrival_if_direct < first_slot_start:
            # Truck early, send to tp3
            yield self.env.timeout(self.road.time_gate_to_tp3())
            yield self.env.process(self.tp3.enter(truck))
            
            # Wait in tp3
            depart_tp3_at = first_slot_start - self.road.time_tp3_to_gha(first_gha)
            hold_tp3 = max(0.0, depart_tp3_at - self.env.now)
            if hold_tp3 > 0:
                yield self.env.timeout(hold_tp3)
            
            # Release from tp3
            self.tp3.release(truck.truck_id)
            
            # Travel to first gha
            yield self.env.timeout(self.road.time_tp3_to_gha(first_gha))
            from_tp3 = True

        else:
            # Truck is not early, go to gha
            yield self.env.timeout(eta_first_gha)
            from_tp3 = False
        
        # ── Stage 5: Visit each gha ──────────────────────────────────────────
        for stop in ordered_stops:
            gha = stop["gha"]
            if not from_tp3 and stop != ordered_stops[0]:
                yield self.env.timeout(self._intra_airport_travel(gha))    # NOTE: must implement specific travel times
                
            from_tp3 = False
            terminal = self.terminals[gha]
            yield self.env.provess(terminal.process_truck(truck, self.dtp))
            
            # Check if truck was redirect to tp3
            if truck.status == Truck.STATUS_AT_TP3:
                yield self.env.process(self._handle_tp3_redirect(truck, gha))
            
        # ── Stage 6: Exit ────────────────────────────────────────────────────
        truck.status = Truck.STATUS_DEPARTED
        self.infra.gate_out(self.env.now, truck)
        
    # ─────────────────────────────────────────────────────────────────────────
    # Handler for the case truck gets redirected to tp3
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_tp3_redirect(self, truck: Truck, gha: str):
        """
        SimPy generator: handles the case where GHATerminal returned STATUS_AT_TP3.
        Truck travels to TP3, parks, waits for a standby signal or its next available
        slot, then travels back to the GHA for another attempt.

        This is the FCFS fallback. In Scenario MO, the Orchestrator calls
        tp3.release() directly with a targeted truck_id, bypassing this timer.
        """
        # Travel from gha to tp3
        yield self.env.timeout(self.road.time_tp3_to_gha(gha))
        yield self.env.process(self.tp3.enter(truck))
        
        MAX_WAIT = 90    # hardcoded
        waited = 0
        
        while waited < MAX_WAIT:
            yield self.env.timeout(1)
            waited += 1
            
            for signal in self.tp3.get_pending_signals():
                if signal["gha"] == gha:
                    self.tp3.release(truck.truck_id)
                    signal["consumed"] = True
                    yield self.env.timeout(self.road.time_tp3_to_gha(gha))
                    terminal = self.terminals[gha]
                    yield self.env.process(terminal.process_truck(truck, self.dtp))
                    return
                
            booking = self.dtp.get_booking(gha, truck.truck_id)
            if booking is not None:
                depart_at = booking - self.road.time_tp3_to_gha(gha)
                if self.env.now >= depart_at:
                    self.tp3.release(truck.truck_id)
                    yield self.env.timeout(self.road.time_tp3_to_gha(gha))
                    terminal = self.terminal[gha]
                    yield self.env.process(terminal.process_truck(truck, self.dtp))
                    return
                
        # Truck gives up on this stop
        self.tp3.release(truck.truck_id)
        truck.complete_stop(gha)

    # ─────────────────────────────────────────────────────────────────────────
    # Travel time helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _origin_to_gate(self, origin_type: str) -> float:
        """Sample travel time from origin to Main Gate."""
        ranges = {
            "rfs": tuple(params["demand"]["origin_travel_time"]["rfs"]),
            "nl": tuple(params["demand"]["origin_travel_time"]["nl"]),
            "forwarder": tuple(params["demand"]["origin_travel_time"]["forwarder"]),
            "tp3": tuple(params["demand"]["origin_travel_time"]["tp3"])
        }
        lo, hi = ranges.get(origin_type, None)
        return float(np.random.uniform(lo, hi))
    
    def _intra_airport_travel(self, next_gha: str) -> float:
        """Travel time between consecutive GHA stops within the airport perimeter."""
        node_map = {"dnata": "N3", "wfs": "N4", "swiss": "N5", "klm": "N6"}
        target_node = node_map[next_gha]
        base = self.road.segments[f"N2_{target_node}"]
        return self.road._apply_stochastic_noise(base)