# =============================================================================
# DIGITAL TRUCK SLOT PLANNING MODULE
# =============================================================================
# DESCRIPTION:
#     Rule engine that manages slot reservations, validates truck arrivals
#     against their booked windows, and handles penalty tracking (no-shows).
#
# KEY DATA STRUCTURES:
#     registry = {
#         "gha": {
#             "slot_start": [
#                 {"truck_id": "TRK-999", "phase": "unbooked"},
#                 {"truck_id": "ABC-123", "phase": "release"},
#                 ...
#             ]
#         }
#     }
#
# BOOKING LIFECYCLE PHASES (stored in registry):
#     "available"  →  slot published, no truck assigned
#     "booked"     →  truck reserved the slot
#     "docked"     →  truck is physically at the dock (R5: Orchestrator cannot touch)
#     "closed"     →  service complete, truck departed
#     "no_show"    →  truck never appeared within the window
#
# TRUCK ARRIVAL PHASES (returned by get_slot_phase, consumed by simulation.py):
#     "unbooked"           →  truck has no booking for this GHA
#     "early"              →  arrived before window opens (>15 min early)
#     "priority"           →  arrived in minutes 0–10 (dock held)
#     "release"            →  arrived in minutes 11–45, dock still free
#     "release_dock_taken" →  arrived in minutes 11–45, dock given to standby
#     "no_show"            →  arrived after slot expiry (> slot_duration from start)
# =============================================================================
import sys
import os
from typing import Dict, Optional, List

import simpy

# Setting base path for local imports.
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
import config.config

# =============================================================================
# PARAMETERS IMPORT
# =============================================================================
params = config.load_params()

# =============================================================================
# DPT PLATFORM MODEL
# =============================================================================
class DTPPlatform:
    """
    Truck Slot Booking logic engine.
    Pure Python logic; relies on the SimPy environment solely to check 'env.now'.
    """
    def __init__(
        self,
        env: simpy.Environment,
        slot_duration: int = params["booking"]["slot_duration"],
        priority_window: int = params["booking"]["priority_window"],
        freeze_time: int = params["booking"]["freeze_time"],
        lead_time: int = params["booking"]["lead_time"]
    ):
        self.env = env
        self.slot_duration = slot_duration
        self.priority_window = priority_window
        self.freeze_time = freeze_time
        self.lead_time = lead_time
        
        self.registry: Dict[str, Dict[int, List[Dict]]] = {
            gha: {} for gha in list(params["gha_docks"])
        }
        
        self.no_shows: Dict[str, int] = {}
        self.late_arrivals: Dict[str, int] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Slot publication
    # ─────────────────────────────────────────────────────────────────────────
    def publish_slot(self, gha: str, slot_start: int) -> bool:
        if gha not in self.registry:
            raise ValueError(f'GHA "{gha}" is not known, please insert a known GHA.')
        
        now = self.env.now
        if slot_start <= now:
            return False
        if slot_start - now > self.lead_time:
            return False
        if slot_start - now < self.freeze_time:
            return False
        
        tot_docks = params["gha"][gha]["total"]
        current_docks = len(self.registry[gha].get(slot_start, []))
        if current_docks >= tot_docks:
            return False
        
        if slot_start not in self.registry[gha]:
            self.registry[gha][slot_start] = []
            
        self.registry[gha][slot_start].append(
            {"truck_id": None, "phase": "available"}
        )
        return True
    
    # ─────────────────────────────────────────────────────────────────────────
    # Slot booking
    # ─────────────────────────────────────────────────────────────────────────
    # NOTE: I have to implement the slot overlapping logic.
    def book_slot(self, gha: str, slot_start: int, truck_id: str) -> bool:
        if gha not in self.registry:
            raise ValueError(f'GHA "{gha}" is not known, please insert a known GHA.')
        
        now = self.env.now
        if slot_start - now < self.freeze_time:
            return False
        
        slots = self.registry[gha].get(slot_start, [])
        for slot in slots:
            if slot["truck_id"] is None and slot["phase"] == "available":
                slot["truck_id"] = truck_id
                slot["phase"] = "booked"
                return True
        return False
    
    def orch_book_slot(self, gha: str, slot_start: int, truck_id: int) -> bool:
        if gha not in self.registry:
            raise ValueError(f'GHA "{gha}" is not known, please insert a known GHA.')
        
        slots = self.registry[gha].get(slot_start, [])
        for slot in slots:
            if slot["truck_id"] is None and slot["phase"] == "available":
                slot["truck_id"] = truck_id
                slot["phase"] = "booked"
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Slot cancellation
    # ─────────────────────────────────────────────────────────────────────────
    def cancel_book(self, gha: str, slot_start: int, truck_id: str) -> bool:
        if gha not in self.registry:
            raise ValueError(f'GHA "{gha}" is not known, please insert a known GHA.')
        
        if slot_start - self.env.now < self.freeze_time:
            return False
        
        return self._free_slot(gha, slot_start, truck_id)
    
    def orch_cancel_book(self, gha: str, slot_start: int, truck_id: str) -> bool:
        if gha not in self.registry:
            raise ValueError(f'GHA "{gha}" is not known, please insert a known GHA.')
        
        if self._is_docked(gha, slot_start, truck_id):
            return False
        
        return self._free_slot(gha, slot_start, truck_id)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Slot modification
    # ─────────────────────────────────────────────────────────────────────────
    def orch_modify_book(
        self,
        truck_id: str,
        from_gha: str,
        from_slot_start: int,
        to_gha: str,
        to_slot_start: int
    ) -> bool:
        if from_gha not in self.registry:
            raise ValueError(f'GHA "{from_gha}" is not known, please insert a known GHA.')
        if to_gha not in self.registry:
            raise ValueError(f'GHA "{to_gha}" is not known, please insert a known GHA.')
        
        if self._is_docked(from_gha, from_slot_start, truck_id):
            return False
        
        new_slots = self.registry[to_gha].get[to_slot_start, []]
        has_space = any(
            slot["truck_id"] is None and slot["phase"] == "available"
            for slot in new_slots
        )
        if not has_space:
            return False
        
        if not self._free_slot(from_gha, from_slot_start, truck_id):
            return False
        
        return self.orch_book_slot(to_gha, to_slot_start, truck_id)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Arrival logic
    # ─────────────────────────────────────────────────────────────────────────
    def get_slot_phase(
        self,
        slot_start: Optional[int],
        arrival_time: int,
        dock_is_free: bool = False
    ) -> str:
        if slot_start is None:
            return "unbooked"    # useful to call the function without creating a specific truck
          
        offset = arrival_time - slot_start
        
        if offset < 0:
            return "early"
        elif offset <= self.priority_window:
            return "priority"
        elif offset <= self.slot_duration:
            return "release" if dock_is_free else "release_dock_taken"
        else:
            return "no_show"
    
    def release_to_standby(self, gha: str, slot_start: int) -> bool:
        if gha not in self.registry:
            raise ValueError(f'GHA "{gha}" is not known, please insert a known GHA.')
        
        now = self.env.now
        if not(slot_start + self.priority_window <= now <= slot_start + self.slot_duration):
            return False
        
        for slot in self.registry[gha].get(slot_start, []):
            if slot["phase"] == "booked":
                return True
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Dock state logic
    # ─────────────────────────────────────────────────────────────────────────
    def mark_docked(self, gha:str, slot_start: int, truck_id: str):
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id:
                slot["phase"] = "docked"
                return
            
    def mark_closed(self, gha:str, slot_start: int, truck_id: str):
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id:
                slot["phase"] = "closed"
                return
    
    # ─────────────────────────────────────────────────────────────────────────
    # Penalty tracking
    # ─────────────────────────────────────────────────────────────────────────
    def record_late(self, truck_id: str):
        self.late_arrivals[truck_id] = self.late_arrivals.get(truck_id, 0) + 1
        
    def record_no_show(self, gha: str, slot_start: int, truck_id: str):
        self.no_shows[truck_id] = self.no_shows.get(truck_id, 0) + 1
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id:
                slot["phase"] = "no_show"
                return

    # ─────────────────────────────────────────────────────────────────────────
    # Public helpers
    # ─────────────────────────────────────────────────────────────────────────
    def get_available_slots(self, gha: str, horizon: int = 480) -> List[int]:
        now = self.env.now
        result = []
        
        for slot_start, slots in self.registry.get(gha, {}).items():
            if slot_start - now < self.freeze_time:
                continue
            if slot_start - now > horizon:
                continue
            if any(
                slot["truck_id"] is None and slot["phase"] == "available"
                for slot in slots
            ):
                result.append(slot_start)
            return sorted(result)
        
    def get_booking(self, gha: str, truck_id: str) -> Optional[int]:
        for slot_start, slots in self.registry.get(gha, {}).items():
            for slot in slots:
                if slot["truck_id"] == truck_id and slot["phase"] in ("booked", "docked"):
                    return slot_start
        return None
    
    def count_available_slots(self, gha: str, horizon: int = 480) -> int:
        return len(self.get_available_slots(gha, horizon))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _free_slot(self, gha: str, slot_start: int, truck_id: str) -> bool:
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id:
                slot["truck_id"] = None
                slot["phase"] = "available"
                return True
        return False
    
    def _is_docked(self, gha: str, slot_start: int, truck_id: str) -> bool:
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id and slot["phase"] == "docked":
                return True
        return False