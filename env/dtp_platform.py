# =============================================================================
# DIGITAL TRUCK SLOT PLANNING MODULE
# =============================================================================
# DESCRIPTION:
#     Rule engine that manages slot reservations, validates truck arrivals
#     against their booked windows, and handles penalty tracking (no-shows).
# =============================================================================
import sys
import os
from typing import Dict, Optional, List

import simpy

# Setting base path for local imports.
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
import config.config

# =============================================================================
# SETTINGS IMPORT
# =============================================================================
params = config.load_config()

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
        
        # Registry Structure: {"gha": {slot_start: [{"truck_id": str|None, "phase": str}, ...]}}
        self.registry: Dict[str, Dict[int, List[Dict]]] = {
            gha: {} for gha in list(params["gha_docks"])
        }
        
        self.no_shows: Dict[str, int] = {}

    def publish_slot(self, gha: str, slot_start: float) -> bool:
        if gha not in self.registry:
            raise ValueError(f'GHA "{gha}" is not known, please insert a known GHA.')
        
        if slot_start - self.env.now > self.lead_time * 60:    # simpy uses seconds
            return False
        
        n_docks = params["gha"][gha]["total"]
        if len(self.registry[gha][slot_start]) >= n_docks:
            return False
        
        if slot_start not in self.registry[gha]:
            self.registry[gha][slot_start] = []
            
        self.registry[gha][slot_start].append(
            {"truck_id": None, "phase": "available"}
        )
        return True

    # NOTE: I have to implement the slot overlapping logic.
    def book_slot(self, gha: str, slot_start: float, truck_id: str) -> bool:
        if slot_start - self.env.now > self.freeze_time:
            slots = self.registry.get(gha, {}).get(slot_start, [])
            for slot in slots:
                if slot["truck_id"] is None:
                    slot["truck_id"] = truck_id
                    slot["phase"] = "booked"
                    return True
            return False
        return False

    def get_slot_phase(
        self,
        gha: str,
        book_start: Optional[int],
        arrival_time: int,
        dock_is_free: bool = False
    ) -> str:
        """
        Determines the status of a truck's arrival relative to its booked slot.
        
        Parameters:
        -----------
        gha : str
        slot_start : int
            The time of the slot start in minutes
        arrival_time : int
            The time at which the truck shown up
        is_dock_full : bool
            Passed by simulation.py to check standby status
        
        Returns:
        --------
        phase : str ("unbooked", "early", "priority", "release", "release_dock_taken", "no_show")
        """
        if book_start is None:
            return "unbooked"    # useful to call the function without creating a specific truck
          
        offset = arrival_time - book_start
        
        if offset < 0:
            return "early"
        elif offset <= self.priority_window:
            return "priority"
        elif offset <= self.slot_duration:
            if dock_is_free:
                return "release"
            return "release_dock_taken"
        return "no_show"
    
    def cancel_book(self, truck_id: str, gha: str, book_start: int) -> bool:
        if gha not in self.registry:
            raise ValueError(f'GHA "{gha}" is not known, please insert a known GHA.')
        if book_start not in self.registry[gha]:
            return False
        
        slots = self.registry[gha][book_start]
        for slot in slots:
            if slot["truck_id"] == truck_id:
                slot["phase"] = "available"
                slot["truck_id"] = None
                return True
        return False
        

    def record_no_show(self, truck_id: str):
        """Logs a no-show infraction for reward penalty and R13 enforcement."""
        self.no_shows[truck_id] = self.no_shows.get(truck_id, 0) + 1

    def is_restricted(self, truck_id: str) -> bool:
        """
        R13 Constraint Implementation.
        If a truck has 3 or more no-shows, it is barred from the DTP app.
        """
        return self.no_shows.get(truck_id, 0) >= 3