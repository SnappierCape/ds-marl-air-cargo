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
import config.config as config

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
        
        # Registry Structure: {"gha": {time: [{"truck_id": str|None, "phase": str}, ...]}}
        self.registry: Dict[str, Dict[int, List[Dict]]] = {
            gha: {} for gha in list(params["gha_docks"])
        }
        
        self.no_shows: Dict[str, int] = {}

    def publish_slot(self, gha: str, time: float) -> bool:
        if gha not in self.registry:
            raise ValueError(f'GHA "{gha}" is not known, please insert a known GHA.')
        
        if time - self.env.now > self.lead_time * 60:    # simpy uses seconds
            return False
        
        n_docks = params["gha"][gha]["total"]
        if len(self.registry[gha][time]) >= n_docks:
            return False
        
        if time not in self.registry[gha]:
            self.registry[gha][time] = []
            
        self.registry[gha][time].append(
            {"truck_id": None, "phase": "available"}
        )
        return True

    def book_slot(self, gha: str, time: float, truck_id: str) -> bool:
        slots = self.registry.get(gha, {}).get(time, [])
        for slot in slots:
            if slot["truck_id"] is None:
                slot["truck_id"] = truck_id
                slot["phase"] = "booked"
                return True
        return False

    def get_slot_phase(self, gha_id: str, slot_start: Optional[float], arrival_time: float, is_dock_full: bool = False) -> str:
        """
        Determines the status of a truck's arrival relative to its booked slot.
        
        Parameters:
        -----------
        gha_id : str
        slot_start : float (The timestamp the slot begins, e.g., 480 for 08:00)
        arrival_time : float (Current simulation time)
        is_dock_full : bool (Passed by simulation.py to check standby status)
        
        Returns:
        --------
        phase : str ("unbooked", "early", "on_time", "late", "late_dock_full", "no_show")
        """
        if slot_start is None:
            return "unbooked"
            
        # Define slot window bounds (Assuming a 30-minute window width)
        # Truck is allowed to arrive 15 minutes early.
        window_start = slot_start - 15.0  
        window_release = slot_start + 15.0 # After 15 mins, slot given to standby
        window_end = slot_start + 30.0    # Absolute end of slot
        
        if arrival_time < window_start:
            return "early"
            
        elif window_start <= arrival_time <= window_release:
            return "on_time"
            
        elif window_release < arrival_time <= window_end:
            # Truck arrived in the second half of its slot. 
            # If a standby truck took the dock, they are redirected to TP3.
            if is_dock_full:
                return "late_dock_full"
            return "late"
            
        else:
            return "no_show"

    def record_no_show(self, truck_id: str):
        """Logs a no-show infraction for reward penalty and R13 enforcement."""
        self.no_shows[truck_id] = self.no_shows.get(truck_id, 0) + 1

    def is_restricted(self, truck_id: str) -> bool:
        """
        R13 Constraint Implementation.
        If a truck has 3 or more no-shows, it is barred from the DTP app.
        """
        return self.no_shows.get(truck_id, 0) >= 3