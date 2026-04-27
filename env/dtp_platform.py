# =============================================================================
# DIGITAL TRUCK SLOT PLANNING MODULE
# =============================================================================
# DESCRIPTION:
#     Rule engine that manages slot reservations, validates truck arrivals
#     against their booked windows, and handles penalty tracking (no-shows).
# =============================================================================
import simpy
from typing import Dict, Optional

# =============================================================================
# DPT PLATFORM MODEL
# =============================================================================
class DTPPlatform:
    """
    Digital Transport Platform logic engine.
    Pure Python logic; relies on the SimPy environment solely to check 'env.now'.
    """
    def __init__(self, env: simpy.Environment):
        self.env = env
        
        # Registry Structure: slots[gha_id][time_min] = {"capacity": int, "booked": int}
        self.registry: Dict[str, Dict[float, Dict]] = {
            "dnata": {}, "klm": {}, "swissport": {}, "menzies_wfs": {}
        }
        
        # Track penalties for the KPI/Reward calculation later
        self.no_shows: Dict[str, int] = {}

    def publish_slot(self, gha_id: str, time_min: float, capacity: int = 10):
        """Initializes a booking slot in the registry."""
        if gha_id in self.registry:
            self.registry[gha_id][time_min] = {
                "capacity": capacity,
                "booked": 0
            }

    def book_slot(self, gha_id: str, time_min: float) -> bool:
        """
        Attempts to reserve a slot. Returns True if successful.
        Called by Transporter Agents or the Orchestrator.
        """
        if gha_id not in self.registry or time_min not in self.registry[gha_id]:
            return False
            
        slot = self.registry[gha_id][time_min]
        if slot["booked"] < slot["capacity"]:
            slot["booked"] += 1
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