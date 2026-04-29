# =============================================================================
# DIGITAL TRUCK SLOT PLANNING MODULE
# =============================================================================
# DESCRIPTION:
#     Rule engine that manages slot reservations, validates truck arrivals
#     against their booked windows, and handles penalty tracking.
#
# DTP RULES IMPLEMENTED:
#     R1:  Booking required to enter (enforced at simulation layer).
#     R2:  45-min windows, one slot = one license plate.
#     R5:  Orchestrator has full authority unless truck is docked.
#     R6:  Frozen window = 2 × slot_duration (90 min). No bookings/cancels inside.
#     R7:  GHAs may publish up to 72h ahead, not inside frozen window.
#     R8:  Priority window (0–10 min), release window (11–45 min).
#     R9:  No-show recorded when truck arrives after slot expiry.
#     R10: Transporter cancels outside frozen window only.
#     R11: GHAs cannot remove a published slot (enforced: no remove_slot method).
#
# REGISTRY STRUCTURE:
#     registry = {
#         gha_id: {
#             slot_start (float, minutes): [
#                 {"truck_id": str|None, "phase": str},
#                 ...   # one entry per available/booked slot in this window
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
# ARRIVAL PHASES (returned by get_slot_phase, consumed by simulation.py):
#     "unbooked"           →  truck has no booking for this GHA
#     "early"              →  arrived before window opens (>15 min early)
#     "priority"           →  arrived in minutes 0–10 (dock held)
#     "release"            →  arrived in minutes 11–45, dock still free
#     "release_dock_taken" →  arrived in minutes 11–45, dock given to standby
#     "no_show"            →  arrived after slot expiry (> slot_duration from start)
# =============================================================================

import sys
import os
from typing import Dict, List, Optional

import simpy

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
import config.config

# =============================================================================
# SETTINGS IMPORT
# =============================================================================
params = config.load_config()

# =============================================================================
# DTP PLATFORM
# =============================================================================
class DTPPlatform:
    """
    Truck slot booking logic engine.
    Pure Python; uses env.now (in minutes) only for time checks.
    SimPy must be configured to run in minutes throughout the project.
    """

    def __init__(
        self,
        env: simpy.Environment,
        slot_duration:    int = params["booking"]["slot_duration"],    # 45 min (R2)
        priority_window:  int = params["booking"]["priority_window"],  # 10 min (R8)
        freeze_time:      int = params["booking"]["freeze_time"],      # 90 min = 2×45 (R6)
        lead_time:        int = params["booking"]["lead_time"],        # 4320 min = 72h (R7)
    ):
        # SimPy environment runs in minutes — no conversion needed.
        self.env              = env
        self.slot_duration    = slot_duration
        self.priority_window  = priority_window
        self.freeze_time      = freeze_time      # inside this window: slots are frozen
        self.lead_time        = lead_time        # max horizon for publishing (72h)

        # Registry: {gha_id: {slot_start: [{"truck_id": ..., "phase": ...}]}}
        self.registry: Dict[str, Dict[float, List[Dict]]] = {
            gha: {} for gha in params["gha_docks"]
        }

        # Penalty tracking — consumed by reward functions
        self.no_shows:     Dict[str, int] = {}   # {truck_id: count}
        self.late_arrivals: Dict[str, int] = {}  # {truck_id: count}

    # =========================================================================
    # SLOT PUBLICATION  (R7, R11)
    # =========================================================================

    def publish_slot(self, gha: str, slot_start: float) -> bool:
        """
        GHA publishes one new available slot for the given time window.

        Rules enforced:
          R7:  Only allowed up to lead_time (72h) ahead and not inside frozen window.
          R2:  Total slots per window capped at GHA's total dock count.
          R11: GHAs cannot remove a published slot — there is no remove_slot method.

        Returns True if slot was successfully published.
        """
        if gha not in self.registry:
            raise ValueError(f"Unknown GHA: '{gha}'. Must be one of {list(self.registry)}")

        now = self.env.now

        # Reject past slots
        if slot_start <= now:
            return False

        # R7: reject if too far in the future (> 72h)
        if slot_start - now > self.lead_time:
            return False

        # R7: reject if inside frozen window (< freeze_time from now)
        # A slot inside the frozen window can no longer be published.
        if slot_start - now < self.freeze_time:
            return False

        # R2: cap at total dock count for this GHA
        n_docks = params["gha_docks"][gha]["total"]
        current_count = len(self.registry[gha].get(slot_start, []))
        if current_count >= n_docks:
            return False

        # Initialise list for this window if first slot
        if slot_start not in self.registry[gha]:
            self.registry[gha][slot_start] = []

        self.registry[gha][slot_start].append(
            {"truck_id": None, "phase": "available"}
        )
        return True

    # =========================================================================
    # BOOKING  (R2, R6)
    # =========================================================================

    def book_slot(self, gha: str, slot_start: float, truck_id: str) -> bool:
        """
        Transporter books an available slot.

        Rules enforced:
          R6: Rejected inside the frozen window (< freeze_time from now).
          R2: One slot per truck — finds first available entry and assigns.

        Returns True if booking succeeded.
        """
        if gha not in self.registry:
            raise ValueError(f"Unknown GHA: '{gha}'.")

        now = self.env.now

        # R6: frozen window check
        if slot_start - now < self.freeze_time:
            return False

        slots = self.registry[gha].get(slot_start, [])
        for slot in slots:
            if slot["truck_id"] is None and slot["phase"] == "available":
                slot["truck_id"] = truck_id
                slot["phase"]    = "booked"
                return True

        return False  # no available entry in this window

    def _internal_book_slot(self, gha: str, slot_start: float, truck_id: str) -> bool:
        """
        Private booking path that bypasses the frozen window check.
        Used ONLY by Orchestrator operations (R5).
        """
        slots = self.registry[gha].get(slot_start, [])
        for slot in slots:
            if slot["truck_id"] is None and slot["phase"] == "available":
                slot["truck_id"] = truck_id
                slot["phase"]    = "booked"
                return True
        return False

    # =========================================================================
    # CANCELLATION  (R10, R5)
    # =========================================================================

    def cancel_book(self, truck_id: str, gha: str, slot_start: float) -> bool:
        """
        Transporter cancels a booking.
        R10: Only allowed outside the frozen window.
        """
        if gha not in self.registry:
            raise ValueError(f"Unknown GHA: '{gha}'.")

        # R10: reject inside frozen window
        if slot_start - self.env.now < self.freeze_time:
            return False

        return self._free_slot(gha, slot_start, truck_id)

    def orch_cancel_book(self, truck_id: str, gha: str, slot_start: float) -> bool:
        """
        Orchestrator cancels a booking. Bypasses frozen window (R5).
        R5: Cannot cancel if truck is already docked.
        """
        if gha not in self.registry:
            raise ValueError(f"Unknown GHA: '{gha}'.")

        # R5: guard — cannot touch a docked truck
        if self._is_truck_docked(gha, slot_start, truck_id):
            return False

        return self._free_slot(gha, slot_start, truck_id)

    def orch_modify_book(
        self,
        truck_id:       str,
        from_gha:       str,
        from_slot_start: float,
        to_gha:         str,
        to_slot_start:  float,
    ) -> bool:
        """
        Orchestrator moves a booking to a different GHA/window (R5).
        Atomic: only cancels original if new booking succeeds.
        Cannot modify if truck is already docked (R5).
        """
        for gha in [from_gha, to_gha]:
            if gha not in self.registry:
                raise ValueError(f"Unknown GHA: '{gha}'.")

        # R5: guard — cannot touch a docked truck
        if self._is_truck_docked(from_gha, from_slot_start, truck_id):
            return False

        # Check new slot has capacity before touching the original
        new_slots = self.registry[to_gha].get(to_slot_start, [])
        has_space = any(
            s["truck_id"] is None and s["phase"] == "available"
            for s in new_slots
        )
        if not has_space:
            return False

        # Atomic swap: free original then book new (bypass frozen window for both)
        if not self._free_slot(from_gha, from_slot_start, truck_id):
            return False

        if not self._internal_book_slot(to_gha, to_slot_start, truck_id):
            # Rollback: restore original booking
            self._internal_book_slot(from_gha, from_slot_start, truck_id)
            return False

        return True

    # =========================================================================
    # ARRIVAL PHASE LOGIC  (R8, R9)
    # =========================================================================

    def get_slot_phase(
        self,
        slot_start:   Optional[float],
        arrival_time: float,
        dock_is_free: bool = True,
    ) -> str:
        """
        Determines which arrival phase a truck is in relative to its booked slot.
        Called by GHATerminal when a truck arrives at the GHA ANPR.

        Parameters
        ----------
        slot_start   : booked slot start time (minutes). None if unbooked.
        arrival_time : current simulation time (minutes).
        dock_is_free : True if the correct dock pool has capacity.

        Returns
        -------
        "unbooked"           — truck has no booking for this GHA
        "early"              — arrived before the slot window opens
        "priority"           — arrived in the priority window (min 0–10)
        "release"            — arrived in release window, dock free → admit
        "release_dock_taken" — arrived in release window, dock taken → TP3
        "no_show"            — arrived after slot expiry → TP3, log penalty (R9)
        """
        if slot_start is None:
            return "unbooked"

        offset = arrival_time - slot_start

        if offset < 0:
            return "early"

        elif offset <= self.priority_window:
            return "priority"

        elif offset <= self.slot_duration:
            return "release" if dock_is_free else "release_dock_taken"

        else:
            return "no_show"

    def should_release_to_standby(self, gha: str, slot_start: float) -> bool:
        """
        Returns True if a slot has entered its release window (minute 10+)
        and the booked truck has not yet appeared (phase still "booked").
        Called by GHATerminal.release_window_monitor() at t = slot_start + priority_window.
        """
        now = self.env.now
        if not (slot_start + self.priority_window <= now <= slot_start + self.slot_duration):
            return False

        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["phase"] == "booked":
                return True
        return False

    # =========================================================================
    # DOCK STATE TRACKING  (R5)
    # =========================================================================

    def mark_docked(self, truck_id: str, gha: str, slot_start: float):
        """
        Called by GHATerminal when a truck backs into the dock.
        Sets phase to "docked" — Orchestrator cannot touch this booking (R5).
        """
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id:
                slot["phase"] = "docked"
                return

    def mark_closed(self, truck_id: str, gha: str, slot_start: float):
        """Called by GHATerminal when a truck pulls out of the dock."""
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id:
                slot["phase"] = "closed"
                return

    # =========================================================================
    # PENALTY TRACKING  (R8, R9)
    # =========================================================================

    def record_no_show(self, truck_id: str, gha: str, slot_start: float):
        """R9: Truck arrived after slot expiry. Log penalty and mark slot."""
        self.no_shows[truck_id] = self.no_shows.get(truck_id, 0) + 1
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id:
                slot["phase"] = "no_show"
                return

    def record_late_arrival(self, truck_id: str):
        """R8: Truck arrived in release window. Small penalty for RL feedback."""
        self.late_arrivals[truck_id] = self.late_arrivals.get(truck_id, 0) + 1

    # =========================================================================
    # QUERY HELPERS
    # =========================================================================

    def get_available_slots(self, gha: str, horizon_min: float = 480) -> List[float]:
        """
        Returns sorted list of slot_start times that have at least one
        available (unbooked) entry, within the bookable window.
        """
        now = self.env.now
        result = []
        for slot_start, entries in self.registry.get(gha, {}).items():
            # Must be bookable (outside frozen window, within horizon)
            if slot_start - now < self.freeze_time:
                continue
            if slot_start - now > horizon_min:
                continue
            if any(s["truck_id"] is None and s["phase"] == "available" for s in entries):
                result.append(slot_start)
        return sorted(result)

    def get_booking(self, truck_id: str, gha: str) -> Optional[float]:
        """Returns slot_start if truck has an active booking at this GHA, else None."""
        for slot_start, entries in self.registry.get(gha, {}).items():
            for slot in entries:
                if slot["truck_id"] == truck_id and slot["phase"] in ("booked", "docked"):
                    return slot_start
        return None

    def count_available_slots(self, gha: str, horizon_min: float = 120) -> int:
        """Count of bookable slots within horizon — used for observation vectors."""
        return len(self.get_available_slots(gha, horizon_min))

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _free_slot(self, gha: str, slot_start: float, truck_id: str) -> bool:
        """Reset a slot entry back to available. Returns True if found."""
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id:
                slot["truck_id"] = None
                slot["phase"]    = "available"
                return True
        return False

    def _is_truck_docked(self, gha: str, slot_start: float, truck_id: str) -> bool:
        """True if the truck's slot is in 'docked' phase (R5 guard)."""
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id and slot["phase"] == "docked":
                return True
        return False