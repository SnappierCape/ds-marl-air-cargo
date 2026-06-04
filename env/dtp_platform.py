# =============================================================================
# DIGITAL TRUCK SLOT PLANNING MODULE
# =============================================================================
# DESCRIPTION:
#     Rule engine for the DTP platform. Manages the slot registry, enforces
#     the DTP booking rules, and answers arrival phase questions.
#
# REGISTRY STRUCTURE:
#     registry[gha][slot_start] = {
#         "available": {flow_type: int},           # uncommitted dock capacity
#         "bookings":  {truck_id: {"phase": str,   # active/terminal bookings
#                                  "flow_type": str}}
#     }
#
#     truck_index[truck_id][gha] = slot_start      # reverse index; only
#                                                  # "booked" / "docked" trucks
#
# BOOKING LIFECYCLE PHASES (live in registry):
#     "available"  →  published, no truck assigned   (tracked in "available" counter)
#     "booked"     →  truck has reserved this slot   (tracked in "bookings" dict)
#     "docked"     →  truck is physically at the dock
#     "closed"     →  service complete
#     "no_show"    →  truck never appeared within the window
#
# ARRIVAL PHASES (returned by get_slot_phase, consumed by objects.py):
#     "early"              →  arrived before the slot window opened
#     "priority"           →  arrived in minutes 0-10, dock held
#     "release"            →  arrived in minutes 11-45, dock still free
#     "release_dock_taken" →  arrived in minutes 11-45, dock given to standby
#     "no_show"            →  arrived after slot expiry
#     "unbooked"           →  truck has no booking for this GHA
# =============================================================================
from typing import Dict, List, Optional

import simpy

from config.config import load_params
params = load_params()

# =============================================================================
# DTP PLATFORM
# =============================================================================
class DTPPlatform:
    def __init__(self, env: simpy.Environment, cfg: Dict = params):
        self.env = env
        self.cfg = cfg
        self.slot_duration = cfg["dtp_rules"]["slot_duration"]
        self.priority_window = cfg["dtp_rules"]["priority_window"]
        self.freeze_time = cfg["dtp_rules"]["freeze_time"]
        self.lead_time = cfg["dtp_rules"]["lead_time"]

        # Registry initialised with one empty dict per known GHA.
        # Each slot_start maps to {"available": {flow_type: int},
        #                          "bookings":  {truck_id: {phase, flow_type}}}
        self.registry: Dict[str, Dict[int, Dict]] = {
            gha: {} for gha in cfg["ghas"].keys()
        }

        # Reverse index: truck_id → {gha → slot_start}.
        # Contains only trucks in "booked" or "docked" state.
        # Maintained by _assign_slot, _free_slot, and _set_phase.
        self.truck_index: Dict[str, Dict[str, int]] = {}

        # Penalty counters — read by reward functions in schiphol_env.py
        self.no_shows: Dict[str, int] = {}       # {truck_id: count}
        self.late_arrivals: Dict[str, int] = {}  # {truck_id: count}

    # ─────────────────────────────────────────────────────────────────────────
    # SLOT PUBLICATION
    # ─────────────────────────────────────────────────────────────────────────
    def publish_slot(self, gha: str, slot_start: int, flow_type: str) -> bool:
        self._validate_gha(gha)

        now = self.env.now
        if slot_start <= now:
            return False
        if slot_start - now >= self.lead_time:
            return False
        if slot_start - now <= self.freeze_time:
            return False

        n_docks = self.cfg["ghas"][gha][flow_type]
        if self._total_published_at(gha, slot_start, flow_type) >= n_docks:
            return False

        if slot_start not in self.registry[gha]:
            self.registry[gha][slot_start] = {"available": {}, "bookings": {}}

        window = self.registry[gha][slot_start]
        window["available"][flow_type] = window["available"].get(flow_type, 0) + 1
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # BOOKING
    # ─────────────────────────────────────────────────────────────────────────
    def book_slot(self, gha: str, slot_start: int, truck_id: str, flow_type: str) -> bool:
        self._validate_gha(gha)

        if slot_start - self.env.now < self.freeze_time:
            return False

        return self._assign_slot(gha, slot_start, truck_id, flow_type)

    def orch_book_slot(self, gha: str, slot_start: int, truck_id: str, flow_type: str) -> bool:
        self._validate_gha(gha)
        return self._assign_slot(gha, slot_start, truck_id, flow_type)

    # ─────────────────────────────────────────────────────────────────────────
    # CANCELLATION
    # ─────────────────────────────────────────────────────────────────────────
    def cancel_book(self, gha: str, slot_start: int, truck_id: str) -> bool:
        self._validate_gha(gha)

        if slot_start - self.env.now < self.freeze_time:
            return False

        return self._free_slot(gha, slot_start, truck_id)

    def orch_cancel_book(self, gha: str, slot_start: int, truck_id: str) -> bool:
        self._validate_gha(gha)

        if self._is_docked(gha, slot_start, truck_id):
            return False

        return self._free_slot(gha, slot_start, truck_id)

    # ─────────────────────────────────────────────────────────────────────────
    # MODIFICATION
    # ─────────────────────────────────────────────────────────────────────────
    def orch_modify_book(
        self,
        truck_id: str,
        from_gha: str,
        from_start: int,
        to_gha: str,
        to_start: int,
        flow_type: str
    ) -> bool:
        """
        Orchestrator moves a booking to a different slot or GHA.
        Atomic: the original is freed only if the new slot is successfully booked.
        Blocked if truck is already docked.
        """
        self._validate_gha(from_gha)
        self._validate_gha(to_gha)

        if self._is_docked(from_gha, from_start, truck_id):
            return False

        if from_gha == to_gha and from_start == to_start:
            return True
        
        # Check destination availability
        taken_slots = self._taken_docks_at(to_gha, to_start, flow_type)
        
        # Check overlapping
        from_end = from_start + self.slot_duration
        to_end = to_start + self.slot_duration
        
        if from_gha == to_gha and (from_start < to_end and to_start < from_end):
            # Own booking will be vacated
            taken_slots -= 1
            
        if taken_slots >= self.cfg["ghas"][to_gha][flow_type]:
            return False
        
        # Save old booking metadata before atomic swap
        from_window = self.registry[from_gha].get(from_start, {})
        
        if truck_id not in from_window.get("bookings", {}):
            return False

        # Atomic swap
        if not self._free_slot(from_gha, from_start, truck_id):
            return False
        
        # Attemp to assign new slot
        if not self._assign_slot(to_gha, to_start, truck_id, flow_type):
            # If assignment fails, force back the truck in the original slot
            # bypassing _assign_slot()
            from_window["available"][flow_type] = max(0, from_window["available"].get(flow_type, 0) - 1)
            from_window["bookings"][truck_id] = {"phase": "booked", "flow_type": flow_type}
            self.truck_index.setdefault(truck_id, {})[from_gha] = from_start
            return False
        
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # ARRIVAL PHASE — called by objects.py when a truck arrives at a GHA
    # ─────────────────────────────────────────────────────────────────────────
    def get_slot_phase(
        self,
        slot_start: Optional[int],
        arrival_time: float,
        dock_is_free: bool,
    ) -> str:
        """Returns the arrival phase string that tells GHATerminal.process_truck() how to handle this truck."""
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

    def release_to_standby(self, gha: str, slot_start: int) -> bool:
        """
        Called by GHATerminal.release_window_watcher() at minute 10 of a slot.
        Returns True if the booked truck has not appeared — meaning the slot
        can be offered to a standby truck from TP3.
        """
        self._validate_gha(gha)

        now = self.env.now
        in_release_window = slot_start + self.priority_window <= now <= slot_start + self.slot_duration
        if not in_release_window:
            return False

        bookings = self.registry[gha].get(slot_start, {}).get("bookings", {})
        return any(v["phase"] == "booked" for v in bookings.values())

    # ─────────────────────────────────────────────────────────────────────────
    # DOCK STATE — called by objects.py to update lifecycle phase
    # ─────────────────────────────────────────────────────────────────────────
    def mark_docked(self, gha: str, slot_start: int, truck_id: str) -> None:
        """Truck has backed into the dock. Orchestrator can no longer touch it."""
        self._set_phase(gha, slot_start, truck_id, "docked")

    def mark_closed(self, gha: str, slot_start: int, truck_id: str) -> None:
        """Truck has pulled out of the dock. Slot lifecycle is complete."""
        self._set_phase(gha, slot_start, truck_id, "closed")

    # ─────────────────────────────────────────────────────────────────────────
    # PENALTY TRACKING — called by objects.py, read by schiphol_env.py
    # ─────────────────────────────────────────────────────────────────────────
    def record_late(self, truck_id: str) -> None:
        """Truck arrived in release window. Small penalty for RL feedback."""
        self.late_arrivals[truck_id] = self.late_arrivals.get(truck_id, 0) + 1

    def record_no_show(self, gha: str, slot_start: int, truck_id: str) -> None:
        """Truck never appeared within the slot window."""
        self.no_shows[truck_id] = self.no_shows.get(truck_id, 0) + 1
        self._set_phase(gha, slot_start, truck_id, "no_show")

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY HELPERS — called by demand.py and schiphol_env.py
    # ─────────────────────────────────────────────────────────────────────────
    @profile
    def get_available_slots(self, gha: str, flow_type: str, horizon: int = 480) -> List[int]:
        self._validate_gha(gha)
        if horizon <= 0:
            raise ValueError(f'Horizon: {horizon}. Please insert positive horizon.')

        now = self.env.now
        result = []

        for slot_start, window in self.registry[gha].items():
            if slot_start - now < self.freeze_time:
                continue
            if slot_start - now > horizon:
                continue
            # O(1): check the available counter directly instead of scanning the list
            if window["available"].get(flow_type, 0) > 0:
                result.append(slot_start)

        return sorted(result)

    def get_booking(self, gha: str, truck_id: str) -> Optional[int]:
        """Returns the slot_start of an active booking for this truck at this GHA."""
        self._validate_gha(gha)
        # O(1): direct reverse-index lookup instead of nested scan
        return self.truck_index.get(truck_id, {}).get(gha)
    
    def upcoming_booking_norm(self, gha: str, n_docks: int, horizon: int) -> float:
        """Fraction of the docks committed in the next 'horizon' minutes."""
        now = self.env.now
        committed = sum(
            1
            for slot_start, window in self.registry.get(gha, {}).items()
            if 0 <= slot_start - now <= horizon
            for v in window["bookings"].values()
            if v["phase"] in ("booked", "docked")
        )
        return min(committed / n_docks, 1.0) if n_docks > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _validate_gha(self, gha: str) -> None:
        if gha not in self.cfg["ghas"].keys():
            raise ValueError(f'Unknown GHA: "{gha}". Please input valid GHA.')

    def _assign_slot(self, gha: str, slot_start: int, truck_id: str, flow_type: str) -> bool:
        """Claim one available dock of the right flow type and register the truck. O(1)."""
        window = self.registry[gha].get(slot_start, {})
        avail = window.get("available", {})

        if avail.get(flow_type, 0) <= 0:
            return False

        avail[flow_type] -= 1
        window["bookings"][truck_id] = {"phase": "booked", "flow_type": flow_type}

        # Keep reverse index current
        self.truck_index.setdefault(truck_id, {})[gha] = slot_start
        return True

    def _free_slot(self, gha: str, slot_start: int, truck_id: str) -> bool:
        """Return the dock to the available pool and remove the truck entry. O(1)."""
        window = self.registry[gha].get(slot_start, {})
        bookings = window.get("bookings", {})

        if truck_id not in bookings:
            return False

        flow_type = bookings.pop(truck_id)["flow_type"]
        window["available"][flow_type] = window["available"].get(flow_type, 0) + 1

        # Remove from reverse index
        truck_ghas = self.truck_index.get(truck_id, {})
        truck_ghas.pop(gha, None)
        return True

    def _set_phase(self, gha: str, slot_start: int, truck_id: str, phase: str) -> None:
        """Update the phase of a specific booking. O(1).
        
        Removes from truck_index on terminal phases ("closed", "no_show") since
        get_booking only tracks active (booked / docked) entries.
        """
        bookings = self.registry[gha][slot_start]["bookings"]
        if truck_id not in bookings:
            return

        bookings[truck_id]["phase"] = phase

        if phase in ("closed", "no_show"):
            self.truck_index.get(truck_id, {}).pop(gha, None)

    def _is_docked(self, gha: str, slot_start: int, truck_id: str) -> bool:
        """True if this truck's slot is in the docked phase. O(1)."""
        bookings = self.registry[gha].get(slot_start, {}).get("bookings", {})
        return bookings.get(truck_id, {}).get("phase") == "docked"

    def _taken_docks_at(self, gha: str, new_start: int, flow_type: str) -> int:
        """
        Counts docks already committed (booked or docked) during the window
        [new_start, new_start + slot_duration). Used to enforce capacity
        across overlapping slot windows.
        """
        count = 0
        new_end = new_start + self.slot_duration

        for existing_start, window in self.registry[gha].items():
            existing_end = existing_start + self.slot_duration
            if existing_start < new_end and new_start < existing_end:
                count += sum(
                    1 for v in window["bookings"].values()
                    if v["phase"] in ("booked", "docked")
                    and v["flow_type"] == flow_type
                )
        return count

    def _total_published_at(self, gha: str, new_start: int, flow_type: str) -> int:
        """
        Counts all slots (available, booked, or docked) during the window.
        This represents the total number of docks 'claimed' by the platform.
        """
        count = 0
        new_end = new_start + self.slot_duration

        for existing_start, window in self.registry[gha].items():
            existing_end = existing_start + self.slot_duration
            if existing_start < new_end and new_start < existing_end:
                # Available capacity is stored as a plain counter — O(1) read
                count += window["available"].get(flow_type, 0)
                # Add committed docks (booked / docked); closed / no_show are free again
                count += sum(
                    1 for v in window["bookings"].values()
                    if v["phase"] in ("booked", "docked")
                    and v["flow_type"] == flow_type
                )
        return count