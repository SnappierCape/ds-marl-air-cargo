# =============================================================================
# DIGITAL TRUCK SLOT PLANNING MODULE
# =============================================================================
# DESCRIPTION:
#     Rule engine for the DTP platform. Manages the slot registry, enforces
#     the DTP booking rules, and answers arrival phase questions.
#
# REGISTRY STRUCTURE:
#     registry[gha][slot_start] = [
#         {"truck_id": str | None, "phase": str},
#         ...   # one entry per published slot in this window
#     ]
#
# BOOKING LIFECYCLE PHASES (live in registry):
#     "available"  →  published, no truck assigned
#     "booked"     →  truck has reserved this slot
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
import sys
import os
from typing import Dict, List, Optional

import simpy

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
import config.config as config

params = config.load_params()

# =============================================================================
# DTP PLATFORM
# =============================================================================
class DTPPlatform:
    def __init__(self, env: simpy.Environment, cfg: Dict = params):
        self.env = env
        rules = cfg["dtp_rules"]
        self.slot_duration = rules["slot_duration"]
        self.priority_window = rules["priority_window"]
        self.freeze_time = rules["freeze_time"]
        self.lead_time = rules["lead_time"]

        # Registry initialised with one empty dict per known GHA
        self.registry: Dict[str, Dict[int, List[Dict]]] = {
            gha: {} for gha in cfg["ghas"].keys()
        }

        # Penalty counters — read by reward functions in schiphol_env.py
        self.no_shows: Dict[str, int] = {}  # {truck_id: count}
        self.late_arrivals: Dict[str, int] = {}  # {truck_id: count}

    # ─────────────────────────────────────────────────────────────────────────
    # SLOT PUBLICATION — called by demand.py at episode start
    # ─────────────────────────────────────────────────────────────────────────
    def publish_slot(self, gha: str, slot_start: int) -> bool:
        self._validate_gha(gha)

        now = self.env.now
        if slot_start <= now:
            return False
        if slot_start - now >= self.lead_time:
            return False
        if slot_start - now <= self.freeze_time:
            return False

        n_docks = params["ghas"][gha]["total"]
        if self._total_published_at(gha, slot_start) >= n_docks:
            return False

        if slot_start not in self.registry[gha]:
            self.registry[gha][slot_start] = []

        self.registry[gha][slot_start].append(
            {"truck_id": None, "phase": "available"}
        )
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # BOOKING — Transporter books, Orchestrator has separate path
    # ─────────────────────────────────────────────────────────────────────────
    def book_slot(self, gha: str, slot_start: int, truck_id: str) -> bool:
        self._validate_gha(gha)

        if slot_start - self.env.now < self.freeze_time:
            return False

        return self._assign_slot(gha, slot_start, truck_id)

    def orch_book_slot(self, gha: str, slot_start: int, truck_id: str) -> bool:
        self._validate_gha(gha)
        return self._assign_slot(gha, slot_start, truck_id)

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
        truck_id:   str,
        from_gha:   str,
        from_start: int,
        to_gha:     str,
        to_start:   int,
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

        # Check space in destination before touching the source
        if self._taken_docks_at(to_gha, to_start) >= params["ghas"][to_gha]["total"]:
            return False

        # Atomic swap
        if not self._free_slot(from_gha, from_start, truck_id):
            return False

        return self._assign_slot(to_gha, to_start, truck_id)

    # ─────────────────────────────────────────────────────────────────────────
    # ARRIVAL PHASE — called by objects.py when a truck arrives at a GHA
    # ─────────────────────────────────────────────────────────────────────────
    def get_slot_phase(
        self,
        slot_start:   Optional[int],
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

        # If any entry is still "booked" (not "docked"), the truck hasn't arrived
        # If even one slot is not docked yet, we caan release it, that's why we don't need
        # to check for a specific truck_id
        return any(
            slot["phase"] == "booked"
            for slot in self.registry[gha].get(slot_start, [])
        )

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
    def get_available_slots(self, gha: str, horizon: int = 480) -> List[int]:
        self._validate_gha(gha)
        if horizon <= 0:
            raise ValueError(f'Horizon: {horizon}. Please insert positive horizon.')

        now = self.env.now
        result = []

        for slot_start, slots in self.registry[gha].items():
            if slot_start - now < self.freeze_time:
                continue
            if slot_start - now > horizon:
                continue
            if any(s["phase"] == "available" for s in slots):
                result.append(slot_start)

        return sorted(result)

    def get_booking(self, gha: str, truck_id: str) -> Optional[int]:
        """Returns the slot_start of an active booking for this truck at this GHA."""
        self._validate_gha(gha)

        for slot_start, slots in self.registry[gha].items():
            for slot in slots:
                if slot["truck_id"] == truck_id and slot["phase"] in ("booked", "docked"):
                    return slot_start
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _validate_gha(self, gha: str) -> None:
        if gha not in self.registry:
            raise ValueError(f'Unknown GHA: "{gha}". Please input valid GHA.')

    def _assign_slot(self, gha: str, slot_start: int, truck_id: str) -> bool:
        """Find the first available entry in a window and assign it."""
        for slot in self.registry[gha].get(slot_start, []):
            if slot["phase"] == "available":
                slot["truck_id"] = truck_id
                slot["phase"] = "booked"
                return True
        return False

    def _free_slot(self, gha: str, slot_start: int, truck_id: str) -> bool:
        """Reset a slot entry back to available."""
        for slot in self.registry[gha].get(slot_start, []):
            if slot["truck_id"] == truck_id:
                slot["truck_id"] = None
                slot["phase"] = "available"
                return True
        return False

    def _set_phase(self, gha: str, slot_start: int, truck_id: str, phase: str) -> None:
        """Update the phase of a specific slot entry."""
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id:
                slot["phase"] = phase
                return

    def _is_docked(self, gha: str, slot_start: int, truck_id: str) -> bool:
        """True if this truck's slot is in the docked phase."""
        for slot in self.registry[gha].get(slot_start, []):
            if slot["truck_id"] == truck_id and slot["phase"] == "docked":
                return True
        return False

    def _taken_docks_at(self, gha: str, new_start: int) -> int:
        """
        Counts docks already committed (booked or docked) during the window
        [new_start, new_start + slot_duration). Used to enforce capacity
        across overlapping slot windows.
        """
        count = 0
        new_end = new_start + self.slot_duration

        for existing_start, slots in self.registry[gha].items():
            existing_end = existing_start + self.slot_duration
            # Check temporal overlap
            if existing_start < new_end and new_start < existing_end:
                count += sum(
                    1 for s in slots
                    if s["phase"] in ("booked", "docked")
                )
        return count

    def _total_published_at(self, gha: str, new_start: int) -> int:
        """
        Counts all slots (available, booked, or docked) during the window.
        This represents the total number of docks 'claimed' by the platform.
        """
        count = 0
        new_end = new_start + self.slot_duration

        for existing_start, slots in self.registry[gha].items():
            existing_end = existing_start + self.slot_duration
            
            # Check for temporal overlap
            if existing_start < new_end and new_start < existing_end:
                count += sum(
                    1 for s in slots
                    if s["phase"] in ("available", "booked", "docked")
                )
        return count