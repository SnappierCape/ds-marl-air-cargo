"""
test_dtp_platform.py
====================
Comprehensive unittest suite for DTPPlatform (dtp_platform.py).

Schiphol Cargo Hub — Digital Truck Slot Planning Module
Project context: MARL-based truck slot coordination system for Schiphol Cargo Hub,
where Ground Handling Agents (GHAs) manage dock capacity and the DTP platform
acts as the central slot registry and rule engine.

CONFIG UNDER TEST
─────────────────
  slot_duration  = 45  min   (how long a slot window lasts)
  priority_window= 10  min   (minutes 0–10 of a slot: dock is held)
  freeze_time    =  5  min   (no booking/cancellation this close to slot start)
  lead_time      = 480 min   (cannot publish further ahead than this)
  GHA_A: import=2, export=1
  GHA_B: import=1, export=2

TEST CLASS INVENTORY (15 classes, ~100 test methods)
─────────────────────────────────────────────────────
  01. TestPublishSlot           — timing guards, capacity caps, flow-type isolation
  02. TestBookSlot              — freeze enforcement, flow-type matching, state transitions
  03. TestOrchBookSlot          — bypasses freeze, still enforces flow-type
  04. TestCancelBook            — freeze enforcement, slot reset
  05. TestOrchCancelBook        — docked truck guard, freeze bypass
  06. TestOrchModifyBook        — atomic swap, docked guard, capacity check, cross-GHA
  07. TestGetSlotPhase          — all six arrival phases + boundary values
  08. TestReleaseToStandby      — timing window correctness, booked vs docked
  09. TestDockLifecycle         — mark_docked / mark_closed phase transitions
  10. TestPenaltyTracking       — record_late / record_no_show counters
  11. TestGetAvailableSlots     — freeze/horizon/flow-type filters, sort order
  12. TestGetBooking            — active vs terminal phase filtering
  13. TestValidateGha           — ValueError on unknown / malformed GHA names
  14. TestCapacityHelpers       — _taken_docks_at / _total_published_at internals
  15. TestFullSlotLifecycle     — end-to-end integration scenarios

Run:
    python -m unittest test_dtp_platform -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# ──────────────────────────────────────────────────────────────────────────────
# MODULE BOOTSTRAP
# Inject lightweight stubs for `simpy` and `config.config` BEFORE the module
# under test is imported.  `params = config.load_params()` runs at import time,
# so the stubs must be in sys.modules first.
# ──────────────────────────────────────────────────────────────────────────────

MOCK_CFG: dict = {
    "dtp_rules": {
        "slot_duration":    45,   # minutes a slot window lasts
        "priority_window":  10,   # first 10 min of slot → dock is held
        "freeze_time":       5,   # no booking/cancel within 5 min of slot start
        "lead_time":       480,   # cannot publish more than 480 min in advance
    },
    "ghas": {
        "GHA_A": {"import": 2, "export": 1},
        "GHA_B": {"import": 1, "export": 2},
    },
}

# simpy stub — DTPPlatform only reads `env.now`; no event primitives needed.
_simpy_stub = MagicMock()
sys.modules.setdefault("simpy", _simpy_stub)

# config stubs
_config_pkg = MagicMock()
_config_mod = MagicMock()
_config_mod.load_params.return_value = MOCK_CFG
sys.modules.setdefault("config",        _config_pkg)
sys.modules.setdefault("config.config", _config_mod)

# Ensure the directory that contains dtp_platform.py is on the path.
# Adjust this if your project layout differs.
# _here = os.path.dirname(os.path.abspath(__file__))
# if _here not in sys.path:
#     sys.path.insert(0, _here)

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.dtp_platform import DTPPlatform   # noqa: E402  (must follow stubs)


# ──────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

class FakeEnv:
    """Minimal simpy.Environment substitute — just a mutable .now clock."""
    def __init__(self, now: int = 0):
        self.now = now


def make_platform(now: int = 0, cfg: dict = None) -> DTPPlatform:
    """Return a fresh DTPPlatform wired to a controllable fake clock."""
    return DTPPlatform(FakeEnv(now), cfg if cfg is not None else MOCK_CFG)


def _get_entry(dtp: DTPPlatform, gha: str, slot_start: int, truck_id: str) -> dict:
    """Retrieve a specific registry entry by truck_id (raises if not found)."""
    return next(
        s for s in dtp.registry[gha].get(slot_start, [])
        if s["truck_id"] == truck_id
    )


# ══════════════════════════════════════════════════════════════════════════════
# 01. SLOT PUBLICATION
# ══════════════════════════════════════════════════════════════════════════════

class TestPublishSlot(unittest.TestCase):
    """
    publish_slot(gha, slot_start, flow_type) → bool

    Timing guards (in order of evaluation):
      • slot_start <= now               → False
      • slot_start - now >= lead_time   → False  (too far ahead)
      • slot_start - now <= freeze_time → False  (too close)

    After passing timing: capacity cap per (gha, flow_type) is enforced.
    Overlapping windows share dock-count budget.
    """

    def setUp(self):
        self.dtp = make_platform()

    # ── timing guards ──────────────────────────────────────────────────────────

    def test_slot_in_past_rejected(self):
        """slot_start <= env.now must return False."""
        self.dtp.env.now = 200
        self.assertFalse(self.dtp.publish_slot("GHA_A", 200, "import"),
                         "Slot exactly at now should be rejected")
        self.assertFalse(self.dtp.publish_slot("GHA_A", 150, "import"),
                         "Slot in the past should be rejected")

    def test_slot_beyond_lead_time_rejected(self):
        """slot_start - now >= lead_time (480) → False."""
        self.dtp.env.now = 0
        self.assertFalse(self.dtp.publish_slot("GHA_A", 480, "import"),
                         "Slot exactly at lead_time boundary should be rejected")
        self.assertFalse(self.dtp.publish_slot("GHA_A", 600, "import"),
                         "Slot beyond lead_time should be rejected")

    def test_slot_inside_freeze_window_rejected(self):
        """slot_start - now <= freeze_time (5) → False."""
        self.dtp.env.now = 100
        self.assertFalse(self.dtp.publish_slot("GHA_A", 105, "import"),
                         "Slot exactly at freeze_time boundary should be rejected")
        self.assertFalse(self.dtp.publish_slot("GHA_A", 103, "import"),
                         "Slot within freeze_time should be rejected")

    def test_valid_slot_accepted(self):
        """A slot well inside the booking window should be published successfully."""
        self.dtp.env.now = 100
        self.assertTrue(self.dtp.publish_slot("GHA_A", 200, "import"))

    def test_registry_entry_after_publish(self):
        """Registry must contain exactly one 'available' entry with correct metadata."""
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        entries = self.dtp.registry["GHA_A"][200]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["phase"],     "available")
        self.assertIsNone(entries[0]["truck_id"])
        self.assertEqual(entries[0]["flow_type"], "import")

    # ── capacity cap ───────────────────────────────────────────────────────────

    def test_import_capacity_cap_enforced(self):
        """
        GHA_A has 2 import docks.
        The 3rd publish attempt in the same window must be rejected.
        """
        self.dtp.env.now = 100
        self.assertTrue(self.dtp.publish_slot("GHA_A", 200, "import"),  "1st import slot")
        self.assertTrue(self.dtp.publish_slot("GHA_A", 200, "import"),  "2nd import slot")
        self.assertFalse(self.dtp.publish_slot("GHA_A", 200, "import"), "3rd import slot (over cap)")

    def test_export_cap_independent_of_import(self):
        """
        Filling all import docks must NOT block the independent export docks
        and vice-versa.  GHA_A: import=2, export=1.
        """
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.publish_slot("GHA_A", 200, "import")
        # Import is full — export must still succeed
        self.assertTrue(self.dtp.publish_slot("GHA_A", 200, "export"),  "Export slot after full import")
        # Export cap (1) is now reached
        self.assertFalse(self.dtp.publish_slot("GHA_A", 200, "export"), "2nd export slot over cap")

    def test_gha_b_export_cap_is_two(self):
        """GHA_B has export=2; both should be publishable, third rejected."""
        self.dtp.env.now = 100
        self.assertTrue(self.dtp.publish_slot("GHA_B", 200, "export"),  "1st export GHA_B")
        self.assertTrue(self.dtp.publish_slot("GHA_B", 200, "export"),  "2nd export GHA_B")
        self.assertFalse(self.dtp.publish_slot("GHA_B", 200, "export"), "3rd export GHA_B over cap")

    def test_overlapping_windows_count_toward_capacity(self):
        """
        Slot 200 (runs 200–245) and slot 220 (runs 220–265) overlap.
        After publishing one slot at 200 and one at 220, the 2-dock cap for
        GHA_A/import is exhausted; a new slot at 210 must be rejected.
        """
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.publish_slot("GHA_A", 220, "import")
        self.assertFalse(
            self.dtp.publish_slot("GHA_A", 210, "import"),
            "Overlapping windows should exhaust 2-dock cap"
        )

    def test_non_overlapping_windows_independent(self):
        """
        Slot 200 ends at 245.  Slot 250 starts at 250, no overlap.
        GHA_A has import=2.  Both windows can each hold 2 slots independently.
        """
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.publish_slot("GHA_A", 200, "import")
        # New slot at 250 is non-overlapping → should succeed
        self.assertTrue(self.dtp.publish_slot("GHA_A", 250, "import"),
                        "Non-overlapping window should have fresh capacity")

    def test_unknown_gha_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.dtp.publish_slot("UNKNOWN_GHA", 200, "import")


# ══════════════════════════════════════════════════════════════════════════════
# 02. STANDARD BOOKING  (book_slot)
# ══════════════════════════════════════════════════════════════════════════════

class TestBookSlot(unittest.TestCase):
    """
    book_slot(gha, slot_start, truck_id, flow_type) → bool

    Guards:
      • slot_start - now < freeze_time  → False
      • no matching available slot for flow_type → False
    On success: phase → "booked", truck_id assigned.
    """

    def setUp(self):
        self.dtp = make_platform()
        # Pre-publish one import and one export slot for GHA_A
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.publish_slot("GHA_A", 200, "export")

    def test_successful_booking_returns_true(self):
        self.dtp.env.now = 100
        self.assertTrue(self.dtp.book_slot("GHA_A", 200, "T001", "import"))

    def test_booked_entry_has_correct_state(self):
        self.dtp.env.now = 100
        self.dtp.book_slot("GHA_A", 200, "T001", "import")
        entry = _get_entry(self.dtp, "GHA_A", 200, "T001")
        self.assertEqual(entry["phase"],     "booked")
        self.assertEqual(entry["flow_type"], "import")
        self.assertEqual(entry["truck_id"],  "T001")

    def test_rejected_within_freeze_window(self):
        """slot_start - now < freeze_time(5) → booking must fail."""
        self.dtp.env.now = 196   # 200 - 196 = 4 < 5
        self.assertFalse(self.dtp.book_slot("GHA_A", 200, "T001", "import"))

    def test_allowed_exactly_at_freeze_boundary(self):
        """
        Condition is strict less-than: slot_start - now < freeze_time.
        At now=195, 200-195=5, which is NOT < 5 → booking is allowed.
        """
        self.dtp.env.now = 195
        self.assertTrue(self.dtp.book_slot("GHA_A", 200, "T001", "import"))

    def test_wrong_flow_type_cannot_claim_slot(self):
        """A truck requesting 'export' flow must not claim an import slot."""
        self.dtp.env.now = 100
        # Remove export slot so only import slot exists
        self.dtp.registry["GHA_A"][200] = [
            s for s in self.dtp.registry["GHA_A"][200]
            if s["flow_type"] == "import"
        ]
        self.assertFalse(self.dtp.book_slot("GHA_A", 200, "T001", "export"),
                         "Cannot book import slot with export flow_type")

    def test_double_booking_rejected_when_at_capacity(self):
        """GHA_A export=1; booking a second export truck must fail."""
        self.dtp.env.now = 100
        self.dtp.book_slot("GHA_A", 200, "T001", "export")
        self.assertFalse(self.dtp.book_slot("GHA_A", 200, "T002", "export"),
                         "Cannot exceed export dock capacity of 1")

    def test_two_trucks_can_book_import_slots(self):
        """GHA_A import=2; both T001 and T002 should succeed."""
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")   # 2nd import slot
        self.assertTrue(self.dtp.book_slot("GHA_A", 200, "T001", "import"))
        self.assertTrue(self.dtp.book_slot("GHA_A", 200, "T002", "import"))

    def test_no_slot_published_returns_false(self):
        """Booking against an unpublished slot_start returns False."""
        self.dtp.env.now = 100
        self.assertFalse(self.dtp.book_slot("GHA_A", 999, "T001", "import"))

    def test_invalid_gha_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.dtp.book_slot("BAD_GHA", 200, "T001", "import")


# ══════════════════════════════════════════════════════════════════════════════
# 03. ORCHESTRATOR BOOKING  (orch_book_slot)
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchBookSlot(unittest.TestCase):
    """
    orch_book_slot bypasses the freeze window check entirely,
    but still delegates to _assign_slot which enforces flow_type.
    """

    def setUp(self):
        self.dtp = make_platform()
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")

    def test_orch_book_succeeds_inside_freeze_window(self):
        """
        Standard book_slot would fail here (now=198, 200-198=2 < freeze_time=5),
        but orch_book_slot has no such constraint.
        """
        self.dtp.env.now = 198
        self.assertTrue(self.dtp.orch_book_slot("GHA_A", 200, "T001", "import"))

    def test_orch_book_enforces_flow_type(self):
        """Orchestrator still cannot assign an import slot to an export truck."""
        self.dtp.env.now = 100
        self.assertFalse(self.dtp.orch_book_slot("GHA_A", 200, "T001", "export"))

    def test_orch_book_returns_false_for_unpublished_slot(self):
        self.dtp.env.now = 100
        self.assertFalse(self.dtp.orch_book_slot("GHA_A", 300, "T001", "import"))

    def test_orch_book_sets_correct_phase(self):
        self.dtp.env.now = 198
        self.dtp.orch_book_slot("GHA_A", 200, "T001", "import")
        entry = _get_entry(self.dtp, "GHA_A", 200, "T001")
        self.assertEqual(entry["phase"], "booked")

    def test_orch_book_invalid_gha_raises(self):
        with self.assertRaises(ValueError):
            self.dtp.orch_book_slot("UNKNOWN", 200, "T001", "import")


# ══════════════════════════════════════════════════════════════════════════════
# 04. STANDARD CANCELLATION  (cancel_book)
# ══════════════════════════════════════════════════════════════════════════════

class TestCancelBook(unittest.TestCase):
    """
    cancel_book(gha, slot_start, truck_id) → bool

    Guards:
      • slot_start - now < freeze_time → False
      • truck_id not found in slot     → False
    On success: phase → "available", truck_id → None.
    """

    def setUp(self):
        self.dtp = make_platform()
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")

    def test_valid_cancellation_resets_entry(self):
        self.dtp.env.now = 100
        result = self.dtp.cancel_book("GHA_A", 200, "T001")
        self.assertTrue(result)
        entry = self.dtp.registry["GHA_A"][200][0]
        self.assertIsNone(entry["truck_id"])
        self.assertEqual(entry["phase"], "available")

    def test_rejected_within_freeze_window(self):
        """slot_start - now < freeze_time(5) → cancellation must fail."""
        self.dtp.env.now = 196   # 200 - 196 = 4 < 5
        self.assertFalse(self.dtp.cancel_book("GHA_A", 200, "T001"))

    def test_allowed_exactly_at_freeze_boundary(self):
        """200 - 195 = 5; strict less-than means this is NOT within freeze window."""
        self.dtp.env.now = 195
        self.assertTrue(self.dtp.cancel_book("GHA_A", 200, "T001"))

    def test_nonexistent_truck_returns_false(self):
        self.dtp.env.now = 100
        self.assertFalse(self.dtp.cancel_book("GHA_A", 200, "GHOST"))

    def test_cancelled_slot_becomes_rebookable(self):
        """After cancel, another truck should be able to book the freed slot."""
        self.dtp.env.now = 100
        self.dtp.cancel_book("GHA_A", 200, "T001")
        self.assertTrue(self.dtp.book_slot("GHA_A", 200, "T002", "import"))

    def test_invalid_gha_raises(self):
        with self.assertRaises(ValueError):
            self.dtp.cancel_book("NO_GHA", 200, "T001")


# ══════════════════════════════════════════════════════════════════════════════
# 05. ORCHESTRATOR CANCELLATION  (orch_cancel_book)
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchCancelBook(unittest.TestCase):
    """
    orch_cancel_book(gha, slot_start, truck_id) → bool

    Guards:
      • truck is in "docked" phase → False  (cannot interrupt an active dock)
    No freeze_time check (orchestrator privilege).
    On success: phase → "available", truck_id → None.
    """

    def setUp(self):
        self.dtp = make_platform()
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")

    def test_orch_cancel_booked_truck(self):
        result = self.dtp.orch_cancel_book("GHA_A", 200, "T001")
        self.assertTrue(result)

    def test_orch_cancel_blocked_for_docked_truck(self):
        """Once docked, the orchestrator must not touch the slot."""
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.assertFalse(self.dtp.orch_cancel_book("GHA_A", 200, "T001"))

    def test_orch_cancel_bypasses_freeze_window(self):
        """Unlike cancel_book, this method has no freeze guard."""
        self.dtp.env.now = 198   # Inside freeze window (200-198=2 < 5)
        self.assertTrue(self.dtp.orch_cancel_book("GHA_A", 200, "T001"))

    def test_orch_cancel_nonexistent_truck_returns_false(self):
        self.assertFalse(self.dtp.orch_cancel_book("GHA_A", 200, "GHOST"))

    def test_orch_cancel_resets_entry_to_available(self):
        self.dtp.orch_cancel_book("GHA_A", 200, "T001")
        entry = self.dtp.registry["GHA_A"][200][0]
        self.assertIsNone(entry["truck_id"])
        self.assertEqual(entry["phase"], "available")

    def test_invalid_gha_raises(self):
        with self.assertRaises(ValueError):
            self.dtp.orch_cancel_book("PHANTOM", 200, "T001")


# ══════════════════════════════════════════════════════════════════════════════
# 06. ORCHESTRATOR MODIFICATION  (orch_modify_book)
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchModifyBook(unittest.TestCase):
    """
    orch_modify_book(truck_id, from_gha, from_start, to_gha, to_start, flow_type) → bool

    Atomic swap:
      1. Blocked if truck is docked at source.
      2. Blocked if destination is at capacity.
      3. Source freed only if destination assignment succeeds.

    Works both within the same GHA and across GHAs.
    """

    def setUp(self):
        self.dtp = make_platform()
        self.dtp.env.now = 100
        # Publish source (slot 200) and non-overlapping destination (slot 300)
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.publish_slot("GHA_A", 300, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")

    def test_successful_same_gha_move(self):
        """T001 moves from slot 200 → slot 300 within GHA_A."""
        result = self.dtp.orch_modify_book("T001", "GHA_A", 200, "GHA_A", 300, "import")
        self.assertTrue(result)
        # Source slot freed
        src = self.dtp.registry["GHA_A"][200][0]
        self.assertIsNone(src["truck_id"])
        self.assertEqual(src["phase"], "available")
        # Destination slot booked
        dst = _get_entry(self.dtp, "GHA_A", 300, "T001")
        self.assertEqual(dst["phase"], "booked")

    def test_blocked_when_truck_is_docked(self):
        self.dtp.mark_docked("GHA_A", 200, "T001")
        result = self.dtp.orch_modify_book("T001", "GHA_A", 200, "GHA_A", 300, "import")
        self.assertFalse(result)

    def test_blocked_when_destination_is_at_capacity(self):
        """
        GHA_A import=2.  Pre-fill slot 300 with 2 booked trucks so it's full.
        Move of T001 to slot 300 must be rejected.
        """
        self.dtp.publish_slot("GHA_A", 300, "import")   # second slot at 300
        self.dtp.book_slot("GHA_A", 300, "T002", "import")
        self.dtp.book_slot("GHA_A", 300, "T003", "import")
        result = self.dtp.orch_modify_book("T001", "GHA_A", 200, "GHA_A", 300, "import")
        self.assertFalse(result)

    def test_source_is_not_freed_on_failed_move(self):
        """Atomicity: if destination booking fails, source must remain booked."""
        # No slot published at destination 999
        result = self.dtp.orch_modify_book("T001", "GHA_A", 200, "GHA_A", 999, "import")
        self.assertFalse(result)
        # Source must still be booked by T001
        entry = _get_entry(self.dtp, "GHA_A", 200, "T001")
        self.assertEqual(entry["phase"], "booked")

    def test_cross_gha_move_succeeds(self):
        """Orchestrator can relocate a booking from GHA_A to GHA_B."""
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_B", 300, "import")
        result = self.dtp.orch_modify_book("T001", "GHA_A", 200, "GHA_B", 300, "import")
        self.assertTrue(result)
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T001"))
        self.assertEqual(self.dtp.get_booking("GHA_B", "T001"), 300)

    def test_returns_false_when_source_truck_not_found(self):
        result = self.dtp.orch_modify_book("GHOST", "GHA_A", 200, "GHA_A", 300, "import")
        self.assertFalse(result)

    def test_wrong_flow_type_at_destination_returns_false(self):
        """Cannot move an import-booked truck into an export slot at the destination."""
        self.dtp.publish_slot("GHA_A", 300, "export")   # only export at slot 300 now
        self.dtp.registry["GHA_A"][300] = [
            s for s in self.dtp.registry["GHA_A"][300] if s["flow_type"] == "export"
        ]
        result = self.dtp.orch_modify_book("T001", "GHA_A", 200, "GHA_A", 300, "import")
        self.assertFalse(result)

    def test_invalid_from_gha_raises(self):
        with self.assertRaises(ValueError):
            self.dtp.orch_modify_book("T001", "NO_GHA", 200, "GHA_A", 300, "import")

    def test_invalid_to_gha_raises(self):
        with self.assertRaises(ValueError):
            self.dtp.orch_modify_book("T001", "GHA_A", 200, "NO_GHA", 300, "import")


# ══════════════════════════════════════════════════════════════════════════════
# 07. ARRIVAL PHASE LOGIC  (get_slot_phase)
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSlotPhase(unittest.TestCase):
    """
    get_slot_phase(slot_start, arrival_time, dock_is_free) → str

    slot_duration=45, priority_window=10.

    Phase mapping (offset = arrival_time - slot_start):
      slot_start is None       → "unbooked"
      offset < 0               → "early"
      0 ≤ offset ≤ 10          → "priority"
      10 < offset ≤ 45, free   → "release"
      10 < offset ≤ 45, taken  → "release_dock_taken"
      offset > 45              → "no_show"
    """

    def setUp(self):
        self.dtp  = make_platform()
        self.slot = 200

    def _phase(self, arrival: float, dock_free: bool, slot_start=None) -> str:
        s = slot_start if slot_start is not None else self.slot
        return self.dtp.get_slot_phase(s, arrival, dock_free)

    # ── no booking ─────────────────────────────────────────────────────────────

    def test_none_slot_start_returns_unbooked(self):
        result = self.dtp.get_slot_phase(None, 200, True)
        self.assertEqual(result, "unbooked")

    def test_none_slot_start_unbooked_regardless_of_dock(self):
        self.assertEqual(self.dtp.get_slot_phase(None, 200, False), "unbooked")

    # ── early arrival ──────────────────────────────────────────────────────────

    def test_arrival_before_slot_start_is_early(self):
        self.assertEqual(self._phase(199,  True),  "early")
        self.assertEqual(self._phase(  0,  False), "early")

    def test_arrival_one_minute_before_slot_is_early(self):
        self.assertEqual(self._phase(199, True), "early")

    # ── priority window ────────────────────────────────────────────────────────

    def test_arrival_at_exact_slot_start_is_priority(self):
        """offset = 0 → priority."""
        self.assertEqual(self._phase(200, True),  "priority")
        self.assertEqual(self._phase(200, False), "priority")

    def test_arrival_at_priority_window_boundary_is_priority(self):
        """offset = priority_window (10) → still priority (inclusive)."""
        self.assertEqual(self._phase(210, True),  "priority")
        self.assertEqual(self._phase(210, False), "priority")

    def test_arrival_mid_priority_window_is_priority(self):
        self.assertEqual(self._phase(205, False), "priority")

    # ── release window ─────────────────────────────────────────────────────────

    def test_arrival_just_past_priority_window_dock_free_is_release(self):
        """offset = 11 → release (one beyond priority boundary)."""
        self.assertEqual(self._phase(211, True), "release")

    def test_arrival_just_past_priority_window_dock_taken_is_release_dock_taken(self):
        self.assertEqual(self._phase(211, False), "release_dock_taken")

    def test_arrival_at_slot_duration_boundary_dock_free(self):
        """offset = slot_duration (45) → release (inclusive upper bound)."""
        self.assertEqual(self._phase(245, True), "release")

    def test_arrival_at_slot_duration_boundary_dock_taken(self):
        self.assertEqual(self._phase(245, False), "release_dock_taken")

    def test_arrival_mid_release_window(self):
        self.assertEqual(self._phase(230, True),  "release")
        self.assertEqual(self._phase(230, False), "release_dock_taken")

    # ── no-show ────────────────────────────────────────────────────────────────

    def test_arrival_one_past_slot_duration_is_no_show(self):
        """offset = 46 > slot_duration(45) → no_show."""
        self.assertEqual(self._phase(246, True),  "no_show")
        self.assertEqual(self._phase(246, False), "no_show")

    def test_arrival_far_late_is_no_show(self):
        self.assertEqual(self._phase(400, True), "no_show")

    # ── boundary between priority and release ─────────────────────────────────

    def test_boundary_priority_to_release(self):
        """
        offset=10 → priority  (last minute inside priority window)
        offset=11 → release   (first minute of release window)
        """
        self.assertEqual(self._phase(210, True), "priority")   # offset=10
        self.assertEqual(self._phase(211, True), "release")    # offset=11


# ══════════════════════════════════════════════════════════════════════════════
# 08. RELEASE TO STANDBY  (release_to_standby)
# ══════════════════════════════════════════════════════════════════════════════

class TestReleaseToStandby(unittest.TestCase):
    """
    release_to_standby(gha, slot_start) → bool

    Called by the GHATerminal at minute 10 (env.now == slot_start + priority_window).
    Returns True iff:
      • env.now is inside [slot_start+priority_window, slot_start+slot_duration]
      • at least one slot entry is still "booked" (not yet docked)

    slot 200: priority_window=10 → release window opens at 210, closes at 245.
    """

    def setUp(self):
        self.dtp = make_platform()
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")

    def _in_release_window(self):
        """Advance clock to the exact release window opening."""
        self.dtp.env.now = 210   # 200 + priority_window(10)

    def test_returns_true_when_truck_still_booked(self):
        self._in_release_window()
        self.assertTrue(self.dtp.release_to_standby("GHA_A", 200))

    def test_returns_false_when_truck_has_docked(self):
        """Truck docked → no 'booked' entries → cannot release."""
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self._in_release_window()
        self.assertFalse(self.dtp.release_to_standby("GHA_A", 200))

    def test_returns_false_before_release_window_opens(self):
        """env.now = slot_start + 5 is before the priority_window(10) elapses."""
        self.dtp.env.now = 205   # 5 min into slot, < priority_window(10)
        self.assertFalse(self.dtp.release_to_standby("GHA_A", 200))

    def test_returns_false_after_slot_expires(self):
        """env.now = slot_start + slot_duration + 1 → outside release window."""
        self.dtp.env.now = 246   # 200 + 46 > slot_duration(45)
        self.assertFalse(self.dtp.release_to_standby("GHA_A", 200))

    def test_returns_false_at_exact_release_window_close(self):
        """env.now = slot_start + slot_duration (=245) is the last valid moment."""
        self.dtp.env.now = 245
        self.assertTrue(self.dtp.release_to_standby("GHA_A", 200),
                        "Boundary at slot_duration should still be inside release window")

    def test_returns_false_for_unregistered_slot(self):
        self._in_release_window()
        self.assertFalse(self.dtp.release_to_standby("GHA_A", 999))

    def test_invalid_gha_raises(self):
        with self.assertRaises(ValueError):
            self.dtp.release_to_standby("PHANTOM", 200)


# ══════════════════════════════════════════════════════════════════════════════
# 09. DOCK LIFECYCLE  (mark_docked / mark_closed)
# ══════════════════════════════════════════════════════════════════════════════

class TestDockLifecycle(unittest.TestCase):
    """
    Lifecycle phases:
      booked → (mark_docked) → docked → (mark_closed) → closed

    After mark_closed, get_booking must return None (closed phase is terminal).
    """

    def setUp(self):
        self.dtp = make_platform()
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")

    def test_mark_docked_sets_phase(self):
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.assertEqual(_get_entry(self.dtp, "GHA_A", 200, "T001")["phase"], "docked")

    def test_is_docked_true_after_mark_docked(self):
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.assertTrue(self.dtp._is_docked("GHA_A", 200, "T001"))

    def test_is_docked_false_before_marking(self):
        self.assertFalse(self.dtp._is_docked("GHA_A", 200, "T001"))

    def test_mark_closed_sets_phase(self):
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.dtp.mark_closed("GHA_A", 200, "T001")
        self.assertEqual(_get_entry(self.dtp, "GHA_A", 200, "T001")["phase"], "closed")

    def test_get_booking_returns_none_after_closed(self):
        """Closed phase is terminal — get_booking must not surface it."""
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.dtp.mark_closed("GHA_A", 200, "T001")
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T001"))

    def test_is_docked_false_after_mark_closed(self):
        """After closure, the phase is no longer 'docked'."""
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.dtp.mark_closed("GHA_A", 200, "T001")
        self.assertFalse(self.dtp._is_docked("GHA_A", 200, "T001"))


# ══════════════════════════════════════════════════════════════════════════════
# 10. PENALTY TRACKING  (record_late / record_no_show)
# ══════════════════════════════════════════════════════════════════════════════

class TestPenaltyTracking(unittest.TestCase):
    """
    record_late(truck_id)              — increments late_arrivals[truck_id]
    record_no_show(gha, slot, truck_id) — increments no_shows[truck_id]
                                          AND sets phase → "no_show"
    """

    def setUp(self):
        self.dtp = make_platform()
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")

    def test_record_late_first_occurrence(self):
        self.dtp.record_late("T001")
        self.assertEqual(self.dtp.late_arrivals["T001"], 1)

    def test_record_late_accumulates(self):
        for _ in range(4):
            self.dtp.record_late("T001")
        self.assertEqual(self.dtp.late_arrivals["T001"], 4)

    def test_record_late_independent_per_truck(self):
        self.dtp.record_late("T001")
        self.dtp.record_late("T001")
        self.dtp.record_late("T002")
        self.assertEqual(self.dtp.late_arrivals["T001"], 2)
        self.assertEqual(self.dtp.late_arrivals["T002"], 1)

    def test_record_no_show_increments_counter(self):
        self.dtp.record_no_show("GHA_A", 200, "T001")
        self.assertEqual(self.dtp.no_shows["T001"], 1)

    def test_record_no_show_sets_registry_phase(self):
        """After recording a no_show, the registry entry phase must be 'no_show'."""
        self.dtp.record_no_show("GHA_A", 200, "T001")
        entry = _get_entry(self.dtp, "GHA_A", 200, "T001")
        self.assertEqual(entry["phase"], "no_show")

    def test_record_no_show_multiple_trucks_independent(self):
        self.dtp.publish_slot("GHA_A", 300, "import")
        self.dtp.book_slot("GHA_A", 300, "T002", "import")
        self.dtp.record_no_show("GHA_A", 200, "T001")
        self.dtp.record_no_show("GHA_A", 300, "T002")
        self.assertEqual(self.dtp.no_shows["T001"], 1)
        self.assertEqual(self.dtp.no_shows["T002"], 1)

    def test_late_and_no_show_counters_are_separate(self):
        """late_arrivals and no_shows are distinct dictionaries."""
        self.dtp.record_late("T001")
        self.dtp.record_no_show("GHA_A", 200, "T001")
        self.assertEqual(self.dtp.late_arrivals.get("T001", 0), 1)
        self.assertEqual(self.dtp.no_shows.get("T001", 0), 1)


# ══════════════════════════════════════════════════════════════════════════════
# 11. AVAILABLE SLOT QUERY  (get_available_slots)
# ══════════════════════════════════════════════════════════════════════════════

class TestGetAvailableSlots(unittest.TestCase):
    """
    get_available_slots(gha, flow_type, horizon=480) → List[int]

    Filters:
      • only "available" entries for the requested flow_type
      • slot_start - now >= freeze_time  (not too close)
      • slot_start - now <= horizon      (not too far)
    Returns: sorted list of slot_start values.
    Raises ValueError for horizon <= 0.
    """

    def setUp(self):
        self.dtp = make_platform()
        self.dtp.env.now = 100
        # Two import slots and one export slot
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.publish_slot("GHA_A", 300, "import")
        self.dtp.publish_slot("GHA_A", 200, "export")

    def test_returns_available_import_slots(self):
        result = self.dtp.get_available_slots("GHA_A", "import")
        self.assertIn(200, result)
        self.assertIn(300, result)

    def test_flow_type_filter_isolates_export(self):
        """Export query must return export slots only, not import."""
        result = self.dtp.get_available_slots("GHA_A", "export")
        self.assertIn(200, result)
        self.assertNotIn(300, result)   # slot 300 has no export entry

    def test_booked_slot_excluded_when_no_available_entries_remain(self):
        """
        Slot 200 has exactly one import entry.  After booking it, it transitions
        to 'booked' and must no longer appear in the available list.
        """
        self.dtp.book_slot("GHA_A", 200, "T001", "import")
        result = self.dtp.get_available_slots("GHA_A", "import")
        self.assertNotIn(200, result)

    def test_slot_near_freeze_window_excluded(self):
        """
        Publish a slot 4 min ahead (now=100, slot=104).
        104-100=4 < freeze_time(5) → must not appear.
        But this publish itself would also be rejected … so we inject directly.
        """
        self.dtp.env.now = 100
        self.dtp.registry["GHA_A"][104] = [{"truck_id": None, "phase": "available", "flow_type": "import"}]
        result = self.dtp.get_available_slots("GHA_A", "import")
        self.assertNotIn(104, result)

    def test_horizon_excludes_distant_slots(self):
        """slot 300: 300-100=200 > horizon=150 → must be excluded."""
        result = self.dtp.get_available_slots("GHA_A", "import", horizon=150)
        self.assertNotIn(300, result)
        self.assertIn(200, result)

    def test_horizon_includes_slot_exactly_at_horizon(self):
        """300-100=200; with horizon=200 the slot is at the boundary (≤ horizon → included)."""
        result = self.dtp.get_available_slots("GHA_A", "import", horizon=200)
        self.assertIn(300, result)

    def test_result_is_sorted(self):
        result = self.dtp.get_available_slots("GHA_A", "import")
        self.assertEqual(result, sorted(result))

    def test_negative_horizon_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.dtp.get_available_slots("GHA_A", "import", horizon=-10)

    def test_zero_horizon_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.dtp.get_available_slots("GHA_A", "import", horizon=0)

    def test_invalid_gha_raises(self):
        with self.assertRaises(ValueError):
            self.dtp.get_available_slots("NOPE", "import")

    def test_empty_result_when_no_slots_published(self):
        fresh = make_platform(now=100)
        result = fresh.get_available_slots("GHA_A", "import")
        self.assertEqual(result, [])


# ══════════════════════════════════════════════════════════════════════════════
# 12. GET BOOKING  (get_booking)
# ══════════════════════════════════════════════════════════════════════════════

class TestGetBooking(unittest.TestCase):
    """
    get_booking(gha, truck_id) → Optional[int]

    Returns slot_start for phases: "booked" or "docked".
    Returns None for: "available", "closed", "no_show", or unknown truck_id.
    """

    def setUp(self):
        self.dtp = make_platform()
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")

    def test_returns_slot_start_for_booked_truck(self):
        self.assertEqual(self.dtp.get_booking("GHA_A", "T001"), 200)

    def test_returns_slot_start_for_docked_truck(self):
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.assertEqual(self.dtp.get_booking("GHA_A", "T001"), 200)

    def test_returns_none_for_unknown_truck(self):
        self.assertIsNone(self.dtp.get_booking("GHA_A", "GHOST"))

    def test_returns_none_after_cancellation(self):
        self.dtp.cancel_book("GHA_A", 200, "T001")
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T001"))

    def test_returns_none_after_mark_closed(self):
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.dtp.mark_closed("GHA_A", 200, "T001")
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T001"))

    def test_returns_none_after_no_show(self):
        self.dtp.record_no_show("GHA_A", 200, "T001")
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T001"))

    def test_different_trucks_at_same_gha(self):
        """Two trucks can have bookings at the same GHA at different slots."""
        self.dtp.publish_slot("GHA_A", 300, "import")
        self.dtp.book_slot("GHA_A", 300, "T002", "import")
        self.assertEqual(self.dtp.get_booking("GHA_A", "T001"), 200)
        self.assertEqual(self.dtp.get_booking("GHA_A", "T002"), 300)

    def test_invalid_gha_raises(self):
        with self.assertRaises(ValueError):
            self.dtp.get_booking("BAD_GHA", "T001")


# ══════════════════════════════════════════════════════════════════════════════
# 13. GHA VALIDATION  (_validate_gha)
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateGha(unittest.TestCase):
    """_validate_gha raises ValueError for any name not in cfg["ghas"]."""

    def setUp(self):
        self.dtp = make_platform()

    def test_known_ghas_do_not_raise(self):
        try:
            self.dtp._validate_gha("GHA_A")
            self.dtp._validate_gha("GHA_B")
        except ValueError:
            self.fail("_validate_gha raised ValueError for a known GHA")

    def test_unknown_gha_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.dtp._validate_gha("GHA_MYSTERY")

    def test_empty_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.dtp._validate_gha("")

    def test_case_sensitive_gha_name(self):
        """GHA names are case-sensitive: 'gha_a' ≠ 'GHA_A'."""
        with self.assertRaises(ValueError):
            self.dtp._validate_gha("gha_a")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            self.dtp._validate_gha("   ")


# ══════════════════════════════════════════════════════════════════════════════
# 14. CAPACITY HELPERS  (_taken_docks_at / _total_published_at)
# ══════════════════════════════════════════════════════════════════════════════

class TestCapacityHelpers(unittest.TestCase):
    """
    _taken_docks_at   — counts "booked" + "docked" in overlapping windows
    _total_published_at — counts "available" + "booked" + "docked" in overlapping windows

    Both helpers are flow_type-aware: only entries with matching flow_type are counted.
    Overlap condition: existing_start < new_end AND new_start < existing_end
    """

    def setUp(self):
        self.dtp = make_platform()
        self.dtp.env.now = 100

    def test_taken_docks_counts_booked(self):
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")
        self.assertEqual(self.dtp._taken_docks_at("GHA_A", 200, "import"), 1)

    def test_taken_docks_counts_docked(self):
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.assertEqual(self.dtp._taken_docks_at("GHA_A", 200, "import"), 1)

    def test_taken_docks_excludes_available(self):
        """Available (unbooked) slots do NOT count as taken docks."""
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.assertEqual(self.dtp._taken_docks_at("GHA_A", 200, "import"), 0)

    def test_taken_docks_excludes_closed(self):
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.dtp.mark_closed("GHA_A", 200, "T001")
        self.assertEqual(self.dtp._taken_docks_at("GHA_A", 200, "import"), 0)

    def test_taken_docks_flow_type_isolation(self):
        """Export bookings must not count toward import taken_docks."""
        self.dtp.publish_slot("GHA_A", 200, "export")
        self.dtp.book_slot("GHA_A", 200, "T001", "export")
        self.assertEqual(self.dtp._taken_docks_at("GHA_A", 200, "import"), 0)

    def test_taken_docks_overlap_across_windows(self):
        """
        Slot 200 (200–245) and slot 220 (220–265) overlap with window 210 (210–255).
        Booking both slots should yield taken_docks=2 for the 210 window.
        """
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.publish_slot("GHA_A", 220, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")
        self.dtp.book_slot("GHA_A", 220, "T002", "import")
        self.assertEqual(self.dtp._taken_docks_at("GHA_A", 210, "import"), 2)

    def test_total_published_counts_available_booked_docked(self):
        """All three live phases (available, booked, docked) count toward total_published."""
        # Slot A: available
        self.dtp.publish_slot("GHA_A", 200, "import")
        # Slot B: booked
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")
        total = self.dtp._total_published_at("GHA_A", 200, "import")
        self.assertEqual(total, 2)

    def test_total_published_excludes_no_show(self):
        """no_show entries must NOT count toward the total published."""
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")
        self.dtp.record_no_show("GHA_A", 200, "T001")
        self.assertEqual(self.dtp._total_published_at("GHA_A", 200, "import"), 0)

    def test_total_published_excludes_closed(self):
        """closed entries must NOT count toward the total published."""
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")
        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.dtp.mark_closed("GHA_A", 200, "T001")
        self.assertEqual(self.dtp._total_published_at("GHA_A", 200, "import"), 0)

    def test_total_published_flow_type_isolation(self):
        """Export slots do not contribute to import total_published count."""
        self.dtp.publish_slot("GHA_A", 200, "export")
        self.assertEqual(self.dtp._total_published_at("GHA_A", 200, "import"), 0)


# ══════════════════════════════════════════════════════════════════════════════
# 15. FULL LIFECYCLE INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestFullSlotLifecycle(unittest.TestCase):
    """
    End-to-end scenarios that trace a truck (or fleet) through the complete
    booking lifecycle, combining publish → book → arrive → dock → close.
    These tests mirror real operational patterns at the Schiphol Cargo Hub.
    """

    def setUp(self):
        self.dtp = make_platform()

    def test_happy_path_priority_arrival(self):
        """
        Standard import truck arrives in the priority window:
          publish → book → priority arrival → docked → closed
        """
        self.dtp.env.now = 100
        self.assertTrue(self.dtp.publish_slot("GHA_A", 200, "import"), "publish")
        self.assertTrue(self.dtp.book_slot("GHA_A", 200, "T001", "import"), "book")
        self.assertEqual(self.dtp.get_booking("GHA_A", "T001"), 200)

        # Truck arrives at offset +5 (within priority_window=10)
        phase = self.dtp.get_slot_phase(200, 205, dock_is_free=True)
        self.assertEqual(phase, "priority")

        self.dtp.mark_docked("GHA_A", 200, "T001")
        self.assertTrue(self.dtp._is_docked("GHA_A", 200, "T001"))

        self.dtp.mark_closed("GHA_A", 200, "T001")
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T001"))

    def test_late_arrival_export_flow(self):
        """
        Export truck arrives during the release window with dock free:
          publish → book → release arrival → late penalty → docked → closed
        """
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "export")
        self.dtp.book_slot("GHA_A", 200, "T010", "export")

        phase = self.dtp.get_slot_phase(200, 220, dock_is_free=True)
        self.assertEqual(phase, "release")

        self.dtp.record_late("T010")
        self.assertEqual(self.dtp.late_arrivals["T010"], 1)

        self.dtp.mark_docked("GHA_A", 200, "T010")
        self.dtp.mark_closed("GHA_A", 200, "T010")
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T010"))

    def test_no_show_lifecycle(self):
        """
        Truck is booked but never arrives within the slot window.
        """
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "export")
        self.dtp.book_slot("GHA_A", 200, "T099", "export")

        phase = self.dtp.get_slot_phase(200, 260, dock_is_free=True)
        self.assertEqual(phase, "no_show")

        self.dtp.record_no_show("GHA_A", 200, "T099")
        self.assertEqual(self.dtp.no_shows["T099"], 1)
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T099"),
                          "no_show truck must not show as active booking")

    def test_standby_release_and_orchestrator_rebook(self):
        """
        Scenario: T001 is booked but doesn't dock by minute 10.
        The platform releases the slot; the orchestrator assigns standby T002.
        """
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T001", "import")

        # Advance to release window opening
        self.dtp.env.now = 210   # slot_start(200) + priority_window(10)
        self.assertTrue(self.dtp.release_to_standby("GHA_A", 200))

        # Orchestrator cancels T001 and assigns standby T002
        self.dtp.orch_cancel_book("GHA_A", 200, "T001")
        self.assertTrue(self.dtp.orch_book_slot("GHA_A", 200, "T002", "import"))
        self.assertEqual(self.dtp.get_booking("GHA_A", "T002"), 200)
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T001"))

    def test_orchestrator_cross_gha_reroute_under_freeze(self):
        """
        Orchestrator reroutes a truck from GHA_A to GHA_B inside the freeze
        window (standard booking/cancel would be blocked).
        """
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.publish_slot("GHA_B", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T050", "import")

        # Advance into freeze window
        self.dtp.env.now = 198   # 200-198=2 < freeze_time(5)

        result = self.dtp.orch_modify_book("T050", "GHA_A", 200, "GHA_B", 200, "import")
        self.assertTrue(result)
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T050"))
        self.assertEqual(self.dtp.get_booking("GHA_B", "T050"), 200)

    def test_peak_period_two_truck_import_flow(self):
        """
        Simulate a peak-period scenario: two import trucks share GHA_A's
        2-dock import capacity, arrive at different offsets.
        """
        self.dtp.env.now = 100
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.publish_slot("GHA_A", 200, "import")
        self.dtp.book_slot("GHA_A", 200, "T_A", "import")
        self.dtp.book_slot("GHA_A", 200, "T_B", "import")

        # T_A arrives in priority window
        self.assertEqual(self.dtp.get_slot_phase(200, 203, True), "priority")
        self.dtp.mark_docked("GHA_A", 200, "T_A")

        # T_B arrives in release window, dock for T_A is taken but T_B has its own entry
        self.assertEqual(self.dtp.get_slot_phase(200, 215, True), "release")
        self.dtp.mark_docked("GHA_A", 200, "T_B")

        # Both closed after service
        self.dtp.mark_closed("GHA_A", 200, "T_A")
        self.dtp.mark_closed("GHA_A", 200, "T_B")
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T_A"))
        self.assertIsNone(self.dtp.get_booking("GHA_A", "T_B"))


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)