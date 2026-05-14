# =============================================================================
# TEST ─ DTP_PLATFORM
# =============================================================================

import sys
import os
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.dtp_platform import DTPPlatform

PASS = "[PASS]"
FAIL = "[FAIL]"

class MockEnv:
    def __init__(self, t=0):
        self.now = t

# ── Helpers ───────────────────────────────────────────────────────────────────
def fresh(t=0):
    """Return a clean DTPPlatform at sim-time t."""
    return DTPPlatform(MockEnv(t))

def publish_and_book(dtp, gha, slot, truck_id):
    dtp.publish_slot(gha, slot)
    return dtp.book_slot(gha, slot, truck_id)

print("=" * 65)
print("  TEST SUITE: DTPPlatform")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# [1] publish_slot
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] publish_slot")

# valid slot
dtp = fresh(0)
ok = dtp.publish_slot("dnata", 100)
print(f"  {PASS if ok else FAIL}  valid slot t+100 accepted: {ok}")

# past slot
ok = dtp.publish_slot("dnata", -10)
print(f"  {PASS if not ok else FAIL}  past slot rejected: {not ok}")

# within freeze window (freeze=90)
ok = dtp.publish_slot("dnata", 50)
print(f"  {PASS if not ok else FAIL}  slot inside freeze window (t+50 < 90) rejected: {not ok}")

# beyond lead time (lead=4320)
ok = dtp.publish_slot("dnata", 5000)
print(f"  {PASS if not ok else FAIL}  slot beyond lead time (t+5000 > 4320) rejected: {not ok}")

# at exact freeze boundary (exactly 90 — still frozen)
ok = dtp.publish_slot("dnata", 90)
print(f"  {PASS if not ok else FAIL}  slot at exact freeze boundary (t+90) rejected: {not ok}")

# capacity enforcement — dnata has 50 docks, fill them all
dtp2 = fresh(0)
published = 0
for i in range(52):
    if dtp2.publish_slot("dnata", 100):
        published += 1
print(f"  {PASS if published == 50 else FAIL}  capacity cap at 50 docks: published={published}/52 attempts")

# invalid GHA
try:
    fresh(0).publish_slot("ghost_gha", 200)
    print(f"  {FAIL}  unknown GHA should raise ValueError")
except ValueError as e:
    print(f"  {PASS}  unknown GHA raised ValueError: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# [2] book_slot
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] book_slot")

dtp = fresh(0)
dtp.publish_slot("klm", 200)

ok = dtp.book_slot("klm", 200, "TRK-001")
print(f"  {PASS if ok else FAIL}  valid booking accepted: {ok}")

# double booking same slot same truck
ok2 = dtp.book_slot("klm", 200, "TRK-002")
print(f"  {PASS if not ok2 else FAIL}  overbooking rejected (no free slot left): {not ok2}")

# booking within freeze window
dtp3 = fresh(0)
dtp3.publish_slot("klm", 200)
dtp3.env.now = 120   # advance time so slot 200 is now 80 min away (< freeze=90)
ok3 = dtp3.book_slot("klm", 200, "TRK-010")
print(f"  {PASS if not ok3 else FAIL}  booking inside freeze window rejected: {not ok3}")

# booking nonexistent slot
dtp4 = fresh(0)
ok4 = dtp4.book_slot("dnata", 999, "TRK-099")
print(f"  {PASS if not ok4 else FAIL}  booking nonexistent slot rejected: {not ok4}")

# ─────────────────────────────────────────────────────────────────────────────
# [3] orch_book_slot — bypasses freeze window
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] orch_book_slot")

dtp = fresh(0)
dtp.publish_slot("wfs", 200)
dtp.env.now = 150   # slot now only 50 min away (inside freeze)

ok = dtp.orch_book_slot("wfs", 200, "TRK-ORCH")
print(f"  {PASS if ok else FAIL}  orchestrator bypasses freeze window: {ok}")

# nonexistent slot
ok2 = dtp.orch_book_slot("wfs", 9999, "TRK-ORCH2")
print(f"  {PASS if not ok2 else FAIL}  orchestrator on nonexistent slot rejected: {not ok2}")

# ─────────────────────────────────────────────────────────────────────────────
# [4] cancel_book
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] cancel_book")

dtp = fresh(0)
dtp.publish_slot("swiss", 300)
dtp.book_slot("swiss", 300, "TRK-C01")

ok = dtp.cancel_book("swiss", 300, "TRK-C01")
print(f"  {PASS if ok else FAIL}  valid cancellation accepted: {ok}")

# slot is now available again
avail = dtp.get_available_slots("swiss", horizon=500)
print(f"  {PASS if 300 in avail else FAIL}  slot 300 back to available after cancel: {300 in avail}")

# cancel inside freeze window
dtp2 = fresh(0)
dtp2.publish_slot("swiss", 300)
dtp2.book_slot("swiss", 300, "TRK-C02")
dtp2.env.now = 220   # 300-220=80 < freeze=90
ok2 = dtp2.cancel_book("swiss", 300, "TRK-C02")
print(f"  {PASS if not ok2 else FAIL}  cancel inside freeze window rejected: {not ok2}")

# cancel nonexistent booking
dtp3 = fresh(0)
dtp3.publish_slot("swiss", 300)
ok3 = dtp3.cancel_book("swiss", 300, "GHOST")
print(f"  {PASS if not ok3 else FAIL}  cancel nonexistent booking rejected: {not ok3}")

# ─────────────────────────────────────────────────────────────────────────────
# [5] orch_cancel_book
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] orch_cancel_book")

# inside freeze window — orchestrator can still cancel
dtp = fresh(0)
dtp.publish_slot("dnata", 200)
dtp.book_slot("dnata", 200, "TRK-OC1")
dtp.env.now = 120   # inside freeze
ok = dtp.orch_cancel_book("dnata", 200, "TRK-OC1")
print(f"  {PASS if ok else FAIL}  orchestrator cancels inside freeze window: {ok}")

# cannot cancel if truck is already docked
dtp2 = fresh(0)
dtp2.publish_slot("dnata", 200)
dtp2.book_slot("dnata", 200, "TRK-OC2")
dtp2.mark_docked("dnata", 200, "TRK-OC2")
ok2 = dtp2.orch_cancel_book("dnata", 200, "TRK-OC2")
print(f"  {PASS if not ok2 else FAIL}  orchestrator blocked from cancelling docked truck: {not ok2}")

# ─────────────────────────────────────────────────────────────────────────────
# [6] orch_modify_book
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] orch_modify_book")

dtp = fresh(0)
dtp.publish_slot("klm", 200)
dtp.publish_slot("klm", 300)
dtp.book_slot("klm", 200, "TRK-M1")

ok = dtp.orch_modify_book("TRK-M1", "klm", 200, "klm", 300)
booking = dtp.get_booking("klm", "TRK-M1")
print(f"  {PASS if ok and booking == 300 else FAIL}  booking moved from 200 to 300: ok={ok} new_slot={booking}")

# cannot move docked truck
dtp2 = fresh(0)
dtp2.publish_slot("klm", 200)
dtp2.publish_slot("klm", 300)
dtp2.book_slot("klm", 200, "TRK-M2")
dtp2.mark_docked("klm", 200, "TRK-M2")
ok2 = dtp2.orch_modify_book("TRK-M2", "klm", 200, "klm", 300)
print(f"  {PASS if not ok2 else FAIL}  modify blocked for docked truck: {not ok2}")

# destination full
dtp3 = fresh(0)
dtp3.publish_slot("wfs", 200)
dtp3.publish_slot("wfs", 400)
dtp3.book_slot("wfs", 200, "TRK-M3")
# fill destination (wfs has 30 docks — publish 30 slots at 400 and book them)
for i in range(30):
    dtp3.publish_slot("wfs", 400)
    dtp3.book_slot("wfs", 400, f"TRK-FILL-{i}")
ok3 = dtp3.orch_modify_book("TRK-M3", "wfs", 200, "wfs", 400)
print(f"  {PASS if not ok3 else FAIL}  modify rejected when destination is full: {not ok3}")

# ─────────────────────────────────────────────────────────────────────────────
# [7] get_slot_phase
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] get_slot_phase")

dtp = fresh(0)
cases = [
    # (slot_start, arrival, dock_free, expected_phase)
    (100, 90,  True,  "early"),
    (100, 100, True,  "priority"),
    (100, 108, True,  "priority"),
    (100, 110, True,  "priority"),           # boundary: priority_window=10
    (100, 111, True,  "release"),
    (100, 130, True,  "release"),
    (100, 111, False, "release_dock_taken"),
    (100, 144, False, "release_dock_taken"),  # slot_end = 100+45=145
    (100, 146, True,  "no_show"),
    (100, 200, True,  "no_show"),
    (None, 50, True,  "unbooked"),
]
for slot_start, arrival, dock_free, expected in cases:
    result = dtp.get_slot_phase(slot_start, arrival, dock_free)
    tag = PASS if result == expected else FAIL
    print(f"  {tag}  slot={str(slot_start):<5} arrival={arrival:<5} "
          f"free={str(dock_free):<5} -> {result:<22} (expected {expected})")

# ─────────────────────────────────────────────────────────────────────────────
# [8] release_to_standby
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] release_to_standby")

# In release window with a booked (not docked) truck -> True
dtp = fresh(0)
dtp.publish_slot("dnata", 100)
dtp.book_slot("dnata", 100, "TRK-R1")
dtp.env.now = 115   # 100+10=110 <= 115 <= 100+45=145
ok = dtp.release_to_standby("dnata", 100)
print(f"  {PASS if ok else FAIL}  release in window with booked truck: {ok}")

# Priority window — not yet release time
dtp2 = fresh(0)
dtp2.publish_slot("dnata", 100)
dtp2.book_slot("dnata", 100, "TRK-R2")
dtp2.env.now = 105   # 105 < 110 (priority_window end)
ok2 = dtp2.release_to_standby("dnata", 100)
print(f"  {PASS if not ok2 else FAIL}  no release in priority window: {not ok2}")

# Truck already docked — should return False (phase != booked)
dtp3 = fresh(0)
dtp3.publish_slot("dnata", 100)
dtp3.book_slot("dnata", 100, "TRK-R3")
dtp3.mark_docked("dnata", 100, "TRK-R3")
dtp3.env.now = 115
ok3 = dtp3.release_to_standby("dnata", 100)
print(f"  {PASS if not ok3 else FAIL}  no release when truck already docked: {not ok3}")

# After slot expiry
dtp4 = fresh(0)
dtp4.publish_slot("dnata", 100)
dtp4.book_slot("dnata", 100, "TRK-R4")
dtp4.env.now = 200   # > 100+45=145
ok4 = dtp4.release_to_standby("dnata", 100)
print(f"  {PASS if not ok4 else FAIL}  no release after slot expiry: {not ok4}")

# ─────────────────────────────────────────────────────────────────────────────
# [9] mark_docked / mark_closed — phase transitions
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] mark_docked / mark_closed")

dtp = fresh(0)
dtp.publish_slot("wfs", 200)
dtp.book_slot("wfs", 200, "TRK-D1")

dtp.mark_docked("wfs", 200, "TRK-D1")
phase_after_dock = dtp.registry["wfs"][200][0]["phase"]
print(f"  {PASS if phase_after_dock == 'docked' else FAIL}  phase after mark_docked: '{phase_after_dock}'")

dtp.mark_closed("wfs", 200, "TRK-D1")
phase_after_close = dtp.registry["wfs"][200][0]["phase"]
print(f"  {PASS if phase_after_close == 'closed' else FAIL}  phase after mark_closed: '{phase_after_close}'")

# mark_docked on nonexistent truck — silent no-op, no crash
try:
    dtp.mark_docked("wfs", 200, "GHOST")
    print(f"  {PASS}  mark_docked on unknown truck_id is silent no-op")
except Exception as e:
    print(f"  {FAIL}  unexpected error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# [10] record_late / record_no_show — penalty counters
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] record_late / record_no_show")

dtp = fresh(0)
dtp.record_late("TRK-L1")
dtp.record_late("TRK-L1")
dtp.record_late("TRK-L2")
print(f"  {PASS if dtp.late_arrivals['TRK-L1'] == 2 else FAIL}  TRK-L1 late count=2: {dtp.late_arrivals['TRK-L1']}")
print(f"  {PASS if dtp.late_arrivals['TRK-L2'] == 1 else FAIL}  TRK-L2 late count=1: {dtp.late_arrivals['TRK-L2']}")

dtp.publish_slot("klm", 200)
dtp.book_slot("klm", 200, "TRK-NS1")
dtp.record_no_show("klm", 200, "TRK-NS1")
phase = dtp.registry["klm"][200][0]["phase"]
count = dtp.no_shows.get("TRK-NS1", 0)
print(f"  {PASS if count == 1 and phase == 'no_show' else FAIL}  no_show counter=1 and phase='no_show': count={count} phase={phase}")

# multiple no-shows same truck
dtp.publish_slot("klm", 300)
dtp.book_slot("klm", 300, "TRK-NS1")
dtp.record_no_show("klm", 300, "TRK-NS1")
print(f"  {PASS if dtp.no_shows['TRK-NS1'] == 2 else FAIL}  repeated no_show increments: count={dtp.no_shows['TRK-NS1']}")

# ─────────────────────────────────────────────────────────────────────────────
# [11] get_available_slots
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] get_available_slots")

dtp = fresh(0)
for t in [100, 200, 300, 400]:
    dtp.publish_slot("swiss", t)

avail = dtp.get_available_slots("swiss", horizon=4320)
print(f"  {PASS if set(avail) == {100,200,300,400} else FAIL}  all 4 published slots visible: {avail}")

# horizon filter
avail_short = dtp.get_available_slots("swiss", horizon=250)
print(f"  {PASS if set(avail_short) == {100,200} else FAIL}  horizon=250 filters to [100,200]: {avail_short}")

# booked slot disappears from available
dtp.book_slot("swiss", 100, "TRK-A1")
avail_after = dtp.get_available_slots("swiss", horizon=4320)
print(f"  {PASS if 100 not in avail_after else FAIL}  booked slot 100 removed from available: {avail_after}")

# frozen slots excluded
dtp2 = fresh(0)
dtp2.publish_slot("swiss", 200)
dtp2.env.now = 115   # 200-115=85 < freeze=90
avail2 = dtp2.get_available_slots("swiss", horizon=500)
print(f"  {PASS if 200 not in avail2 else FAIL}  frozen slot excluded from available: {avail2}")

# invalid horizon
try:
    dtp.get_available_slots("swiss", horizon=0)
    print(f"  {FAIL}  horizon=0 should raise ValueError")
except ValueError as e:
    print(f"  {PASS}  horizon=0 raised ValueError: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# [12] get_booking
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] get_booking")

dtp = fresh(0)
dtp.publish_slot("dnata", 150)
dtp.book_slot("dnata", 150, "TRK-GB1")

slot = dtp.get_booking("dnata", "TRK-GB1")
print(f"  {PASS if slot == 150 else FAIL}  get_booking returns correct slot: {slot}")

# no booking
slot2 = dtp.get_booking("dnata", "TRK-GHOST")
print(f"  {PASS if slot2 is None else FAIL}  get_booking returns None for unknown truck: {slot2}")

# closed booking not returned
dtp.mark_docked("dnata", 150, "TRK-GB1")
dtp.mark_closed("dnata", 150, "TRK-GB1")
slot3 = dtp.get_booking("dnata", "TRK-GB1")
print(f"  {PASS if slot3 is None else FAIL}  get_booking returns None after slot closed: {slot3}")

# docked booking IS still returned
dtp2 = fresh(0)
dtp2.publish_slot("dnata", 150)
dtp2.book_slot("dnata", 150, "TRK-GB2")
dtp2.mark_docked("dnata", 150, "TRK-GB2")
slot4 = dtp2.get_booking("dnata", "TRK-GB2")
print(f"  {PASS if slot4 == 150 else FAIL}  get_booking returns slot while docked: {slot4}")

# ─────────────────────────────────────────────────────────────────────────────
# [13] _validate_gha
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] _validate_gha")

dtp = fresh(0)
for bad in ["fakeGHA", "", "DNATA", "Dnata", "menzies"]:
    try:
        dtp._validate_gha(bad)
        print(f"  {FAIL}  '{bad}' should raise ValueError")
    except ValueError as e:
        print(f"  {PASS}  '{bad}' raised ValueError: {e}")

for good in ["dnata", "klm", "wfs", "swiss"]:
    try:
        dtp._validate_gha(good)
        print(f"  {PASS}  '{good}' is valid GHA — no error")
    except ValueError:
        print(f"  {FAIL}  '{good}' unexpectedly raised ValueError")

# ─────────────────────────────────────────────────────────────────────────────
# [14] _taken_docks_at — overlapping windows
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14] _taken_docks_at — overlapping window capacity")

dtp = fresh(0)
# Publish and book 3 slots all overlapping at t=200 (slot_duration=45 -> window 200-245)
for i in range(3):
    dtp.publish_slot("dnata", 200)
    dtp.book_slot("dnata", 200, f"TRK-T{i}")

count = dtp._taken_docks_at("dnata", 200)
print(f"  {PASS if count == 3 else FAIL}  3 overlapping booked slots counted: {count}")

# Non-overlapping slot (at 300, window 300-345 does not overlap 200-245)
count2 = dtp._taken_docks_at("dnata", 300)
print(f"  {PASS if count2 == 0 else FAIL}  non-overlapping slot at 300 gives 0: {count2}")

# Partially overlapping (slot at 230, window 230-275 overlaps 200-245)
dtp.publish_slot("dnata", 230)
dtp.book_slot("dnata", 230, "TRK-OVERLAP")
count3 = dtp._taken_docks_at("dnata", 230)
print(f"  {PASS if count3 == 4 else FAIL}  slot 230 sees 4 overlapping docks (3 from 200 + itself): {count3}")

print("\n" + "=" * 65)
print("  DTPPlatform tests complete")
print("=" * 65)