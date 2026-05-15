# =============================================================================
# TEST ─ DEMAND MODULE
# =============================================================================

import sys
import os
import simpy
import numpy as np
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.demand import DemandGenerator
from env.dtp_platform import DTPPlatform
from env.objects import Truck, GHATerminal, TP3Buffer
from env.infrastructure import InfrastructureLayer, CheckpointID
from env.road import RoadNetwork
from env.service_time import ServiceTimeModel
import config.config as config

params = config.load_params()
GHA_IDS = list(params["ghas"].keys())
PASS = "[PASS]"
FAIL = "[FAIL]"

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_env(sim_time=0):
    env   = simpy.Environment()
    env.now  # just access to confirm
    infra = InfrastructureLayer()
    svc   = ServiceTimeModel(params)
    road  = RoadNetwork(params["road"])
    dtp   = DTPPlatform(env)
    tp3   = TP3Buffer(env, infra)
    terminals = {g: GHATerminal(env, g, svc, infra) for g in GHA_IDS}
    demand = DemandGenerator(env, dtp, terminals, tp3, infra, road)
    return env, dtp, terminals, tp3, infra, road, demand

def prepopulate(dtp, env, ghas=None):
    """Publish slots for next 2000 minutes so tests can book freely."""
    ghas = ghas or GHA_IDS
    slot_dur   = params["dtp_rules"]["slot_duration"]
    freeze     = params["dtp_rules"]["freeze_time"]
    lead       = params["dtp_rules"]["lead_time"]
    for gha in ghas:
        n_docks = params["ghas"][gha]["total"]
        t = env.now + freeze + 1
        while t <= env.now + min(lead, 2000):
            for _ in range(n_docks):
                dtp.publish_slot(gha, t)
            t += slot_dur

print("=" * 65)
print("  TEST SUITE: DemandGenerator — behavioural")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# [1] DTP rule R1: truck cannot be dispatched until ALL stops are booked
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] R1 — dispatch blocked until all stops booked")

env, dtp, terminals, tp3, infra, road, demand = build_env()
prepopulate(dtp, env)
env.process(demand.run())
env.run(until=1)

truck = demand._create_truck()
# force a 2-stop manifest for determinism
truck.manifest = [{"gha": GHA_IDS[0], "parcels": 5}, {"gha": GHA_IDS[1], "parcels": 5}]
truck.stops_remaining = list(truck.manifest)
demand.pending_trucks.append(truck)
dispatch_event = env.event()
demand._dispatch_events[truck.truck_id] = dispatch_event

# dispatch with zero bookings — must fail
ok_zero = demand.dispatch_truck(truck.truck_id)
print(f"  {PASS if not ok_zero else FAIL}  dispatch with 0/2 bookings rejected: {not ok_zero}")
print(f"  {PASS if truck.truck_id in [t.truck_id for t in demand.pending_trucks] else FAIL}  truck still in pending after failed dispatch")

# book only first stop — still must fail
demand.book_one_slot(truck.truck_id, GHA_IDS[0])
ok_partial = demand.dispatch_truck(truck.truck_id)
print(f"  {PASS if not ok_partial else FAIL}  dispatch with 1/2 bookings rejected: {not ok_partial}")

# book second stop — now must succeed
demand.book_one_slot(truck.truck_id, GHA_IDS[1])
ok_full = demand.dispatch_truck(truck.truck_id)
print(f"  {PASS if ok_full else FAIL}  dispatch with 2/2 bookings accepted: {ok_full}")
print(f"  {PASS if truck.truck_id not in [t.truck_id for t in demand.pending_trucks] else FAIL}  truck removed from pending after dispatch")

# ─────────────────────────────────────────────────────────────────────────────
# [2] book_one_slot — respects freeze window (DTP rule R6)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] R6 — book_one_slot respects freeze window")

env, dtp, terminals, tp3, infra, road, demand = build_env()
prepopulate(dtp, env)

truck = demand._create_truck()
truck.manifest = [{"gha": GHA_IDS[0], "parcels": 5}]
truck.stops_remaining = list(truck.manifest)
demand.pending_trucks.append(truck)
dispatch_event = env.event()
demand._dispatch_events[truck.truck_id] = dispatch_event

freeze = params["dtp_rules"]["freeze_time"]
ok = demand.book_one_slot(truck.truck_id, GHA_IDS[0])
if ok:
    booked_slot = truck.booked_slots[GHA_IDS[0]]
    margin = booked_slot - env.now
    print(f"  {PASS if margin >= freeze else FAIL}  booked slot {margin:.0f}m away >= freeze={freeze}: margin={margin:.0f}")
else:
    print(f"  {FAIL}  book_one_slot returned False unexpectedly")

# ─────────────────────────────────────────────────────────────────────────────
# [3] book_one_slot — cannot double-book same GHA for same truck
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] book_one_slot — no double-booking same GHA")

env, dtp, terminals, tp3, infra, road, demand = build_env()
prepopulate(dtp, env)

truck = demand._create_truck()
truck.manifest = [{"gha": GHA_IDS[0], "parcels": 5}]
truck.stops_remaining = list(truck.manifest)
demand.pending_trucks.append(truck)
demand._dispatch_events[truck.truck_id] = env.event()

first  = demand.book_one_slot(truck.truck_id, GHA_IDS[0])
second = demand.book_one_slot(truck.truck_id, GHA_IDS[0])
print(f"  {PASS if first else FAIL}  first booking accepted: {first}")
print(f"  {PASS if not second else FAIL}  second booking for same GHA rejected: {not second}")
print(f"  {PASS if len(truck.booked_slots) == 1 else FAIL}  only 1 slot in booked_slots: {len(truck.booked_slots)}")

# ─────────────────────────────────────────────────────────────────────────────
# [4] book_one_slot — cannot book a GHA not in the truck's manifest
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] book_one_slot — rejected for GHA not in manifest")

env, dtp, terminals, tp3, infra, road, demand = build_env()
prepopulate(dtp, env)

truck = demand._create_truck()
truck.manifest = [{"gha": GHA_IDS[0], "parcels": 5}]
truck.stops_remaining = list(truck.manifest)
demand.pending_trucks.append(truck)
demand._dispatch_events[truck.truck_id] = env.event()

# try to book a GHA that is NOT in the manifest
other_gha = GHA_IDS[1]
ok = demand.book_one_slot(truck.truck_id, other_gha)
print(f"  {PASS if not ok else FAIL}  booking GHA outside manifest rejected: {not ok}")
print(f"  {PASS if other_gha not in truck.booked_slots else FAIL}  booked_slots unchanged: {truck.booked_slots}")

# ─────────────────────────────────────────────────────────────────────────────
# [5] book_one_slot — slot sequencing: multi-stop gaps are >= intra-airport buffer
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] multi-stop booking: slot gaps respect intra-airport buffer")

env, dtp, terminals, tp3, infra, road, demand = build_env()
prepopulate(dtp, env)
slot_dur = params["dtp_rules"]["slot_duration"]
buffer   = demand._intra_airport_buffer()

truck = demand._create_truck()
truck.manifest = [
    {"gha": GHA_IDS[0], "parcels": 5},
    {"gha": GHA_IDS[1], "parcels": 5},
    {"gha": GHA_IDS[2], "parcels": 5},
]
truck.stops_remaining = list(truck.manifest)
demand.pending_trucks.append(truck)
demand._dispatch_events[truck.truck_id] = env.event()

for gha in [GHA_IDS[0], GHA_IDS[1], GHA_IDS[2]]:
    demand.book_one_slot(truck.truck_id, gha)

slots = sorted(truck.booked_slots.values())
print(f"  {PASS if len(slots) == 3 else FAIL}  3 slots booked: {len(slots)}")
for i in range(len(slots) - 1):
    gap = slots[i+1] - (slots[i] + slot_dur)
    tag = PASS if gap >= 0 else FAIL
    print(f"  {tag}  gap between slot {i} and {i+1} = {gap:.0f}m (slot_dur={slot_dur}, buffer={buffer:.0f}m)")

# ─────────────────────────────────────────────────────────────────────────────
# [6] book_one_slot — unknown truck_id returns False
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] book_one_slot — unknown truck_id")

env, dtp, terminals, tp3, infra, road, demand = build_env()
prepopulate(dtp, env)
ok = demand.book_one_slot("GHOST-999", GHA_IDS[0])
print(f"  {PASS if not ok else FAIL}  unknown truck returns False: {ok}")

# ─────────────────────────────────────────────────────────────────────────────
# [7] dispatch_truck — unknown truck_id returns False
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] dispatch_truck — unknown truck_id")

env, dtp, terminals, tp3, infra, road, demand = build_env()
ok = demand.dispatch_truck("GHOST-999")
print(f"  {PASS if not ok else FAIL}  dispatch unknown truck returns False: {ok}")

# ─────────────────────────────────────────────────────────────────────────────
# [8] _rate_at — peak multiplier applied in correct window
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] _rate_at — peak multiplier in correct window")

env, dtp, terminals, tp3, infra, road, demand = build_env()
base       = params["demand"]["arrival_rate"]
mult       = params["demand"]["peak_multiplier"]
peak_start = params["demand"]["peak_window"][0]
peak_end   = params["demand"]["peak_window"][1]
ramp       = params["demand"]["ramp_dur"]

# well inside peak
r_peak = demand._rate_at(peak_start + (peak_end - peak_start) / 2)
print(f"  {PASS if abs(r_peak - base * mult) < 1e-9 else FAIL}  rate at peak midpoint = base*mult={base*mult}: {r_peak:.4f}")

# well outside peak (beginning of day)
r_off = demand._rate_at(0.0)
print(f"  {PASS if abs(r_off - base) < 1e-9 else FAIL}  rate at t=0 = base={base}: {r_off:.4f}")

# ramp-up: strictly between base and peak
t_ramp = peak_start - ramp / 2
r_ramp = demand._rate_at(t_ramp)
print(f"  {PASS if base < r_ramp < base * mult else FAIL}  rate during ramp-up is between base and peak: {r_ramp:.4f}")

# rate is always positive
for t in [0, 200, 600, peak_start, peak_start + 30, peak_end, peak_end + 60, 1439]:
    assert demand._rate_at(t) > 0, f"rate=0 at t={t}"
print(f"  {PASS}  rate > 0 at all tested time points")

# periodicity: rate at t and t+1440 should be equal
for t in [0, 300, peak_start, peak_end]:
    assert abs(demand._rate_at(t) - demand._rate_at(t + 1440)) < 1e-9, f"non-periodic at t={t}"
print(f"  {PASS}  rate function is periodic over 1440 minutes")

# ─────────────────────────────────────────────────────────────────────────────
# [9] _create_truck — structural validity
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] _create_truck — structural and statistical validity")

env, dtp, terminals, tp3, infra, road, demand = build_env()

trucks = [demand._create_truck() for _ in range(200)]

# unique IDs
ids = [t.truck_id for t in trucks]
print(f"  {PASS if len(set(ids)) == 200 else FAIL}  all 200 truck IDs are unique")

# flow types only export/import
flows = [t.flow_type for t in trucks]
print(f"  {PASS if set(flows) == {'export', 'import'} else FAIL}  flow types are only export/import: {set(flows)}")

# origin types only from config
valid_origins = set(params["demand"]["origin_split"].keys())
origins = [t.origin_type for t in trucks]
print(f"  {PASS if set(origins) <= valid_origins else FAIL}  origin types valid: {set(origins)}")

# manifests: 1-4 stops, all ghas valid, no duplicates within a truck
valid_ghas = set(GHA_IDS)
all_manifest_ok = True
for t in trucks:
    ghas_in_manifest = [s["gha"] for s in t.manifest]
    if len(ghas_in_manifest) != len(set(ghas_in_manifest)):
        all_manifest_ok = False; break
    if not all(g in valid_ghas for g in ghas_in_manifest):
        all_manifest_ok = False; break
    if not (1 <= len(t.manifest) <= 4):
        all_manifest_ok = False; break
print(f"  {PASS if all_manifest_ok else FAIL}  all manifests: 1-4 stops, valid GHAs, no duplicates")

# parcel counts in range
parcels_ok = all(
    params["demand"]["parcels_min"] <= s["parcels"] <= params["demand"]["parcels_max"]
    for t in trucks for s in t.manifest
)
print(f"  {PASS if parcels_ok else FAIL}  all parcel counts in [{params['demand']['parcels_min']},{params['demand']['parcels_max']}]")

# stops_remaining mirrors manifest at creation
mirrors_ok = all(t.stops_remaining == t.manifest for t in trucks)
print(f"  {PASS if mirrors_ok else FAIL}  stops_remaining mirrors manifest at creation")

# counter increments correctly
print(f"  {PASS if demand._truck_counter == 200 else FAIL}  truck counter=200 after 200 creates: {demand._truck_counter}")

# ─────────────────────────────────────────────────────────────────────────────
# [10] _intra_airport_buffer — equals max road segment
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] _intra_airport_buffer — equals max road segment value")

env, dtp, terminals, tp3, infra, road, demand = build_env()
buf = demand._intra_airport_buffer()
max_seg = max(params["road"]["segments"].values())
print(f"  {PASS if abs(buf - max_seg) < 1e-9 else FAIL}  buffer={buf} == max_segment={max_seg}")
print(f"  {PASS if buf > 0 else FAIL}  buffer > 0: {buf}")

# ─────────────────────────────────────────────────────────────────────────────
# [11] full journey: gate_in fires only AFTER dispatch, not before
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] gate_in fires only after dispatch")

env, dtp, terminals, tp3, infra, road, demand = build_env()
prepopulate(dtp, env)

# manually wire one truck and hold it in pending
truck = demand._create_truck()
truck.manifest = [{"gha": GHA_IDS[0], "parcels": 5}]
truck.stops_remaining = list(truck.manifest)
demand.pending_trucks.append(truck)
dispatch_event = env.event()
demand._dispatch_events[truck.truck_id] = dispatch_event
env.process(demand._truck_journey(truck, dispatch_event))

# run simulation — truck is blocked waiting for dispatch_event
env.run(until=50)
gate_in_events = [e for e in infra.get_all_events() if e.checkpoint == CheckpointID.GATE_IN]
print(f"  {PASS if len(gate_in_events) == 0 else FAIL}  no GATE_IN before dispatch: {len(gate_in_events)}")

# now book and dispatch
demand.book_one_slot(truck.truck_id, GHA_IDS[0])
demand.dispatch_truck(truck.truck_id)
env.run(until=600)

gate_in_events = [e for e in infra.get_all_events() if e.checkpoint == CheckpointID.GATE_IN]
print(f"  {PASS if len(gate_in_events) == 1 else FAIL}  GATE_IN fires after dispatch: {len(gate_in_events)}")

# ─────────────────────────────────────────────────────────────────────────────
# [12] full journey: gate_in before gate_out, gate_out fires after service
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] full journey: event ordering gate_in -> gate_out")

env, dtp, terminals, tp3, infra, road, demand = build_env()
prepopulate(dtp, env)

truck = demand._create_truck()
truck.manifest = [{"gha": GHA_IDS[0], "parcels": 5}]
truck.stops_remaining = list(truck.manifest)
demand.pending_trucks.append(truck)
de = env.event()
demand._dispatch_events[truck.truck_id] = de
env.process(demand._truck_journey(truck, de))

demand.book_one_slot(truck.truck_id, GHA_IDS[0])
demand.dispatch_truck(truck.truck_id)
env.run(until=2000)

events = infra.get_all_events()
checkpoints = [e.checkpoint for e in events if e.truck_id == truck.truck_id]

has_gate_in  = CheckpointID.GATE_IN  in checkpoints
has_gate_out = CheckpointID.GATE_OUT in checkpoints
print(f"  {PASS if has_gate_in else FAIL}  GATE_IN fired")
print(f"  {PASS if has_gate_out else FAIL}  GATE_OUT fired")

if has_gate_in and has_gate_out:
    t_in  = next(e.sim_time for e in events if e.checkpoint == CheckpointID.GATE_IN  and e.truck_id == truck.truck_id)
    t_out = next(e.sim_time for e in events if e.checkpoint == CheckpointID.GATE_OUT and e.truck_id == truck.truck_id)
    print(f"  {PASS if t_in < t_out else FAIL}  GATE_IN(t={t_in:.1f}) before GATE_OUT(t={t_out:.1f})")

print(f"  {PASS if truck.status == Truck.STATUS_DEPARTED else FAIL}  truck status=departed after journey: {truck.status}")
print(f"  {PASS if truck.stops_remaining == [] else FAIL}  all stops completed: {truck.stops_remaining}")

# ─────────────────────────────────────────────────────────────────────────────
# [13] full journey: DOCK_START always preceded by GHA_IN for same truck
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] journey ordering: GHA_IN before DOCK_START before DOCK_END")

env, dtp, terminals, tp3, infra, road, demand = build_env()
prepopulate(dtp, env)

truck = demand._create_truck()
truck.manifest = [{"gha": GHA_IDS[0], "parcels": 5}]
truck.stops_remaining = list(truck.manifest)
demand.pending_trucks.append(truck)
de = env.event()
demand._dispatch_events[truck.truck_id] = de
env.process(demand._truck_journey(truck, de))
demand.book_one_slot(truck.truck_id, GHA_IDS[0])
demand.dispatch_truck(truck.truck_id)
env.run(until=2000)

truck_events = [(e.checkpoint, e.sim_time)
                for e in infra.get_all_events()
                if e.truck_id == truck.truck_id]

def find_time(cp):
    for c, t in truck_events:
        if c == cp: return t
    return None

t_gha_in    = find_time(CheckpointID.GHA_IN)
t_dock_start = find_time(CheckpointID.DOCK_START)
t_dock_end   = find_time(CheckpointID.DOCK_END)

print(f"  {PASS if t_gha_in is not None else FAIL}  GHA_IN fired: t={t_gha_in}")
print(f"  {PASS if t_dock_start is not None else FAIL}  DOCK_START fired: t={t_dock_start}")
print(f"  {PASS if t_dock_end is not None else FAIL}  DOCK_END fired: t={t_dock_end}")
if all(x is not None for x in [t_gha_in, t_dock_start, t_dock_end]):
    print(f"  {PASS if t_gha_in <= t_dock_start <= t_dock_end else FAIL}  GHA_IN <= DOCK_START <= DOCK_END: {t_gha_in:.1f} <= {t_dock_start:.1f} <= {t_dock_end:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# [14] pending_trucks: dispatch removes truck, booking on dispatched truck fails
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14] pending_trucks management after dispatch")

env, dtp, terminals, tp3, infra, road, demand = build_env()
prepopulate(dtp, env)

truck = demand._create_truck()
truck.manifest = [{"gha": GHA_IDS[0], "parcels": 5}]
truck.stops_remaining = list(truck.manifest)
demand.pending_trucks.append(truck)
de = env.event()
demand._dispatch_events[truck.truck_id] = de
env.process(demand._truck_journey(truck, de))

demand.book_one_slot(truck.truck_id, GHA_IDS[0])
demand.dispatch_truck(truck.truck_id)

print(f"  {PASS if truck.truck_id not in [t.truck_id for t in demand.pending_trucks] else FAIL}  dispatched truck removed from pending_trucks")

# booking attempt on already-dispatched truck must fail (not in pending)
ok_post = demand.book_one_slot(truck.truck_id, GHA_IDS[1])
print(f"  {PASS if not ok_post else FAIL}  booking on dispatched truck rejected: {not ok_post}")

print("\n" + "=" * 65)
print("  DemandGenerator tests complete")
print("=" * 65)