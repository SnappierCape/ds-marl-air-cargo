# =============================================================================
# TEST ─ INFRASTRUCTURE
# =============================================================================

import sys
import os
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.infrastructure import InfrastructureLayer, CheckpointID, SensorEvent

PASS = "[PASS]"
FAIL = "[FAIL]"

# ─────────────────────────────────────────────────────────────────────────────
# Mock truck
# ─────────────────────────────────────────────────────────────────────────────
class MockTruck:
    def __init__(self, truck_id="TRK-001", flow_type="export", parcels=10, slots=None):
        self.truck_id = truck_id
        self.flow_type = flow_type
        self.timestamps = {}
        self.booked_slots = slots or {"dnata": 480}
        self._parcels = parcels

    def total_parcels(self):
        return self._parcels

    def parcels_for(self, gha_id):
        return self._parcels

    def next_slot(self):
        return list(self.booked_slots.values())[0] if self.booked_slots else None

def fresh():
    return InfrastructureLayer()

print("=" * 65)
print("  TEST SUITE: InfrastructureLayer")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# [1] gate_in
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] gate_in")

infra = fresh()
truck = MockTruck("TRK-001", "export", parcels=15, slots={"klm": 300})
infra.gate_in(100.0, truck)

events = infra.get_all_events()
e = events[0]
print(f"  {PASS if len(events) == 1 else FAIL}  exactly 1 event logged: {len(events)}")
print(f"  {PASS if e.checkpoint == CheckpointID.GATE_IN else FAIL}  checkpoint is GATE_IN: {e.checkpoint}")
print(f"  {PASS if e.sim_time == 100.0 else FAIL}  sim_time=100.0: {e.sim_time}")
print(f"  {PASS if e.truck_id == 'TRK-001' else FAIL}  truck_id='TRK-001': {e.truck_id}")
print(f"  {PASS if e.flow_type == 'export' else FAIL}  flow_type='export': {e.flow_type}")
print(f"  {PASS if e.n_parcels == 15 else FAIL}  n_parcels=15: {e.n_parcels}")
print(f"  {PASS if e.slot_window == 300 else FAIL}  slot_window=300: {e.slot_window}")
print(f"  {PASS if e.gha_id is None else FAIL}  gha_id=None (gate event): {e.gha_id}")
print(f"  {PASS if e.dock_id is None else FAIL}  dock_id=None (gate event): {e.dock_id}")
print(f"  {PASS if truck.timestamps.get('gate_in') == 100.0 else FAIL}  truck.timestamps['gate_in']=100.0: {truck.timestamps.get('gate_in')}")

# multiple trucks gate_in
infra2 = fresh()
trucks = [MockTruck(f"TRK-{i:03d}", "import", parcels=i*2) for i in range(1, 6)]
for i, t in enumerate(trucks):
    infra2.gate_in(float(i * 10), t)
print(f"  {PASS if len(infra2.get_all_events()) == 5 else FAIL}  5 gate_in events logged for 5 trucks: {len(infra2.get_all_events())}")

# ─────────────────────────────────────────────────────────────────────────────
# [2] gate_out
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] gate_out")

infra = fresh()
truck = MockTruck("TRK-002", "import")
infra.gate_in(50.0, truck)
infra.gate_out(200.0, truck)

events = infra.get_all_events()
e_out = events[1]
print(f"  {PASS if e_out.checkpoint == CheckpointID.GATE_OUT else FAIL}  checkpoint is GATE_OUT: {e_out.checkpoint}")
print(f"  {PASS if e_out.sim_time == 200.0 else FAIL}  sim_time=200.0: {e_out.sim_time}")
print(f"  {PASS if e_out.n_parcels is None else FAIL}  n_parcels=None (gate_out carries no parcel count): {e_out.n_parcels}")
print(f"  {PASS if e_out.slot_window is None else FAIL}  slot_window=None: {e_out.slot_window}")
print(f"  {PASS if truck.timestamps.get('gate_out') == 200.0 else FAIL}  truck.timestamps['gate_out']=200.0: {truck.timestamps.get('gate_out')}")

# ─────────────────────────────────────────────────────────────────────────────
# [3] tp3_in
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] tp3_in")

infra = fresh()
truck = MockTruck("TRK-003", "export")
infra.tp3_in(75.0, truck)

e = infra.get_all_events()[0]
print(f"  {PASS if e.checkpoint == CheckpointID.TP3_IN else FAIL}  checkpoint is TP3_IN: {e.checkpoint}")
print(f"  {PASS if e.sim_time == 75.0 else FAIL}  sim_time=75.0: {e.sim_time}")
print(f"  {PASS if e.n_parcels is None else FAIL}  n_parcels=None: {e.n_parcels}")
print(f"  {PASS if e.gha_id is None else FAIL}  gha_id=None: {e.gha_id}")
print(f"  {PASS if truck.timestamps.get('tp3_in') == 75.0 else FAIL}  truck.timestamps['tp3_in']=75.0: {truck.timestamps.get('tp3_in')}")

# ─────────────────────────────────────────────────────────────────────────────
# [4] tp3_out
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] tp3_out")

infra = fresh()
truck = MockTruck("TRK-004", "import")
infra.tp3_in(80.0, truck)
infra.tp3_out(120.0, truck)

events = infra.get_all_events()
e_out = events[1]
print(f"  {PASS if e_out.checkpoint == CheckpointID.TP3_OUT else FAIL}  checkpoint is TP3_OUT: {e_out.checkpoint}")
print(f"  {PASS if e_out.sim_time == 120.0 else FAIL}  sim_time=120.0: {e_out.sim_time}")
print(f"  {PASS if e_out.n_parcels is None else FAIL}  n_parcels=None: {e_out.n_parcels}")
print(f"  {PASS if truck.timestamps.get('tp3_out') == 120.0 else FAIL}  truck.timestamps['tp3_out']=120.0: {truck.timestamps.get('tp3_out')}")

# ─────────────────────────────────────────────────────────────────────────────
# [5] gha_in
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] gha_in")

infra = fresh()
truck = MockTruck("TRK-005", "export", parcels=12, slots={"dnata": 480, "klm": 540})
infra.gha_in(130.0, truck, "dnata")

e = infra.get_all_events()[0]
print(f"  {PASS if e.checkpoint == CheckpointID.GHA_IN else FAIL}  checkpoint is GHA_IN: {e.checkpoint}")
print(f"  {PASS if e.gha_id == 'dnata' else FAIL}  gha_id='dnata': {e.gha_id}")
print(f"  {PASS if e.n_parcels == 12 else FAIL}  n_parcels=12: {e.n_parcels}")
print(f"  {PASS if e.slot_window == 480 else FAIL}  slot_window=480: {e.slot_window}")
print(f"  {PASS if e.dock_id is None else FAIL}  dock_id=None: {e.dock_id}")
print(f"  {PASS if truck.timestamps.get('gha_in_dnata') == 130.0 else FAIL}  truck.timestamps['gha_in_dnata']=130.0: {truck.timestamps.get('gha_in_dnata')}")

# multi-stop: second gha_in
infra.gha_in(200.0, truck, "klm")
e2 = infra.get_all_events()[1]
print(f"  {PASS if e2.gha_id == 'klm' else FAIL}  second gha_in gha_id='klm': {e2.gha_id}")
print(f"  {PASS if e2.slot_window == 540 else FAIL}  second gha_in slot_window=540: {e2.slot_window}")
print(f"  {PASS if truck.timestamps.get('gha_in_klm') == 200.0 else FAIL}  truck.timestamps['gha_in_klm']=200.0: {truck.timestamps.get('gha_in_klm')}")

# gha with no booking for that stop
truck_nobooking = MockTruck("TRK-NB", "export", slots={"wfs": 600})
infra.gha_in(250.0, truck_nobooking, "dnata")   # not in booked_slots
e3 = infra.get_all_events()[-1]
print(f"  {PASS if e3.slot_window is None else FAIL}  slot_window=None when gha not in booked_slots: {e3.slot_window}")

# ─────────────────────────────────────────────────────────────────────────────
# [6] dock_start
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] dock_start")

infra = fresh()
truck = MockTruck("TRK-006", "export", parcels=8, slots={"wfs": 480})
infra.dock_start(160.0, truck, "wfs", dock_id=3)

e = infra.get_all_events()[0]
print(f"  {PASS if e.checkpoint == CheckpointID.DOCK_START else FAIL}  checkpoint is DOCK_START: {e.checkpoint}")
print(f"  {PASS if e.gha_id == 'wfs' else FAIL}  gha_id='wfs': {e.gha_id}")
print(f"  {PASS if e.dock_id == 3 else FAIL}  dock_id=3: {e.dock_id}")
print(f"  {PASS if e.n_parcels == 8 else FAIL}  n_parcels=8: {e.n_parcels}")
print(f"  {PASS if e.slot_window == 480 else FAIL}  slot_window=480: {e.slot_window}")
print(f"  {PASS if truck.timestamps.get('dock_start_wfs') == 160.0 else FAIL}  truck.timestamps['dock_start_wfs']=160.0: {truck.timestamps.get('dock_start_wfs')}")

# multiple docks, multiple trucks
infra2 = fresh()
for dock_id in range(5):
    t = MockTruck(f"TRK-D{dock_id}", "import", parcels=5, slots={"swiss": 500})
    infra2.dock_start(float(dock_id * 5), t, "swiss", dock_id=dock_id)
events = infra2.get_all_events()
dock_ids_logged = [e.dock_id for e in events]
print(f"  {PASS if dock_ids_logged == list(range(5)) else FAIL}  dock_ids [0..4] correctly logged: {dock_ids_logged}")

# ─────────────────────────────────────────────────────────────────────────────
# [7] dock_end
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] dock_end")

infra = fresh()
truck = MockTruck("TRK-007", "import", slots={"klm": 400})
infra.dock_start(200.0, truck, "klm", dock_id=1)
infra.dock_end(240.0, truck, "klm", dock_id=1)

events = infra.get_all_events()
e_end = events[1]
print(f"  {PASS if e_end.checkpoint == CheckpointID.DOCK_END else FAIL}  checkpoint is DOCK_END: {e_end.checkpoint}")
print(f"  {PASS if e_end.sim_time == 240.0 else FAIL}  sim_time=240.0: {e_end.sim_time}")
print(f"  {PASS if e_end.n_parcels is None else FAIL}  n_parcels=None (already logged at dock_start): {e_end.n_parcels}")
print(f"  {PASS if e_end.gha_id == 'klm' else FAIL}  gha_id='klm': {e_end.gha_id}")
print(f"  {PASS if e_end.dock_id == 1 else FAIL}  dock_id=1: {e_end.dock_id}")
print(f"  {PASS if truck.timestamps.get('dock_end_klm') == 240.0 else FAIL}  truck.timestamps['dock_end_klm']=240.0: {truck.timestamps.get('dock_end_klm')}")

# ─────────────────────────────────────────────────────────────────────────────
# [8] flush_step_buffer
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] flush_step_buffer")

infra = fresh()
t1 = MockTruck("TRK-F1", "export", slots={"dnata": 100})
t2 = MockTruck("TRK-F2", "import", slots={"klm": 200})

infra.gate_in(10.0, t1)
infra.gate_in(20.0, t2)
infra.tp3_in(30.0, t1)

# first flush should return 3 events
buf1 = infra.flush_step_buffer()
print(f"  {PASS if len(buf1) == 3 else FAIL}  flush returns 3 events (step 1): {len(buf1)}")

# buffer should be empty now
buf2 = infra.flush_step_buffer()
print(f"  {PASS if len(buf2) == 0 else FAIL}  second flush returns 0 events: {len(buf2)}")

# full log still has all 3
all_ev = infra.get_all_events()
print(f"  {PASS if len(all_ev) == 3 else FAIL}  event_log retains all 3 events after flush: {len(all_ev)}")

# add more events and flush again
infra.tp3_out(50.0, t1)
infra.gha_in(60.0, t1, "dnata")
buf3 = infra.flush_step_buffer()
print(f"  {PASS if len(buf3) == 2 else FAIL}  flush returns 2 new events (step 2): {len(buf3)}")
print(f"  {PASS if len(infra.get_all_events()) == 5 else FAIL}  total event_log = 5: {len(infra.get_all_events())}")

# ─────────────────────────────────────────────────────────────────────────────
# [9] get_all_events — ordering and completeness
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] get_all_events — ordering and completeness")

infra = fresh()
truck = MockTruck("TRK-FULL", "export", parcels=20, slots={"wfs": 480})

steps = [
    ("gate_in",   (50.0, truck),),
    ("tp3_in",    (55.0, truck),),
    ("tp3_out",   (70.0, truck),),
    ("gha_in",    (75.0, truck, "wfs"),),
    ("dock_start",(80.0, truck, "wfs", 2),),
    ("dock_end",  (120.0, truck, "wfs", 2),),
    ("gate_out",  (130.0, truck),),
]

for method, args in steps:
    getattr(infra, method)(*args)

all_ev = infra.get_all_events()
expected_checkpoints = [
    CheckpointID.GATE_IN, CheckpointID.TP3_IN, CheckpointID.TP3_OUT,
    CheckpointID.GHA_IN, CheckpointID.DOCK_START, CheckpointID.DOCK_END,
    CheckpointID.GATE_OUT,
]
actual_checkpoints = [e.checkpoint for e in all_ev]
print(f"  {PASS if len(all_ev) == 7 else FAIL}  full journey logged 7 events: {len(all_ev)}")
print(f"  {PASS if actual_checkpoints == expected_checkpoints else FAIL}  checkpoint order correct: {[c.value for c in actual_checkpoints]}")

times = [e.sim_time for e in all_ev]
is_sorted = all(times[i] <= times[i+1] for i in range(len(times)-1))
print(f"  {PASS if is_sorted else FAIL}  events in chronological order: {times}")

all_truck_ids = [e.truck_id for e in all_ev]
print(f"  {PASS if all(tid == 'TRK-FULL' for tid in all_truck_ids) else FAIL}  all events carry correct truck_id")

# ─────────────────────────────────────────────────────────────────────────────
# [10] multi-truck interleaved journey
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] multi-truck interleaved journey")

infra = fresh()
ta = MockTruck("TRK-A", "export", slots={"dnata": 300})
tb = MockTruck("TRK-B", "import", slots={"klm": 350})
tc = MockTruck("TRK-C", "export", slots={"wfs": 400})

infra.gate_in(10.0, ta)
infra.gate_in(15.0, tb)
infra.tp3_in(20.0, tc)
infra.gha_in(25.0, ta, "dnata")
infra.dock_start(30.0, ta, "dnata", 0)
infra.gha_in(32.0, tb, "klm")
infra.dock_end(60.0, ta, "dnata", 0)
infra.gate_out(65.0, ta)
infra.tp3_out(70.0, tc)

all_ev = infra.get_all_events()
buf = infra.flush_step_buffer()

ev_by_truck = {}
for e in all_ev:
    ev_by_truck.setdefault(e.truck_id, []).append(e.checkpoint)

print(f"  {PASS if len(all_ev) == 9 else FAIL}  total 9 events across 3 trucks: {len(all_ev)}")
print(f"  {PASS if len(ev_by_truck['TRK-A']) == 5 else FAIL}  TRK-A has 5 events: {len(ev_by_truck['TRK-A'])}")
print(f"  {PASS if len(ev_by_truck['TRK-B']) == 2 else FAIL}  TRK-B has 2 events: {len(ev_by_truck['TRK-B'])}")
print(f"  {PASS if len(ev_by_truck['TRK-C']) == 2 else FAIL}  TRK-C has 2 events: {len(ev_by_truck['TRK-C'])}")
print(f"  {PASS if len(buf) == 0 else FAIL}  step_buffer empty after flush: {len(buf)}")

# ─────────────────────────────────────────────────────────────────────────────
# [11] SensorEvent field integrity — spot checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] SensorEvent field integrity")

infra = fresh()
truck = MockTruck("TRK-SI", "import", parcels=7, slots={"swiss": 600})
infra.gate_in(0.0, truck)
infra.gha_in(5.0, truck, "swiss")
infra.dock_start(10.0, truck, "swiss", dock_id=7)
infra.dock_end(50.0, truck, "swiss", dock_id=7)
infra.gate_out(60.0, truck)

for e in infra.get_all_events():
    has_truck_id = e.truck_id == "TRK-SI"
    has_flow = e.flow_type == "import"
    tag = PASS if has_truck_id and has_flow else FAIL
    print(f"  {tag}  {e.checkpoint.value:<20} truck_id OK={has_truck_id}  flow_type OK={has_flow}  "
          f"sim_time={e.sim_time}  gha_id={e.gha_id}  dock_id={e.dock_id}  n_parcels={e.n_parcels}")

print("\n" + "=" * 65)
print("  InfrastructureLayer tests complete")
print("=" * 65)