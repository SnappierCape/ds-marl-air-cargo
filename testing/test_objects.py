# =============================================================================
# TEST ─ OBJECTS MODULE
# =============================================================================

import sys
import os
import simpy
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.objects import Truck, GHATerminal, TP3Buffer
from env.infrastructure import InfrastructureLayer, CheckpointID
import config.config as config

params = config.load_params()
PASS = "[PASS]"
FAIL = "[FAIL]"

# ── Mocks ─────────────────────────────────────────────────────────────────────
class MockSvcTm:
    def __init__(self, t=20.0):
        self._t = t
    def sample(self, flow_type):
        return self._t

class MockDTP:
    def __init__(self, phase="priority", standby=False):
        self._phase   = phase
        self._standby = standby
        self.no_shows     = {}
        self.late_arrivals = {}
        self.priority_window = 10
        self.slot_duration   = 45
        self.registry = {g: {} for g in params["ghas"].keys()}

    def get_slot_phase(self, slot_start, arrival_time, dock_is_free):
        return self._phase
    def mark_docked(self, gha, slot_start, truck_id): pass
    def mark_closed(self, gha, slot_start, truck_id): pass
    def record_late(self, truck_id):
        self.late_arrivals[truck_id] = self.late_arrivals.get(truck_id, 0) + 1
    def record_no_show(self, gha, slot_start, truck_id):
        self.no_shows[truck_id] = self.no_shows.get(truck_id, 0) + 1
        for slot in self.registry.get(gha, {}).get(slot_start, []):
            if slot["truck_id"] == truck_id:
                slot["phase"] = "no_show"
    def release_to_standby(self, gha, slot_start):
        return self._standby

def make_truck(tid="TRK-001", flow="export", manifest=None, slots=None):
    m = manifest or [{"gha": "dnata", "parcels": 10}]
    t = Truck(tid, flow, "rfs", m)
    t.booked_slots = slots or {s["gha"]: 0 for s in m}
    return t

def run_process_truck(phase, flow="export", slot_at=0, service_t=20.0, sim_until=300):
    env      = simpy.Environment()
    infra    = InfrastructureLayer()
    dtp      = MockDTP(phase=phase)
    terminal = GHATerminal(env, "dnata", MockSvcTm(service_t), infra)
    truck    = make_truck(tid="TRK-PT", flow=flow, slots={"dnata": slot_at})
    env.process(terminal.process_truck(truck, dtp))
    env.run(until=sim_until)
    return truck, terminal, infra, dtp

class SmallTP3(TP3Buffer):
    CAPACITY = 3

print("=" * 65)
print("  TEST SUITE: objects — Truck, GHATerminal, TP3Buffer")
print("=" * 65)

# =============================================================================
# TRUCK
# =============================================================================
print("\n━━━━ Truck ━━━━")

# ─────────────────────────────────────────────────────────────────────────────
# [1] __post_init__ — stops_remaining mirrors manifest
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] __post_init__ — stops_remaining mirrors manifest")

manifest = [{"gha": "dnata", "parcels": 10}, {"gha": "klm", "parcels": 5}]
truck = Truck("T1", "export", "rfs", manifest)
print(f"  {PASS if truck.stops_remaining == manifest else FAIL}  stops_remaining equals manifest: {truck.stops_remaining}")
print(f"  {PASS if truck.stops_remaining is not truck.manifest else FAIL}  stops_remaining is a copy (not same object)")
print(f"  {PASS if truck.status == 'in_transit' else FAIL}  default status='in_transit': {truck.status}")
print(f"  {PASS if truck.booked_slots == {} else FAIL}  booked_slots empty by default: {truck.booked_slots}")
print(f"  {PASS if truck.timestamps == {} else FAIL}  timestamps empty by default: {truck.timestamps}")

# empty manifest
truck_empty = Truck("T2", "import", "rfs", [])
print(f"  {PASS if truck_empty.stops_remaining == [] else FAIL}  empty manifest -> empty stops_remaining")

# ─────────────────────────────────────────────────────────────────────────────
# [2] total_parcels
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] total_parcels")

truck = Truck("T1", "export", "rfs", [
    {"gha": "dnata", "parcels": 10},
    {"gha": "klm",   "parcels": 5},
    {"gha": "wfs",   "parcels": 20},
])
print(f"  {PASS if truck.total_parcels() == 35 else FAIL}  total_parcels=35: {truck.total_parcels()}")

truck_one = Truck("T2", "import", "rfs", [{"gha": "dnata", "parcels": 7}])
print(f"  {PASS if truck_one.total_parcels() == 7 else FAIL}  single-stop total_parcels=7: {truck_one.total_parcels()}")

truck_zero = Truck("T3", "export", "rfs", [])
print(f"  {PASS if truck_zero.total_parcels() == 0 else FAIL}  empty manifest total_parcels=0: {truck_zero.total_parcels()}")

# ─────────────────────────────────────────────────────────────────────────────
# [3] parcels_for
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] parcels_for")

truck = Truck("T1", "export", "rfs", [
    {"gha": "dnata", "parcels": 10},
    {"gha": "klm",   "parcels": 5},
])
print(f"  {PASS if truck.parcels_for('dnata') == 10 else FAIL}  parcels_for('dnata')=10: {truck.parcels_for('dnata')}")
print(f"  {PASS if truck.parcels_for('klm') == 5 else FAIL}  parcels_for('klm')=5: {truck.parcels_for('klm')}")
print(f"  {PASS if truck.parcels_for('wfs') == 0 else FAIL}  parcels_for unknown gha=0: {truck.parcels_for('wfs')}")
print(f"  {PASS if truck.parcels_for('') == 0 else FAIL}  parcels_for empty string=0: {truck.parcels_for('')}")

# ─────────────────────────────────────────────────────────────────────────────
# [4] next_slot — min slot among remaining stops
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] next_slot")

truck = Truck("T1", "export", "rfs", [
    {"gha": "dnata", "parcels": 10},
    {"gha": "klm",   "parcels": 5},
    {"gha": "wfs",   "parcels": 8},
])
truck.booked_slots = {"dnata": 300, "klm": 100, "wfs": 500}
print(f"  {PASS if truck.next_slot() == 100 else FAIL}  next_slot=100 (min of 300,100,500): {truck.next_slot()}")

# after completing klm stop
truck.complete_stop("klm")
print(f"  {PASS if truck.next_slot() == 300 else FAIL}  next_slot=300 after klm done: {truck.next_slot()}")

# after completing all stops
truck.complete_stop("dnata")
truck.complete_stop("wfs")
print(f"  {PASS if truck.next_slot() is None else FAIL}  next_slot=None with no remaining stops: {truck.next_slot()}")

# no bookings at all
truck2 = Truck("T2", "import", "rfs", [{"gha": "dnata", "parcels": 5}])
print(f"  {PASS if truck2.next_slot() is None else FAIL}  next_slot=None with no booked_slots: {truck2.next_slot()}")

# booking exists but for completed stop (edge case: stop removed but slot remains in dict)
truck3 = Truck("T3", "export", "rfs", [{"gha": "dnata", "parcels": 5}, {"gha": "klm", "parcels": 3}])
truck3.booked_slots = {"dnata": 200, "klm": 400}
truck3.complete_stop("dnata")   # only klm remains
print(f"  {PASS if truck3.next_slot() == 400 else FAIL}  next_slot ignores completed stop: {truck3.next_slot()}")

# ─────────────────────────────────────────────────────────────────────────────
# [5] next_stop
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] next_stop")

truck = Truck("T1", "export", "rfs", [
    {"gha": "dnata", "parcels": 10},
    {"gha": "klm",   "parcels": 5},
])
print(f"  {PASS if truck.next_stop() == {'gha': 'dnata', 'parcels': 10} else FAIL}  next_stop is first manifest entry: {truck.next_stop()}")

truck.complete_stop("dnata")
print(f"  {PASS if truck.next_stop() == {'gha': 'klm', 'parcels': 5} else FAIL}  next_stop after first complete: {truck.next_stop()}")

truck.complete_stop("klm")
print(f"  {PASS if truck.next_stop() is None else FAIL}  next_stop=None when all done: {truck.next_stop()}")

# ─────────────────────────────────────────────────────────────────────────────
# [6] complete_stop — removes only the target gha
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] complete_stop")

truck = Truck("T1", "export", "rfs", [
    {"gha": "dnata", "parcels": 10},
    {"gha": "klm",   "parcels": 5},
    {"gha": "wfs",   "parcels": 8},
])
truck.complete_stop("klm")
remaining_ghas = [s["gha"] for s in truck.stops_remaining]
print(f"  {PASS if 'klm' not in remaining_ghas else FAIL}  'klm' removed from stops_remaining: {remaining_ghas}")
print(f"  {PASS if len(truck.stops_remaining) == 2 else FAIL}  2 stops remaining: {len(truck.stops_remaining)}")

# completing non-existent gha is silent no-op
truck.complete_stop("swiss")
print(f"  {PASS if len(truck.stops_remaining) == 2 else FAIL}  completing unknown gha is no-op: {len(truck.stops_remaining)}")

# complete same gha twice — second call is no-op
truck.complete_stop("dnata")
truck.complete_stop("dnata")
print(f"  {PASS if len(truck.stops_remaining) == 1 else FAIL}  duplicate complete_stop is safe: {len(truck.stops_remaining)}")

# =============================================================================
# GHATerminal
# =============================================================================
print("\n━━━━ GHATerminal ━━━━")

# ─────────────────────────────────────────────────────────────────────────────
# [7] _dock_pool and _queue routing
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] _dock_pool and _queue routing")

env      = simpy.Environment()
infra    = InfrastructureLayer()
terminal = GHATerminal(env, "dnata", MockSvcTm(), infra)

exp_pool = terminal._dock_pool("export")
imp_pool = terminal._dock_pool("import")
print(f"  {PASS if exp_pool is terminal.docks_exp else FAIL}  export -> docks_exp")
print(f"  {PASS if imp_pool is terminal.docks_imp else FAIL}  import -> docks_imp")
print(f"  {PASS if exp_pool is not imp_pool else FAIL}  export and import pools are distinct")

exp_q = terminal._queue("export")
imp_q = terminal._queue("import")
print(f"  {PASS if exp_q is terminal.queue_exp else FAIL}  export -> queue_exp")
print(f"  {PASS if imp_q is terminal.queue_imp else FAIL}  import -> queue_imp")
print(f"  {PASS if exp_q is not imp_q else FAIL}  export and import queues are distinct")

# ─────────────────────────────────────────────────────────────────────────────
# [8] exp_occupancy / imp_occupancy — reflect simpy resource count
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] exp_occupancy / imp_occupancy")

for gha in params["ghas"].keys():
    env2 = simpy.Environment()
    t2   = GHATerminal(env2, gha, MockSvcTm(), InfrastructureLayer())
    n_exp = params["ghas"][gha]["export"]
    n_imp = params["ghas"][gha]["import"]
    print(f"  {PASS if t2.exp_occupancy() == 0.0 else FAIL}  {gha} exp_occupancy=0 at init: {t2.exp_occupancy()}")
    print(f"  {PASS if t2.imp_occupancy() == 0.0 else FAIL}  {gha} imp_occupancy=0 at init: {t2.imp_occupancy()}")
    print(f"  {PASS if t2.n_exp == n_exp else FAIL}  {gha} n_exp={n_exp}: {t2.n_exp}")
    print(f"  {PASS if t2.n_imp == n_imp else FAIL}  {gha} n_imp={n_imp}: {t2.n_imp}")

# ─────────────────────────────────────────────────────────────────────────────
# [9] exp_queue_norm / imp_queue_norm
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] exp_queue_norm / imp_queue_norm")

env = simpy.Environment()
t   = GHATerminal(env, "dnata", MockSvcTm(), InfrastructureLayer())
print(f"  {PASS if t.exp_queue_norm() == 0.0 else FAIL}  exp_queue_norm=0 at init: {t.exp_queue_norm()}")
print(f"  {PASS if t.imp_queue_norm() == 0.0 else FAIL}  imp_queue_norm=0 at init: {t.imp_queue_norm()}")

# manually add trucks to queue
dummy_trucks = [make_truck(f"DUMMY-{i}") for i in range(3)]
t.queue_exp.extend(dummy_trucks)
max_q = params["ghas"]["dnata"]["export"]
expected_norm = min(3 / max_q, 1.0)
print(f"  {PASS if abs(t.exp_queue_norm() - expected_norm) < 1e-9 else FAIL}  exp_queue_norm=3/{max_q}={expected_norm:.4f}: {t.exp_queue_norm():.4f}")

# capped at 1.0 when queue > capacity
big_queue = [make_truck(f"BIG-{i}") for i in range(max_q + 10)]
t.queue_exp.extend(big_queue)
print(f"  {PASS if t.exp_queue_norm() == 1.0 else FAIL}  exp_queue_norm capped at 1.0: {t.exp_queue_norm()}")

# ─────────────────────────────────────────────────────────────────────────────
# [10] upcoming_bookings_norm
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] upcoming_bookings_norm")

env = simpy.Environment()
t   = GHATerminal(env, "dnata", MockSvcTm(), InfrastructureLayer())
dtp = MockDTP()
# no bookings -> 0.0
print(f"  {PASS if t.upcoming_bookings_norm(dtp, 120) == 0.0 else FAIL}  no bookings -> 0.0: {t.upcoming_bookings_norm(dtp, 120)}")

# add booked slots within horizon
dtp.registry["dnata"] = {
    50:  [{"truck_id": "T1", "phase": "booked"}],
    80:  [{"truck_id": "T2", "phase": "docked"}],
    200: [{"truck_id": "T3", "phase": "booked"}],   # outside horizon=100
}
result = t.upcoming_bookings_norm(dtp, horizon=100)
total_docks = params["ghas"]["dnata"]["total"]
expected = min(2 / total_docks, 1.0)   # T1 and T2 within horizon
print(f"  {PASS if abs(result - expected) < 1e-9 else FAIL}  2 bookings in horizon=100: {result:.4f} (expected {expected:.4f})")

# 'available' phase not counted
dtp2 = MockDTP()
dtp2.registry["dnata"] = {50: [{"truck_id": None, "phase": "available"}]}
print(f"  {PASS if t.upcoming_bookings_norm(dtp2, 120) == 0.0 else FAIL}  'available' phase not counted: {t.upcoming_bookings_norm(dtp2, 120)}")

# ─────────────────────────────────────────────────────────────────────────────
# [11] process_truck — "priority" phase: normal service
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] process_truck — 'priority' phase")

truck, terminal, infra, dtp = run_process_truck("priority", flow="export", service_t=20.0)

events = infra.get_all_events()
checkpoints = [e.checkpoint for e in events]
print(f"  {PASS if CheckpointID.GHA_IN in checkpoints else FAIL}  GHA_IN event fired")
print(f"  {PASS if CheckpointID.DOCK_START in checkpoints else FAIL}  DOCK_START event fired")
print(f"  {PASS if CheckpointID.DOCK_END in checkpoints else FAIL}  DOCK_END event fired")
print(f"  {PASS if truck.status == Truck.STATUS_IN_TRANSIT else FAIL}  truck status=in_transit after service: {truck.status}")
print(f"  {PASS if truck.stops_remaining == [] else FAIL}  stop completed: {truck.stops_remaining}")
print(f"  {PASS if terminal.stats['export']['processed'] == 1 else FAIL}  processed counter=1: {terminal.stats['export']['processed']}")
print(f"  {PASS if terminal.stats['export']['tot_serv'] == 20.0 else FAIL}  tot_serv=20.0: {terminal.stats['export']['tot_serv']}")
print(f"  {PASS if dtp.late_arrivals == {} else FAIL}  no late penalty in priority phase: {dtp.late_arrivals}")

# ─────────────────────────────────────────────────────────────────────────────
# [12] process_truck — "early" phase: waits until slot opens
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] process_truck — 'early' phase: waits until slot_start")

env      = simpy.Environment()
infra    = InfrastructureLayer()
dtp      = MockDTP(phase="early")
terminal = GHATerminal(env, "dnata", MockSvcTm(20.0), infra)
truck    = make_truck(slots={"dnata": 50})   # slot opens at t=50

env.process(terminal.process_truck(truck, dtp))
env.run(until=200)

events      = infra.get_all_events()
dock_start  = next((e for e in events if e.checkpoint == CheckpointID.DOCK_START), None)
print(f"  {PASS if dock_start is not None else FAIL}  DOCK_START event fired")
print(f"  {PASS if dock_start is not None and dock_start.sim_time >= 50.0 else FAIL}  DOCK_START at t>=50 (waited for slot): t={dock_start.sim_time if dock_start else 'N/A'}")
print(f"  {PASS if truck.status == Truck.STATUS_IN_TRANSIT else FAIL}  truck completed service: {truck.status}")

# ─────────────────────────────────────────────────────────────────────────────
# [13] process_truck — "release" phase: late penalty, proceeds
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] process_truck — 'release' phase: late penalty, proceeds")

truck, terminal, infra, dtp = run_process_truck("release", flow="import", service_t=15.0)

print(f"  {PASS if truck.status == Truck.STATUS_IN_TRANSIT else FAIL}  truck completed service: {truck.status}")
print(f"  {PASS if dtp.late_arrivals.get('TRK-PT', 0) == 1 else FAIL}  late_arrival recorded: {dtp.late_arrivals}")
print(f"  {PASS if terminal.stats['import']['processed'] == 1 else FAIL}  import processed=1: {terminal.stats['import']['processed']}")

# ─────────────────────────────────────────────────────────────────────────────
# [14] process_truck — "release_dock_taken": redirect to TP3
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14] process_truck — 'release_dock_taken': redirect to TP3")

truck, terminal, infra, dtp = run_process_truck("release_dock_taken")

print(f"  {PASS if truck.status == Truck.STATUS_AT_TP3 else FAIL}  truck status=at_tp3: {truck.status}")
print(f"  {PASS if terminal.stats['export']['processed'] == 0 else FAIL}  no service performed: {terminal.stats['export']['processed']}")
# GHA_IN fires but DOCK_START should not
events = infra.get_all_events()
checkpoints = [e.checkpoint for e in events]
print(f"  {PASS if CheckpointID.GHA_IN in checkpoints else FAIL}  GHA_IN fired before redirect")
print(f"  {PASS if CheckpointID.DOCK_START not in checkpoints else FAIL}  DOCK_START not fired (redirected): {checkpoints}")
print(f"  {PASS if dtp.late_arrivals.get('TRK-PT', 0) == 1 else FAIL}  late penalty recorded: {dtp.late_arrivals}")

# ─────────────────────────────────────────────────────────────────────────────
# [15] process_truck — "no_show": redirect to TP3, no_show recorded
# ─────────────────────────────────────────────────────────────────────────────
print("\n[15] process_truck — 'no_show': redirect to TP3, penalty")

truck, terminal, infra, dtp = run_process_truck("no_show")

print(f"  {PASS if truck.status == Truck.STATUS_AT_TP3 else FAIL}  truck status=at_tp3: {truck.status}")
print(f"  {PASS if dtp.no_shows.get('TRK-PT', 0) == 1 else FAIL}  no_show recorded: {dtp.no_shows}")
print(f"  {PASS if terminal.stats['export']['processed'] == 0 else FAIL}  no service performed: {terminal.stats['export']['processed']}")

# ─────────────────────────────────────────────────────────────────────────────
# [16] process_truck — multiple trucks, FIFO dock queue
# ─────────────────────────────────────────────────────────────────────────────
print("\n[16] process_truck — multiple trucks FIFO queuing")

env      = simpy.Environment()
infra    = InfrastructureLayer()
terminal = GHATerminal(env, "dnata", MockSvcTm(30.0), infra)
dtp      = MockDTP("priority")

# dnata has 25 export docks — fill all then add 3 more that must queue
n_docks = params["ghas"]["dnata"]["export"]
trucks  = [make_truck(f"TRK-{i:03d}", flow="export") for i in range(n_docks + 3)]

for tr in trucks:
    env.process(terminal.process_truck(tr, dtp))

env.run(until=500)
processed = terminal.stats["export"]["processed"]
print(f"  {PASS if processed == n_docks + 3 else FAIL}  all {n_docks + 3} trucks processed: {processed}")
print(f"  {PASS if all(t.stops_remaining == [] for t in trucks) else FAIL}  all stops completed")

# ─────────────────────────────────────────────────────────────────────────────
# [17] release_window_watcher — fires at priority_window, signals TP3
# ─────────────────────────────────────────────────────────────────────────────
print("\n[17] release_window_watcher — signals TP3 when standby=True")

env      = simpy.Environment()
infra    = InfrastructureLayer()
tp3      = TP3Buffer(env, infra)
terminal = GHATerminal(env, "dnata", MockSvcTm(), infra)
dtp      = MockDTP(standby=True)

env.process(terminal.release_window_watcher(100, dtp, tp3))
env.run(until=200)   # 100 + priority_window(10) = 110

print(f"  {PASS if len(tp3.standby_opportunities) == 1 else FAIL}  1 standby signal fired: {len(tp3.standby_opportunities)}")
opp = tp3.standby_opportunities[0]
print(f"  {PASS if opp['gha'] == 'dnata' else FAIL}  signal gha='dnata': {opp['gha']}")
print(f"  {PASS if opp['slot_start'] == 100 else FAIL}  signal slot_start=100: {opp['slot_start']}")
print(f"  {PASS if opp['consumed'] == False else FAIL}  signal not yet consumed: {opp['consumed']}")

# standby=False — no signal emitted
env2      = simpy.Environment()
tp3_2     = TP3Buffer(env2, InfrastructureLayer())
terminal2 = GHATerminal(env2, "dnata", MockSvcTm(), InfrastructureLayer())
dtp2      = MockDTP(standby=False)
env2.process(terminal2.release_window_watcher(100, dtp2, tp3_2))
env2.run(until=200)
print(f"  {PASS if len(tp3_2.standby_opportunities) == 0 else FAIL}  no signal when release_to_standby=False: {len(tp3_2.standby_opportunities)}")

# =============================================================================
# TP3Buffer
# =============================================================================
print("\n━━━━ TP3Buffer ━━━━")

# ─────────────────────────────────────────────────────────────────────────────
# [18] enter — parks truck, updates status and infra
# ─────────────────────────────────────────────────────────────────────────────
print("\n[18] enter — parks trucks, status and infra events")

env  = simpy.Environment()
infra = InfrastructureLayer()
tp3  = TP3Buffer(env, infra)

trucks = [make_truck(f"TRK-E{i}") for i in range(3)]
for tr in trucks:
    env.process(tp3.enter(tr))
env.run(until=10)

print(f"  {PASS if tp3.n_parked() == 3 else FAIL}  3 trucks parked: {tp3.n_parked()}")
print(f"  {PASS if all(t.status == Truck.STATUS_AT_TP3 for t in trucks) else FAIL}  all trucks status=at_tp3")
print(f"  {PASS if tp3.occupancy_ratio() == 3 / TP3Buffer.CAPACITY else FAIL}  occupancy_ratio correct: {tp3.occupancy_ratio():.4f}")

tp3_in_events = [e for e in infra.get_all_events() if e.checkpoint == CheckpointID.TP3_IN]
print(f"  {PASS if len(tp3_in_events) == 3 else FAIL}  3 TP3_IN events in infra: {len(tp3_in_events)}")

# ─────────────────────────────────────────────────────────────────────────────
# [19] enter — overflow queuing when TP3 full
# ─────────────────────────────────────────────────────────────────────────────
print("\n[19] enter — overflow queue when TP3 full (capacity=3)")

env   = simpy.Environment()
infra = InfrastructureLayer()
tp3   = SmallTP3(env, infra)

trucks = [make_truck(f"TRK-OV{i}") for i in range(5)]   # 5 > capacity=3
for tr in trucks:
    env.process(tp3.enter(tr))
env.run(until=10)

print(f"  {PASS if tp3.n_parked() == 3 else FAIL}  only 3 parked (at capacity): {tp3.n_parked()}")
print(f"  {PASS if tp3.n_overflow() == 2 else FAIL}  2 trucks in overflow queue: {tp3.n_overflow()}")

# release one slot — an overflow truck should enter
tp3.release(trucks[0].truck_id)
env.run(until=20)
print(f"  {PASS if tp3.n_parked() == 3 else FAIL}  after release, overflow fills in: {tp3.n_parked()}")
print(f"  {PASS if tp3.n_overflow() == 1 else FAIL}  overflow queue reduces to 1: {tp3.n_overflow()}")

# ─────────────────────────────────────────────────────────────────────────────
# [20] release — targeted by truck_id
# ─────────────────────────────────────────────────────────────────────────────
print("\n[20] release — targeted by truck_id")

env   = simpy.Environment()
infra = InfrastructureLayer()
tp3   = TP3Buffer(env, infra)
trucks = [make_truck(f"TRK-R{i}") for i in range(4)]
for tr in trucks:
    env.process(tp3.enter(tr))
env.run(until=10)

released = tp3.release("TRK-R1")
print(f"  {PASS if released is not None and released.truck_id == 'TRK-R1' else FAIL}  release returns correct truck: {released.truck_id if released else None}")
print(f"  {PASS if tp3.n_parked() == 3 else FAIL}  parked count reduced to 3: {tp3.n_parked()}")
tp3_out = [e for e in infra.get_all_events() if e.checkpoint == CheckpointID.TP3_OUT]
print(f"  {PASS if len(tp3_out) == 1 else FAIL}  1 TP3_OUT event fired: {len(tp3_out)}")
print(f"  {PASS if tp3_out[0].truck_id == 'TRK-R1' else FAIL}  TP3_OUT for correct truck: {tp3_out[0].truck_id}")

# release unknown truck_id -> None
result = tp3.release("GHOST")
print(f"  {PASS if result is None else FAIL}  release unknown truck_id -> None: {result}")

# ─────────────────────────────────────────────────────────────────────────────
# [21] release_next — FCFS by gha booking
# ─────────────────────────────────────────────────────────────────────────────
print("\n[21] release_next — FCFS release by gha")

env   = simpy.Environment()
infra = InfrastructureLayer()
tp3   = TP3Buffer(env, infra)

ta = make_truck("TA", slots={"dnata": 100, "klm": 200})
tb = make_truck("TB", slots={"klm": 150})
tc = make_truck("TC", slots={"wfs": 300})

for tr in [ta, tb, tc]:
    env.process(tp3.enter(tr))
env.run(until=10)

# release first truck with klm booking
released = tp3.release_next("klm")
print(f"  {PASS if released is not None else FAIL}  release_next found a truck: {released.truck_id if released else None}")
print(f"  {PASS if released is not None and "klm" in released.booked_slots else FAIL}  released truck has klm booking: {released.booked_slots if released else None}")

# no truck with swiss -> None
released2 = tp3.release_next("swiss")
print(f"  {PASS if released2 is None else FAIL}  release_next returns None for unbooked gha: {released2}")

# release_next depletes matching trucks
tp3.release_next("klm")   # should get TC (wfs) — NO, TC has wfs not klm
# ta was already released. tb had klm. So now no truck with klm remains.
released3 = tp3.release_next("klm")
print(f"  {PASS if released3 is None else FAIL}  release_next returns None when no more klm trucks: {released3}")

# ─────────────────────────────────────────────────────────────────────────────
# [22] signal_standby_opportunity / get_pending_signals
# ─────────────────────────────────────────────────────────────────────────────
print("\n[22] signal_standby_opportunity / get_pending_signals")

env  = simpy.Environment()
tp3  = TP3Buffer(env, InfrastructureLayer())

tp3.signal_standby_opportunity("dnata", 100, 95.0)
tp3.signal_standby_opportunity("klm",   200, 195.0)
tp3.signal_standby_opportunity("dnata", 150, 140.0)

signals = tp3.get_pending_signals()
print(f"  {PASS if len(signals) == 3 else FAIL}  3 pending signals: {len(signals)}")

# consume one
signals[0]["consumed"] = True
pending = tp3.get_pending_signals()
print(f"  {PASS if len(pending) == 2 else FAIL}  2 pending after consuming 1: {len(pending)}")

# filter by gha manually (as demand.py does)
dnata_signals = [s for s in tp3.get_pending_signals() if s["gha"] == "dnata"]
print(f"  {PASS if len(dnata_signals) == 1 else FAIL}  1 dnata signal remaining: {len(dnata_signals)}")
print(f"  {PASS if dnata_signals[0]['slot_start'] == 150 else FAIL}  correct slot_start=150: {dnata_signals[0]['slot_start']}")

# ─────────────────────────────────────────────────────────────────────────────
# [23] parked_by_flow_type
# ─────────────────────────────────────────────────────────────────────────────
print("\n[23] parked_by_flow_type")

env   = simpy.Environment()
infra = InfrastructureLayer()
tp3   = TP3Buffer(env, infra)

exp_trucks = [make_truck(f"EXP-{i}", flow="export") for i in range(4)]
imp_trucks = [make_truck(f"IMP-{i}", flow="import") for i in range(3)]
for tr in exp_trucks + imp_trucks:
    env.process(tp3.enter(tr))
env.run(until=10)

print(f"  {PASS if tp3.parked_by_flow_type('export') == 4 else FAIL}  4 export trucks parked: {tp3.parked_by_flow_type('export')}")
print(f"  {PASS if tp3.parked_by_flow_type('import') == 3 else FAIL}  3 import trucks parked: {tp3.parked_by_flow_type('import')}")
print(f"  {PASS if tp3.parked_by_flow_type('export') + tp3.parked_by_flow_type('import') == tp3.n_parked() else FAIL}  flow counts sum to n_parked")

# ─────────────────────────────────────────────────────────────────────────────
# [24] get_parked_trucks — returns all parked truck objects
# ─────────────────────────────────────────────────────────────────────────────
print("\n[24] get_parked_trucks")

env   = simpy.Environment()
infra = InfrastructureLayer()
tp3   = TP3Buffer(env, infra)

input_trucks = [make_truck(f"TRK-GP{i}") for i in range(5)]
for tr in input_trucks:
    env.process(tp3.enter(tr))
env.run(until=10)

parked = tp3.get_parked_trucks()
parked_ids = {t.truck_id for t in parked}
input_ids  = {t.truck_id for t in input_trucks}
print(f"  {PASS if len(parked) == 5 else FAIL}  get_parked_trucks returns 5 trucks: {len(parked)}")
print(f"  {PASS if parked_ids == input_ids else FAIL}  all truck ids match: {parked_ids == input_ids}")
print(f"  {PASS if all(isinstance(t, Truck) for t in parked) else FAIL}  all items are Truck instances")

# after release, truck disappears from list
tp3.release("TRK-GP2")
parked_after = tp3.get_parked_trucks()
print(f"  {PASS if 'TRK-GP2' not in {t.truck_id for t in parked_after} else FAIL}  released truck not in get_parked_trucks: {len(parked_after)}")

# ─────────────────────────────────────────────────────────────────────────────
# [25] occupancy_ratio / n_parked / n_overflow edge cases
# ─────────────────────────────────────────────────────────────────────────────
print("\n[25] occupancy_ratio / n_parked / n_overflow edge cases")

env = simpy.Environment()
tp3 = TP3Buffer(env, InfrastructureLayer())
print(f"  {PASS if tp3.occupancy_ratio() == 0.0 else FAIL}  occupancy_ratio=0 at init: {tp3.occupancy_ratio()}")
print(f"  {PASS if tp3.n_parked() == 0 else FAIL}  n_parked=0 at init: {tp3.n_parked()}")
print(f"  {PASS if tp3.n_overflow() == 0 else FAIL}  n_overflow=0 at init: {tp3.n_overflow()}")

# occupancy_ratio should be between 0 and 1
env2 = simpy.Environment()
tp3_2 = TP3Buffer(env2, InfrastructureLayer())
trucks2 = [make_truck(f"OCC-{i}") for i in range(10)]
for tr in trucks2:
    env2.process(tp3_2.enter(tr))
env2.run(until=10)
ratio = tp3_2.occupancy_ratio()
print(f"  {PASS if 0.0 <= ratio <= 1.0 else FAIL}  occupancy_ratio in [0,1]: {ratio:.4f}")
print(f"  {PASS if ratio == 10 / TP3Buffer.CAPACITY else FAIL}  occupancy_ratio=10/{TP3Buffer.CAPACITY}: {ratio:.4f}")

print("\n" + "=" * 65)
print("  Objects tests complete")
print("=" * 65)