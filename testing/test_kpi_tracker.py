# =============================================================================
# TEST ─ KPI_TRACKER MODULE
# =============================================================================

import sys
import os
import math
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.kpi_tracker import KPITracker
from env.infrastructure import SensorEvent, CheckpointID
import config.config as config

params = config.load_params()
W = params["marl"]["reward_weights"]
PEAK_START = params["demand"]["peak_window"][0]
PEAK_END   = params["demand"]["peak_window"][1]
PASS = "[PASS]"
FAIL = "[FAIL]"

# ── Event factories ───────────────────────────────────────────────────────────
def ev_gate_in(tid, t, flow="export", parcels=10, slot=480):
    return SensorEvent(t, CheckpointID.GATE_IN, tid, flow, None, None, parcels, slot)

def ev_gha_in(tid, t, gha, flow="export", parcels=10, slot=480):
    return SensorEvent(t, CheckpointID.GHA_IN, tid, flow, gha, None, parcels, slot)

def ev_dock_start(tid, t, gha, dock=0, flow="export", parcels=10, slot=480):
    return SensorEvent(t, CheckpointID.DOCK_START, tid, flow, gha, dock, parcels, slot)

def ev_dock_end(tid, t, gha, dock=0, flow="export", slot=480):
    return SensorEvent(t, CheckpointID.DOCK_END, tid, flow, gha, dock, None, slot)

def ev_gate_out(tid, t, flow="export"):
    return SensorEvent(t, CheckpointID.GATE_OUT, tid, flow, None, None, None, None)

def approx(a, b, tol=1e-9):
    return abs(a - b) < tol

# ── Mock objects ──────────────────────────────────────────────────────────────
class MockTerminal:
    def __init__(self, exp=0.6, imp=0.4, exp_q=0.1, imp_q=0.2, proc_exp=5, proc_imp=3):
        self._exp, self._imp = exp, imp
        self._eq, self._iq = exp_q, imp_q
        self.stats = {
            "export": {"processed": proc_exp, "tot_wait": 0.0, "tot_serv": 0.0},
            "import": {"processed": proc_imp, "tot_wait": 0.0, "tot_serv": 0.0},
        }
    def exp_occupancy(self): return self._exp
    def imp_occupancy(self): return self._imp
    def exp_queue_norm(self): return self._eq
    def imp_queue_norm(self): return self._iq

class MockDTP:
    def __init__(self, no_shows=None, late=None):
        self.no_shows      = no_shows or {}
        self.late_arrivals = late or {}

print("=" * 65)
print("  TEST SUITE: KPITracker")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# [1] ingest — GATE_IN initialises truck state
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] ingest — GATE_IN initialises truck state")

kpi = KPITracker()
kpi.ingest([ev_gate_in("T1", 100.0, parcels=15)])
state = kpi._truck.get("T1")
print(f"  {PASS if state is not None else FAIL}  truck state created after GATE_IN: {state is not None}")
print(f"  {PASS if state['gate_in'] == 100.0 else FAIL}  gate_in timestamp=100.0: {state['gate_in']}")
print(f"  {PASS if state['n_parcels'] == 15 else FAIL}  n_parcels=15: {state['n_parcels']}")

# second gate_in same truck overwrites
kpi.ingest([ev_gate_in("T1", 200.0, parcels=5)])
state2 = kpi._truck["T1"]
print(f"  {PASS if state2['gate_in'] == 200.0 else FAIL}  second GATE_IN overwrites state: gate_in={state2['gate_in']}")

# ─────────────────────────────────────────────────────────────────────────────
# [2] ingest — GHA_IN stores arrival time per gha
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] ingest — GHA_IN timestamps per GHA")

kpi = KPITracker()
kpi.ingest([
    ev_gate_in("T1", 0.0),
    ev_gha_in("T1", 10.0, "dnata"),
    ev_gha_in("T1", 80.0, "klm"),
])
state = kpi._truck["T1"]
print(f"  {PASS if state['gha_in'].get('dnata') == 10.0 else FAIL}  gha_in['dnata']=10.0: {state['gha_in'].get('dnata')}")
print(f"  {PASS if state['gha_in'].get('klm') == 80.0 else FAIL}  gha_in['klm']=80.0: {state['gha_in'].get('klm')}")

# gha_in for unknown truck is silently ignored
kpi.ingest([ev_gha_in("GHOST", 50.0, "dnata")])
print(f"  {PASS if 'GHOST' not in kpi._truck else FAIL}  gha_in for unknown truck silently ignored")

# ─────────────────────────────────────────────────────────────────────────────
# [3] ingest — DOCK_START accumulates wait time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] ingest — DOCK_START accumulates wait time")

kpi = KPITracker()
kpi.ingest([
    ev_gate_in("T1", 0.0),
    ev_gha_in("T1", 10.0, "dnata"),
    ev_dock_start("T1", 25.0, "dnata"),   # wait = 25-10 = 15
])
print(f"  {PASS if approx(kpi._total_wait, 15.0) else FAIL}  total_wait=15.0: {kpi._total_wait}")

# two trucks
kpi.ingest([
    ev_gate_in("T2", 0.0),
    ev_gha_in("T2", 5.0, "klm"),
    ev_dock_start("T2", 20.0, "klm"),    # wait = 20-5 = 15
])
print(f"  {PASS if approx(kpi._total_wait, 30.0) else FAIL}  cumulative total_wait=30.0: {kpi._total_wait}")

# dock_start with no prior gha_in defaults wait to 0
kpi2 = KPITracker()
kpi2.ingest([ev_gate_in("TX", 0.0), ev_dock_start("TX", 50.0, "dnata")])
print(f"  {PASS if approx(kpi2._total_wait, 0.0) else FAIL}  dock_start with no gha_in -> wait=0: {kpi2._total_wait}")

# ─────────────────────────────────────────────────────────────────────────────
# [4] ingest — DOCK_END accumulates service time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] ingest — DOCK_END accumulates service time")

kpi = KPITracker()
kpi.ingest([
    ev_gate_in("T1", 0.0),
    ev_gha_in("T1", 10.0, "dnata"),
    ev_dock_start("T1", 20.0, "dnata"),   # wait=10
    ev_dock_end("T1", 60.0, "dnata"),     # service=40
])
print(f"  {PASS if approx(kpi._total_service, 40.0) else FAIL}  total_service=40.0: {kpi._total_service}")
print(f"  {PASS if approx(kpi._total_wait, 10.0) else FAIL}  total_wait=10.0: {kpi._total_wait}")

# dock_end with no prior dock_start: service defaults to 0
kpi2 = KPITracker()
kpi2.ingest([ev_gate_in("TX", 0.0), ev_dock_end("TX", 50.0, "dnata")])
print(f"  {PASS if approx(kpi2._total_service, 0.0) else FAIL}  dock_end with no dock_start -> service=0: {kpi2._total_service}")

# ─────────────────────────────────────────────────────────────────────────────
# [5] ingest — GATE_OUT closes truck and updates nttp
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] ingest — GATE_OUT closes truck, updates nttp")

kpi = KPITracker()
kpi.ingest([
    ev_gate_in("T1", 0.0, parcels=10),
    ev_gate_out("T1", 100.0),
])
expected_nttp = (100.0 - 0.0) / 10.0   # 10.0
print(f"  {PASS if approx(kpi._nttp_sum, expected_nttp) else FAIL}  nttp_sum=10.0: {kpi._nttp_sum}")
print(f"  {PASS if kpi._n_completed == 1 else FAIL}  n_completed=1: {kpi._n_completed}")
print(f"  {PASS if 'T1' not in kpi._truck else FAIL}  truck state cleared after gate_out")

# gate_out with 0 parcels should not count
kpi2 = KPITracker()
kpi2.ingest([ev_gate_in("T2", 0.0, parcels=0), ev_gate_out("T2", 100.0)])
print(f"  {PASS if kpi2._n_completed == 0 else FAIL}  gate_out with 0 parcels not counted: n_completed={kpi2._n_completed}")

# gate_out for unknown truck (e.g. missed gate_in) is silently ignored
kpi3 = KPITracker()
kpi3.ingest([ev_gate_out("GHOST", 100.0)])
print(f"  {PASS if kpi3._n_completed == 0 else FAIL}  gate_out for unknown truck silently ignored")

# ─────────────────────────────────────────────────────────────────────────────
# [6] ingest — peak window accumulation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] ingest — peak window accumulation")

kpi = KPITracker()
# outside peak
kpi.ingest([
    ev_gate_in("NP", 0.0),
    ev_gha_in("NP", 10.0, "dnata"),
    ev_dock_start("NP", 20.0, "dnata"),   # t=20, outside peak
    ev_dock_end("NP", 50.0, "dnata"),
])
# inside peak
peak_t = PEAK_START + 10
kpi.ingest([
    ev_gate_in("P1", float(peak_t), parcels=5),
    ev_gha_in("P1", float(peak_t + 5), "klm"),
    ev_dock_start("P1", float(peak_t + 15), "klm"),   # wait=10, inside peak
    ev_dock_end("P1", float(peak_t + 45), "klm"),     # service=30, inside peak
])
print(f"  {PASS if approx(kpi._peak_wait, 10.0) else FAIL}  peak_wait=10.0: {kpi._peak_wait}")
print(f"  {PASS if approx(kpi._peak_service, 30.0) else FAIL}  peak_service=30.0: {kpi._peak_service}")
print(f"  {PASS if approx(kpi._total_wait, 20.0) else FAIL}  total_wait=10+10=20.0: {kpi._total_wait}")

# ─────────────────────────────────────────────────────────────────────────────
# [7] wpr — formula and edge cases
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] wpr — formula and edge cases")

kpi = KPITracker()
kpi.ingest([
    ev_gate_in("T1", 0.0),
    ev_gha_in("T1", 0.0, "dnata"),
    ev_dock_start("T1", 10.0, "dnata"),   # wait=10
    ev_dock_end("T1", 40.0, "dnata"),     # service=30
])
expected = 10.0 / 30.0
print(f"  {PASS if approx(kpi.wpr(), expected) else FAIL}  wpr=10/30={expected:.6f}: {kpi.wpr():.6f}")

# zero service time -> wpr = 0
kpi2 = KPITracker()
print(f"  {PASS if approx(kpi2.wpr(), 0.0) else FAIL}  wpr=0.0 when no events: {kpi2.wpr()}")

# multiple trucks
kpi3 = KPITracker()
for i in range(5):
    tid = f"T{i}"
    kpi3.ingest([
        ev_gate_in(tid, 0.0),
        ev_gha_in(tid, 0.0, "dnata"),
        ev_dock_start(tid, float(i * 5), "dnata"),      # wait = i*5
        ev_dock_end(tid, float(i * 5 + 20), "dnata"),   # service = 20
    ])
total_w = sum(i * 5 for i in range(5))
total_s = 5 * 20
expected3 = total_w / total_s
print(f"  {PASS if approx(kpi3.wpr(), expected3) else FAIL}  multi-truck wpr={expected3:.4f}: {kpi3.wpr():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# [8] peak_wpr — only counts events in peak window
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] peak_wpr — only peak events")

kpi = KPITracker()
# off-peak truck
kpi.ingest([
    ev_gate_in("OFF", 0.0),
    ev_gha_in("OFF", 0.0, "dnata"),
    ev_dock_start("OFF", 5.0, "dnata"),
    ev_dock_end("OFF", 35.0, "dnata"),
])
# peak truck
pt = float(PEAK_START + 5)
kpi.ingest([
    ev_gate_in("ON", pt),
    ev_gha_in("ON", pt, "dnata"),
    ev_dock_start("ON", pt + 12.0, "dnata"),  # wait=12
    ev_dock_end("ON", pt + 52.0, "dnata"),    # service=40
])
expected_pwpr = 12.0 / 40.0
print(f"  {PASS if approx(kpi.peak_wpr(), expected_pwpr) else FAIL}  peak_wpr=12/40={expected_pwpr:.4f}: {kpi.peak_wpr():.4f}")
print(f"  {PASS if kpi.peak_wpr() != kpi.wpr() else FAIL}  peak_wpr differs from overall wpr")

# no peak events -> 0
kpi2 = KPITracker()
print(f"  {PASS if approx(kpi2.peak_wpr(), 0.0) else FAIL}  peak_wpr=0 with no peak events: {kpi2.peak_wpr()}")

# ─────────────────────────────────────────────────────────────────────────────
# [9] nttp — normalized turnaround
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] nttp — normalized turnaround time per parcel")

kpi = KPITracker()
kpi.ingest([ev_gate_in("T1", 0.0, parcels=10), ev_gate_out("T1", 100.0)])
print(f"  {PASS if approx(kpi.nttp(), 10.0) else FAIL}  nttp=100/10=10.0: {kpi.nttp()}")

kpi.ingest([ev_gate_in("T2", 50.0, parcels=5), ev_gate_out("T2", 100.0)])
expected = (10.0 + 10.0) / 2    # (100/10 + 50/5) / 2
print(f"  {PASS if approx(kpi.nttp(), expected) else FAIL}  nttp=(10+10)/2=10.0: {kpi.nttp()}")

# varying parcel counts
kpi2 = KPITracker()
kpi2.ingest([ev_gate_in("TA", 0.0, parcels=20), ev_gate_out("TA", 200.0)])   # 10.0
kpi2.ingest([ev_gate_in("TB", 0.0, parcels=4),  ev_gate_out("TB", 80.0)])    # 20.0
expected2 = (200.0/20 + 80.0/4) / 2   # (10+20)/2 = 15
print(f"  {PASS if approx(kpi2.nttp(), expected2) else FAIL}  nttp=(10+20)/2=15.0: {kpi2.nttp():.2f}")

# zero completed -> 0.0
kpi3 = KPITracker()
print(f"  {PASS if approx(kpi3.nttp(), 0.0) else FAIL}  nttp=0.0 with no completed trucks: {kpi3.nttp()}")

# ─────────────────────────────────────────────────────────────────────────────
# [10] snapshot_utilization and utilization_std
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] snapshot_utilization and utilization_std")

gha_keys = list(params["ghas"].keys())

kpi = KPITracker()
# uniform occupancy across all GHAs -> std should be ~0
terminals_uniform = {g: MockTerminal(exp=0.5, imp=0.5) for g in gha_keys}
for _ in range(10):
    kpi.snapshot_utilization(terminals_uniform)
std = kpi.utilization_std()
print(f"  {PASS if std < 0.01 else FAIL}  uniform occupancy -> util_std≈0: {std:.6f}")

# varied occupancy -> nonzero std
kpi2 = KPITracker()
varied = {
    gha_keys[0]: MockTerminal(exp=0.9, imp=0.9),
    gha_keys[1]: MockTerminal(exp=0.1, imp=0.1),
    gha_keys[2]: MockTerminal(exp=0.5, imp=0.5),
    gha_keys[3]: MockTerminal(exp=0.5, imp=0.5),
}
for _ in range(10):
    kpi2.snapshot_utilization(varied)
std2 = kpi2.utilization_std()
print(f"  {PASS if std2 > 0.1 else FAIL}  varied occupancy -> util_std>0.1: {std2:.4f}")

# single snapshot still works
kpi3 = KPITracker()
kpi3.snapshot_utilization(terminals_uniform)
std3 = kpi3.utilization_std()
print(f"  {PASS if isinstance(std3, float) else FAIL}  utilization_std returns float after single snapshot: {std3}")

# no snapshots -> 0.0
kpi4 = KPITracker()
print(f"  {PASS if approx(kpi4.utilization_std(), 0.0) else FAIL}  util_std=0.0 with no snapshots: {kpi4.utilization_std()}")

# ─────────────────────────────────────────────────────────────────────────────
# [11] global_reward
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] global_reward")

# zero state -> 0.0
kpi = KPITracker()
print(f"  {PASS if approx(kpi.global_reward(), 0.0) else FAIL}  global_reward=0 with no events: {kpi.global_reward()}")

# known wpr, zero util_std
kpi = KPITracker()
kpi.ingest([
    ev_gate_in("T1", 0.0),
    ev_gha_in("T1", 0.0, "dnata"),
    ev_dock_start("T1", 10.0, "dnata"),
    ev_dock_end("T1", 40.0, "dnata"),
])
expected_gr = -(W["wpr_global"] * (10.0/30.0) + W["util_std"] * 0.0)
print(f"  {PASS if approx(kpi.global_reward(), expected_gr) else FAIL}  global_reward with wpr=1/3: expected={expected_gr:.6f} got={kpi.global_reward():.6f}")

# reward is always <= 0
for _ in range(10):
    kpi_r = KPITracker()
    kpi_r.ingest([
        ev_gate_in("T1", 0.0),
        ev_gha_in("T1", 0.0, "dnata"),
        ev_dock_start("T1", float(_ * 3), "dnata"),
        ev_dock_end("T1", float(_ * 3 + 20), "dnata"),
    ])
    assert kpi_r.global_reward() <= 0.0
print(f"  {PASS}  global_reward <= 0 across 10 random scenarios")

# ─────────────────────────────────────────────────────────────────────────────
# [12] transporter_reward — delta-based, step by step
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] transporter_reward — delta tracking")

kpi = KPITracker()
dtp0 = MockDTP()
r0 = kpi.transporter_reward(dtp0)
print(f"  {PASS if approx(r0, 0.0) else FAIL}  zero state -> transporter_reward=0: {r0}")

# add wait
kpi.ingest([
    ev_gate_in("T1", 0.0),
    ev_gha_in("T1", 0.0, "dnata"),
    ev_dock_start("T1", 20.0, "dnata"),   # wait=20
])
dtp1 = MockDTP()
r1 = kpi.transporter_reward(dtp1)
expected_r1 = -(W["wait_per_min"] * 20.0)
print(f"  {PASS if approx(r1, expected_r1) else FAIL}  wait=20 -> reward={expected_r1:.4f}: {r1:.4f}")

# second call: wait delta should be 0 (no new wait)
r2 = kpi.transporter_reward(dtp1)
print(f"  {PASS if approx(r2, 0.0) else FAIL}  second call with no new wait -> reward=0: {r2}")

# no_show penalty
kpi2 = KPITracker()
dtp_ns = MockDTP(no_shows={"TRK-NS": 2}, late={"TRK-L": 1})
r_ns = kpi2.transporter_reward(dtp_ns)
expected_ns = -(W["no_show"] * 2 + W["missed_slot"] * 1)
print(f"  {PASS if approx(r_ns, expected_ns) else FAIL}  2 no_shows + 1 late: expected={expected_ns:.2f} got={r_ns:.2f}")

# delta: second call same dtp should be 0
r_ns2 = kpi2.transporter_reward(dtp_ns)
print(f"  {PASS if approx(r_ns2, 0.0) else FAIL}  same dtp second call -> delta=0: {r_ns2}")

# cumulative: add more no_shows
dtp_ns2 = MockDTP(no_shows={"TRK-NS": 3}, late={"TRK-L": 1})
r_ns3 = kpi2.transporter_reward(dtp_ns2)
expected_ns3 = -(W["no_show"] * 1)   # only 1 new no_show
print(f"  {PASS if approx(r_ns3, expected_ns3) else FAIL}  1 new no_show: expected={expected_ns3:.2f} got={r_ns3:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# [13] gha_reward — delta processed, queue, utilization
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] gha_reward — utilization + delta_proc - queue")

gha = gha_keys[0]

# first call from zero
kpi = KPITracker()
t = MockTerminal(exp=0.8, imp=0.6, exp_q=0.1, imp_q=0.2, proc_exp=10, proc_imp=5)
r = kpi.gha_reward(gha, t)
util  = (0.8 + 0.6) / 2
q     = 0.1 + 0.2
delta = 10 + 5   # first call, prev=0
expected_r = W["dock_util"] * util + W["parcel_on_time"] * delta - W["queue_per_step"] * q
print(f"  {PASS if approx(r, expected_r) else FAIL}  first call: expected={expected_r:.4f} got={r:.4f}")

# second call same terminal: delta_proc=0
r2 = kpi.gha_reward(gha, t)
expected_r2 = W["dock_util"] * util - W["queue_per_step"] * q
print(f"  {PASS if approx(r2, expected_r2) else FAIL}  second call, no new procs: expected={expected_r2:.4f} got={r2:.4f}")

# new processed trucks
t2 = MockTerminal(exp=0.8, imp=0.6, exp_q=0.1, imp_q=0.2, proc_exp=13, proc_imp=7)
r3 = kpi.gha_reward(gha, t2)
delta3 = (13 + 7) - (10 + 5)
expected_r3 = W["dock_util"] * util + W["parcel_on_time"] * delta3 - W["queue_per_step"] * q
print(f"  {PASS if approx(r3, expected_r3) else FAIL}  +5 processed: expected={expected_r3:.4f} got={r3:.4f}")

# zero occupancy, zero queue, zero processed -> reward=0
kpi2 = KPITracker()
t_zero = MockTerminal(exp=0.0, imp=0.0, exp_q=0.0, imp_q=0.0, proc_exp=0, proc_imp=0)
r_zero = kpi2.gha_reward(gha, t_zero)
print(f"  {PASS if approx(r_zero, 0.0) else FAIL}  zero state -> gha_reward=0: {r_zero}")

# ─────────────────────────────────────────────────────────────────────────────
# [14] summary — returns expected keys
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14] summary")

kpi = KPITracker()
kpi.ingest([
    ev_gate_in("T1", 0.0, parcels=10),
    ev_gha_in("T1", 10.0, "dnata"),
    ev_dock_start("T1", 20.0, "dnata"),
    ev_dock_end("T1", 50.0, "dnata"),
    ev_gate_out("T1", 60.0),
])
s = kpi.summary()
expected_keys = {"wpr", "peak_wpr", "nttp", "util_std", "n_completed", "global_reward"}
print(f"  {PASS if expected_keys == set(s.keys()) else FAIL}  summary has correct keys: {set(s.keys())}")
print(f"  {PASS if s['n_completed'] == 1 else FAIL}  n_completed=1: {s['n_completed']}")
print(f"  {PASS if approx(s['nttp'], 60.0/10) else FAIL}  nttp=6.0: {s['nttp']}")
print(f"  {PASS if approx(s['wpr'], 10.0/30.0) else FAIL}  wpr=1/3: {s['wpr']:.4f}")
print(f"  {PASS if s['global_reward'] <= 0 else FAIL}  global_reward <= 0: {s['global_reward']:.4f}")

# empty episode
kpi2 = KPITracker()
s2 = kpi2.summary()
for k in expected_keys:
    tag = PASS if k in s2 else FAIL
    print(f"  {tag}  '{k}' present in empty-episode summary: {s2[k]}")

print("\n" + "=" * 65)
print("  KPITracker tests complete")
print("=" * 65)