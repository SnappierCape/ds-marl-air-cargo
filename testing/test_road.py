# =============================================================================
# TEST ─ ROAD MODULE
# =============================================================================

import sys
import os
import numpy as np
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.road import RoadNetwork

params_road = {
    "sigma": 0.20,
    "lbound": 0.5,
    "ubound": 3.0,
    "segments": {
        "N0_N1": 2.0, "N0_N2": 2.0, "N0_N3": 1.0, "N0_N4": 3.0, "N0_N5": 5.0,
        "N1_N2": 0.5, "N1_N3": 2.0, "N1_N4": 4.0, "N1_N5": 6.0,
        "N2_N3": 2.0, "N2_N4": 4.0, "N2_N5": 6.0,
        "N3_N4": 2.0, "N3_N5": 4.0,
        "N4_N5": 2.0
    },
    "nodes": {
        "gate": "N0", "tp3": "N1", "dnata": "N2",
        "klm": "N3", "wfs": "N4", "swiss": "N5"
    }
}

road = RoadNetwork(params_road)
SAMPLES = 500
PASS = "[PASS]"
FAIL = "[FAIL]"

print("=" * 60)
print("  TEST SUITE: RoadNetwork")
print("=" * 60)

# ── Test _apply_noise ────────────────────────────────────────────────────────
print("\n[1] _apply_noise — bounds and positivity")
errors = []
for base in [0.5, 1.0, 2.0, 5.0, 10.0]:
    samples = [road._apply_noise(base) for _ in range(SAMPLES)]
    lo, hi = base * road.lbound, base * road.ubound
    oob = [s for s in samples if s < lo or s > hi]
    non_pos = [s for s in samples if s <= 0]
    tag = PASS if not oob and not non_pos else FAIL
    print(f"  {tag}  base={base:.1f}m | "
          f"min={min(samples):.3f} max={max(samples):.3f} mean={np.mean(samples):.3f} | "
          f"out-of-bounds={len(oob)}/{SAMPLES}  non-positive={len(non_pos)}")
    if oob or non_pos:
        errors.append(f"base={base} produced {len(oob)} OOB and {len(non_pos)} non-positive samples")

print(f"\n  {PASS if not errors else FAIL}  _apply_noise summary: {'all checks passed' if not errors else errors}")

# ── Test _apply_noise — invalid input ────────────────────────────────────────
print("\n[2] _apply_noise — invalid input (base_time <= 0)")
for bad in [0, -1.0, -100]:
    try:
        road._apply_noise(bad)
        print(f"  {FAIL}  base={bad} — expected ValueError, got none")
    except ValueError as e:
        print(f"  {PASS}  base={bad} — raised ValueError: {e}")

# ── Test time_from_to — all valid pairs ──────────────────────────────────────
print("\n[3] time_from_to — all valid node pairs")
node_names = list(params_road["nodes"].keys())
all_ok = True
for a in node_names:
    for b in node_names:
        if a == b:
            continue
        try:
            t = road.time_from_to(a, b)
            t_rev = road.time_from_to(b, a)
            if t <= 0 or t_rev <= 0:
                print(f"  {FAIL}  {a} -> {b}: non-positive time {t:.3f}")
                all_ok = False
        except Exception as e:
            print(f"  {FAIL}  {a} -> {b}: unexpected error: {e}")
            all_ok = False
if all_ok:
    print(f"  {PASS}  all {len(node_names)*(len(node_names)-1)} directed pairs returned positive travel times")

# ── Test time_from_to — stochastic variability ───────────────────────────────
print("\n[4] time_from_to — stochastic variability per route")
routes = [("gate", "tp3"), ("gate", "dnata"), ("tp3", "klm"), ("wfs", "swiss")]
for a, b in routes:
    samples = [road.time_from_to(a, b) for _ in range(SAMPLES)]
    std = np.std(samples)
    tag = PASS if std > 0 else FAIL
    print(f"  {tag}  {a:<10} -> {b:<10} | "
          f"min={min(samples):.2f} max={max(samples):.2f} "
          f"mean={np.mean(samples):.2f} std={std:.3f}")

# ── Test time_from_to — invalid nodes ────────────────────────────────────────
print("\n[5] time_from_to — invalid node names")
bad_pairs = [("gate", "narnia"), ("foo", "tp3"), ("xyz", "abc")]
for a, b in bad_pairs:
    try:
        road.time_from_to(a, b)
        print(f"  {FAIL}  ({a}, {b}) — expected ValueError, got none")
    except ValueError as e:
        print(f"  {PASS}  ({a}, {b}) — raised ValueError: {e}")

# ── Test symmetry (key normalization) ────────────────────────────────────────
print("\n[6] time_from_to — segment key symmetry (A->B same segment as B->A)")
sym_pairs = [("gate", "tp3"), ("dnata", "klm"), ("wfs", "swiss")]
for a, b in sym_pairs:
    samples_ab = [road.time_from_to(a, b) for _ in range(SAMPLES)]
    samples_ba = [road.time_from_to(b, a) for _ in range(SAMPLES)]
    mean_ab, mean_ba = np.mean(samples_ab), np.mean(samples_ba)
    tag = PASS if abs(mean_ab - mean_ba) < 0.5 else FAIL
    print(f"  {tag}  {a} <-> {b} | mean(a->b)={mean_ab:.3f}  mean(b->a)={mean_ba:.3f}")

print("\n" + "=" * 60)
print("  RoadNetwork tests complete")
print("=" * 60)