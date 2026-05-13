# =============================================================================
# TEST ─ SERVICE_TIME MODULE
# =============================================================================

import sys
import os
import numpy as np
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.service_time import ServiceTimeModel

cfg = {
    "service_time": {
        "export": {"mu": 3.5, "sigma": 0.4, "bounds": [11, 180]},
        "import": {"mu": 2.8, "sigma": 0.3, "bounds": [8,  120]}
    }
}

model = ServiceTimeModel(cfg)
SAMPLES = 1000
PASS = "[PASS]"
FAIL = "[FAIL]"

print("=" * 60)
print("  TEST SUITE: ServiceTimeModel")
print("=" * 60)

# ── Test sample — output bounds ───────────────────────────────────────────────
print("\n[1] sample — output within configured bounds")
for flow in ("export", "import"):
    lo, hi = cfg["service_time"][flow]["bounds"]
    samples = [model.sample(flow) for _ in range(SAMPLES)]
    oob = [s for s in samples if s < lo or s > hi]
    non_pos = [s for s in samples if s <= 0]
    tag = PASS if not oob and not non_pos else FAIL
    print(f"  {tag}  flow={flow:<8} | bounds=[{lo}, {hi}] | "
          f"min={min(samples):.2f} max={max(samples):.2f} mean={np.mean(samples):.2f} "
          f"std={np.std(samples):.2f} | out-of-bounds={len(oob)}/{SAMPLES}")

# ── Test sample — return type ─────────────────────────────────────────────────
print("\n[2] sample — return type is float")
for flow in ("export", "import"):
    result = model.sample(flow)
    tag = PASS if isinstance(result, float) else FAIL
    print(f"  {tag}  flow={flow:<8} | type={type(result).__name__}  value={result:.4f}")

# ── Test sample — stochastic (not constant) ───────────────────────────────────
print("\n[3] sample — output is stochastic (non-zero variance)")
for flow in ("export", "import"):
    samples = [model.sample(flow) for _ in range(SAMPLES)]
    std = np.std(samples)
    tag = PASS if std > 0.1 else FAIL
    print(f"  {tag}  flow={flow:<8} | std={std:.4f}  (expected > 0.1)")

# ── Test sample — export > import on average (mu difference) ─────────────────
print("\n[4] sample — export mean > import mean (mu_exp=3.5 > mu_imp=2.8)")
exp_samples = [model.sample("export") for _ in range(SAMPLES)]
imp_samples = [model.sample("import") for _ in range(SAMPLES)]
exp_mean, imp_mean = np.mean(exp_samples), np.mean(imp_samples)
tag = PASS if exp_mean > imp_mean else FAIL
print(f"  {tag}  export mean={exp_mean:.2f}  import mean={imp_mean:.2f}")

# ── Test sample — invalid flow type ──────────────────────────────────────────
print("\n[5] sample — invalid flow type raises ValueError")
for bad in ["Export", "IMPORT", "cargo", "", "none"]:
    try:
        model.sample(bad)
        print(f"  {FAIL}  flow='{bad}' — expected ValueError, got none")
    except ValueError as e:
        print(f"  {PASS}  flow='{bad}' — raised ValueError: {e}")

# ── Test mean — analytical vs empirical ──────────────────────────────────────
print("\n[6] mean — analytical value close to empirical sample mean")
for flow in ("export", "import"):
    analytical = model.mean(flow)
    empirical = np.mean([model.sample(flow) for _ in range(5000)])
    rel_err = abs(analytical - empirical) / analytical
    tag = PASS if rel_err < 0.10 else FAIL
    print(f"  {tag}  flow={flow:<8} | analytical={analytical:.3f}  "
          f"empirical={empirical:.3f}  rel_error={rel_err:.2%}")

# ── Test mean — return type is float ─────────────────────────────────────────
print("\n[7] mean — return type is float")
for flow in ("export", "import"):
    result = model.mean(flow)
    tag = PASS if isinstance(result, float) else FAIL
    print(f"  {tag}  flow={flow:<8} | type={type(result).__name__}  value={result:.4f}")

# ── Test mean — invalid flow type ────────────────────────────────────────────
print("\n[8] mean — invalid flow type raises ValueError")
for bad in ["Export", "IMPORT", "cargo"]:
    try:
        model.mean(bad)
        print(f"  {FAIL}  flow='{bad}' — expected ValueError, got none")
    except ValueError as e:
        print(f"  {PASS}  flow='{bad}' — raised ValueError: {e}")

# ── Test mean — export > import (larger mu) ───────────────────────────────────
print("\n[9] mean — export analytical mean > import analytical mean")
exp_mean_a = model.mean("export")
imp_mean_a = model.mean("import")
tag = PASS if exp_mean_a > imp_mean_a else FAIL
print(f"  {tag}  export={exp_mean_a:.3f}  import={imp_mean_a:.3f}")

print("\n" + "=" * 60)
print("  ServiceTimeModel tests complete")
print("=" * 60)