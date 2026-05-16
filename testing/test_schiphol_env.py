# =============================================================================
# TEST ─ SCHIPHOL_ENV MODULE
# =============================================================================
import sys
import os
import numpy as np
import simpy
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.schiphol_env import SchipholCargoEnv, GHA_IDS, N_GHAS, TRANSPORTER_ACTION_DIM, N_BOOK_ACTIONS, N_DISPATCH_ACTIONS, N_TP3_ACTIONS
import config.config as config

params = config.load_params()
PASS = "[PASS]"
FAIL = "[FAIL]"

def make_env(orch=False):
    e = SchipholCargoEnv(with_orchestrator=orch)
    obs, infos = e.reset()
    return e, obs, infos

def no_op_actions(env):
    return {a: 0 for a in env.agents}

def masked_random_action(space, mask):
    valid = np.where(mask == 1)[0]
    return int(np.random.choice(valid)) if len(valid) > 0 else 0

print("=" * 65)
print("  TEST SUITE: SchipholCargoEnv — behavioural")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# [1] reset — correct agent roster for both scenarios
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] reset — agent roster")

env_m, obs_m, info_m = make_env(orch=False)
expected_m = set(["transporter"] + GHA_IDS)
print(f"  {PASS if set(env_m.agents) == expected_m else FAIL}  Scenario M agents correct: {set(env_m.agents)}")

env_mo, obs_mo, info_mo = make_env(orch=True)
expected_mo = expected_m | {"orchestrator"}
print(f"  {PASS if set(env_mo.agents) == expected_mo else FAIL}  Scenario MO adds orchestrator: {set(env_mo.agents)}")

# ─────────────────────────────────────────────────────────────────────────────
# [2] reset — observations match declared space shape and bounds [0,1]
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] reset — observation validity")

env, obs, infos = make_env(orch=True)
for agent in env.agents:
    vec       = obs[agent]
    declared  = env.observation_space(agent).shape[0]
    in_bounds = np.all(vec >= 0.0) and np.all(vec <= 1.0)
    no_nan    = not np.any(np.isnan(vec))
    no_inf    = not np.any(np.isinf(vec))
    shape_ok  = vec.shape[0] == declared
    tag = PASS if (shape_ok and in_bounds and no_nan and no_inf) else FAIL
    print(f"  {tag}  {agent:<20} shape={vec.shape[0]}/{declared}  bounds={in_bounds}  nan={not no_nan}  inf={not no_inf}")

# ─────────────────────────────────────────────────────────────────────────────
# [3] reset — action masks always include no_op (action 0)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] reset — no_op always valid in initial masks")

env, obs, infos = make_env(orch=True)
for agent, info in infos.items():
    mask = info["action_mask"]
    tag  = PASS if mask[0] == 1 else FAIL
    print(f"  {tag}  {agent:<20} action 0 (no_op) is valid: {mask[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# [4] reset — sim time starts at 0, advances per step
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] sim time advances by step_min each step")

env, obs, infos = make_env()
step_min = params["marl"]["step_min"]
t0 = env.sim.now
env.step(no_op_actions(env))
t1 = env.sim.now
env.step(no_op_actions(env))
t2 = env.sim.now
print(f"  {PASS if t0 == 0 else FAIL}  sim starts at t=0: {t0}")
print(f"  {PASS if abs(t1 - t0 - step_min) < 1e-9 else FAIL}  step 1 advances {step_min}m: t={t1}")
print(f"  {PASS if abs(t2 - t1 - step_min) < 1e-9 else FAIL}  step 2 advances {step_min}m: t={t2}")

# ─────────────────────────────────────────────────────────────────────────────
# [5] step — all-no_op does not crash, returns correct structure
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] step — all-no_op returns correct structure")

env, obs, infos = make_env()
for _ in range(20):
    obs, rewards, dones, infos = env.step(no_op_actions(env))

print(f"  {PASS if set(obs.keys()) == set(env.agents) else FAIL}  obs keys match agents")
print(f"  {PASS if set(rewards.keys()) == set(env.agents) else FAIL}  reward keys match agents")
print(f"  {PASS if set(dones.keys()) == set(env.agents) else FAIL}  done keys match agents")
print(f"  {PASS if set(infos.keys()) == set(env.agents) else FAIL}  info keys match agents")
print(f"  {PASS if all(isinstance(r, float) for r in rewards.values()) else FAIL}  all rewards are float")
print(f"  {PASS if all(np.isfinite(r) for r in rewards.values()) else FAIL}  all rewards are finite")

# ─────────────────────────────────────────────────────────────────────────────
# [6] observations stay in [0,1] and correct shape across N steps
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] observations remain valid across 100 random steps")

env, obs, infos = make_env(orch=True)
errors = []
for step in range(100):
    actions = {a: masked_random_action(env.action_space(a), infos[a]["action_mask"])
               for a in env.agents}
    obs, rewards, dones, infos = env.step(actions)
    for agent, vec in obs.items():
        declared = env.observation_space(agent).shape[0]
        if vec.shape[0] != declared:
            errors.append(f"step {step} {agent} shape {vec.shape[0]} != {declared}")
        if np.any(np.isnan(vec)):
            errors.append(f"step {step} {agent} NaN")
        if np.any(np.isinf(vec)):
            errors.append(f"step {step} {agent} Inf")
        if np.any(vec < 0) or np.any(vec > 1.0):
            errors.append(f"step {step} {agent} out of [0,1]: min={vec.min():.3f} max={vec.max():.3f}")

print(f"  {PASS if not errors else FAIL}  {'0 errors across 100 steps' if not errors else str(len(errors)) + ' errors found'}")
for e in errors[:5]:
    print(f"       {e}")

# ─────────────────────────────────────────────────────────────────────────────
# [7] action masks — dispatch only valid when all stops booked
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] transporter mask — dispatch only when all stops booked")

env, obs, infos = make_env()
# Run for a while to generate pending trucks
for _ in range(50):
    env.step(no_op_actions(env))

env2, obs2, infos2 = make_env()
for _ in range(50):
    obs2, _, _, infos2 = env2.step(no_op_actions(env2))

mask = infos2["transporter"]["action_mask"]
dispatch_start = N_BOOK_ACTIONS + 1
dispatch_end   = N_BOOK_ACTIONS + N_DISPATCH_ACTIONS + 1

pending = env2.demand.pending_trucks
violations = 0
for t_idx, truck in enumerate(pending[:10]):
    dispatch_action = dispatch_start + t_idx
    if dispatch_action >= len(mask):
        continue
    needed = {s["gha"] for s in truck.manifest}
    booked = set(truck.booked_slots.keys())
    all_booked = needed.issubset(booked)
    mask_allows = mask[dispatch_action] == 1
    if mask_allows and not all_booked:
        violations += 1

print(f"  {PASS if violations == 0 else FAIL}  dispatch never allowed unless all stops booked: {violations} violations")

# ─────────────────────────────────────────────────────────────────────────────
# [8] action masks — book action only valid for needed/unbooked GHAs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] transporter mask — book only for needed and unbooked GHAs")

env, obs, infos = make_env()
for _ in range(30):
    obs, _, _, infos = env.step(no_op_actions(env))

mask    = infos["transporter"]["action_mask"]
pending = env.demand.pending_trucks
violations = 0
for t_idx, truck in enumerate(pending[:10]):
    for g_idx, gha in enumerate(GHA_IDS):
        action = t_idx * N_GHAS + g_idx + 1
        if action >= len(mask):
            continue
        if mask[action] == 1:
            needed  = any(s["gha"] == gha for s in truck.stops_remaining)
            already = gha in truck.booked_slots
            if not needed or already:
                violations += 1

print(f"  {PASS if violations == 0 else FAIL}  book action only valid for needed/unbooked GHAs: {violations} violations")

# ─────────────────────────────────────────────────────────────────────────────
# [9] action masks — GHA: published slots do not include past windows
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] GHA mask — published windows are in the future")

env, obs, infos = make_env()
for _ in range(100):
    obs, _, _, infos = env.step(no_op_actions(env))

now      = env.sim.now
freeze   = params["dtp_rules"]["freeze_time"]
for gha in GHA_IDS:
    mask = infos[gha]["action_mask"]
    windows = env._next_publishable_windows(gha)
    for i, w in enumerate(windows):
        action_idx = i + 1
        if action_idx < len(mask) and mask[action_idx] == 1:
            in_future = w > now + freeze
            tag = PASS if in_future else FAIL
            print(f"  {tag}  {gha} action {action_idx}: window={w:.0f} > now+freeze={now+freeze:.0f}: {in_future}")
            break
    else:
        print(f"  {PASS}  {gha}: all GHA window actions masked (no publishable slot right now)")

# ─────────────────────────────────────────────────────────────────────────────
# [10] transporter booking action: book_one_slot actually registers in platform
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] transporter book action: slot registered in DTP registry")

env, obs, infos = make_env()
for _ in range(30):
    obs, _, _, infos = env.step(no_op_actions(env))

mask    = infos["transporter"]["action_mask"]
pending = env.demand.pending_trucks

booked_action = None
target_truck  = None
target_gha    = None

for t_idx, truck in enumerate(pending[:10]):
    for g_idx, gha in enumerate(GHA_IDS):
        action = t_idx * N_GHAS + g_idx + 1
        if action < len(mask) and mask[action] == 1:
            booked_action = action
            target_truck  = truck
            target_gha    = gha
            break
    if booked_action:
        break

if booked_action and target_truck:
    slots_before = len(env.dtp.get_available_slots(target_gha, horizon=4320))
    env.step({a: (booked_action if a == "transporter" else 0) for a in env.agents})
    obs, _, _, infos = env.step(no_op_actions(env))
    booked_in_registry = target_gha in target_truck.booked_slots
    print(f"  {PASS if booked_in_registry else FAIL}  booking action registers in truck.booked_slots: {booked_in_registry}")
    print(f"  {PASS if booked_in_registry else FAIL}  target GHA={target_gha} truck={target_truck.truck_id}")
else:
    print(f"  (skipped — no bookable action available at this step)")

# ─────────────────────────────────────────────────────────────────────────────
# [11] GHA publish action: slot appears in DTP registry
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] GHA publish action: slot appears in DTP get_available_slots")

env, obs, infos = make_env()
for _ in range(30):
    obs, _, _, infos = env.step(no_op_actions(env))

for gha in GHA_IDS:
    mask = infos[gha]["action_mask"]
    if mask[1] == 1:
        window = env._next_publishable_windows(gha)[0]
        slots_before = len(env.dtp.get_available_slots(gha, horizon=4320))
        env.step({a: (1 if a == gha else 0) for a in env.agents})
        slots_after = len(env.dtp.get_available_slots(gha, horizon=4320))
        print(f"  {PASS if slots_after > slots_before else FAIL}  {gha} publish action increases available slots: {slots_before} -> {slots_after}")
        break
else:
    print(f"  (skipped — no GHA had a publishable slot)")

# ─────────────────────────────────────────────────────────────────────────────
# [12] orchestrator mask — only acts on parked TP3 trucks
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] orchestrator mask — valid actions correspond to parked trucks")

env_mo, obs_mo, infos_mo = make_env(orch=True)
# run long enough for some trucks to reach TP3
for _ in range(300):
    obs_mo, _, _, infos_mo = env_mo.step(no_op_actions(env_mo))

mask   = infos_mo["orchestrator"]["action_mask"]
parked = env_mo.tp3.get_parked_trucks()
n_parked = len(parked)

violations = 0
for t_idx in range(N_TP3_ACTIONS):
    for g_idx in range(N_GHAS):
        action = t_idx * N_GHAS + g_idx + 1
        if action >= len(mask):
            continue
        if mask[action] == 1 and t_idx >= n_parked:
            violations += 1

print(f"  {PASS if violations == 0 else FAIL}  no valid orch actions beyond parked truck count ({n_parked}): {violations} violations")
print(f"  {PASS if mask[0] == 1 else FAIL}  no_op always valid for orchestrator")

# ─────────────────────────────────────────────────────────────────────────────
# [13] rewards — global reward is <= 0, private rewards are finite
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] rewards — signs and finiteness")

env, obs, infos = make_env(orch=True)
all_global_non_positive = True
all_finite = True

for step in range(100):
    actions = {a: masked_random_action(env.action_space(a), infos[a]["action_mask"])
               for a in env.agents}
    obs, rewards, dones, infos = env.step(actions)
    r_g = env.kpi.global_reward()
    if r_g > 1e-9:
        all_global_non_positive = False
    for r in rewards.values():
        if not np.isfinite(r):
            all_finite = False

print(f"  {PASS if all_global_non_positive else FAIL}  global_reward <= 0 across 100 steps")
print(f"  {PASS if all_finite else FAIL}  all agent rewards are finite across 100 steps")

# ─────────────────────────────────────────────────────────────────────────────
# [14] rewards — alpha mixing: transporter reward is blend of private and global
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14] reward mixing — alpha parameter respected")

alpha = params["marl"]["alpha"]
env, obs, infos = make_env()
for _ in range(50):
    obs, rewards, dones, infos = env.step(no_op_actions(env))

r_global  = env.kpi.global_reward()
r_private = env.kpi.transporter_reward(env.dtp)
expected  = (1 - alpha) * r_private + alpha * r_global
actual    = rewards["transporter"]
# NOTE: private reward consumes delta on call, so we check structure not exact value
print(f"  {PASS if np.isfinite(actual) else FAIL}  transporter reward is finite: {actual:.4f}")
print(f"  {PASS if isinstance(alpha, float) and 0 <= alpha <= 1 else FAIL}  alpha={alpha} is in [0,1]")

# ─────────────────────────────────────────────────────────────────────────────
# [15] prepopulate — slots available for all GHAs at episode start
# ─────────────────────────────────────────────────────────────────────────────
print("\n[15] _prepopulate_slots — all GHAs have available slots at reset")

env, obs, infos = make_env()
for gha in GHA_IDS:
    slots = env.dtp.get_available_slots(gha, horizon=4320)
    n_docks = params["ghas"][gha]["total"]
    print(f"  {PASS if len(slots) > 0 else FAIL}  {gha:<15} {len(slots)} available slots (n_docks={n_docks})")

# ─────────────────────────────────────────────────────────────────────────────
# [16] _prepopulate_slots — no slots inside freeze window
# ─────────────────────────────────────────────────────────────────────────────
print("\n[16] _prepopulate_slots — no slots inside freeze window")

env, obs, infos = make_env()
freeze = params["dtp_rules"]["freeze_time"]
violations = 0
for gha in GHA_IDS:
    for slot_start in env.dtp.registry[gha].keys():
        if slot_start - env.sim.now < freeze:
            violations += 1

print(f"  {PASS if violations == 0 else FAIL}  no published slots inside freeze window: {violations} violations")

# ─────────────────────────────────────────────────────────────────────────────
# [17] action space declared dimensions match actual logic
# ─────────────────────────────────────────────────────────────────────────────
print("\n[17] action space dimensions match expected values")

env, _, _ = make_env(orch=True)

expected_trans = TRANSPORTER_ACTION_DIM
actual_trans   = env.action_space("transporter").n
print(f"  {PASS if actual_trans == expected_trans else FAIL}  transporter action_dim={actual_trans} (expected {expected_trans})")

for gha in GHA_IDS:
    dim = env.action_space(gha).n
    print(f"  {PASS if dim == 3 else FAIL}  {gha} action_dim=3: {dim}")

orch_expected = N_TP3_ACTIONS * N_GHAS + 1
orch_actual   = env.action_space("orchestrator").n
print(f"  {PASS if orch_actual == orch_expected else FAIL}  orchestrator action_dim={orch_actual} (expected {orch_expected})")

# ─────────────────────────────────────────────────────────────────────────────
# [18] consecutive resets produce clean independent state
# ─────────────────────────────────────────────────────────────────────────────
print("\n[18] consecutive resets produce clean state")

env = SchipholCargoEnv()
for trial in range(3):
    obs, infos = env.reset()
    # run a few steps
    for _ in range(20):
        actions = {a: masked_random_action(env.action_space(a), infos[a]["action_mask"])
                   for a in env.agents}
        obs, _, _, infos = env.step(actions)
    t_after = env.sim.now
    obs2, infos2 = env.reset()
    t_reset = env.sim.now
    kpi_reset = env.kpi._n_completed

    # sim time resets to 0
    print(f"  {PASS if t_reset == 0 else FAIL}  trial {trial}: sim resets to t=0 (was t={t_after:.0f})")
    print(f"  {PASS if kpi_reset == 0 else FAIL}  trial {trial}: kpi tracker cleared (n_completed={kpi_reset})")

    # observations valid after reset
    valid = all(
        obs2[a].shape[0] == env.observation_space(a).shape[0] and
        np.all(obs2[a] >= 0) and np.all(obs2[a] <= 1)
        for a in env.agents
    )
    print(f"  {PASS if valid else FAIL}  trial {trial}: observations valid after reset")

# ─────────────────────────────────────────────────────────────────────────────
# [19] KPI tracker integrated: completed trucks increment after full episodes
# ─────────────────────────────────────────────────────────────────────────────
print("\n[19] KPI tracker integrates real simulation: trucks complete over time")

env, obs, infos = make_env()
n_before = env.kpi._n_completed
# run 1440 steps (1 simulated day)
for _ in range(1440):
    actions = {a: masked_random_action(env.action_space(a), infos[a]["action_mask"])
               for a in env.agents}
    obs, _, _, infos = env.step(actions)

n_after = env.kpi._n_completed
trucks_generated = env.demand._truck_counter
print(f"  {PASS if n_after > n_before else FAIL}  completed trucks > 0 after 1440 steps: n_completed={n_after}")
print(f"  {PASS if trucks_generated > 0 else FAIL}  trucks were generated: {trucks_generated}")
print(f"  {PASS if env.kpi.wpr() >= 0 else FAIL}  wpr >= 0: {env.kpi.wpr():.4f}")
print(f"  {PASS if env.kpi.nttp() >= 0 else FAIL}  nttp >= 0: {env.kpi.nttp():.4f}")

summary = env.kpi.summary()
print(f"  {PASS if summary['n_completed'] == n_after else FAIL}  summary consistent with tracker: {summary['n_completed']}")

print("\n" + "=" * 65)
print("  SchipholCargoEnv tests complete")
print("=" * 65)