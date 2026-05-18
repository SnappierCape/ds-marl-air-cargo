# =============================================================================
# SANITY CHECK SCRIPT
# =============================================================================
# DESCRIPTION:
#     Runs one full episode with random actions to verify the environment
#     produces valid observations, rewards, and doesn't crash.
#
# HOW TO RUN:
#     uv run testing/simulation.py
#     uv run testing/simulation.py --orchestrator (Scenario MO)
#     uv run testing/simulation.py --steps 5000 (longer episode)
# =============================================================================
import sys
import argparse

import numpy as np

from env.schiphol_env import SchipholCargoEnv

from config.config import load_params
params = load_params()

# =============================================================================
# MAIN TEST LOGIC
# =============================================================================
def check_obs(obs: dict, step: int) -> list:
    """Returns list of error strings found in observations."""
    errors = []
    for agent, vec in obs.items():
        if np.any(np.isnan(vec)):
            errors.append(f"step {step} | {agent} | NaN in observation")
        if np.any(np.isinf(vec)):
            errors.append(f"step {step} | {agent} | Inf in observation")
        if np.any(vec < 0) or np.any(vec > 1):
            errors.append(
                f"step {step} | {agent} | values outside [0,1]: "
                f"min={vec.min():.3f} max={vec.max():.3f}"
            )
    return errors

def check_rewards(rewards: dict, step: int) -> list:
    """Returns list of error strings found in rewards."""
    errors = []
    for agent, r in rewards.items():
        if np.isnan(r):
            errors.append(f"step {step} | {agent} | NaN reward")
        if np.isinf(r):
            errors.append(f"step {step} | {agent} | Inf reward")
    return errors

def check_masks(infos: dict, action_spaces: dict, step: int) -> list:
    """Returns list of error strings found in action masks."""
    errors = []
    for agent, info in infos.items():
        mask = info.get("action_mask")
        if mask is None:
            errors.append(f"step {step} | {agent} | missing action_mask")
            continue
        expected_len = action_spaces[agent].n
        if len(mask) != expected_len:
            errors.append(
                f"step {step} | {agent} | mask length {len(mask)} "
                f"!= action_space size {expected_len}"
            )
        if not np.any(mask):
            errors.append(
                f"step {step} | {agent} | all actions masked "
                f"(no valid action available)"
            )
    return errors

def sample_masked_action(action_space, mask) -> int:
    """Sample a random valid action respecting the mask."""
    valid = np.where(mask == 1)[0]
    if len(valid) == 0:
        return 0    # fallback to no_op
    return int(np.random.choice(valid))

def run_sanity_check(with_orchestrator: bool, n_steps: int):
    print("=" * 60)
    print(f"  Sanity Check — {'Scenario MO' if with_orchestrator else 'Scenario M'}")
    print(f"  Steps: {n_steps} | Step duration: {params['marl']['step_min']} min")
    print("=" * 60)

    # ── Instantiate environment ───────────────────────────────────────────────
    try:
        env = SchipholCargoEnv(with_orchestrator=with_orchestrator)
        print("[OK] Environment instantiated")
    except Exception as e:
        print(f"[FAIL] Environment instantiation crashed: {e}")
        raise

    # ── Reset ─────────────────────────────────────────────────────────────────
    try:
        obs, infos = env.reset()
        print(f"[OK] reset() returned {len(obs)} agents: {list(obs.keys())}")
    except Exception as e:
        print(f"[FAIL] reset() crashed: {e}")
        raise

    # Store action spaces for mask validation
    action_spaces = {a: env.action_space(a) for a in env.agents}
    obs_spaces = {a: env.observation_space(a) for a in env.agents}

    # ── Check initial observations ────────────────────────────────────────────
    errors = check_obs(obs, step=0)
    errors += check_masks(infos, action_spaces, step=0)

    # Check observation dimensions
    for agent, vec in obs.items():
        expected = obs_spaces[agent].shape[0]
        if vec.shape[0] != expected:
            errors.append(
                f"step 0 | {agent} | obs shape {vec.shape[0]} "
                f"!= declared {expected}"
            )

    if errors:
        print("[FAIL] Errors in initial observations:")
        for e in errors: print(f"       {e}")
    else:
        print("[OK] Initial observations valid")

    # ── Run episode ───────────────────────────────────────────────────────────
    all_errors = []
    total_rews = {a: 0.0 for a in env.agents}
    action_counts = {a: {} for a in env.agents}

    print(f"\nRunning {n_steps} steps...")

    for step in range(1, n_steps + 1):
        # Sample one valid random action per agent using action mask
        actions = {
            agent: sample_masked_action(
                action_spaces[agent],
                infos[agent]["action_mask"]
            )
            for agent in env.agents
        }

        # Track action distribution
        for agent, action in actions.items():
            action_counts[agent][action] = action_counts[agent].get(action, 0) + 1

        try:
            obs, rewards, dones, infos = env.step(actions)
        except Exception as e:
            print(f"\n[FAIL] step() crashed at step {step}: {e}")
            raise

        # Accumulate rewards
        for agent, r in rewards.items():
            total_rews[agent] += r

        # Validate outputs
        step_errors = check_obs(obs, step)
        step_errors += check_rewards(rewards, step)
        step_errors += check_masks(infos, action_spaces, step)
        all_errors += step_errors

        # Print progress every 100 steps
        if step % 100 == 0:
            status = "ERR" if step_errors else " OK"
            sim_time_h = env.sim.now / 60
            tp3_occ = env.tp3.occupancy_ratio()
            print(
                f"  [{status}] step {step:>4} | "
                f"sim time {sim_time_h:.1f}h | "
                f"TP3 {tp3_occ:.0%} | "
                f"trucks generated: {env.demand._truck_counter}"
            )

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    if all_errors:
        print(f"[FAIL] {len(all_errors)} error(s) found:")
        for e in all_errors[:20]:    # cap at 20 to avoid flooding
            print(f"       {e}")
        if len(all_errors) > 20:
            print(f"       ... and {len(all_errors) - 20} more")
    else:
        print("[OK] No errors found across all steps")

    print("\n  Cumulative rewards:")
    for agent, r in total_rews.items():
        print(f"    {agent:<25} {r:+.3f}")

    print("\n  Action distribution (top 3 per agent):")
    for agent, counts in action_counts.items():
        top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
        top3_str = "  ".join(f"action {a}={c}x" for a, c in top3)
        print(f"    {agent:<25} {top3_str}")

    print("\n  Episode summary (KPI tracker):")
    summary = env.kpi.summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"    {k:<25} {v:.4f}")
        else:
            print(f"    {k:<25} {v}")

    print("\n  DTP registry snapshot:")
    for gha in list(params["ghas"].keys()):
        total_slots  = sum(len(v) for v in env.dtp.registry[gha].values())
        booked_slots = sum(
            1 for entries in env.dtp.registry[gha].values()
            for s in entries if s["phase"] in ("booked", "docked")
        )
        print(f"    {gha:<15} {total_slots} slots published, {booked_slots} booked/docked")

    print("\n  Infrastructure event log:")
    from env.infrastructure import CheckpointID
    event_log = env.infra.get_all_events()
    counts = {}
    for e in event_log:
        counts[e.checkpoint.value] = counts.get(e.checkpoint.value, 0) + 1
    for checkpoint, count in sorted(counts.items()):
        print(f"    {checkpoint:<25} {count} events")

    print("=" * 60)
    passed = len(all_errors) == 0
    print(f"  {'PASSED' if passed else 'FAILED'}")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orchestrator", action="store_true",
        help="Run Scenario MO (with Orchestrator)"
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of steps to run (default: 200)"
    )
    args = parser.parse_args()

    passed = run_sanity_check(
        with_orchestrator=args.orchestrator,
        n_steps=args.steps
    )
    sys.exit(0 if passed else 1)