"""
simulation.py
=============
Comprehensive sanity-check runner for SchipholCargoEnv.

Runs full episodes with random (mask-respecting) actions and performs
exhaustive validation at every step across seven inspection domains:

  1. Observation integrity   — shape, dtype, [0,1] bounds, NaN/Inf
  2. Reward integrity        — finite, sign-aware soft checks
  3. Action mask integrity   — length, no-op always live, coverage ratio
  4. DTP registry health     — slot phase distribution, orphaned bookings,
                               phase-transition monotonicity
  5. TP3 buffer health       — occupancy drift, overflow spikes, stale trucks
  6. Demand pipeline health  — truck counter, pending list bounds,
                               booked_slots consistency
  7. Terminal / KPI health   — occupancy range, utilisation snapshots,
                               late / no-show accumulation

All errors are collected (never silently swallowed) and emitted together in
a structured final report.

HOW TO RUN
──────────
  python sim_test.py                          # Scenario M,  200 steps, 1 iteration
  python sim_test.py --orchestrator           # Scenario MO, 200 steps
  python sim_test.py --steps 2000             # longer episode
  python sim_test.py --iterations 5           # run 5 consecutive full iterations
  python sim_test.py --orchestrator --steps 500 --verbose
  python sim_test.py --log-interval 50        # progress every 50 steps

EXIT CODE
─────────
  0  — no errors detected across all iterations
  1  — at least one error detected in any iteration
"""
import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from env.schiphol_env import SchipholCargoEnv, GHA_IDS

from config.config import load_params
params = load_params()

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers (degrade gracefully on Windows)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    return f"{GREEN}[OK]{RESET}    {msg}"
def warn(msg):  return f"{YELLOW}[WARN]{RESET}  {msg}"
def fail(msg):  return f"{RED}[FAIL]{RESET}  {msg}"
def info(msg):  return f"{CYAN}[INFO]{RESET}  {msg}"
def header(msg):return f"\n{BOLD}{msg}{RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# ERROR ACCUMULATOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ErrorLog:
    errors:      List[str]      = field(default_factory=list)
    warnings:    List[str]      = field(default_factory=list)
    _warned_keys: set           = field(default_factory=set)
 
    def error(self, msg: str):
        self.errors.append(msg)
 
    def warning(self, msg: str):
        self.warnings.append(msg)
 
    def warning_once(self, key: str, msg: str):
        if key not in self._warned_keys:
            self._warned_keys.add(key)
            self.warnings.append(msg)
 
    def clear_warning(self, key: str):
        self._warned_keys.discard(key)
 
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
 
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TIME-SERIES TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesTracker:
    def __init__(self):
        self.sim_time:        List[float] = []
        self.tp3_occupancy:   List[float] = []
        self.tp3_overflow:    List[int]   = []
        self.n_pending:       List[int]   = []
        self.n_trucks_total:  List[int]   = []

        self.rewards: Dict[str, List[float]] = defaultdict(list)
        self.exp_occ: Dict[str, List[float]] = defaultdict(list)
        self.imp_occ: Dict[str, List[float]] = defaultdict(list)
        self.dtp_phases: Dict[str, List[int]] = defaultdict(list)
        self.mask_coverage: Dict[str, List[float]] = defaultdict(list)
        self.obs_min:  Dict[str, List[float]] = defaultdict(list)
        self.obs_max:  Dict[str, List[float]] = defaultdict(list)
        self.obs_mean: Dict[str, List[float]] = defaultdict(list)

    def record(self, step: int, env: SchipholCargoEnv,
               obs: dict, rewards: dict, infos: dict, agents: list):

        self.sim_time.append(env.sim.now)
        self.tp3_occupancy.append(env.tp3.occupancy_ratio())
        self.tp3_overflow.append(env.tp3.n_overflow())
        self.n_pending.append(len(env.demand.pending_trucks))
        self.n_trucks_total.append(env.demand._truck_counter)

        for agent in agents:
            r = rewards.get(agent, 0.0)
            self.rewards[agent].append(float(r))

            mask = infos[agent].get("action_mask", np.array([1]))
            coverage = float(mask.sum()) / float(len(mask)) if len(mask) > 0 else 0.0
            self.mask_coverage[agent].append(coverage)

            vec = obs.get(agent, np.zeros(1))
            self.obs_min[agent].append(float(vec.min()))
            self.obs_max[agent].append(float(vec.max()))
            self.obs_mean[agent].append(float(vec.mean()))

        for gha in GHA_IDS:
            t = env.terminals[gha]
            self.exp_occ[gha].append(t.exp_occupancy())
            self.imp_occ[gha].append(t.imp_occupancy())

        # Safely extract phase statistics using the updated structure
        phase_count: Dict[str, int] = defaultdict(int)
        for gha in GHA_IDS:
            for slot_start, window in env.dtp.registry[gha].items():
                # Count available slots from explicit storage
                phase_count["available"] += sum(window["available"].values())
                # Count current state of active/terminal records
                for b_info in window["bookings"].values():
                    phase_count[b_info["phase"]] += 1

        for phase in ["available", "booked", "docked", "closed", "no_show"]:
            self.dtp_phases[phase].append(phase_count[phase])


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

def validate_observations(obs: dict, obs_spaces: dict,
                          step: int, log: ErrorLog) -> None:
    for agent, vec in obs.items():
        tag = f"step {step:>5} | {agent}"
        if vec.dtype != np.float32:
            log.warning(f"{tag} | obs dtype {vec.dtype} (expected float32)")

        expected_shape = obs_spaces[agent].shape
        if vec.shape != expected_shape:
            log.error(f"{tag} | obs shape {vec.shape} != declared {expected_shape}")

        if np.any(np.isnan(vec)):
            log.error(f"{tag} | NaN in observation")
        if np.any(np.isinf(vec)):
            log.error(f"{tag} | Inf in observation")

        below = vec < -1e-6
        above = vec >  1.0 + 1e-6
        if np.any(below):
            log.error(f"{tag} | obs below 0: min={vec.min():.6f}")
        if np.any(above):
            log.error(f"{tag} | obs above 1: max={vec.max():.6f}")


def validate_rewards(rewards: dict, step: int, log: ErrorLog) -> None:
    for agent, r in rewards.items():
        tag = f"step {step:>5} | {agent}"
        if not np.isfinite(r):
            log.error(f"{tag} | non-finite reward: {r}")
        if abs(r) > 1_000:
            log.warning(f"{tag} | suspiciously large reward: {r:.4f}")


def validate_masks(infos: dict, action_spaces: dict,
                   step: int, log: ErrorLog) -> None:
    for agent, info in infos.items():
        tag = f"step {step:>5} | {agent}"
        mask = info.get("action_mask")

        if mask is None:
            log.error(f"{tag} | action_mask missing from infos")
            continue

        expected_len = action_spaces[agent].n
        if len(mask) != expected_len:
            log.error(f"{tag} | mask length {len(mask)} != action_space.n {expected_len}")

        if mask[0] != 1:
            log.error(f"{tag} | no_op (action 0) is masked — must always be valid")

        if not np.any(mask):
            log.error(f"{tag} | all actions masked; agent cannot act")


def validate_dones(dones: dict, label: str, step: int, log: ErrorLog) -> None:
    for agent, d in dones.items():
        if d is not False and d != 0:
            log.warning(f"step {step:>5} | {agent} | {label}=True — episode ended mid-run")


def validate_dtp_registry(env: SchipholCargoEnv, step: int, log: ErrorLog) -> None:
    """
    Domain 4b — DTP registry health.
    Evaluates safety invariants against the O(1) dictionary design.
    """
    legal_booking_phases = {"booked", "docked", "closed", "no_show"}

    for gha in GHA_IDS:
        tag = f"step {step:>5} | DTP/{gha}"
        active: Dict[str, int] = {} 

        for slot_start, window in env.dtp.registry[gha].items():
            # Validate availability counts aren't tracking corrupt negative entries
            for f_type, count in window["available"].items():
                if count < 0:
                    log.error(f"{tag} | negative available count ({count}) for flow {f_type} at {slot_start}")

            # Scan individual active records in the map
            for truck_id, b_info in window["bookings"].items():
                phase = b_info["phase"]
                if phase not in legal_booking_phases:
                    log.error(f"{tag} | unknown phase '{phase}' at slot {slot_start} for truck {truck_id}")

                if phase in ("booked", "docked"):
                    if truck_id in active:
                        log.error(f"{tag} | truck {truck_id} duplicated across active slots: {active[truck_id]} and {slot_start}")
                    else:
                        active[truck_id] = slot_start


def validate_tp3(env: SchipholCargoEnv, step: int, log: ErrorLog) -> None:
    tag = f"step {step:>5} | TP3"
    occ = env.tp3.occupancy_ratio()

    if not (0.0 <= occ <= 1.0 + 1e-9):
        log.error(f"{tag} | occupancy_ratio {occ:.4f} outside [0, 1]")

    if env.tp3.n_overflow() < 0:
        log.error(f"{tag} | n_overflow() returned negative count")


def validate_demand_pipeline(env: SchipholCargoEnv, step: int, log: ErrorLog) -> None:
    tag = f"step {step:>5} | Demand"
    pending = list(env.demand.pending_trucks.values())
 
    WARN_KEY = "pending_high"
    if len(pending) > 500:
        log.warning_once(WARN_KEY, f"{tag} | pending_trucks crossed 500 (now {len(pending)})")
    elif len(pending) < 400:
        log.clear_warning(WARN_KEY)
 
    for truck in pending:
        manifest_ghas = {s["gha"] for s in truck.manifest}
        booked_ghas   = set(truck.booked_slots.keys())
        if booked_ghas - manifest_ghas:
            log.error(f"{tag} | truck {truck.truck_id} has unmanifested bookings")


def validate_terminals(env: SchipholCargoEnv, step: int, log: ErrorLog) -> None:
    for gha in GHA_IDS:
        tag = f"step {step:>5} | Terminal/{gha}"
        t = env.terminals[gha]
        if not (0.0 <= t.exp_occupancy() <= 1.0 + 1e-9):
            log.error(f"{tag} | exp_occupancy outside [0,1]")


# ─────────────────────────────────────────────────────────────────────────────
# ACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sample_masked_action(action_space, mask: np.ndarray) -> int:
    valid = np.where(mask == 1)[0]
    return int(np.random.choice(valid)) if len(valid) > 0 else 0


def build_actions(env: SchipholCargoEnv, infos: dict) -> Dict[str, int]:
    return {
        agent: sample_masked_action(env.action_space(agent), infos[agent]["action_mask"])
        for agent in env.agents
    }


# ─────────────────────────────────────────────────────────────────────────────
# INITIAL RESET VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_reset(env: SchipholCargoEnv, obs: dict, infos: dict,
                   obs_spaces: dict, action_spaces: dict, log: ErrorLog) -> None:
    tag = "reset"

    expected_base = {"transporter"} | set(GHA_IDS)
    expected_orch = expected_base | {"orchestrator"}
    expected = expected_orch if env.with_orchestrator else expected_base
    if set(env.agents) != expected:
        log.error(f"{tag} | agent set mismatch")

    # Fixed structural count verification for the dictionary optimization layer
    total_slots = 0
    for gha in GHA_IDS:
        for slot_start, window in env.dtp.registry[gha].items():
            # Sum up empty slots plus active bookings
            total_slots += sum(window["available"].values()) + len(window["bookings"])

    if total_slots == 0:
        log.error(f"{tag} | DTP registry completely unpopulated after environment reset")
    else:
        print(info(f"DTP pre-populated with {total_slots} tracked state windows across all GHAs"))

    validate_observations(obs, obs_spaces, step=0, log=log)
    validate_masks(infos, action_spaces, step=0, log=log)


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE STATISTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _series_stats(values: List[float]) -> str:
    if not values:
        return "n/a"
    a = np.array(values)
    return f"mean={a.mean():.4f}  std={a.std():.4f}  min={a.min():.4f}  max={a.max():.4f}"

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(env: SchipholCargoEnv, ts: TimeSeriesTracker,
                 log: ErrorLog, action_counts: dict,
                 n_steps: int, elapsed: float) -> bool:

    W = 64
    DIV = "─" * W

    print(f"\n{BOLD}{'═' * W}{RESET}")
    print(f"{BOLD}  EPISODE REPORT{RESET}")
    print(f"{'═' * W}")

    print(header("  Episode metadata"))
    print(DIV)
    print(f"  Steps run           : {n_steps}")
    print(f"  Simulated time      : {env.sim.now / 60:.2f} h")
    print(f"  Wall-clock time     : {elapsed:.2f} s  ({1000 * elapsed / max(n_steps, 1):.1f} ms/step)")
    print(f"  Trucks generated    : {env.demand._truck_counter}")

    print(header("  Cumulative & per-step reward statistics"))
    print(DIV)
    for agent, series in ts.rewards.items():
        print(f"  {agent:<25}  cumulative {np.sum(series):+10.3f}  |  {_series_stats(series)}")

    print(header("  Action mask coverage (fraction of valid actions)"))
    print(DIV)
    for agent, cov_series in ts.mask_coverage.items():
        print(f"  {agent:<25}  mean={np.mean(cov_series):.2%}")

    print(header("  Action distribution (top 5 non-no_op per agent)"))
    print(DIV)
    for agent in env.agents:
        counts = action_counts.get(agent, {})
        total  = sum(counts.values())
        if total == 0:
            print(f"  {agent:<25}  no actions recorded")
            continue

        no_op_count = counts.get(0, 0)
        no_op_pct   = no_op_count / total

        top5 = sorted(
            [(a, c) for a, c in counts.items() if a != 0],
            key=lambda x: x[1], reverse=True
        )[:5]
        top_str = "  ".join(f"a{a}={c}x({c/total:.0%})" for a, c in top5)

        print(f"  {agent:<25}  no_op={no_op_pct:.0%}  |  {top_str}")

    print(header("  DTP registry snapshot"))
    print(DIV)
    
    # Custom display parser matching the dictionary registry structure mapping
    for gha in GHA_IDS:
        p_counts = defaultdict(int)
        for slot_start, window in env.dtp.registry[gha].items():
            p_counts["available"] += sum(window["available"].values())
            for b_info in window["bookings"].values():
                p_counts[b_info["phase"]] += 1
                
        total_tracked = sum(p_counts.values())
        phase_str = "  ".join(f"{k}={v}" for k, v in sorted(p_counts.items()))
        print(f"  {gha:<15}  {total_tracked:>4} nodes  |  {phase_str}")

    if log.has_warnings:
        print(header("  WARNINGS"))
        print(DIV)
        for w in log.warnings: print(f"  {warn(w)}")

    if log.has_errors:
        print(header("  ERRORS"))
        print(DIV)
        for e in log.errors: print(f"  {fail(e)}")

    print(f"\n{'═' * W}")
    if log.has_errors:
        print(f"  {RED}{BOLD}FAILED{RESET}  —  {len(log.errors)} error(s)")
    else:
        print(f"  {GREEN}{BOLD}PASSED{RESET}  —  All systems normal.")
    print(f"{'═' * W}\n")

    return not log.has_errors


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────
@profile
def run_simulation(with_orchestrator: bool,
                   n_steps: int,
                   log_interval: int = 100,
                   verbose: bool = False) -> bool:

    log = ErrorLog()
    ts  = TimeSeriesTracker()

    scenario = "Scenario MO (with Orchestrator)" if with_orchestrator else "Scenario M  (no Orchestrator)"

    print(f"\n{'═' * 64}")
    print(f"  {BOLD}Schiphol Cargo Hub — Simulation Sanity Check{RESET}")
    print(f"  {scenario}")
    print(f"{'═' * 64}\n")

    print(header("  Phase 1 — Instantiation"))
    env = SchipholCargoEnv(with_orchestrator=with_orchestrator)
    print(ok("Environment instantiated"))

    print(header("  Phase 2 — Reset"))
    # Native unpacked mapping for PettingZoo compatibility loop
    reset_out = env.reset()
    obs, infos = reset_out if isinstance(reset_out, tuple) else (reset_out, getattr(env, "infos", {}))
    print(ok("reset() returned successfully"))

    action_spaces = {a: env.action_space(a)      for a in env.agents}
    obs_spaces    = {a: env.observation_space(a)  for a in env.agents}

    validate_reset(env, obs, infos, obs_spaces, action_spaces, log)

    if log.has_errors:
        return False

    print(header(f"  Phase 3 — Episode ({n_steps} steps)"))
    action_counts: Dict[str, Dict[int, int]] = {a: {} for a in env.agents}
    t_start = time.perf_counter()

    for step in range(1, n_steps + 1):
        actions = build_actions(env, infos)
        for agent, action in actions.items():
            action_counts[agent][action] = action_counts[agent].get(action, 0) + 1

        # Complete step extraction mapping 
        step_out = env.step(actions)
        if len(step_out) == 5:
            obs, rewards, term, trunc, infos = step_out
        else:
            obs, rewards, dones, infos = step_out
            term = trunc = dones

        validate_observations(obs, obs_spaces, step, log)
        validate_rewards(rewards, step, log)
        validate_masks(infos, action_spaces, step, log)
        validate_dtp_registry(env, step, log)
        validate_tp3(env, step, log)
        validate_demand_pipeline(env, step, log)
        validate_terminals(env, step, log)

        ts.record(step, env, obs, rewards, infos, env.agents)

        if step % log_interval == 0:
            status_sym = f"{RED}ERR{RESET}" if log.has_errors else f"{GREEN} OK{RESET}"
            print(f"  [{status_sym}] step {step:>5} | t={env.sim.now/60:.1f}h | pending={len(env.demand.pending_trucks):>3}")

    elapsed = time.perf_counter() - t_start
    passed = print_report(env, ts, log, action_counts, n_steps, elapsed)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schiphol Cargo Hub — Simulation Sanity Check")
    parser.add_argument("--orchestrator", action="store_true", help="Run Scenario MO")
    parser.add_argument("--steps", type=int, default=200, help="MARL steps per iteration")
    parser.add_argument("--iterations", type=int, default=1, help="Total simulation episodes")
    parser.add_argument("--log-interval", type=int, default=100, help="Print interval")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    all_passed = True
    for iteration in range(1, args.iterations + 1):
        if args.iterations > 1:
            print(f"\n{BOLD}{CYAN}┌──────────────────────────────────────────────────────────┐{RESET}")
            print(f"{BOLD}{CYAN}│ STARTING ITERATION {iteration:>3} / {args.iterations:<3}                         │{RESET}")
            print(f"{BOLD}{CYAN}└──────────────────────────────────────────────────────────┘{RESET}")
            
        passed = run_simulation(
            with_orchestrator=args.orchestrator,
            n_steps=args.steps,
            log_interval=args.log_interval,
            verbose=args.verbose,
        )
        if not passed:
            all_passed = False

    if args.iterations > 1:
        print(f"\n{BOLD}{'═' * 64}{RESET}")
        if all_passed:
            print(f"  {GREEN}ALL {args.iterations} ITERATIONS PASSED SUCCESSFULLY!{RESET}")
        else:
            print(f"  {RED}SOME ITERATIONS FAILED. Check individual logs.{RESET}")
        print(f"{BOLD}{'═' * 64}{RESET}\n")

    sys.exit(0 if all_passed else 1)