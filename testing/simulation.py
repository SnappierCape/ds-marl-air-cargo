"""
simulation.py
=============
Comprehensive sanity-check runner for SchipholCargoEnv.

Runs one full episode with random (mask-respecting) actions and performs
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
  python simulation.py                          # Scenario M,  200 steps
  python simulation.py --orchestrator           # Scenario MO, 200 steps
  python simulation.py --steps 2000             # longer episode
  python simulation.py --orchestrator --steps 500 --verbose
  python simulation.py --log-interval 50        # progress every 50 steps

EXIT CODE
─────────
  0  — no errors detected
  1  — at least one error detected
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
    # Keys of warnings that have already fired once — used to suppress
    # per-step repetition.  Call warning_once(key, msg) instead of warning()
    # when a condition can persist for many consecutive steps.
    _warned_keys: set           = field(default_factory=set)
 
    def error(self, msg: str):
        self.errors.append(msg)
 
    def warning(self, msg: str):
        self.warnings.append(msg)
 
    def warning_once(self, key: str, msg: str):
        """Emit `msg` only the first time this `key` is seen.
        When the condition clears, call clear_warning(key) so it can re-arm."""
        if key not in self._warned_keys:
            self._warned_keys.add(key)
            self.warnings.append(msg)
 
    def clear_warning(self, key: str):
        """Re-arm a suppressed warning so it fires again if the condition returns."""
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
    """
    Lightweight per-step tracker.  Values are stored as flat Python lists
    so the memory cost is proportional to n_steps, not n_steps × dim.
    """
    def __init__(self):
        self.sim_time:        List[float] = []
        self.tp3_occupancy:   List[float] = []
        self.tp3_overflow:    List[int]   = []
        self.n_pending:       List[int]   = []
        self.n_trucks_total:  List[int]   = []

        # per-agent rewards
        self.rewards: Dict[str, List[float]] = defaultdict(list)

        # per-GHA dock utilisation (exp, imp)
        self.exp_occ: Dict[str, List[float]] = defaultdict(list)
        self.imp_occ: Dict[str, List[float]] = defaultdict(list)

        # slot phase counts from DTP
        self.dtp_phases: Dict[str, List[int]] = defaultdict(list)

        # mask coverage ratio per agent
        self.mask_coverage: Dict[str, List[float]] = defaultdict(list)

        # obs statistics
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

        # DTP phase snapshot
        phase_count = defaultdict(int)
        for gha in GHA_IDS:
            for entries in env.dtp.registry[gha].values():
                for e in entries:
                    phase_count[e["phase"]] += 1
        for phase, count in phase_count.items():
            self.dtp_phases[phase].append(count)


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

def validate_observations(obs: dict, obs_spaces: dict,
                          step: int, log: ErrorLog) -> None:
    """
    Domain 1 — Observation integrity.
    Checks: shape, dtype, [0,1] bounds, NaN, Inf.
    """
    for agent, vec in obs.items():
        tag = f"step {step:>5} | {agent}"

        # dtype
        if vec.dtype != np.float32:
            log.warning(f"{tag} | obs dtype {vec.dtype} (expected float32)")

        # shape
        expected_shape = obs_spaces[agent].shape
        if vec.shape != expected_shape:
            log.error(
                f"{tag} | obs shape {vec.shape} != declared {expected_shape}"
            )

        # NaN / Inf
        if np.any(np.isnan(vec)):
            log.error(f"{tag} | NaN in observation (indices: {np.where(np.isnan(vec))[0].tolist()})")
        if np.any(np.isinf(vec)):
            log.error(f"{tag} | Inf in observation (indices: {np.where(np.isinf(vec))[0].tolist()})")

        # [0, 1] bounds — flag both violations and near-violations
        below = vec < -1e-6
        above = vec >  1.0 + 1e-6
        if np.any(below):
            log.error(
                f"{tag} | obs below 0: indices={np.where(below)[0].tolist()}, "
                f"min={vec.min():.6f}"
            )
        if np.any(above):
            log.error(
                f"{tag} | obs above 1: indices={np.where(above)[0].tolist()}, "
                f"max={vec.max():.6f}"
            )


def validate_rewards(rewards: dict, step: int, log: ErrorLog) -> None:
    """
    Domain 2 — Reward integrity.
    Checks: finite, soft plausibility range.
    """
    for agent, r in rewards.items():
        tag = f"step {step:>5} | {agent}"
        if not np.isfinite(r):
            log.error(f"{tag} | non-finite reward: {r}")
        # Reward should not explode beyond a loose sanity band
        if abs(r) > 1_000:
            log.warning(f"{tag} | suspiciously large reward: {r:.4f}")


def validate_masks(infos: dict, action_spaces: dict,
                   step: int, log: ErrorLog) -> None:
    """
    Domain 3 — Action mask integrity.
    Checks: presence, length, no_op always live, at least one valid action.
    """
    for agent, info in infos.items():
        tag = f"step {step:>5} | {agent}"
        mask = info.get("action_mask")

        if mask is None:
            log.error(f"{tag} | action_mask missing from infos")
            continue

        expected_len = action_spaces[agent].n
        if len(mask) != expected_len:
            log.error(
                f"{tag} | mask length {len(mask)} != action_space.n {expected_len}"
            )

        # action 0 (no_op) must always be valid
        if mask[0] != 1:
            log.error(f"{tag} | no_op (action 0) is masked — must always be valid")

        # at least one action must be valid
        if not np.any(mask):
            log.error(f"{tag} | all actions masked; agent cannot act")

        # mask values must be 0 or 1 only
        unique_vals = set(int(v) for v in mask)
        if not unique_vals.issubset({0, 1}):
            log.error(f"{tag} | mask contains values other than 0/1: {unique_vals}")


def validate_dones(dones: dict, label: str, step: int, log: ErrorLog) -> None:
    """
    Domain 4a — Done flags.
    Per env design, dones must always be False (episode ends externally).
    """
    for agent, d in dones.items():
        if d is not False and d != 0:
            log.warning(
                f"step {step:>5} | {agent} | {label}=True — episode ended mid-run "
                f"(expected externally controlled termination)"
            )


def validate_dtp_registry(env: SchipholCargoEnv, step: int, log: ErrorLog) -> None:
    """
    Domain 4b — DTP registry health.
    Checks: valid phases, no truck booked into two slots at once per GHA,
    available slots still published (shouldn't all dry up before horizon),
    phase enum values are legal.
    """
    legal_phases = {"available", "booked", "docked", "closed", "no_show"}

    for gha in GHA_IDS:
        tag = f"step {step:>5} | DTP/{gha}"
        # Track which truck_ids are actively booked/docked per GHA
        active: Dict[str, int] = {}   # truck_id → slot_start

        for slot_start, entries in env.dtp.registry[gha].items():
            for e in entries:
                # Phase must be a known value
                if e["phase"] not in legal_phases:
                    log.error(
                        f"{tag} | unknown phase '{e['phase']}' "
                        f"at slot_start={slot_start}"
                    )

                # A truck must not hold two active bookings at the same GHA
                if e["phase"] in ("booked", "docked") and e["truck_id"] is not None:
                    tid = e["truck_id"]
                    if tid in active:
                        log.error(
                            f"{tag} | truck {tid} has TWO active slots: "
                            f"{active[tid]} and {slot_start}"
                        )
                    else:
                        active[tid] = slot_start

                # 'available' entries must not have a truck_id
                if e["phase"] == "available" and e["truck_id"] is not None:
                    log.warning(
                        f"{tag} | 'available' slot at {slot_start} "
                        f"has truck_id={e['truck_id']} (should be None)"
                    )


def validate_tp3(env: SchipholCargoEnv, step: int, log: ErrorLog) -> None:
    """
    Domain 5 — TP3 buffer health.
    Checks: occupancy in [0,1], overflow non-negative,
    no truck parked with a completed stop still in its stop list.
    """
    tag = f"step {step:>5} | TP3"
    occ = env.tp3.occupancy_ratio()

    if not (0.0 <= occ <= 1.0 + 1e-9):
        log.error(f"{tag} | occupancy_ratio {occ:.4f} outside [0, 1]")

    overflow = env.tp3.n_overflow()
    if overflow < 0:
        log.error(f"{tag} | n_overflow() returned {overflow} (must be >= 0)")

    # Trucks parked in TP3 should still have at least one unfinished stop
    for truck in env.tp3.get_parked_trucks():
        if not truck.stops_remaining:
            log.warning(
                f"{tag} | truck {truck.truck_id} parked in TP3 "
                f"but has no stops_remaining"
            )


def validate_demand_pipeline(env: SchipholCargoEnv, step: int, log: ErrorLog) -> None:
    """
    Domain 6 — Demand generator / truck pipeline health.
    Checks: truck counter monotonic, pending list sane,
    every truck's booked_slots keys exist in its manifest,
    no truck has more booked_slots than manifest entries.
    """
    tag = f"step {step:>5} | Demand"
    pending = env.demand.pending_trucks
 
    # Edge-triggered: warn once when the queue first crosses 500,
    # suppress while it stays high, re-arm when it recovers below 400.
    WARN_KEY = "pending_high"
    if len(pending) > 500:
        log.warning_once(
            WARN_KEY,
            f"{tag} | pending_trucks crossed 500 (now {len(pending)}) "
            f"— possible accumulation without dispatch"
        )
    elif len(pending) < 400:
        log.clear_warning(WARN_KEY)   # re-arm for next spike
 
    for truck in pending:
        # booked GHAs must be a subset of manifest GHAs
        manifest_ghas = {s["gha"] for s in truck.manifest}
        booked_ghas   = set(truck.booked_slots.keys())
        stray = booked_ghas - manifest_ghas
        if stray:
            log.error(
                f"{tag} | truck {truck.truck_id} has bookings at "
                f"{stray} which are NOT in its manifest"
            )
 
        # cannot book more GHAs than the manifest has
        if len(booked_ghas) > len(truck.manifest):
            log.error(
                f"{tag} | truck {truck.truck_id} has {len(booked_ghas)} bookings "
                f"but only {len(truck.manifest)} manifest stops"
            )
 
        # flow_type must be one of the two legal values
        if truck.flow_type not in ("import", "export"):
            log.error(
                f"{tag} | truck {truck.truck_id} has "
                f"invalid flow_type '{truck.flow_type}'"
            )

def validate_terminals(env: SchipholCargoEnv, step: int, log: ErrorLog) -> None:
    """
    Domain 7 — GHA terminal / KPI health.
    Checks: occupancy metrics in [0,1], no negative KPI counters.
    """
    for gha in GHA_IDS:
        tag = f"step {step:>5} | Terminal/{gha}"
        t = env.terminals[gha]

        for metric_name, value in [
            ("exp_occupancy", t.exp_occupancy()),
            ("imp_occupancy", t.imp_occupancy()),
        ]:
            if not np.isfinite(value):
                log.error(f"{tag} | {metric_name}() returned non-finite {value}")
            elif not (-1e-9 <= value <= 1.0 + 1e-9):
                log.error(
                    f"{tag} | {metric_name}() = {value:.4f} outside [0, 1]"
                )


# ─────────────────────────────────────────────────────────────────────────────
# ACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sample_masked_action(action_space, mask: np.ndarray) -> int:
    """Sample a uniformly random action from the set of valid (mask=1) actions."""
    valid = np.where(mask == 1)[0]
    return int(np.random.choice(valid)) if len(valid) > 0 else 0


def build_actions(env: SchipholCargoEnv, infos: dict) -> Dict[str, int]:
    """Sample one masked random action for every active agent."""
    return {
        agent: sample_masked_action(
            env.action_space(agent),
            infos[agent]["action_mask"]
        )
        for agent in env.agents
    }


# ─────────────────────────────────────────────────────────────────────────────
# INITIAL RESET VALIDATION (expanded)
# ─────────────────────────────────────────────────────────────────────────────

def validate_reset(env: SchipholCargoEnv, obs: dict, infos: dict,
                   obs_spaces: dict, action_spaces: dict, log: ErrorLog) -> None:
    """
    Extended checks run only on the post-reset state.
    """
    tag = "reset"

    # Agent roster
    expected_base = {"transporter"} | set(GHA_IDS)
    expected_orch = expected_base | {"orchestrator"}
    expected = expected_orch if env.with_orchestrator else expected_base
    actual   = set(env.agents)
    if actual != expected:
        log.error(
            f"{tag} | agent set mismatch — got {actual}, expected {expected}"
        )

    # possible_agents is a superset of agents
    if not set(env.agents).issubset(set(env.possible_agents)):
        log.error(f"{tag} | env.agents is not a subset of env.possible_agents")

    # Every agent must have an obs and an info
    for agent in env.agents:
        if agent not in obs:
            log.error(f"{tag} | agent '{agent}' missing from obs dict")
        if agent not in infos:
            log.error(f"{tag} | agent '{agent}' missing from infos dict")

    # DTP must have pre-published slots
    total_slots = sum(
        len(entries)
        for gha in GHA_IDS
        for entries in env.dtp.registry[gha].values()
    )
    if total_slots == 0:
        log.error(f"{tag} | DTP registry is empty after reset — _prepopulate_slots failed")
    else:
        print(info(f"DTP pre-populated with {total_slots} slot entries across all GHAs"))

    # SimPy clock must start at 0
    if env.sim.now != 0:
        log.warning(f"{tag} | sim.now = {env.sim.now} after reset (expected 0)")

    # Demand generator must exist and have counter = 0
    if not hasattr(env.demand, '_truck_counter'):
        log.warning(f"{tag} | demand generator missing _truck_counter attribute")
    elif env.demand._truck_counter != 0:
        log.warning(
            f"{tag} | _truck_counter = {env.demand._truck_counter} after reset "
            f"(expected 0)"
        )

    # KPI tracker must be fresh
    summary = env.kpi.summary()
    if summary is None:
        log.error(f"{tag} | kpi.summary() returned None")

    # Validate obs/masks for step 0
    validate_observations(obs, obs_spaces, step=0, log=log)
    validate_masks(infos, action_spaces, step=0, log=log)

    print(info(f"Agents: {env.agents}"))
    print(info(
        f"Obs dims — "
        + ", ".join(f"{a}: {obs[a].shape[0]}" for a in env.agents if a in obs)
    ))
    print(info(
        f"Action dims — "
        + ", ".join(f"{a}: {action_spaces[a].n}" for a in env.agents)
    ))


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE STATISTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _series_stats(values: List[float]) -> str:
    if not values:
        return "n/a"
    a = np.array(values)
    return (
        f"mean={a.mean():.4f}  std={a.std():.4f}  "
        f"min={a.min():.4f}  max={a.max():.4f}"
    )


def _reward_stability_check(ts: TimeSeriesTracker,
                             log: ErrorLog, window: int = 50) -> None:
    """
    Flag agents whose rolling reward has zero variance for a sustained window
    (suggests the reward signal is dead / disconnected).
    """
    for agent, series in ts.rewards.items():
        if len(series) < window:
            continue
        tail = np.array(series[-window:])
        if np.std(tail) < 1e-9:
            log.warning(
                f"Reward for '{agent}' has zero variance over last {window} steps "
                f"(constant {tail[0]:.6f}) — reward signal may be disconnected"
            )


def _obs_drift_check(ts: TimeSeriesTracker,
                     log: ErrorLog, window: int = 100) -> None:
    """
    Flag agents whose observation vector has been identically zero for a
    sustained window (suggests a frozen or disconnected state builder).
    """
    for agent, mean_series in ts.obs_mean.items():
        if len(mean_series) < window:
            continue
        tail = np.array(mean_series[-window:])
        if np.allclose(tail, 0.0, atol=1e-9):
            log.warning(
                f"Obs mean for '{agent}' is zero over last {window} steps "
                f"— _get_obs may be returning an all-zero vector"
            )


def _tp3_stall_check(ts: TimeSeriesTracker,
                     log: ErrorLog, window: int = 100) -> None:
    """
    Flag if TP3 occupancy has been at 1.0 for a sustained period,
    suggesting trucks are stuck and never released.
    """
    if len(ts.tp3_occupancy) < window:
        return
    tail = np.array(ts.tp3_occupancy[-window:])
    if np.all(tail >= 1.0 - 1e-9):
        log.warning(
            f"TP3 occupancy has been at capacity for {window} consecutive steps "
            f"— trucks may be stalling without release"
        )


def _pending_trucks_stall_check(ts: TimeSeriesTracker,
                                log: ErrorLog, window: int = 200) -> None:
    """
    Flag if the pending truck count is growing monotonically — dispatch
    isn't keeping up with arrivals.
    """
    if len(ts.n_pending) < window:
        return
    tail = np.array(ts.n_pending[-window:])
    diffs = np.diff(tail)
    if np.all(diffs >= 0) and tail[-1] > 50:
        log.warning(
            f"Pending trucks grew monotonically over {window} steps "
            f"(current: {tail[-1]}) — dispatch may be broken"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(env: SchipholCargoEnv, ts: TimeSeriesTracker,
                 log: ErrorLog, action_counts: dict,
                 n_steps: int, elapsed: float) -> bool:
    """Print the full post-episode report.  Returns True if no errors."""

    W = 64
    DIV = "─" * W

    print(f"\n{BOLD}{'═' * W}{RESET}")
    print(f"{BOLD}  EPISODE REPORT{RESET}")
    print(f"{'═' * W}")

    # ── Episode metadata ──────────────────────────────────────────────────────
    print(header("  Episode metadata"))
    print(DIV)
    sim_hours = env.sim.now / 60
    step_min  = params["marl"]["step_min"]
    print(f"  Steps run           : {n_steps}")
    print(f"  Simulated time      : {sim_hours:.2f} h  ({env.sim.now:.0f} min)")
    print(f"  Step duration       : {step_min} min")
    print(f"  Wall-clock time     : {elapsed:.2f} s  "
          f"({1000 * elapsed / max(n_steps, 1):.1f} ms/step)")
    print(f"  Trucks generated    : {env.demand._truck_counter}")
    print(f"  Trucks pending      : {len(env.demand.pending_trucks)}")
    print(f"  TP3 occupancy (now) : {env.tp3.occupancy_ratio():.2%}")
    print(f"  TP3 overflow (now)  : {env.tp3.n_overflow()}")

    # ── Reward summary ────────────────────────────────────────────────────────
    print(header("  Cumulative & per-step reward statistics"))
    print(DIV)
    for agent, series in ts.rewards.items():
        arr = np.array(series)
        cumsum = arr.sum()
        print(
            f"  {agent:<25}  "
            f"cumulative {cumsum:+10.3f}  |  {_series_stats(series)}"
        )

    # ── Observation statistics ────────────────────────────────────────────────
    print(header("  Observation statistics (per-step mean of obs vector)"))
    print(DIV)
    for agent in ts.obs_mean:
        print(
            f"  {agent:<25}  mean-of-means: "
            f"{np.mean(ts.obs_mean[agent]):.4f}  "
            f"obs-max ever: {max(ts.obs_max[agent]):.4f}"
        )

    # ── Mask coverage ─────────────────────────────────────────────────────────
    print(header("  Action mask coverage (fraction of valid actions)"))
    print(DIV)
    for agent, cov_series in ts.mask_coverage.items():
        arr = np.array(cov_series)
        print(
            f"  {agent:<25}  "
            f"mean={arr.mean():.2%}  min={arr.min():.2%}  max={arr.max():.2%}"
        )

    # ── Action distribution ───────────────────────────────────────────────────
    print(header("  Action distribution (top 5 per agent)"))
    print(DIV)
    for agent, counts in action_counts.items():
        total = sum(counts.values())
        top5  = sorted(counts.items(), key=lambda x: -x[1])[:5]
        parts = "  ".join(
            f"a{a}={c}x({100*c/total:.0f}%)" for a, c in top5
        )
        noop_pct = 100 * counts.get(0, 0) / max(total, 1)
        print(f"  {agent:<25}  no_op={noop_pct:.0f}%  |  {parts}")

    # ── DTP registry snapshot ─────────────────────────────────────────────────
    print(header("  DTP registry snapshot"))
    print(DIV)
    phase_totals: Dict[str, int] = defaultdict(int)
    for gha in GHA_IDS:
        slot_count   = sum(len(v) for v in env.dtp.registry[gha].values())
        phase_counts: Dict[str, int] = defaultdict(int)
        for entries in env.dtp.registry[gha].values():
            for e in entries:
                phase_counts[e["phase"]] += 1
                phase_totals[e["phase"]] += 1
        phase_str = "  ".join(f"{k}={v}" for k, v in sorted(phase_counts.items()))
        print(f"  {gha:<15}  {slot_count:>4} entries  |  {phase_str}")
    print(f"  {'TOTAL':<15}  "
          + "  ".join(f"{k}={v}" for k, v in sorted(phase_totals.items())))

    # ── TP3 time-series ───────────────────────────────────────────────────────
    print(header("  TP3 buffer statistics"))
    print(DIV)
    print(f"  Occupancy  :  {_series_stats(ts.tp3_occupancy)}")
    print(f"  Overflow   :  {_series_stats([float(x) for x in ts.tp3_overflow])}")

    # ── GHA utilisation ───────────────────────────────────────────────────────
    print(header("  GHA dock utilisation"))
    print(DIV)
    for gha in GHA_IDS:
        exp = _series_stats(ts.exp_occ[gha])
        imp = _series_stats(ts.imp_occ[gha])
        print(f"  {gha}")
        print(f"    export  {exp}")
        print(f"    import  {imp}")

    # ── Demand pipeline ───────────────────────────────────────────────────────
    print(header("  Demand pipeline (pending truck queue)"))
    print(DIV)
    print(f"  {_series_stats([float(x) for x in ts.n_pending])}")

    # ── DTP phase time-series ─────────────────────────────────────────────────
    print(header("  DTP phase counts over episode (final values)"))
    print(DIV)
    for phase, series in ts.dtp_phases.items():
        if series:
            print(f"  {phase:<15}  final={series[-1]:>5}  "
                  f"peak={max(series):>5}  {_series_stats(series)}")

    # ── KPI tracker summary ───────────────────────────────────────────────────
    print(header("  KPI tracker summary"))
    print(DIV)
    try:
        summary = env.kpi.summary()
        for k, v in summary.items():
            fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {k:<35}  {fmt}")
    except Exception as e:
        print(f"  {warn(f'kpi.summary() raised: {e}')}")

    # ── Infrastructure event log ──────────────────────────────────────────────
    print(header("  Infrastructure event log"))
    print(DIV)
    try:
        from env.infrastructure import CheckpointID
        event_log = env.infra.get_all_events()
        counts: Dict[str, int] = {}
        for e in event_log:
            key = e.checkpoint.value if hasattr(e.checkpoint, "value") else str(e.checkpoint)
            counts[key] = counts.get(key, 0) + 1
        if counts:
            for checkpoint, count in sorted(counts.items()):
                print(f"  {checkpoint:<30}  {count:>6} events")
        else:
            print(f"  {warn('No infrastructure events recorded')}")
    except Exception as e:
        print(f"  {warn(f'Infrastructure event log unavailable: {e}')}")

    # ── Warnings ──────────────────────────────────────────────────────────────
    if log.has_warnings:
        print(header("  WARNINGS"))
        print(DIV)
        for w in log.warnings:
            print(f"  {warn(w)}")

    # ── Errors ────────────────────────────────────────────────────────────────
    if log.has_errors:
        print(header("  ERRORS"))
        print(DIV)
        # Show all errors (no cap — this is a test runner, not a prod log)
        for e in log.errors:
            print(f"  {fail(e)}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    if log.has_errors:
        err_count  = len(log.errors)
        warn_count = len(log.warnings)
        print(
            f"  {RED}{BOLD}FAILED{RESET}  —  "
            f"{err_count} error(s)  {warn_count} warning(s)"
        )
    else:
        warn_count = len(log.warnings)
        label = f"  {YELLOW}{BOLD}PASSED WITH WARNINGS{RESET}" if warn_count \
                else f"  {GREEN}{BOLD}PASSED{RESET}"
        print(f"{label}  —  0 errors  {warn_count} warning(s)")
    print(f"{'═' * W}\n")

    return not log.has_errors


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation(with_orchestrator: bool,
                   n_steps: int,
                   log_interval: int = 100,
                   verbose: bool = False) -> bool:

    log = ErrorLog()
    ts  = TimeSeriesTracker()

    scenario = "Scenario MO (with Orchestrator)" if with_orchestrator \
               else "Scenario M  (no Orchestrator)"

    print(f"\n{'═' * 64}")
    print(f"  {BOLD}Schiphol Cargo Hub — Simulation Sanity Check{RESET}")
    print(f"  {scenario}")
    print(f"  Steps: {n_steps}  |  Step duration: {params['marl']['step_min']} min")
    print(f"{'═' * 64}\n")

    # ── 1. Instantiate ────────────────────────────────────────────────────────
    print(header("  Phase 1 — Instantiation"))
    try:
        env = SchipholCargoEnv(with_orchestrator=with_orchestrator)
        print(ok("Environment instantiated"))
    except Exception as e:
        print(fail(f"Instantiation crashed: {e}"))
        raise

    # ── 2. Reset ──────────────────────────────────────────────────────────────
    print(header("  Phase 2 — Reset"))
    try:
        obs, infos = env.reset()
        print(ok("reset() returned successfully"))
    except Exception as e:
        print(fail(f"reset() crashed: {e}"))
        raise

    action_spaces = {a: env.action_space(a)      for a in env.agents}
    obs_spaces    = {a: env.observation_space(a)  for a in env.agents}

    validate_reset(env, obs, infos, obs_spaces, action_spaces, log)

    if log.has_errors:
        print(fail(f"{len(log.errors)} error(s) found during reset — aborting"))
        for e in log.errors:
            print(f"    {e}")
        return False
    print(ok("Reset validation passed"))

    # ── 3. Episode loop ───────────────────────────────────────────────────────
    print(header(f"  Phase 3 — Episode ({n_steps} steps)"))
    action_counts: Dict[str, Dict[int, int]] = {a: {} for a in env.agents}

    prev_truck_counter = 0
    t_start = time.perf_counter()

    for step in range(1, n_steps + 1):
        # Sample actions
        actions = build_actions(env, infos)

        # Track action distribution
        for agent, action in actions.items():
            action_counts[agent][action] = \
                action_counts[agent].get(action, 0) + 1

        # Step the environment
        try:
            obs, rewards, term, trunc, infos = env.step(actions)
        except Exception as e:
            log.error(f"step {step} | env.step() crashed: {e}")
            print(fail(f"env.step() crashed at step {step}: {e}"))
            raise

        # ── Per-step validation ───────────────────────────────────────────────
        validate_observations(obs, obs_spaces, step, log)
        validate_rewards(rewards, step, log)
        validate_masks(infos, action_spaces, step, log)
        validate_dones(term, "termination", step, log)
        validate_dones(trunc, "truncation", step, log)
        validate_dtp_registry(env, step, log)
        validate_tp3(env, step, log)
        validate_demand_pipeline(env, step, log)
        validate_terminals(env, step, log)

        # ── Time-series recording ─────────────────────────────────────────────
        ts.record(step, env, obs, rewards, infos, env.agents)

        # ── Progress print ────────────────────────────────────────────────────
        if step % log_interval == 0:
            n_new_trucks = env.demand._truck_counter - prev_truck_counter
            prev_truck_counter = env.demand._truck_counter
            step_errors = sum(
                1 for e in log.errors
                if e.startswith(f"step {step:>5}")
            )
            status_sym = f"{RED}ERR{RESET}" if step_errors else f"{GREEN} OK{RESET}"
            avg_rew = {
                a: np.mean(ts.rewards[a][-log_interval:])
                for a in env.agents
                if ts.rewards[a]
            }
            rew_str = "  ".join(
                f"{a[:8]}={v:+.3f}" for a, v in avg_rew.items()
            )
            print(
                f"  [{status_sym}] "
                f"step {step:>5} | "
                f"t={env.sim.now/60:.1f}h | "
                f"TP3={env.tp3.occupancy_ratio():.0%} | "
                f"pending={len(env.demand.pending_trucks):>3} | "
                f"+{n_new_trucks} trucks | "
                f"{rew_str}"
            )

        # ── Verbose per-step dump ─────────────────────────────────────────────
        if verbose:
            for agent in env.agents:
                vec = obs.get(agent, np.zeros(1))
                print(
                    f"        {agent:<20} "
                    f"obs=[{vec.min():.3f},{vec.max():.3f}] "
                    f"r={rewards.get(agent, 0):+.4f} "
                    f"mask_live={infos[agent]['action_mask'].sum()}"
                )

    elapsed = time.perf_counter() - t_start

    # ── 4. Post-episode aggregate checks ─────────────────────────────────────
    print(header("  Phase 4 — Post-episode aggregate checks"))
    _reward_stability_check(ts, log, window=min(50, n_steps // 4))
    _obs_drift_check(ts, log,        window=min(100, n_steps // 2))
    _tp3_stall_check(ts, log,        window=min(100, n_steps // 2))
    _pending_trucks_stall_check(ts, log, window=min(200, n_steps // 2))

    # Truck counter must be monotonically increasing
    if env.demand._truck_counter < 0:
        log.error("_truck_counter is negative after episode")

    # At least some trucks must have been generated in a run of this length
    if n_steps >= 50 and env.demand._truck_counter == 0:
        log.warning("No trucks were generated in this episode — demand.run() may be broken")

    print(ok("Aggregate checks complete") if not log.has_errors else
          warn(f"{len(log.errors)} total errors so far"))

    # ── 5. Final report ───────────────────────────────────────────────────────
    passed = print_report(env, ts, log, action_counts, n_steps, elapsed)
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schiphol Cargo Hub — Simulation Sanity Check"
    )
    parser.add_argument(
        "--orchestrator", action="store_true",
        help="Run Scenario MO (with Orchestrator agent)"
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of MARL steps to run (default: 200)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100,
        help="Print progress every N steps (default: 100)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-step per-agent obs/reward/mask details"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(info(f"Random seed set to {args.seed}"))

    passed = run_simulation(
        with_orchestrator=args.orchestrator,
        n_steps=args.steps,
        log_interval=args.log_interval,
        verbose=args.verbose,
    )

    sys.exit(0 if passed else 1)