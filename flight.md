# Flight Work Guide — Schiphol MARL Project
# Modules: kpi_tracker.py · benchmarl_task.py
# Estimated total time: ~3.5 hours

---

## Before You Start: Read This Carefully

This guide gives you five self-contained tasks ordered by priority and
independence. Each task has:
- A diagnosis explaining WHY it needs to be done
- An exact specification of WHAT to write
- A verification checklist you can run mentally without executing code

Do tasks in order. Tasks 1–3 are in kpi_tracker.py. Tasks 4–5 are in
benchmarl_task.py. Task 3 depends on Task 2. Everything else is independent.

---

## TASK 1 — Fix the eval=0 bug by auditing reward weights
**File:** `config/params.yaml` and `kpi_tracker.py`
**Time:** 30 minutes
**Why:** Every eval metric was exactly 0 across all agents. This means
the reward function returns 0 when agents take no actions. This is the
single most dangerous bug in the current system — a no-op policy is
optimal under the current reward design.

### Diagnosis

During BenchMARL evaluation, deterministic argmax is used. With random
network weights, argmax returns action 0 (no-op) for every agent at every
step. When no-op is selected for 1440 steps:

- No trucks are dispatched → pending_trucks grows but trucks never reach GHAs
- No DOCK_START/DOCK_END events fire → _total_wait=0, _total_service=0
- No terminal occupancy → exp_occupancy()=0, imp_occupancy()=0

This causes:
- `global_reward()`: wpr()=0 and util_std()=0 → returns exactly 0
- `gha_reward()`: util=0, delta_proc=0, q=0 → returns exactly 0
- `transporter_reward()`: delta_wait=0, delta_no_shows=0, delta_late=0,
  and ONLY the pending_trucks term can fire

The pending_trucks term in transporter_reward is:
```python
w["pending_trucks"] * len(demand.pending_trucks)
```
Open params.yaml right now and find `reward_weights`. Check the value of
`pending_trucks`. If it is 0.0, commented out, or missing — that is the
root cause of eval=0 for the transporter.

For GHAs and orchestrator, even a nonzero pending_trucks weight doesn't
help because they use `gha_reward()` and `orchestrator_reward()`, which
have NO terms that fire on a no-op step.

### What to implement

#### Step 1.1: Add per-step idle penalty to global_reward()

The global reward fires every step for every agent. Add a term that
penalises trucks sitting unprocessed in the system. This means both
pending_trucks (not yet dispatched) and trucks currently parked at TP3
waiting to be released.

```python
def global_reward(self, demand: DemandGenerator, dtp: DTPPlatform) -> float:
    """
    Called every step for every agent group.
    Now requires demand and dtp references to compute idle truck count.
    """
    w = self.w
    
    # Trucks waiting to be dispatched (never left the staging area)
    n_pending = len(demand.pending_trucks)
    
    # Trucks parked at TP3 waiting for orchestrator to release them
    # If no orchestrator: tp3 buffer is bypassed, this is 0
    n_tp3 = len(dtp.tp3_buffer) if hasattr(dtp, 'tp3_buffer') else 0
    
    idle_penalty = w.get("idle_truck", 0.01) * (n_pending + n_tp3)
    
    return -(
        w["wpr_global"] * self.wpr() +
        w["util_std"] * self.utilization_std() +
        idle_penalty
    )
```

IMPORTANT: You changed the signature of global_reward(). You must update
every call site in schiphol_env.py. Search for `global_reward()` and add
the demand and dtp arguments.

#### Step 1.2: Add "idle_truck" weight to params.yaml

In the `reward_weights` section of params.yaml, add:
```yaml
idle_truck: 0.01
```

The value 0.01 means each idle truck costs 0.01 reward per step. Over
1440 steps, one truck idle the whole episode costs 14.4 reward units.
This is in the same order of magnitude as your current per-step rewards
(mean = -0.88) so it will be visible without dominating.

#### Step 1.3: Verify the pending_trucks weight is nonzero

In transporter_reward(), the line:
```python
w["pending_trucks"] * len(demand.pending_trucks)
```
Check that w["pending_trucks"] is at least 0.005. If it is 0, set it to
0.005 in params.yaml.

#### Step 1.4: Add a queue penalty to gha_reward() that fires even on no-op

Currently gha_reward returns 0 on no-op because all its terms require
activity. Add a term for the per-step cost of having trucks queued at the
terminal entrance — this fires even when the GHA agent does nothing:

```python
def gha_reward(self, gha: str, terminal) -> float:
    w = self.w
    util = (terminal.exp_occupancy() + terminal.imp_occupancy()) / 2
    q = terminal.exp_queue_norm() + terminal.imp_queue_norm()
    total_proc = (
        terminal.stats["export"]["processed"] +
        terminal.stats["import"]["processed"]
    )
    delta_proc = total_proc - self._prev_proc[gha]
    self._prev_proc[gha] = total_proc

    # This fires every step regardless of GHA action
    # It penalises the GHA for having trucks stuck in its queue
    queue_cost = w.get("queue_per_step", 0.02) * q

    return (
        w["dock_util"] * util +
        w["parcel_on_time"] * delta_proc -
        queue_cost
    )
```

Note: if q is 0 because no trucks ever arrive (due to transporter no-op),
this still returns 0. That is CORRECT — the GHA cannot be penalised for a
queue that the transporter failed to create. The GHA's incentive to
publish slots is tied to the util and delta_proc terms that reward it when
trucks DO arrive.

### Verification checklist (mental)
- [ ] params.yaml has `idle_truck: 0.01` under reward_weights
- [ ] params.yaml has `pending_trucks` > 0 (suggest 0.005)
- [ ] global_reward() now accepts demand and dtp parameters
- [ ] Every call to self.kpi.global_reward() in schiphol_env.py passes
      (self.demand, self.dtp) as arguments
- [ ] The no-op scenario now produces: global_reward < 0 (idle_penalty
      fires), transporter_reward < 0 (pending_trucks fires),
      gha_reward = 0 (no queue since transporter never dispatched)

---

## TASK 2 — Add per-flow-type KPI accumulators
**File:** `kpi_tracker.py`
**Time:** 45 minutes
**Why:** Now that slots and trucks are typed (export/import), the tracker
has no way to tell you whether the system treats both flow types equitably.
This is diagnostic, not a bug — but without it you cannot tell whether
your policy learned to prioritise one flow type over another.

Also: SensorEvent carries a flow_type field that is currently ignored
entirely by the tracker.

### What to implement

#### Step 2.1: Add flow-type accumulators in __init__

After the existing episode-level accumulators, add:

```python
# Per-flow-type accumulators — mirrors of _total_* split by flow type
self._export_wait: float = 0.0
self._import_wait: float = 0.0
self._export_service: float = 0.0
self._import_service: float = 0.0
self._export_completed: int = 0
self._import_completed: int = 0
self._export_nttp_sum: float = 0.0
self._import_nttp_sum: float = 0.0
```

#### Step 2.2: Capture flow_type at GATE_IN

The truck's flow_type is known when it enters. Store it in the per-truck
state dict so it is available at every subsequent event for that truck.

In the `if e.checkpoint == CheckpointID.GATE_IN:` block, change:

```python
# BEFORE
self._truck[tid] = {
    "gate_in": e.sim_time,
    "n_parcels": e.n_parcels or 0,
    "gha_in": {},
    "dock_start": {},
    "dock_end": {},
}

# AFTER
self._truck[tid] = {
    "gate_in": e.sim_time,
    "n_parcels": e.n_parcels or 0,
    "flow_type": e.flow_type or "unknown",   # "export" or "import"
    "gha_in": {},
    "dock_start": {},
    "dock_end": {},
}
```

Note: SensorEvent must have a flow_type field for this to work. Check
infrastructure.py to confirm SensorEvent has this attribute. If it does
not, you need to add it — SensorEvent is likely a dataclass or namedtuple
and you add `flow_type: str = ""` to its fields.

#### Step 2.3: Accumulate wait time by flow_type at DOCK_START

In the `elif e.checkpoint == CheckpointID.DOCK_START:` block, after the
existing lines:

```python
# existing lines
self._total_wait += wait
if self._peak_start <= e.sim_time <= self._peak_end:
    self._peak_wait += wait

# ADD THESE LINES
state = self._truck.get(tid)
if state:
    if state["flow_type"] == "export":
        self._export_wait += wait
    else:
        self._import_wait += wait
```

#### Step 2.4: Accumulate service time by flow_type at DOCK_END

In the `elif e.checkpoint == CheckpointID.DOCK_END:` block, after the
existing lines:

```python
# existing lines
self._total_service += service
if self._peak_start <= e.sim_time <= self._peak_end:
    self._peak_service += service

# ADD THESE LINES
state = self._truck.get(tid)
if state:
    if state["flow_type"] == "export":
        self._export_service += service
    else:
        self._import_service += service
```

#### Step 2.5: Accumulate completed trucks and NTTP by flow_type at GATE_OUT

In the `elif e.checkpoint == CheckpointID.GATE_OUT:` block, replace the
existing nttp computation:

```python
# BEFORE
if state and state["n_parcels"] > 0:
    turnaround = e.sim_time - state["gate_in"]
    self._nttp_sum  += turnaround / state["n_parcels"]
    self._n_completed += 1

# AFTER
if state and state["n_parcels"] > 0:
    turnaround = e.sim_time - state["gate_in"]
    nttp_contrib = turnaround / state["n_parcels"]
    self._nttp_sum += nttp_contrib
    self._n_completed += 1
    if state["flow_type"] == "export":
        self._export_nttp_sum += nttp_contrib
        self._export_completed += 1
    else:
        self._import_nttp_sum += nttp_contrib
        self._import_completed += 1
```

#### Step 2.6: Add flow-type KPI properties

After the existing wpr(), peak_wpr(), nttp() properties, add:

```python
def export_wpr(self) -> float:
    """WPR for export trucks only."""
    return (0.0 if self._export_service == 0
            else self._export_wait / self._export_service)

def import_wpr(self) -> float:
    """WPR for import trucks only."""
    return (0.0 if self._import_service == 0
            else self._import_wait / self._import_service)

def export_nttp(self) -> float:
    return (0.0 if self._export_completed == 0
            else self._export_nttp_sum / self._export_completed)

def import_nttp(self) -> float:
    return (0.0 if self._import_completed == 0
            else self._import_nttp_sum / self._import_completed)

def flow_type_wpr_gap(self) -> float:
    """
    Absolute difference between export and import WPR.
    A healthy system keeps this near 0. Large values mean one flow type
    is being systematically disadvantaged.
    """
    return abs(self.export_wpr() - self.import_wpr())
```

#### Step 2.7: Add flow-type KPIs to summary()

```python
def summary(self) -> Dict:
    return {
        "wpr": self.wpr(),
        "peak_wpr": self.peak_wpr(),
        "nttp": self.nttp(),
        "export_wpr": self.export_wpr(),
        "import_wpr": self.import_wpr(),
        "export_nttp": self.export_nttp(),
        "import_nttp": self.import_nttp(),
        "flow_type_wpr_gap": self.flow_type_wpr_gap(),
        "util_std": self.utilization_std(),
        "n_completed": self._n_completed,
        "export_completed": self._export_completed,
        "import_completed": self._import_completed,
        "global_reward": self.global_reward(None, None),  # NOTE: update signature
    }
```

Wait — after Task 1, global_reward() requires demand and dtp. The summary
is called at episode end in schiphol_env.py, not inside KPITracker itself.
So summary() should NOT call global_reward() — remove that line from
summary() entirely. The schiphol_env.py already has access to everything
it needs to log rewards separately.

### Verification checklist (mental)
- [ ] _export_wait, _import_wait, _export_service, _import_service all
      start at 0.0 in __init__
- [ ] flow_type is stored in _truck[tid] at GATE_IN
- [ ] DOCK_START handler updates both _total_wait AND the correct
      flow-type wait accumulator
- [ ] DOCK_END handler updates both _total_service AND the correct
      flow-type service accumulator
- [ ] GATE_OUT handler updates both _n_completed AND the correct
      flow-type completed counter
- [ ] export_wpr() + import_wpr() guard against division by zero
- [ ] summary() does NOT call global_reward() (signature changed in Task 1)

---

## TASK 3 — Add orchestrator_reward() to KPITracker
**File:** `kpi_tracker.py`
**Time:** 45 minutes
**Why:** The orchestrator currently has no dedicated reward function in
KPITracker. Looking at how schiphol_env.py is likely structured, the
orchestrator probably receives the global_reward() like everyone else,
which means it has no signal specific to its job (releasing trucks from
TP3 at the right time). This explains why the orchestrator's critic loss
is 6x larger than the GHAs' — its return distribution is the same as the
global signal but its actions have no direct connection to it.

The orchestrator's job is:
1. Assign parked trucks at TP3 to specific GHAs at the right time
2. Prevent TP3 overflow (trucks arriving with nowhere to go)
3. Time releases so trucks arrive at their booked slot window

### What to implement

#### Step 3.1: Add TP3 buffer size tracker in __init__

```python
# Add to __init__ after the existing _prev_* trackers:
self._prev_tp3_size: int = 0
self._prev_dispatched: int = 0   # total trucks released from TP3 so far
```

#### Step 3.2: Add orchestrator_reward() method

Add this after gha_reward():

```python
def orchestrator_reward(self, dtp: DTPPlatform, tp3_buffer: list,
                         max_tp3_capacity: int) -> float:
    """
    Per-step reward for the orchestrator agent.
    
    Args:
        dtp: the DTPPlatform instance (for slot timing info)
        tp3_buffer: current list of trucks parked at TP3
        max_tp3_capacity: maximum number of trucks TP3 can hold
                         (from params["tp3"]["capacity"] or similar)
    
    Returns:
        float: step reward for orchestrator
    """
    w = self.w
    current_tp3_size = len(tp3_buffer)

    # ── Term 1: TP3 overflow pressure ────────────────────────────────
    # Penalise proportionally to how full TP3 is.
    # At 0% full: 0 penalty. At 100% full: full penalty.
    # This incentivises the orchestrator to release trucks proactively.
    if max_tp3_capacity > 0:
        occupancy_ratio = current_tp3_size / max_tp3_capacity
    else:
        occupancy_ratio = 0.0
    overflow_penalty = w.get("tp3_pressure", 0.05) * occupancy_ratio

    # ── Term 2: TP3 size delta — reward for releasing trucks ─────────
    # If TP3 shrank this step, the orchestrator released a truck.
    # Reward that positively (it means it acted instead of waiting).
    # If TP3 grew, that is demand-driven (new truck arrived) — neutral.
    delta_tp3 = self._prev_tp3_size - current_tp3_size  # positive when shrinking
    release_reward = w.get("tp3_release", 0.02) * max(0, delta_tp3)
    self._prev_tp3_size = current_tp3_size

    # ── Term 3: Late arrivals delta ───────────────────────────────────
    # Re-use the late arrival tracking already done in transporter_reward.
    # Late arrivals are partly the orchestrator's fault (released too late).
    # NOTE: This double-counts with transporter_reward. You can remove this
    # term if you want clean separation, but including it gives the
    # orchestrator a direct signal about timing quality.
    current_late = sum(dtp.late_arrivals.values())
    delta_late = max(0, current_late - self._prev_late)
    # NOTE: Do NOT update self._prev_late here — transporter_reward owns it.
    # If orchestrator_reward is called AFTER transporter_reward in the step,
    # self._prev_late was already updated and delta_late will be 0 here.
    # If called BEFORE, it will see the same delta. 
    # DECISION: Remove the late term from here to avoid double-counting.
    # The orchestrator's influence on late arrivals is already captured
    # indirectly through the transporter's reward signal.

    return -(overflow_penalty) + release_reward
```

IMPORTANT CALL-ORDER NOTE: After writing this, open schiphol_env.py and
find where transporter_reward() is called. orchestrator_reward() must be
called AFTER transporter_reward() in the same step so that _prev_late is
already updated. Otherwise the late delta is double-counted.

The safe pattern in schiphol_env.py should be:

```python
# In _get_rewards() or wherever rewards are computed:
rewards["transporter"] = (
    self.kpi.global_reward(self.demand, self.dtp) +
    self.kpi.transporter_reward(self.dtp, self.demand)
)
rewards["orchestrator"] = (
    self.kpi.global_reward(self.demand, self.dtp) +
    self.kpi.orchestrator_reward(
        self.dtp,
        self.tp3.get_parked_trucks(),     # or however TP3 is accessed
        params["tp3"]["capacity"]
    )
)
for gha in GHA_IDS:
    rewards[gha] = (
        self.kpi.global_reward(self.demand, self.dtp) +
        self.kpi.gha_reward(gha, self.terminals[gha])
    )
```

#### Step 3.3: Add TP3 weights to params.yaml

```yaml
reward_weights:
  # ... existing weights ...
  tp3_pressure: 0.05   # per-step cost per unit of TP3 occupancy ratio
  tp3_release: 0.02    # per-step reward per truck released from TP3
```

#### Step 3.4: Add tp3 tracking to __init__

```python
self._prev_tp3_size: int = 0
```

### Verification checklist (mental)
- [ ] orchestrator_reward() accepts dtp, tp3_buffer, max_tp3_capacity
- [ ] _prev_tp3_size is initialised to 0 in __init__
- [ ] The overflow term is in [0, tp3_pressure] per step (bounded)
- [ ] The release_reward only fires when tp3_buffer shrank (max(0, delta))
- [ ] orchestrator_reward is called AFTER transporter_reward in schiphol_env
- [ ] params.yaml has tp3_pressure and tp3_release under reward_weights
- [ ] schiphol_env.py passes orchestrator_reward() to rewards["orchestrator"]

---

## TASK 4 — Add reset() method to KPITracker
**File:** `kpi_tracker.py`
**Time:** 20 minutes
**Why:** When schiphol_env.reset() is called (start of new episode),
the KPI state must be wiped. Currently this is likely done by creating
a new KPITracker() instance, which works but allocates memory on every
reset and is harder to reason about. A reset() method is the clean pattern.

Also: after Tasks 1–3, there are now more fields to reset and doing it in
one place prevents subtle bugs where a new field is added but the
schiphol_env.py instantiation pattern misses it.

### What to implement

Add this method to KPITracker, ideally right after __init__:

```python
def reset(self) -> None:
    """
    Wipe all episode state. Called at the start of each new episode.
    Keeps structural attributes (peak window, weights, gha list) intact.
    """
    # Per-truck working state
    self._truck.clear()

    # Episode-level accumulators
    self._total_wait = 0.0
    self._total_service = 0.0
    self._peak_wait = 0.0
    self._peak_service = 0.0
    self._nttp_sum = 0.0
    self._n_completed = 0

    # Per-flow-type accumulators (added in Task 2)
    self._export_wait = 0.0
    self._import_wait = 0.0
    self._export_service = 0.0
    self._import_service = 0.0
    self._export_completed = 0
    self._import_completed = 0
    self._export_nttp_sum = 0.0
    self._import_nttp_sum = 0.0

    # Utilization snapshots — reset the lists but keep the structure
    for gha in self._ghas:
        self._util[gha]["export"].clear()
        self._util[gha]["import"].clear()

    # Delta trackers
    self._prev_proc = {gha: 0 for gha in self._ghas}
    self._prev_total_wait = 0.0
    self._prev_no_shows = 0
    self._prev_late = 0
    self._prev_tp3_size = 0   # added in Task 3
```

Then open schiphol_env.py and find the reset() method. Look for the line
that creates a new KPITracker. It likely looks like:

```python
self.kpi = KPITracker()
```

Replace it with:

```python
self.kpi.reset()
```

But ONLY if self.kpi already exists (i.e., this is not the first call).
The safe pattern is:

```python
if hasattr(self, 'kpi'):
    self.kpi.reset()
else:
    self.kpi = KPITracker()
```

### Verification checklist (mental)
- [ ] reset() clears _truck dict (not reassign — use .clear())
- [ ] reset() resets ALL fields added in Tasks 2 and 3
- [ ] reset() preserves _peak_start, _peak_end, _ghas, w (structural)
- [ ] schiphol_env.py uses self.kpi.reset() instead of self.kpi = KPITracker()
- [ ] The first call to schiphol_env.reset() still creates KPITracker via __init__

---

## TASK 5 — Harden benchmarl_task.py
**File:** `benchmarl_task.py`
**Time:** 60 minutes
**Why:** The task module has several subtle issues that will cause silent
failures during training rather than clean error messages. These are
architectural decisions that are hard to diagnose at runtime.

### Issue 5.1: RewardSum transform key mismatch

The current `get_reward_sum_transform` builds reward keys as tuples:
```python
reward_keys = [(group, "reward") for group in group_map.keys()]
```

TorchRL's RewardSum expects these as NestedKeys, which in practice means
either strings or tuples. The tuple format `("transporter", "reward")`
maps to a nested tensordict key `td["transporter"]["reward"]`. This is
correct IF the PettingZooWrapper puts rewards under that nesting. Verify
by adding a debug print in make_env() during your next test run:

```python
# Add temporarily to make_env() after creating torchrl_env:
obs = torchrl_env.reset()
print("TensorDict keys:", obs.keys(include_nested=True, leaves_only=False))
```

The output will tell you the exact key structure. If rewards appear as
`td["transporter", "reward"]` (tuple key) vs `td["transporter"]["reward"]`
(nested), the RewardSum keys need to match.

For the flight: review the current code and add a comment above
get_reward_sum_transform explaining the expected key format and where to
verify it. Add an assertion:

```python
def get_reward_sum_transform(self, env: EnvBase):
    group_map = self.group_map(env)
    reward_keys = [(group, "reward") for group in group_map.keys()]
    ep_reward_keys = [(group, "episode_reward") for group in group_map.keys()]
    
    # IMPORTANT: reward_keys must match PettingZooWrapper's output key format.
    # PettingZooWrapper nests rewards as td[group][reward_key].
    # TorchRL tuple keys (group, "reward") access td[group]["reward"].
    # These are equivalent. If W&B shows episode_reward=0 at eval but not
    # at collection, the key format is wrong — debug with the print above.
    
    return RewardSum(in_keys=reward_keys, out_keys=ep_reward_keys)
```

### Issue 5.2: StepCounter placement conflict

The current make_env() applies StepCounter inside the factory:
```python
torchrl_env = TransformedEnv(torchrl_env)
torchrl_env.append_transform(StepCounter(task_cfg.max_steps))
```

BenchMARL's experiment infrastructure also adds a StepCounter. If both
are active, the step count is incremented twice per step and the episode
ends at max_steps/2. This is a known issue with custom PettingZoo tasks.

**The fix:** Remove the StepCounter from make_env() and let BenchMARL's
own infrastructure handle it via the max_steps() method. Your max_steps()
method already returns the correct value:

```python
def max_steps(self, env: EnvBase) -> int:
    return self.config.max_steps
```

BenchMARL calls this and adds its own StepCounter. So remove these two
lines from make_env():

```python
# REMOVE THESE:
torchrl_env = TransformedEnv(torchrl_env)
torchrl_env.append_transform(StepCounter(task_cfg.max_steps))
```

If get_env_transforms() or get_reward_sum_transform() need a
TransformedEnv wrapper, BenchMARL handles that too — your task's
transform methods are called after the base env is created.

CAVEAT: If your last test run produced correct 1440-step episodes, then
the double-StepCounter issue is not occurring (BenchMARL may be smarter
about this in your version). In that case, leave make_env() as is and
only remove StepCounter if you see episodes ending at 720 steps instead
of 1440. Add a comment explaining this risk.

### Issue 5.3: num_envs parameter is ignored

The get_env_fun() signature has:
```python
def get_env_fun(self, seed, device, continuous_actions, num_envs) -> Callable:
```

But make_env() creates exactly one environment and ignores num_envs.
For a non-vectorized PettingZoo env this is correct — BenchMARL handles
the "multiple envs" concept by calling get_env_fun() multiple times
(once per worker). The num_envs parameter is intended for vectorized
environments (like VMAS) that can batch internally.

Add a comment making this explicit so you don't confuse yourself later:

```python
def get_env_fun(self, seed: Optional[int], device: str,
                continuous_actions: bool, num_envs: int) -> Callable[[], EnvBase]:
    """
    Returns a callable factory for one environment instance.
    
    num_envs is intentionally ignored. SchipholCargoEnv is a non-vectorized
    PettingZoo environment backed by SimPy. It cannot be batched internally.
    BenchMARL handles parallelism by calling this factory multiple times
    across worker processes (controlled by on_policy_n_envs_per_worker).
    Each worker gets its own independent env instance.
    """
```

### Issue 5.4: group_map() reads from env.base_env which may not exist

```python
def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
    return env.base_env.group_map
```

`env.base_env` works when env is a TransformedEnv wrapping a
PettingZooWrapper. But if BenchMARL passes the raw PettingZooWrapper
(not wrapped in TransformedEnv), `base_env` does not exist and you get
an AttributeError.

Also, `group_map` on PettingZooWrapper is the dict you passed in at
construction time — it exists as an attribute only if PettingZooWrapper
stores it. Verify that PettingZooWrapper exposes it. If not, you need
to reconstruct it from the env's agent list.

**Safe version:**

```python
def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
    """
    Returns the agent grouping used for this task.
    
    Traverses the env wrapper chain to find the PettingZooWrapper,
    then reads the group_map it was initialised with.
    """
    # Unwrap TransformedEnv layers to reach the PettingZooWrapper
    base = env
    while hasattr(base, 'base_env'):
        base = base.base_env
    
    # PettingZooWrapper stores group_map as set during construction
    if hasattr(base, 'group_map'):
        return base.group_map
    
    # Fallback: reconstruct from agent list
    # This happens if PettingZooWrapper does not expose group_map
    with_orch = self.config.with_orchestrator
    gha_ids = list(params["ghas"].keys())
    groups = {
        "transporter": ["transporter"],
        "ghas": gha_ids,
    }
    if with_orch:
        groups["orchestrator"] = ["orchestrator"]
    return groups
```

### Issue 5.5: associated_class() — verify BenchMARL's dispatch mechanism

Your SchipholTask inherits from BenchMARL's Task (which is an Enum) and
defines:

```python
class SchipholTask(Task):
    SCENARIO_M = "scenario_m"
    SCENARIO_MO = "scenario_mo"
    
    @staticmethod
    def associated_class():
        return SchipholTaskImplementation
```

BenchMARL's Task enum uses associated_class() to know which implementation
class provides the env factory and spec methods. When you call:

```python
task = SchipholTask.SCENARIO_MO.get_task(config={...})
```

BenchMARL internally does something like:
```python
impl_class = self.associated_class()
return impl_class(name=self.name, config=config)
```

This returns a SchipholTaskImplementation instance. Your W&B run succeeded
(iter=0 completed), so this dispatch is working. However add a comment
explaining the pattern:

```python
@staticmethod
def associated_class():
    """
    BenchMARL uses this to find the implementation class for this task.
    SchipholTaskImplementation provides all the env factory logic.
    SchipholTask (this class) is the Enum used in train.py to select scenarios.
    
    The string values ("scenario_m", "scenario_mo") must match the YAML
    filenames under conf/task/schiphol/ if you use Hydra-based config loading.
    When using get_task(config={...}) directly (as in train.py), the YAML
    files are not needed.
    """
    return SchipholTaskImplementation
```

### Verification checklist (mental)
- [ ] StepCounter is either removed from make_env() OR you have a comment
      explaining why double-application is not occurring in your version
- [ ] num_envs parameter has a docstring explaining why it is ignored
- [ ] group_map() has a safe fallback that does not crash if base_env
      chain traversal fails
- [ ] associated_class() has a comment explaining the dispatch pattern
- [ ] get_reward_sum_transform() has a comment about key format verification

---

## FINAL: Ordering and interdependencies

```
Task 1 (reward weights + idle penalty)
    └── Changes global_reward() signature
        └── Must update all call sites in schiphol_env.py

Task 2 (flow-type KPIs)
    └── Requires SensorEvent.flow_type field to exist
        └── Check infrastructure.py — add if missing

Task 3 (orchestrator_reward)
    └── Depends on Task 4 (_prev_tp3_size in __init__)
    └── Requires call-order fix in schiphol_env.py

Task 4 (reset method)
    └── Must include ALL fields from Tasks 1, 2, 3

Task 5 (benchmarl_task hardening)
    └── Fully independent — do this last or first, your choice
```

The minimum viable set before the next training run is Task 1 + Task 4.
Tasks 2, 3, 5 are improvements that do not block training.

---

## Quick reference: reward_weights that must exist in params.yaml

After all tasks are complete, your reward_weights section must have:

```yaml
reward_weights:
  wpr_global:      0.5     # global WPR penalty (all agents)
  util_std:        0.3     # dock utilization imbalance penalty (all agents)
  idle_truck:      0.01    # per idle truck per step (all agents) [NEW Task 1]
  wait_per_min:    0.1     # per minute of wait accumulation (transporter)
  no_show:         1.0     # per no-show event (transporter)
  missed_slot:     0.5     # per late arrival event (transporter)
  pending_trucks:  0.005   # per pending truck per step (transporter) [CHECK]
  dock_util:       1.0     # dock utilization reward (gha)
  parcel_on_time:  0.5     # per truck processed (gha)
  queue_per_step:  0.02    # per queue unit per step (gha)
  tp3_pressure:    0.05    # TP3 occupancy ratio penalty (orchestrator) [NEW Task 3]
  tp3_release:     0.02    # reward per truck released from TP3 (orchestrator) [NEW Task 3]
```

Values above are suggestions based on the reward scales observed in your
runs (-0.88 mean per step). Adjust after seeing the first multi-iteration
training run.