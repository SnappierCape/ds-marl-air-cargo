# =============================================================================
# TEST SUITE — PETTINGZOO ENVIRONMENT WRAPPER  (schiphol_env.py)
# =============================================================================
# All external packages (simpy, gymnasium, pettingzoo, numpy) and every
# env.* / config.* import are stubbed so the suite runs with no Schiphol
# code except the file under test.
#
# BUGS DELIBERATELY TARGETED (found by static analysis):
#
#   BUG-A  _apply_action() signature declares THREE parameters
#          (self, agent, action, flow_type) but step() calls it with
#          TWO: self._apply_action(agent, action).
#          → TypeError on every step() call.
#
#   BUG-B  step() returns a 4-tuple (obs, rewards, dones, infos).
#          PettingZoo ParallelEnv.step() must return a 5-tuple:
#          (obs, rewards, terminations, truncations, infos).
#          → Training framework receives wrong positional arguments.
#
#   BUG-C  _get_mask() dispatch block checks
#              needed = {s["gha"] for s in truck.manifest}
#          instead of truck.stops_remaining.
#          A truck that has already completed a stop still has that GHA
#          in manifest, so dispatch is permanently blocked for trucks
#          that have partially executed their manifest.
#
#   BUG-D  _get_obs() for the Transporter reads
#              params["ghas"][gha]["total"]
#          but the params schema only has "export" and "import" keys
#          per GHA.  → KeyError on every transporter observation build.
#
#   BUG-E  Inside the Orchestrator branch of _apply_action(), the local
#          variable `flow_type` is used at lines 207 and 209, but it is
#          only reachable via step(), which never passes flow_type to
#          _apply_action().  Even if the call were corrected, flow_type
#          is never given a concrete value inside the orchestrator branch.
#          → NameError / UnboundLocalError in orchestrator actions.
#
#   BUG-F  _obs_dim() and _action_dim() raise ValueError for any unknown
#          agent string but _get_reward() does NOT — it falls through to
#          the else branch which blindly calls self.terminals[agent],
#          raising a silent KeyError instead of a clear ValueError.
#
#   BUG-G  _get_mask() dispatch condition uses truck.manifest for
#          "needed" GHAs.  If the manifest contains a GHA that has
#          already been served (stop completed), that GHA will never be
#          in booked_slots, so needed ⊄ booked and dispatch is masked
#          off permanently — truck is stuck.
#
# TEST CLASSES:
#   TestSchipholEnvInit             — constructor, possible_agents
#   TestObsAndActionSpaces          — space types, shapes, dtypes
#   TestObsDimHelper                — _obs_dim() per agent + ValueError
#   TestActionDimHelper             — _action_dim() per agent + ValueError
#   TestResetReturnShape            — reset() tuple, obs shape, infos keys
#   TestResetAgentList              — agents reset, with/without orch.
#   TestStepReturnShape             — BUG-B 4-tuple vs 5-tuple
#   TestApplyActionSignature        — BUG-A TypeError on step()
#   TestApplyActionNoOp             — action 0 is always valid no-op
#   TestApplyActionTransporterBook  — book action decoding, bounds
#   TestApplyActionTransporterDispatch — dispatch action decoding
#   TestApplyActionGHA              — GHA publish actions
#   TestApplyActionOrchestrator     — BUG-E flow_type undefined
#   TestGetObsTransporter           — shape, bounds, BUG-D KeyError
#   TestGetObsGHA                   — shape, bounds, tod encoding
#   TestGetObsOrchestrator          — shape, bounds
#   TestGetReward                   — alpha mixing, orchestrator path, BUG-F
#   TestGetMaskNoOp                 — action 0 always unmasked
#   TestGetMaskTransporterBook      — valid book actions enabled
#   TestGetMaskTransporterDispatch  — BUG-C / BUG-G completed-stop bug
#   TestGetMaskGHA                  — window mask length
#   TestGetMaskOrchestrator         — parked-truck mask
#   TestPrepopulateSlots            — slot count per GHA/flow
#   TestNextPublishableWindows      — freeze, rounding, spacing
#   TestAllAvailableSlots           — sorted, capped at N_SLOT_ACTIONS
#   TestFindUnbookingTruck          — needs GHA / no booking filter
#   TestStressAndEdge               — many steps, all no-ops, all agents
# =============================================================================

import sys
import types
import math
import unittest
from unittest.mock import MagicMock, patch, call, PropertyMock


# ─────────────────────────────────────────────────────────────────────────────
# 0.  numpy — real (needed for the obs array arithmetic)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Stub every import that schiphol_env.py makes before we import it
# ─────────────────────────────────────────────────────────────────────────────

# ── simpy ──────────────────────────────────────────────────────────────────
simpy_mod = types.ModuleType("simpy")

class _SimPyEnv:
    def __init__(self):
        self.now = 0.0
        self._procs = []
    def process(self, gen):
        proc = MagicMock()
        self._procs.append(proc)
        # exhaust the generator so __init__ side-effects run
        try:
            next(gen)
        except StopIteration:
            pass
        return proc
    def run(self, until=None):
        self.now = until if until is not None else self.now

simpy_mod.Environment = _SimPyEnv
sys.modules["simpy"] = simpy_mod

# ── gymnasium ──────────────────────────────────────────────────────────────
gym_mod    = types.ModuleType("gymnasium")
spaces_mod = types.ModuleType("gymnasium.spaces")

class _Box:
    def __init__(self, low, high, shape, dtype):
        self.low = low; self.high = high
        self.shape = shape; self.dtype = dtype
class _Discrete:
    def __init__(self, n): self.n = n

spaces_mod.Box      = _Box
spaces_mod.Discrete = _Discrete
gym_mod.spaces      = spaces_mod
gym_mod.Space       = object
sys.modules["gymnasium"]        = gym_mod
sys.modules["gymnasium.spaces"] = spaces_mod

# ── pettingzoo ─────────────────────────────────────────────────────────────
pz_mod  = types.ModuleType("pettingzoo")
class _ParallelEnv:
    def __init__(self): pass
pz_mod.ParallelEnv = _ParallelEnv
sys.modules["pettingzoo"] = pz_mod

# ── config.config ──────────────────────────────────────────────────────────
config_mod        = types.ModuleType("config")
config_config_mod = types.ModuleType("config.config")

GHA_IDS_TEST = ["GHA_A", "GHA_B", "GHA_C"]

DEFAULT_PARAMS = {
    "ghas": {
        "GHA_A": {"export": 3, "import": 2},
        "GHA_B": {"export": 4, "import": 4},
        "GHA_C": {"export": 2, "import": 2},
    },
    "tp3":  {"capacity": 10},
    "road": {"speed": 60},
    "marl": {
        "step_min": 5,
        "alpha": 0.3,
        "reward_weights": {
            "wpr_global": 1.0, "util_std": 1.0,
            "wait_per_min": 1.0, "no_show": 5.0, "missed_slot": 3.0,
            "dock_util": 2.0, "parcel_on_time": 1.0, "queue_per_step": 0.5,
        },
    },
    "demand": {"peak_window": [60, 120]},
    "dtp_rules": {
        "slot_duration": 45,
        "lead_time": 4320,   # 72h in minutes
        "freeze_time": 15,
    },
}
config_config_mod.load_params = lambda: DEFAULT_PARAMS
sys.modules["config"]        = config_mod
sys.modules["config.config"] = config_config_mod

# ── env sub-modules ────────────────────────────────────────────────────────
def _make_env_stub(name, classes):
    mod = types.ModuleType(name)
    for cls_name, cls in classes.items():
        setattr(mod, cls_name, cls)
    sys.modules[name] = mod
    return mod

# env (parent)
# sys.modules["env"] = types.ModuleType("env")

class _Truck:
    STATUS_IN_TRANSIT = "in_transit"
    STATUS_AT_TP3     = "at_tp3"
    STATUS_QUEUED     = "queued"
    STATUS_DOCKED     = "docked"
    STATUS_DEPARTED   = "departed"
    def __init__(self, truck_id="T1", flow_type="export", manifest=None, booked_slots=None):
        self.truck_id      = truck_id
        self.flow_type     = flow_type
        self.manifest      = manifest or [{"gha": "GHA_A", "parcels": 5}]
        self.booked_slots  = booked_slots or {}
        self.stops_remaining = list(self.manifest)
        self.status        = self.STATUS_IN_TRANSIT

class _GHATerminal:
    def __init__(self, *a, **kw): pass
    def exp_occupancy(self): return 0.0
    def imp_occupancy(self): return 0.0
    def exp_queue_norm(self): return 0.0
    def imp_queue_norm(self): return 0.0
    def upcoming_bookings_norm(self, dtp, horizon): return 0.0

class _TP3Buffer:
    def __init__(self, *a, **kw):
        self._parked = []
    def occupancy_ratio(self): return 0.0
    def n_overflow(self): return 0
    def get_parked_trucks(self): return []
    def release(self, tid): pass

class _DTPPlatform:
    def __init__(self, *a, **kw):
        self.registry = {}
    def publish_slot(self, gha, t, ft): pass
    def get_available_slots(self, gha, horizon=None): return []
    def book_one_slot(self, *a): pass
    def dispatch_truck(self, *a): pass
    def get_slot_phase(self, *a): return "on_time"
    def orch_book_slot(self, *a): pass

class _InfrastructureLayer:
    def flush_step_buffer(self): return []

class _ServiceTimeModel:
    def __init__(self, p): pass
    def sample(self, ft): return 10.0

class _RoadNetwork:
    def __init__(self, cfg): pass

class _DemandGenerator:
    def __init__(self, *a, **kw):
        self.pending_trucks = []
    def run(self):
        return iter([])
    def book_one_slot(self, *a): pass
    def dispatch_truck(self, tid): pass

class _KPITracker:
    def __init__(self): pass
    def ingest(self, events): pass
    def snapshot_utilization(self, terminals): pass
    def global_reward(self): return 0.0
    def transporter_reward(self, dtp): return 0.0
    def gha_reward(self, agent, terminal): return 0.0

_make_env_stub("env.objects",       {"Truck": _Truck, "GHATerminal": _GHATerminal, "TP3Buffer": _TP3Buffer})
_make_env_stub("env.dtp_platform",  {"DTPPlatform": _DTPPlatform})
_make_env_stub("env.infrastructure",{"InfrastructureLayer": _InfrastructureLayer})
_make_env_stub("env.service_time",  {"ServiceTimeModel": _ServiceTimeModel})
_make_env_stub("env.road",          {"RoadNetwork": _RoadNetwork})
_make_env_stub("env.demand",        {"DemandGenerator": _DemandGenerator})
_make_env_stub("env.kpi_tracker",   {"KPITracker": _KPITracker})

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Import the module under test
# ─────────────────────────────────────────────────────────────────────────────
from env.schiphol_env import (               # noqa: E402
    SchipholCargoEnv,
    N_SLOT_ACTIONS, N_TP3_ACTIONS, N_PENDING_TRUCKS,
    N_GHAS, N_BOOK_ACTIONS, N_DISPATCH_ACTIONS,
    TRANSPORTER_ACTION_DIM, GHA_IDS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Shared factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_env(with_orchestrator=False):
    """Create a fully reset SchipholCargoEnv with all mocks wired."""
    env = SchipholCargoEnv(with_orchestrator=with_orchestrator)
    # Inject mock internals so we control them
    env.sim    = _SimPyEnv()
    env.infra  = MagicMock(spec=_InfrastructureLayer)
    env.infra.flush_step_buffer.return_value = []
    env.kpi    = MagicMock(spec=_KPITracker)
    env.kpi.global_reward.return_value      = 0.0
    env.kpi.transporter_reward.return_value = 0.0
    env.kpi.gha_reward.return_value         = 0.0
    env.dtp    = MagicMock(spec=_DTPPlatform)
    env.dtp.registry = {}
    env.dtp.get_available_slots.return_value = []
    env.tp3    = MagicMock(spec=_TP3Buffer)
    env.tp3.occupancy_ratio.return_value = 0.0
    env.tp3.n_overflow.return_value      = 0
    env.tp3.get_parked_trucks.return_value = []
    env.demand = MagicMock(spec=_DemandGenerator)
    env.demand.pending_trucks = []
    env.svc_tm = MagicMock(spec=_ServiceTimeModel)
    env.terminals = {
        gha: MagicMock(spec=_GHATerminal) for gha in GHA_IDS
    }
    for t in env.terminals.values():
        t.exp_occupancy.return_value = 0.0
        t.imp_occupancy.return_value = 0.0
        t.exp_queue_norm.return_value = 0.0
        t.imp_queue_norm.return_value = 0.0
        t.upcoming_bookings_norm.return_value = 0.0
    env.agents = env.possible_agents[:]
    env.step_min = 5
    env.alpha    = 0.3
    return env


def make_env_patched_apply(with_orchestrator=False):
    """
    Same as make_env() but with _apply_action monkey-patched to a 2-arg
    no-op.  Use this in tests that want to exercise step() internals
    WITHOUT triggering BUG-A (the missing flow_type argument), since
    BUG-A is already explicitly covered in TestApplyActionSignature.
    """
    env = make_env(with_orchestrator=with_orchestrator)
    env._apply_action = lambda agent, action: None   # bypass BUG-A
    return env


def make_truck(truck_id="T1", flow_type="export", manifest=None, booked_slots=None):
    m = manifest or [{"gha": g, "parcels": 3} for g in GHA_IDS]
    t = _Truck(truck_id=truck_id, flow_type=flow_type, manifest=m, booked_slots=booked_slots or {})
    return t


def no_op_actions(env):
    return {a: 0 for a in env.agents}


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class TestSchipholEnvInit(unittest.TestCase):

    def test_possible_agents_includes_transporter(self):
        env = SchipholCargoEnv()
        self.assertIn("transporter", env.possible_agents)

    def test_possible_agents_includes_all_ghas(self):
        env = SchipholCargoEnv()
        for gha in GHA_IDS:
            self.assertIn(gha, env.possible_agents)

    def test_possible_agents_excludes_orchestrator_by_default(self):
        env = SchipholCargoEnv(with_orchestrator=False)
        self.assertNotIn("orchestrator", env.possible_agents)

    def test_possible_agents_includes_orchestrator_when_requested(self):
        env = SchipholCargoEnv(with_orchestrator=True)
        self.assertIn("orchestrator", env.possible_agents)

    def test_agent_count_without_orchestrator(self):
        env = SchipholCargoEnv(with_orchestrator=False)
        # transporter + N_GHAS
        self.assertEqual(len(env.possible_agents), 1 + N_GHAS)

    def test_agent_count_with_orchestrator(self):
        env = SchipholCargoEnv(with_orchestrator=True)
        self.assertEqual(len(env.possible_agents), 1 + N_GHAS + 1)

    def test_step_min_loaded_from_params(self):
        env = SchipholCargoEnv()
        self.assertEqual(env.step_min, DEFAULT_PARAMS["marl"]["step_min"])

    def test_alpha_loaded_from_params(self):
        env = SchipholCargoEnv()
        self.assertAlmostEqual(env.alpha, DEFAULT_PARAMS["marl"]["alpha"])

    def test_metadata_name(self):
        self.assertEqual(SchipholCargoEnv.metadata["name"], "schiphol_cargo_v0")

    def test_with_orchestrator_flag_stored(self):
        env = SchipholCargoEnv(with_orchestrator=True)
        self.assertTrue(env.with_orchestrator)


# ---------------------------------------------------------------------------
class TestObsAndActionSpaces(unittest.TestCase):

    def setUp(self):
        self.env = make_env(with_orchestrator=True)

    def test_observation_space_returns_box(self):
        sp = self.env.observation_space("transporter")
        self.assertIsInstance(sp, _Box)

    def test_observation_space_shape_matches_obs_dim(self):
        for agent in self.env.agents:
            sp = self.env.observation_space(agent)
            self.assertEqual(sp.shape, (self.env._obs_dim(agent),),
                msg=f"Space shape mismatch for {agent}")

    def test_observation_space_dtype_float32(self):
        sp = self.env.observation_space("transporter")
        self.assertEqual(sp.dtype, np.float32)

    def test_observation_space_bounds_0_to_1(self):
        sp = self.env.observation_space("transporter")
        self.assertEqual(sp.low,  0.0)
        self.assertEqual(sp.high, 1.0)

    def test_action_space_returns_discrete(self):
        sp = self.env.action_space("transporter")
        self.assertIsInstance(sp, _Discrete)

    def test_action_space_n_matches_action_dim(self):
        for agent in self.env.agents:
            sp = self.env.action_space(agent)
            self.assertEqual(sp.n, self.env._action_dim(agent),
                msg=f"Action space mismatch for {agent}")


# ---------------------------------------------------------------------------
class TestObsDimHelper(unittest.TestCase):

    def setUp(self):
        self.env = make_env(with_orchestrator=True)

    def test_transporter_obs_dim_correct(self):
        expected = 4 + 2 * N_GHAS + 2 * N_GHAS + 4 * N_PENDING_TRUCKS
        self.assertEqual(self.env._obs_dim("transporter"), expected)

    def test_gha_obs_dim_correct(self):
        expected = 9 + 2 * (N_GHAS - 1)
        for gha in GHA_IDS:
            self.assertEqual(self.env._obs_dim(gha), expected,
                msg=f"obs_dim wrong for {gha}")

    def test_orchestrator_obs_dim_correct(self):
        expected = 4 + 5 * N_GHAS
        self.assertEqual(self.env._obs_dim("orchestrator"), expected)

    def test_obs_dim_raises_value_error_for_unknown_agent(self):
        with self.assertRaises(ValueError):
            self.env._obs_dim("ghost_agent")

    def test_obs_dim_raises_value_error_for_empty_string(self):
        with self.assertRaises(ValueError):
            self.env._obs_dim("")

    def test_obs_dim_gha_grows_with_n_ghas(self):
        """More GHAs → more cross-occupancy features in GHA obs."""
        base = 9 + 2 * (N_GHAS - 1)
        self.assertEqual(self.env._obs_dim(GHA_IDS[0]), base)


# ---------------------------------------------------------------------------
class TestActionDimHelper(unittest.TestCase):

    def setUp(self):
        self.env = make_env(with_orchestrator=True)

    def test_transporter_action_dim(self):
        self.assertEqual(self.env._action_dim("transporter"), TRANSPORTER_ACTION_DIM)

    def test_gha_action_dim_is_3(self):
        for gha in GHA_IDS:
            self.assertEqual(self.env._action_dim(gha), 3,
                msg=f"action_dim wrong for {gha}")

    def test_orchestrator_action_dim(self):
        expected = N_TP3_ACTIONS * N_GHAS + 1
        self.assertEqual(self.env._action_dim("orchestrator"), expected)

    def test_action_dim_raises_value_error_for_unknown_agent(self):
        with self.assertRaises(ValueError):
            self.env._action_dim("unknown_agent")

    def test_action_dim_raises_value_error_for_empty_string(self):
        with self.assertRaises(ValueError):
            self.env._action_dim("")

    def test_transporter_action_dim_formula(self):
        # 1 (no_op) + N_BOOK_ACTIONS + N_DISPATCH_ACTIONS
        self.assertEqual(TRANSPORTER_ACTION_DIM, 1 + N_BOOK_ACTIONS + N_DISPATCH_ACTIONS)


# ---------------------------------------------------------------------------
class TestResetReturnShape(unittest.TestCase):
    """reset() must return a 2-tuple (obs, infos)."""

    def _make_patchable_env(self, orch=False):
        """Patch all constructors that reset() calls, then call reset()."""
        env = SchipholCargoEnv(with_orchestrator=orch)
        # Patch the methods reset() calls that we don't want running for real
        env._prepopulate_slots = MagicMock()
        env._get_obs  = lambda a: np.zeros(env._obs_dim(a), dtype=np.float32)
        env._get_mask = lambda a: np.ones(env._action_dim(a), dtype=np.int8)

        with patch("env.schiphol_env.simpy.Environment", return_value=_SimPyEnv()), \
             patch("env.schiphol_env.InfrastructureLayer", return_value=MagicMock()), \
             patch("env.schiphol_env.ServiceTimeModel",    return_value=MagicMock()), \
             patch("env.schiphol_env.RoadNetwork",         return_value=MagicMock()), \
             patch("env.schiphol_env.DTPPlatform",         return_value=MagicMock()), \
             patch("env.schiphol_env.TP3Buffer",           return_value=MagicMock()), \
             patch("env.schiphol_env.KPITracker",          return_value=MagicMock()), \
             patch("env.schiphol_env.GHATerminal",         return_value=MagicMock()), \
             patch("env.schiphol_env.DemandGenerator",     return_value=MagicMock(spec=_DemandGenerator)):
            result = env.reset()
        return result, env

    def test_reset_returns_two_element_tuple(self):
        result, _ = self._make_patchable_env()
        self.assertEqual(len(result), 2)

    def test_reset_first_element_is_obs_dict(self):
        (obs, infos), env = self._make_patchable_env()
        self.assertIsInstance(obs, dict)

    def test_reset_obs_keys_match_agents(self):
        (obs, infos), env = self._make_patchable_env()
        self.assertEqual(set(obs.keys()), set(env.agents))

    def test_reset_obs_values_are_numpy_arrays(self):
        (obs, infos), env = self._make_patchable_env()
        for agent, arr in obs.items():
            self.assertIsInstance(arr, np.ndarray, msg=f"{agent} obs not ndarray")

    def test_reset_infos_contains_action_mask(self):
        (obs, infos), env = self._make_patchable_env()
        for agent in env.agents:
            self.assertIn("action_mask", infos[agent])

    def test_reset_agents_equals_possible_agents(self):
        _, env = self._make_patchable_env()
        self.assertEqual(env.agents, env.possible_agents)

    def test_reset_with_orchestrator_includes_orchestrator_obs(self):
        (obs, infos), env = self._make_patchable_env(orch=True)
        self.assertIn("orchestrator", obs)


# ---------------------------------------------------------------------------
class TestResetAgentList(unittest.TestCase):

    def test_agents_list_repopulated_on_second_reset(self):
        env = make_env()
        env.agents = []    # simulate end-of-episode clearing
        env._prepopulate_slots = MagicMock()
        env._get_obs  = lambda a: np.zeros(env._obs_dim(a), dtype=np.float32)
        env._get_mask = lambda a: np.ones(env._action_dim(a), dtype=np.int8)
        with patch("env.schiphol_env.simpy.Environment", return_value=_SimPyEnv()), \
             patch("env.schiphol_env.InfrastructureLayer", return_value=MagicMock()), \
             patch("env.schiphol_env.ServiceTimeModel",    return_value=MagicMock()), \
             patch("env.schiphol_env.RoadNetwork",         return_value=MagicMock()), \
             patch("env.schiphol_env.DTPPlatform",         return_value=MagicMock()), \
             patch("env.schiphol_env.TP3Buffer",           return_value=MagicMock()), \
             patch("env.schiphol_env.KPITracker",          return_value=MagicMock()), \
             patch("env.schiphol_env.GHATerminal",         return_value=MagicMock()), \
             patch("env.schiphol_env.DemandGenerator",     return_value=MagicMock(spec=_DemandGenerator)):
            env.reset()
        self.assertEqual(set(env.agents), set(env.possible_agents))


# ---------------------------------------------------------------------------
class TestStepReturnShape(unittest.TestCase):
    """
    Tests for step() return structure and side-effects.

    NOTE: All tests here use make_env_patched_apply() which bypasses BUG-A
    (_apply_action called with wrong number of args from step()).  BUG-A is
    explicitly tested in TestApplyActionSignature.  Using the patched env
    here lets us isolate all OTHER step() behaviour without BUG-A masking it.

    BUG-B: step() returns a 4-tuple (obs, rewards, dones, infos).
    PettingZoo ParallelEnv.step() must return a 5-tuple:
    (obs, rewards, terminations, truncations, infos).
    → Training framework receives wrong positional arguments.
    """

    def setUp(self):
        self.env = make_env_patched_apply()

    def test_step_obs_is_dict(self):
        obs, *_ = self.env.step(no_op_actions(self.env))
        self.assertIsInstance(obs, dict)

    def test_step_rewards_is_dict(self):
        _, rewards, *_ = self.env.step(no_op_actions(self.env))
        self.assertIsInstance(rewards, dict)

    def test_step_obs_keys_match_agents(self):
        obs, *_ = self.env.step(no_op_actions(self.env))
        self.assertEqual(set(obs.keys()), set(self.env.agents))

    def test_step_rewards_keys_match_agents(self):
        _, rewards, *_ = self.env.step(no_op_actions(self.env))
        self.assertEqual(set(rewards.keys()), set(self.env.agents))

    def test_step_infos_contain_action_mask(self):
        *_, infos = self.env.step(no_op_actions(self.env))
        for agent in self.env.agents:
            self.assertIn("action_mask", infos[agent])

    def test_step_advances_simpy_by_step_min(self):
        t_before = self.env.sim.now
        self.env.step(no_op_actions(self.env))
        self.assertAlmostEqual(self.env.sim.now, t_before + self.env.step_min)

    def test_step_calls_kpi_ingest(self):
        self.env.step(no_op_actions(self.env))
        self.env.kpi.ingest.assert_called_once()

    def test_step_calls_kpi_snapshot_utilization(self):
        self.env.step(no_op_actions(self.env))
        self.env.kpi.snapshot_utilization.assert_called_once_with(self.env.terminals)

    def test_step_calls_kpi_global_reward(self):
        self.env.step(no_op_actions(self.env))
        self.env.kpi.global_reward.assert_called_once()

    def test_step_obs_arrays_correct_shape(self):
        obs, *_ = self.env.step(no_op_actions(self.env))
        for agent, arr in obs.items():
            expected = self.env._obs_dim(agent)
            self.assertEqual(arr.shape, (expected,),
                msg=f"{agent} obs shape mismatch after step()")


# ---------------------------------------------------------------------------
class TestApplyActionSignature(unittest.TestCase):
    """
    BUG-A: _apply_action() declares (self, agent, action, flow_type) but
    step() calls self._apply_action(agent, action) — missing flow_type.
    """

    def setUp(self):
        self.env = make_env()

    def test_apply_action_no_op_does_not_raise(self):
        """Action 0 (no_op) returns immediately before flow_type is needed."""
        # Should NOT raise even with the bug, because the function returns early
        try:
            self.env._apply_action("transporter", 0)
        except Exception as e:
            self.fail(f"no_op raised unexpectedly: {e}")

    def test_apply_action_accepts_three_positional_args(self):
        """Verify the raw signature takes three args (documents correct form)."""
        import inspect
        sig = inspect.signature(self.env._apply_action)
        params = list(sig.parameters.keys())
        self.assertEqual(len(params), 2,
            "_apply_action must have exactly 2 parameters: agent, action")


# ---------------------------------------------------------------------------
class TestApplyActionNoOp(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_no_op_does_not_call_dtp_methods(self):
        self.env._apply_action("transporter", 0)
        self.env.dtp.publish_slot.assert_not_called()
        self.env.dtp.book_one_slot.assert_not_called()

    def test_no_op_valid_for_every_agent(self):
        for agent in self.env.agents:
            try:
                self.env._apply_action(agent, 0)
            except Exception as e:
                self.fail(f"no_op for {agent} raised: {e}")

    def test_no_op_does_not_change_sim_time(self):
        t_before = self.env.sim.now
        self.env._apply_action("transporter", 0)
        self.assertEqual(self.env.sim.now, t_before)


# ---------------------------------------------------------------------------
class TestApplyActionTransporterBook(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_book_action_calls_book_one_slot_when_truck_exists(self):
        truck = make_truck("T1")
        self.env.demand.pending_trucks = [truck]
        # Action 1 → truck_idx=0, gha_idx=0
        self.env._apply_action("transporter", 1)
        self.env.demand.book_one_slot.assert_called_once_with("T1", GHA_IDS[0], "export")

    def test_book_action_out_of_range_truck_index_is_silently_ignored(self):
        """Truck index beyond pending list — no crash, no call."""
        self.env.demand.pending_trucks = []
        self.env._apply_action("transporter", 1)
        self.env.demand.book_one_slot.assert_not_called()

    def test_book_action_gha_index_decoded_correctly(self):
        """Action = truck*N_GHAS + gha + 1.  Verify gha selection."""
        truck = make_truck("T1")
        self.env.demand.pending_trucks = [truck]
        # Action 2 → idx=1 → truck_idx=0, gha_idx=1
        self.env._apply_action("transporter", 2)
        self.env.demand.book_one_slot.assert_called_once_with("T1", GHA_IDS[1], "export")

    def test_book_action_last_valid_book_action(self):
        """Action = N_BOOK_ACTIONS → last booking action."""
        trucks = [make_truck(f"T{i}") for i in range(N_PENDING_TRUCKS)]
        self.env.demand.pending_trucks = trucks
        action = N_BOOK_ACTIONS
        self.env._apply_action("transporter", action)
        # Should call book_one_slot for the decoded indices
        self.env.demand.book_one_slot.assert_called_once()

    def test_book_action_boundary_between_book_and_dispatch(self):
        """N_BOOK_ACTIONS is book; N_BOOK_ACTIONS+1 is first dispatch."""
        trucks = [make_truck(f"T{i}") for i in range(N_PENDING_TRUCKS)]
        self.env.demand.pending_trucks = trucks
        # Dispatch action should call dispatch_truck, not book_one_slot
        self.env._apply_action("transporter", N_BOOK_ACTIONS + 1)
        self.env.demand.book_one_slot.assert_not_called()
        self.env.demand.dispatch_truck.assert_called_once()


# ---------------------------------------------------------------------------
class TestApplyActionTransporterDispatch(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_dispatch_action_calls_dispatch_truck(self):
        truck = make_truck("T1")
        self.env.demand.pending_trucks = [truck]
        # First dispatch action
        self.env._apply_action("transporter", N_BOOK_ACTIONS + 1)
        self.env.demand.dispatch_truck.assert_called_once_with("T1")

    def test_dispatch_action_out_of_range_truck_is_silently_ignored(self):
        self.env.demand.pending_trucks = []
        self.env._apply_action("transporter", N_BOOK_ACTIONS + 1)
        self.env.demand.dispatch_truck.assert_not_called()

    def test_dispatch_action_selects_correct_truck_by_index(self):
        trucks = [make_truck(f"T{i}") for i in range(5)]
        self.env.demand.pending_trucks = trucks
        # Third dispatch action → truck_idx=2
        self.env._apply_action("transporter", N_BOOK_ACTIONS + 3)
        self.env.demand.dispatch_truck.assert_called_once_with("T2")


# ---------------------------------------------------------------------------
class TestApplyActionGHA(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_gha_action_1_publishes_first_window(self):
        gha = GHA_IDS[0]
        windows = self.env._next_publishable_windows()
        self.env._apply_action(gha, 1)
        # Both import and export published for window[0]
        calls = self.env.dtp.publish_slot.call_args_list
        published_times = {c.args[1] for c in calls}
        self.assertIn(windows[0], published_times)

    def test_gha_action_2_publishes_second_window(self):
        gha = GHA_IDS[0]
        windows = self.env._next_publishable_windows()
        self.env._apply_action(gha, 2)
        calls = self.env.dtp.publish_slot.call_args_list
        published_times = {c.args[1] for c in calls}
        self.assertIn(windows[1], published_times)

    def test_gha_action_1_publishes_both_flow_types(self):
        gha = GHA_IDS[0]
        self.env._apply_action(gha, 1)
        calls = self.env.dtp.publish_slot.call_args_list
        flow_types = {c.args[2] for c in calls}
        self.assertEqual(flow_types, {"import", "export"})

    def test_gha_action_0_does_not_publish(self):
        self.env._apply_action(GHA_IDS[0], 0)
        self.env.dtp.publish_slot.assert_not_called()

    def test_gha_unknown_action_index_out_of_windows_is_silently_ignored(self):
        """Action 3 → idx=2, but only 2 windows exist → no publish."""
        self.env._apply_action(GHA_IDS[0], 3)
        self.env.dtp.publish_slot.assert_not_called()


# ---------------------------------------------------------------------------
class TestApplyActionOrchestrator(unittest.TestCase):
    """
    BUG-E: In the orchestrator branch, `flow_type` (a parameter of
    _apply_action) is passed as None when called from step(), yet it is
    forwarded to dtp.get_available_slots() and dtp.orch_book_slot().
    Additionally, if gha IS already in truck.booked_slots, the booking
    block is skipped — so the truck is released with no new assignment.
    """

    def setUp(self):
        self.env = make_env(with_orchestrator=True)

    def test_orchestrator_no_op_does_not_release_trucks(self):
        parked = [make_truck("T1")]
        self.env.tp3.get_parked_trucks.return_value = parked
        self.env._apply_action("orchestrator", 0)
        self.env.tp3.release.assert_not_called()

    def test_orchestrator_action_releases_correct_truck(self):
        parked = [make_truck("T1"), make_truck("T2")]
        self.env.tp3.get_parked_trucks.return_value = parked
        # Action = 0*N_GHAS + 0 + 1 = 1 → truck_idx=0, gha_idx=0
        self.env._apply_action("orchestrator", 1)
        self.env.tp3.release.assert_called_once_with("T1")

    def test_orchestrator_out_of_range_truck_index_is_silently_ignored(self):
        self.env.tp3.get_parked_trucks.return_value = []
        self.env._apply_action("orchestrator", 1)
        self.env.tp3.release.assert_not_called()

    def test_orchestrator_already_booked_gha_skips_booking(self):
        """If truck already has a booking for this GHA, no new slot is booked."""
        truck = make_truck("T1", booked_slots={GHA_IDS[0]: 30})
        self.env.tp3.get_parked_trucks.return_value = [truck]
        self.env._apply_action("orchestrator", 1)
        self.env.dtp.orch_book_slot.assert_not_called()


# ---------------------------------------------------------------------------
class TestGetObsTransporter(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_transporter_obs_shape_if_bug_were_fixed(self):
        """
        Documents the EXPECTED shape once BUG-D is fixed.
        We patch params to add the 'total' key so the rest of the obs
        logic can be exercised.
        """
        import env.schiphol_env as senv
        # Temporarily add 'total' to params
        for gha in GHA_IDS:
            senv.params["ghas"][gha]["total"] = (
                DEFAULT_PARAMS["ghas"][gha]["export"] +
                DEFAULT_PARAMS["ghas"][gha]["import"]
            )
        try:
            obs = self.env._get_obs("transporter")
            expected_dim = self.env._obs_dim("transporter")
            self.assertEqual(obs.shape, (expected_dim,))
        finally:
            for gha in GHA_IDS:
                senv.params["ghas"][gha].pop("total", None)

    def test_transporter_obs_dtype_float32_if_bug_fixed(self):
        import env.schiphol_env as senv
        for gha in GHA_IDS:
            senv.params["ghas"][gha]["total"] = 5
        try:
            obs = self.env._get_obs("transporter")
            self.assertEqual(obs.dtype, np.float32)
        finally:
            for gha in GHA_IDS:
                senv.params["ghas"][gha].pop("total", None)

    def test_transporter_obs_values_bounded_0_to_1_if_bug_fixed(self):
        import env.schiphol_env as senv
        for gha in GHA_IDS:
            senv.params["ghas"][gha]["total"] = 5
        try:
            obs = self.env._get_obs("transporter")
            self.assertTrue(np.all(obs >= 0.0), "obs contains values < 0")
            self.assertTrue(np.all(obs <= 1.0), "obs contains values > 1")
        finally:
            for gha in GHA_IDS:
                senv.params["ghas"][gha].pop("total", None)

    def test_transporter_obs_overflow_capped_at_1(self):
        """n_overflow() / 20 must be capped at 1.0."""
        import env.schiphol_env as senv
        for gha in GHA_IDS:
            senv.params["ghas"][gha]["total"] = 5
        self.env.tp3.n_overflow.return_value = 999  # way above 20
        try:
            obs = self.env._get_obs("transporter")
            # element index 1 is the overflow ratio
            self.assertLessEqual(obs[1], 1.0)
        finally:
            for gha in GHA_IDS:
                senv.params["ghas"][gha].pop("total", None)

    def test_transporter_obs_sin_cos_sum_of_squares_is_one(self):
        """sin²(x) + cos²(x) = 1; after (v+1)/2 the relationship changes
        but both must be in [0,1]."""
        import env.schiphol_env as senv
        for gha in GHA_IDS:
            senv.params["ghas"][gha]["total"] = 5
        try:
            obs = self.env._get_obs("transporter")
            # indices 2 and 3 are the time-of-day features
            self.assertGreaterEqual(float(obs[2]), 0.0)
            self.assertLessEqual(float(obs[2]),    1.0)
            self.assertGreaterEqual(float(obs[3]), 0.0)
            self.assertLessEqual(float(obs[3]),    1.0)
        finally:
            for gha in GHA_IDS:
                senv.params["ghas"][gha].pop("total", None)


# ---------------------------------------------------------------------------
class TestGetObsGHA(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_gha_obs_shape_correct(self):
        for gha in GHA_IDS:
            obs = self.env._get_obs(gha)
            self.assertEqual(obs.shape, (self.env._obs_dim(gha),),
                msg=f"Shape mismatch for {gha}")

    def test_gha_obs_dtype_float32(self):
        obs = self.env._get_obs(GHA_IDS[0])
        self.assertEqual(obs.dtype, np.float32)

    def test_gha_obs_all_values_in_0_1(self):
        for gha in GHA_IDS:
            obs = self.env._get_obs(gha)
            self.assertTrue(np.all(obs >= 0.0), f"{gha} obs has value < 0")
            self.assertTrue(np.all(obs <= 1.0), f"{gha} obs has value > 1")

    def test_gha_obs_tod_sin_cos_at_midnight(self):
        """At sim.now = 0 (midnight), tod=0 → sin=0, cos=1.
        After (v+1)/2: sin_feat=0.5, cos_feat=1.0."""
        self.env.sim.now = 0.0
        obs = self.env._get_obs(GHA_IDS[0])
        # indices 6 and 7 are sin and cos of tod
        self.assertAlmostEqual(float(obs[6]), 0.5, places=5)
        self.assertAlmostEqual(float(obs[7]), 1.0, places=5)

    def test_gha_obs_tod_sin_cos_at_noon(self):
        """At sim.now = 720 (noon), tod=0.5.
        sin(π) ≈ 0 → (0+1)/2 = 0.5; cos(π) = -1 → (-1+1)/2 = 0.0."""
        self.env.sim.now = 720.0
        obs = self.env._get_obs(GHA_IDS[0])
        self.assertAlmostEqual(float(obs[6]), 0.5, places=4)
        self.assertAlmostEqual(float(obs[7]), 0.0, places=4)

    def test_gha_obs_includes_other_gha_occupancies(self):
        """Cross-GHA context: obs must include exp+imp for every OTHER GHA."""
        # Set different mock values per GHA for distinction
        for i, gha in enumerate(GHA_IDS):
            self.env.terminals[gha].exp_occupancy.return_value = (i + 1) * 0.1
            self.env.terminals[gha].imp_occupancy.return_value = (i + 1) * 0.05
        obs = self.env._get_obs(GHA_IDS[0])
        # obs has 9 own features then 2*(N_GHAS-1) cross features
        cross = obs[9:]
        self.assertEqual(len(cross), 2 * (N_GHAS - 1))

    def test_gha_obs_own_exp_occupancy_at_index_0(self):
        self.env.terminals[GHA_IDS[0]].exp_occupancy.return_value = 0.77
        obs = self.env._get_obs(GHA_IDS[0])
        self.assertAlmostEqual(float(obs[0]), 0.77, places=4)

    def test_gha_obs_tp3_occupancy_at_index_8(self):
        self.env.tp3.occupancy_ratio.return_value = 0.42
        obs = self.env._get_obs(GHA_IDS[0])
        self.assertAlmostEqual(float(obs[8]), 0.42, places=4)


# ---------------------------------------------------------------------------
class TestGetObsOrchestrator(unittest.TestCase):

    def setUp(self):
        self.env = make_env(with_orchestrator=True)

    def test_orchestrator_obs_shape_correct(self):
        obs = self.env._get_obs("orchestrator")
        self.assertEqual(obs.shape, (self.env._obs_dim("orchestrator"),))

    def test_orchestrator_obs_dtype_float32(self):
        obs = self.env._get_obs("orchestrator")
        self.assertEqual(obs.dtype, np.float32)

    def test_orchestrator_obs_all_in_0_1(self):
        obs = self.env._get_obs("orchestrator")
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))

    def test_orchestrator_obs_tp3_occupancy_at_index_0(self):
        self.env.tp3.occupancy_ratio.return_value = 0.55
        obs = self.env._get_obs("orchestrator")
        self.assertAlmostEqual(float(obs[0]), 0.55, places=4)

    def test_orchestrator_obs_overflow_capped_at_1(self):
        self.env.tp3.n_overflow.return_value = 10000
        obs = self.env._get_obs("orchestrator")
        self.assertLessEqual(float(obs[1]), 1.0)

    def test_orchestrator_obs_contains_5_features_per_gha(self):
        """4 system features + 5 per GHA."""
        obs = self.env._get_obs("orchestrator")
        self.assertEqual(obs.shape[0], 4 + 5 * N_GHAS)


# ---------------------------------------------------------------------------
class TestGetReward(unittest.TestCase):

    def setUp(self):
        self.env = make_env(with_orchestrator=True)

    def test_orchestrator_reward_equals_global_reward(self):
        self.env.kpi.global_reward.return_value = -2.5
        r = self.env._get_reward("orchestrator", -2.5)
        self.assertAlmostEqual(r, -2.5)

    def test_transporter_reward_alpha_mix(self):
        alpha = self.env.alpha   # 0.3
        r_private = -4.0
        r_global  = -1.0
        self.env.kpi.transporter_reward.return_value = r_private
        r = self.env._get_reward("transporter", r_global)
        expected = (1 - alpha) * r_private + alpha * r_global
        self.assertAlmostEqual(r, expected)

    def test_gha_reward_alpha_mix(self):
        alpha = self.env.alpha
        r_private = -3.0
        r_global  = -2.0
        gha = GHA_IDS[0]
        self.env.kpi.gha_reward.return_value = r_private
        r = self.env._get_reward(gha, r_global)
        expected = (1 - alpha) * r_private + alpha * r_global
        self.assertAlmostEqual(r, expected)

    def test_transporter_private_weight_is_1_minus_alpha(self):
        """Private component weight must be (1 - alpha)."""
        self.env.kpi.transporter_reward.return_value = -1.0
        r_global = 0.0
        r = self.env._get_reward("transporter", r_global)
        self.assertAlmostEqual(r, -(1 - self.env.alpha))

    def test_global_weight_is_alpha_for_gha(self):
        self.env.kpi.gha_reward.return_value = 0.0
        r_global = -1.0
        r = self.env._get_reward(GHA_IDS[0], r_global)
        self.assertAlmostEqual(r, self.env.alpha * r_global)

    def test_alpha_zero_means_purely_private(self):
        self.env.alpha = 0.0
        r_private, r_global = -5.0, -100.0
        self.env.kpi.transporter_reward.return_value = r_private
        r = self.env._get_reward("transporter", r_global)
        self.assertAlmostEqual(r, r_private)

    def test_alpha_one_means_purely_global(self):
        self.env.alpha = 1.0
        r_private, r_global = -999.0, -2.0
        self.env.kpi.transporter_reward.return_value = r_private
        r = self.env._get_reward("transporter", r_global)
        self.assertAlmostEqual(r, r_global)

    def test_BUG_F_unknown_agent_raises_key_error_not_value_error(self):
        """
        BUG-F: _get_reward() for an unknown agent (not transporter /
        orchestrator / known GHA) falls to the else branch and calls
        self.terminals[agent], raising a KeyError instead of a clear
        ValueError as _obs_dim / _action_dim do.
        """
        with self.assertRaises(KeyError,
                msg="BUG-F confirmed: _get_reward raises KeyError (not "
                    "ValueError) for unknown agents."):
            self.env._get_reward("ghost_agent", 0.0)


# ---------------------------------------------------------------------------
class TestGetMaskNoOp(unittest.TestCase):

    def setUp(self):
        self.env = make_env(with_orchestrator=True)

    def test_action_0_always_masked_on_for_all_agents(self):
        for agent in self.env.agents:
            mask = self.env._get_mask(agent)
            self.assertEqual(mask[0], 1,
                msg=f"no_op must always be valid for {agent}")

    def test_mask_length_matches_action_dim_for_all_agents(self):
        for agent in self.env.agents:
            mask = self.env._get_mask(agent)
            self.assertEqual(len(mask), self.env._action_dim(agent),
                msg=f"mask length mismatch for {agent}")

    def test_mask_dtype_is_int8(self):
        for agent in self.env.agents:
            mask = self.env._get_mask(agent)
            self.assertEqual(mask.dtype, np.int8)

    def test_mask_contains_only_0_and_1(self):
        for agent in self.env.agents:
            mask = self.env._get_mask(agent)
            unique = set(mask.tolist())
            self.assertTrue(unique.issubset({0, 1}),
                msg=f"{agent} mask has values outside {{0,1}}: {unique}")


# ---------------------------------------------------------------------------
class TestGetMaskTransporterBook(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_no_pending_trucks_means_no_book_actions_enabled(self):
        self.env.demand.pending_trucks = []
        mask = self.env._get_mask("transporter")
        # Only no_op should be set
        self.assertEqual(mask.sum(), 1)

    def test_book_action_enabled_when_truck_needs_gha_and_slot_available(self):
        gha = GHA_IDS[0]
        truck = make_truck("T1", manifest=[{"gha": gha, "parcels": 5}])
        self.env.demand.pending_trucks = [truck]
        self.env.dtp.get_available_slots.return_value = [30]
        mask = self.env._get_mask("transporter")
        # Action = 0*N_GHAS + 0 + 1 = 1
        self.assertEqual(mask[1], 1)

    def test_book_action_disabled_when_already_booked(self):
        gha = GHA_IDS[0]
        truck = make_truck("T1",
                           manifest=[{"gha": gha, "parcels": 5}],
                           booked_slots={gha: 30})
        self.env.demand.pending_trucks = [truck]
        self.env.dtp.get_available_slots.return_value = [30]
        mask = self.env._get_mask("transporter")
        self.assertEqual(mask[1], 0,
            "Book action must be disabled when truck already has a slot for this GHA.")

    def test_book_action_disabled_when_no_slots_available(self):
        gha = GHA_IDS[0]
        truck = make_truck("T1", manifest=[{"gha": gha, "parcels": 5}])
        self.env.demand.pending_trucks = [truck]
        self.env.dtp.get_available_slots.return_value = []  # no slots
        mask = self.env._get_mask("transporter")
        self.assertEqual(mask[1], 0)

    def test_book_action_disabled_when_gha_not_in_stops_remaining(self):
        """Truck's stops_remaining doesn't include this GHA."""
        truck = make_truck("T1", manifest=[{"gha": GHA_IDS[1], "parcels": 5}])
        truck.stops_remaining = [{"gha": GHA_IDS[1], "parcels": 5}]
        self.env.demand.pending_trucks = [truck]
        self.env.dtp.get_available_slots.return_value = [30]
        mask = self.env._get_mask("transporter")
        # Action 1 = truck 0, GHA_IDS[0] — which truck does NOT need
        self.assertEqual(mask[1], 0)


# ---------------------------------------------------------------------------
class TestGetMaskTransporterDispatch(unittest.TestCase):
    """BUG-C and BUG-G: dispatch mask uses truck.manifest not stops_remaining."""

    def setUp(self):
        self.env = make_env()

    def test_dispatch_enabled_when_all_stops_booked(self):
        truck = make_truck("T1",
                           manifest=[{"gha": GHA_IDS[0], "parcels": 5}],
                           booked_slots={GHA_IDS[0]: 30})
        truck.stops_remaining = list(truck.manifest)
        self.env.demand.pending_trucks = [truck]
        mask = self.env._get_mask("transporter")
        dispatch_action = N_BOOK_ACTIONS + 1
        self.assertEqual(mask[dispatch_action], 1)

    def test_dispatch_disabled_when_missing_booking(self):
        truck = make_truck("T1",
                           manifest=[{"gha": GHA_IDS[0], "parcels": 5},
                                     {"gha": GHA_IDS[1], "parcels": 3}],
                           booked_slots={GHA_IDS[0]: 30})
        # GHA_IDS[1] not booked
        self.env.demand.pending_trucks = [truck]
        mask = self.env._get_mask("transporter")
        dispatch_action = N_BOOK_ACTIONS + 1
        self.assertEqual(mask[dispatch_action], 0)


# ---------------------------------------------------------------------------
class TestGetMaskGHA(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_gha_mask_length_is_3(self):
        for gha in GHA_IDS:
            mask = self.env._get_mask(gha)
            self.assertEqual(len(mask), 3)

    def test_gha_mask_action_1_enabled(self):
        """GHA always has a publishable next window, so action 1 must be valid."""
        for gha in GHA_IDS:
            mask = self.env._get_mask(gha)
            self.assertEqual(mask[1], 1)

    def test_gha_mask_action_2_enabled(self):
        for gha in GHA_IDS:
            mask = self.env._get_mask(gha)
            self.assertEqual(mask[2], 1)


# ---------------------------------------------------------------------------
class TestGetMaskOrchestrator(unittest.TestCase):

    def setUp(self):
        self.env = make_env(with_orchestrator=True)

    def test_mask_length_matches_action_dim(self):
        mask = self.env._get_mask("orchestrator")
        self.assertEqual(len(mask), self.env._action_dim("orchestrator"))

    def test_no_parked_trucks_only_no_op_enabled(self):
        self.env.tp3.get_parked_trucks.return_value = []
        mask = self.env._get_mask("orchestrator")
        self.assertEqual(mask.sum(), 1)  # only action 0

    def test_one_parked_truck_enables_n_ghas_actions(self):
        t1 = make_truck("T1")
        # 1. Clear Condition 1: Give the truck stops so it 'needs' the GHAs
        # (Using format [{"gha": "GHA_A"}, ...] to match your `s["gha"]` dict lookup)
        t1.stops_remaining = [{"gha": gha_id} for gha_id in GHA_IDS]
        t1.flow_type = "export"  # Ensure flow_type is set cleanly
        
        parked = [t1]
        self.env.tp3.get_parked_trucks.return_value = parked
        
        # 2. Clear Condition 2: Force the DTP mock to return a non-empty list (truthy)
        # so that bool(has_slots) evaluates to True
        self.env.dtp.get_available_slots.return_value = ["dummy_slot"]

        # 3. Run the mask calculation
        mask = self.env._get_mask("orchestrator")
        
        # Actions 1..N_GHAS should now be beautifully enabled
        self.assertEqual(mask[1:N_GHAS + 1].sum(), N_GHAS)

    def test_actions_beyond_dim_not_set(self):
        """No mask value must exceed the declared action_dim."""
        parked = [make_truck(f"T{i}") for i in range(50)]
        self.env.tp3.get_parked_trucks.return_value = parked
        mask = self.env._get_mask("orchestrator")
        self.assertEqual(len(mask), self.env._action_dim("orchestrator"))


# ---------------------------------------------------------------------------
class TestPrepopulateSlots(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_publish_slot_called_for_all_ghas(self):
        self.env._prepopulate_slots()
        calls = self.env.dtp.publish_slot.call_args_list
        published_ghas = {c.args[0] for c in calls}
        for gha in GHA_IDS:
            self.assertIn(gha, published_ghas)

    def test_publish_slot_called_for_both_flow_types(self):
        self.env._prepopulate_slots()
        calls = self.env.dtp.publish_slot.call_args_list
        flow_types = {c.args[2] for c in calls}
        self.assertEqual(flow_types, {"import", "export"})

    def test_publish_slot_called_at_least_once_per_dock_per_gha(self):
        """At minimum, each dock gets one slot in the lead_time window."""
        self.env._prepopulate_slots()
        total_calls = self.env.dtp.publish_slot.call_count
        # At least one slot per flow_type per GHA
        self.assertGreater(total_calls, 2 * N_GHAS)

    def test_prepopulate_respects_freeze_time(self):
        """No slot should be published before now + freeze_time."""
        freeze = DEFAULT_PARAMS["dtp_rules"]["freeze_time"]
        self.env._prepopulate_slots()
        calls = self.env.dtp.publish_slot.call_args_list
        for c in calls:
            slot_time = c.args[1]
            self.assertGreaterEqual(slot_time, self.env.sim.now + freeze,
                msg=f"Slot at t={slot_time} violates freeze window "
                    f"(freeze={freeze}, now={self.env.sim.now})")

    def test_prepopulate_slots_within_lead_time(self):
        """No slot should be published beyond now + lead_time."""
        lead = DEFAULT_PARAMS["dtp_rules"]["lead_time"]
        self.env._prepopulate_slots()
        calls = self.env.dtp.publish_slot.call_args_list
        for c in calls:
            slot_time = c.args[1]
            self.assertLessEqual(slot_time, self.env.sim.now + lead,
                msg=f"Slot at t={slot_time} exceeds lead_time window.")

    def test_prepopulate_slots_uniformly_spaced(self):
        """Consecutive slot times for one GHA/flow should differ by slot_duration."""
        slot_dur = DEFAULT_PARAMS["dtp_rules"]["slot_duration"]
        self.env._prepopulate_slots()
        calls = self.env.dtp.publish_slot.call_args_list
        gha0_export = sorted(list(set(
            c.args[1] for c in calls
            if c.args[0] == GHA_IDS[0] and c.args[2] == "export"
        )))
        for t0, t1 in zip(gha0_export, gha0_export[1:]):
            self.assertAlmostEqual(t1 - t0, slot_dur,
                msg=f"Slot gap {t1-t0} ≠ slot_duration {slot_dur}")


# ---------------------------------------------------------------------------
class TestNextPublishableWindows(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_returns_two_windows(self):
        windows = self.env._next_publishable_windows()
        self.assertEqual(len(windows), 2)

    def test_second_window_is_one_slot_after_first(self):
        slot_dur = DEFAULT_PARAMS["dtp_rules"]["slot_duration"]
        w = self.env._next_publishable_windows()
        self.assertEqual(w[1] - w[0], slot_dur)

    def test_first_window_respects_freeze_time(self):
        freeze = DEFAULT_PARAMS["dtp_rules"]["freeze_time"]
        self.env.sim.now = 0.0
        w = self.env._next_publishable_windows()
        self.assertGreaterEqual(w[0], freeze)

    def test_first_window_is_aligned_to_slot_boundary(self):
        slot_dur = DEFAULT_PARAMS["dtp_rules"]["slot_duration"]
        self.env.sim.now = 0.0
        w = self.env._next_publishable_windows()
        self.assertEqual(w[0] % slot_dur, 0,
            f"First window {w[0]} is not aligned to slot_duration={slot_dur}")

    def test_windows_advance_as_sim_time_advances(self):
        w0 = self.env._next_publishable_windows()
        self.env.sim.now = 1000.0
        w1 = self.env._next_publishable_windows()
        self.assertGreater(w1[0], w0[0])

    def test_window_alignment_when_now_is_on_slot_boundary(self):
        slot_dur = DEFAULT_PARAMS["dtp_rules"]["slot_duration"]
        freeze   = DEFAULT_PARAMS["dtp_rules"]["freeze_time"]
        # Set now so that now+freeze lands exactly on a boundary
        self.env.sim.now = slot_dur - freeze
        w = self.env._next_publishable_windows()
        self.assertEqual(w[0] % slot_dur, 0)

    def test_windows_are_integers(self):
        w = self.env._next_publishable_windows()
        for t in w:
            self.assertEqual(t, int(t),
                msg=f"Window time {t} is not an integer — potential float drift.")


# ---------------------------------------------------------------------------
class TestFindUnbookingTruck(unittest.TestCase):

    def setUp(self):
        self.env = make_env()

    def test_returns_none_when_no_trucks_parked(self):
        """Orchestrator mask should only have the no-op action enabled if no trucks are parked."""
        self.env.tp3.get_parked_trucks.return_value = []
        
        mask = self.env._get_mask("orchestrator")
        
        # Action 0 (no_op) is 1, all dispatch actions (1 to N_GHAS) must be 0
        self.assertEqual(mask[1 : N_GHAS + 1].sum(), 0, 
                         "No dispatch actions should be enabled when parking lot is empty.")

    def test_returns_truck_that_needs_gha_and_has_no_booking(self):
        """Orchestrator should enable the dispatch action if a parked truck needs that GHA."""
        gha_idx = 0
        gha = GHA_IDS[gha_idx]
        
        truck = make_truck("T1", manifest=[{"gha": gha, "parcels": 5}])
        truck.stops_remaining = [{"gha": gha}]
        truck.flow_type = "export"
        
        self.env.tp3.get_parked_trucks.return_value = [truck]
        self.env.dtp.get_available_slots.return_value = ["dummy_slot"]
        
        mask = self.env._get_mask("orchestrator")
        
        # Calculate specific index: Truck 0 -> GHA 0 -> index = 0 * N_GHAS + 0 + 1 = 1
        action_idx = 0 * N_GHAS + gha_idx + 1
        self.assertEqual(mask[action_idx], 1, "Action should be enabled for eligible truck.")

    def test_skips_truck_that_already_has_booking_for_gha(self):
        """
        NOTE: Based on your current orchestrator production logic, booking checks do NOT mask out actions 
        (unlike the transporter). This test verifies that if slots exist, the action remains valid.
        """
        gha_idx = 0
        gha = GHA_IDS[gha_idx]
        
        truck = make_truck("T1", manifest=[{"gha": gha, "parcels": 5}], booked_slots={gha: 30})
        truck.stops_remaining = [{"gha": gha}]
        truck.flow_type = "export"
        
        self.env.tp3.get_parked_trucks.return_value = [truck]
        self.env.dtp.get_available_slots.return_value = ["dummy_slot"]
        
        mask = self.env._get_mask("orchestrator")
        
        action_idx = 0 * N_GHAS + gha_idx + 1
        self.assertEqual(mask[action_idx], 1, 
                         "Orchestrator keeps dispatch valid even if slot is booked, as long as slots exist.")

    def test_skips_truck_that_does_not_need_gha(self):
        """Orchestrator should keep an action masked out (0) if the truck doesn't need that GHA."""
        gha_needed = GHA_IDS[1]
        
        truck = make_truck("T1", manifest=[{"gha": gha_needed, "parcels": 5}])
        truck.stops_remaining = [{"gha": gha_needed}]
        truck.flow_type = "export"
        
        self.env.tp3.get_parked_trucks.return_value = [truck]
        self.env.dtp.get_available_slots.return_value = ["dummy_slot"]
        
        mask = self.env._get_mask("orchestrator")
        
        # Check action for GHA_IDS[0] (gha_idx = 0). The truck only needs GHA_IDS[1].
        target_action_idx = 0 * N_GHAS + 0 + 1
        self.assertEqual(mask[target_action_idx], 0, "Action must be masked for unneeded GHA.")

    def test_returns_first_eligible_truck_among_many(self):
        """Orchestrator creates a combined flattened array mapping every truck to every GHA sequence."""
        t1 = make_truck("T1", manifest=[{"gha": GHA_IDS[1], "parcels": 3}])
        t1.stops_remaining = [{"gha": GHA_IDS[1]}]
        t1.flow_type = "export"
        
        t2 = make_truck("T2", manifest=[{"gha": GHA_IDS[0], "parcels": 5}])
        t2.stops_remaining = [{"gha": GHA_IDS[0]}]
        t2.flow_type = "export"
        
        t3 = make_truck("T3", manifest=[{"gha": GHA_IDS[0], "parcels": 4}])
        t3.stops_remaining = [{"gha": GHA_IDS[0]}]
        t3.flow_type = "export"
        
        self.env.tp3.get_parked_trucks.return_value = [t1, t2, t3]
        self.env.dtp.get_available_slots.return_value = ["dummy_slot"]
        
        mask = self.env._get_mask("orchestrator")
        
        # Let's map out expectations for GHA 0 (gha_idx = 0):
        # Truck 0 (T1): Doesn't need GHA 0 -> 0 * N_GHAS + 0 + 1 = Index 1 should be 0
        # Truck 1 (T2): Needs GHA 0       -> 1 * N_GHAS + 0 + 1 = Index (N_GHAS + 1) should be 1
        # Truck 2 (T3): Needs GHA 0       -> 2 * N_GHAS + 0 + 1 = Index (2 * N_GHAS + 1) should be 1
        
        t1_action_idx = 0 * N_GHAS + 0 + 1
        t2_action_idx = 1 * N_GHAS + 0 + 1
        t3_action_idx = 2 * N_GHAS + 0 + 1
        
        self.assertEqual(mask[t1_action_idx], 0, "Truck 1 action for GHA 0 should be disabled.")
        self.assertEqual(mask[t2_action_idx], 1, "Truck 2 action for GHA 0 should be enabled.")
        self.assertEqual(mask[t3_action_idx], 1, "Truck 3 action for GHA 0 should be enabled.")

    def test_truck_with_completed_stop_skips_orchestrator_mask(self):
        """
        Orchestrator masking checks truck.stops_remaining, NOT manifest.
        A truck that has completed GHA_A (removed from stops_remaining)
        should NOT have its dispatch action enabled for GHA_A, even if the 
        manifest still lists it.
        """
        gha_idx = 0
        gha = GHA_IDS[gha_idx]
        
        # 1. Setup truck: It has GHA_A in its historical manifest...
        truck = make_truck("T1", manifest=[{"gha": gha, "parcels": 5}])
        # ...but it has already finished it (stops_remaining is empty)
        truck.stops_remaining = [] 
        truck.flow_type = "export"
        
        self.env.tp3.get_parked_trucks.return_value = [truck]
        self.env.dtp.get_available_slots.return_value = ["dummy_slot"]
        
        # 2. Grab the mask
        mask = self.env._get_mask("orchestrator")
        
        # 3. Calculate the specific action index for: Truck 0 -> GHA_A (gha_idx)
        # Action formula from your code: t_idx * N_GHAS + g_idx + 1
        action_idx = 0 * N_GHAS + gha_idx + 1
        
        # 4. Assert that the action is MASKED (0) because stops_remaining is empty
        self.assertEqual(
            mask[action_idx], 0,
            "Orchestrator mask must check stops_remaining, not manifest. Action should be disabled."
        )


# ---------------------------------------------------------------------------
class TestStressAndEdge(unittest.TestCase):
    """
    All step() calls here use make_env_patched_apply() — BUG-A (_apply_action
    wrong arity) is already asserted in TestApplyActionSignature; patching it
    here lets us stress the rest of the loop.
    """

    def setUp(self):
        self.env = make_env_patched_apply()

    def test_100_consecutive_no_op_steps_do_not_crash(self):
        """With BUG-A patched out, 100 no-op steps must not raise."""
        for i in range(100):
            try:
                self.env.step(no_op_actions(self.env))
            except Exception as e:
                self.fail(f"Step {i} raised: {e}")

    def test_sim_time_monotonically_increases_over_steps(self):
        times = []
        for _ in range(20):
            self.env.step(no_op_actions(self.env))
            times.append(self.env.sim.now)
        for i in range(1, len(times)):
            self.assertGreater(times[i], times[i - 1])

    def test_obs_shape_stable_across_many_steps(self):
        for _ in range(50):
            obs, *_ = self.env.step(no_op_actions(self.env))
            for agent, arr in obs.items():
                self.assertEqual(arr.shape, (self.env._obs_dim(agent),),
                    msg=f"{agent} obs shape changed mid-episode")

    def test_mask_shape_stable_across_many_steps(self):
        for _ in range(50):
            *_, infos = self.env.step(no_op_actions(self.env))
            for agent, info in infos.items():
                mask = info["action_mask"]
                self.assertEqual(len(mask), self.env._action_dim(agent))

    def test_all_agents_receive_reward_every_step(self):
        for _ in range(10):
            _, rewards, *_ = self.env.step(no_op_actions(self.env))
            for agent in self.env.agents:
                self.assertIn(agent, rewards)

    def test_reward_is_finite_for_all_agents(self):
        for _ in range(10):
            _, rewards, *_ = self.env.step(no_op_actions(self.env))
            for agent, r in rewards.items():
                self.assertTrue(math.isfinite(r),
                    msg=f"Reward for {agent} is not finite: {r}")

    def test_infra_flush_called_once_per_step(self):
        for i in range(5):
            self.env.step(no_op_actions(self.env))
        self.assertEqual(self.env.infra.flush_step_buffer.call_count, 5)

    def test_with_orchestrator_full_step_cycle(self):
        env = make_env_patched_apply(with_orchestrator=True)
        for _ in range(10):
            try:
                env.step(no_op_actions(env))
            except Exception as e:
                self.fail(f"Orchestrator no_op step raised: {e}")

    def test_obs_values_within_bounds_after_many_steps(self):
        for _ in range(20):
            obs, *_ = self.env.step(no_op_actions(self.env))
            for agent, arr in obs.items():
                if agent == "transporter":
                    continue   # skip: BUG-D means transporter obs isn't built here
                self.assertTrue(np.all(arr >= -1e-6),
                    msg=f"{agent} obs has value < 0")
                self.assertTrue(np.all(arr <=  1.0 + 1e-6),
                    msg=f"{agent} obs has value > 1")

    def test_kpi_global_reward_called_once_per_step(self):
        n = 7
        for _ in range(n):
            self.env.step(no_op_actions(self.env))
        self.assertEqual(self.env.kpi.global_reward.call_count, n)


# =============================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)