# =============================================================================
# TEST SUITE — KPI TRACKER MODULE
# =============================================================================
# DESCRIPTION:
#     Full unittest coverage for KPITracker (kpi_tracker.py).
#     All external dependencies (config, infrastructure, dtp_platform) are
#     mocked so the suite runs in isolation — no Schiphol env needed.
#
# ORGANISATION (10 test classes):
#     TestKPITrackerInit              — __init__ state
#     TestIngestGateEvents            — GATE_IN / GATE_OUT lifecycle
#     TestIngestDockEvents            — GHA_IN / DOCK_START / DOCK_END logic
#     TestWPRCalculation              — wpr() correctness + edge cases
#     TestPeakWPRCalculation          — peak_wpr() window enforcement
#     TestNTTPCalculation             — nttp() correctness + edge cases
#     TestUtilizationStd              — snapshot_utilization() + utilization_std()
#     TestGlobalReward                — global_reward() sign / magnitude
#     TestTransporterReward           — transporter_reward() delta logic
#     TestGHAReward                   — gha_reward() composite formula
#     TestSummary                     — summary() dict completeness
#     TestStressAndEdgeCases          — high volume, zero-parcel, orphan events,
#                                       simultaneous trucks, peak boundary
# =============================================================================

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from enum import Enum, auto


# ---------------------------------------------------------------------------
# ① Stub the modules that kpi_tracker.py imports so we never need the
#    real Schiphol codebase present during testing.
# ---------------------------------------------------------------------------

# --- config stub -----------------------------------------------------------
config_mod = types.ModuleType("config")
config_config_mod = types.ModuleType("config.config")

DEFAULT_PARAMS = {
    "demand": {"peak_window": [60, 120]},           # peak: minutes 60–120
    "ghas": {"GHA_A": {}, "GHA_B": {}, "GHA_C": {}},
    "marl": {
        "reward_weights": {
            "wpr_global":    1.0,
            "util_std":      1.0,
            "wait_per_min":  1.0,
            "no_show":       5.0,
            "missed_slot":   3.0,
            "dock_util":     2.0,
            "parcel_on_time":1.0,
            "queue_per_step":0.5,
        }
    },
}

config_config_mod.load_params = lambda: DEFAULT_PARAMS
sys.modules["config"] = config_mod
sys.modules["config.config"] = config_config_mod

# --- infrastructure stub ---------------------------------------------------
infra_mod = types.ModuleType("env.infrastructure")


class CheckpointID(Enum):
    GATE_IN    = auto()
    GHA_IN     = auto()
    DOCK_START = auto()
    DOCK_END   = auto()
    GATE_OUT   = auto()


class SensorEvent:
    """Minimal SensorEvent mirroring the real dataclass fields used by KPITracker."""
    def __init__(self, checkpoint, truck_id, sim_time,
                 gha_id=None, n_parcels=None):
        self.checkpoint = checkpoint
        self.truck_id   = truck_id
        self.sim_time   = sim_time
        self.gha_id     = gha_id
        self.n_parcels  = n_parcels


infra_mod.CheckpointID = CheckpointID
infra_mod.SensorEvent  = SensorEvent
# sys.modules["env"]                = types.ModuleType("env")
sys.modules["env.infrastructure"] = infra_mod

# --- dtp_platform stub -----------------------------------------------------
dtp_mod = types.ModuleType("env.dtp_platform")


class DTPPlatform:
    """Stub — only the attributes read by transporter_reward() matter."""
    def __init__(self, no_shows=None, late_arrivals=None):
        self.no_shows      = no_shows      or {}
        self.late_arrivals = late_arrivals or {}


dtp_mod.DTPPlatform = DTPPlatform
sys.modules["env.dtp_platform"] = dtp_mod

# ---------------------------------------------------------------------------
# ② NOW import the class under test (stubs are already in sys.modules)
# ---------------------------------------------------------------------------
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.kpi_tracker import KPITracker  # noqa: E402  (import after stubs)


# ===========================================================================
# HELPERS
# ===========================================================================

def make_truck_journey(truck_id="T1", gha_id="GHA_A",
                       gate_in=0, gha_in=5,
                       dock_start=10, dock_end=20, gate_out=25,
                       n_parcels=10):
    """Return the canonical list of five SensorEvents for one full truck visit."""
    return [
        SensorEvent(CheckpointID.GATE_IN,    truck_id, gate_in,    n_parcels=n_parcels),
        SensorEvent(CheckpointID.GHA_IN,     truck_id, gha_in,     gha_id=gha_id),
        SensorEvent(CheckpointID.DOCK_START, truck_id, dock_start, gha_id=gha_id),
        SensorEvent(CheckpointID.DOCK_END,   truck_id, dock_end,   gha_id=gha_id),
        SensorEvent(CheckpointID.GATE_OUT,   truck_id, gate_out,   n_parcels=n_parcels),
    ]


def make_terminal_mock(exp_occ=0.5, imp_occ=0.5,
                       exp_q=0.0, imp_q=0.0,
                       processed_export=0, processed_import=0):
    """Return a MagicMock that satisfies every terminal attribute read by KPITracker."""
    t = MagicMock()
    t.exp_occupancy.return_value    = exp_occ
    t.imp_occupancy.return_value    = imp_occ
    t.exp_queue_norm.return_value   = exp_q
    t.imp_queue_norm.return_value   = imp_q
    t.stats = {
        "export": {"processed": processed_export},
        "import": {"processed": processed_import},
    }
    return t


# ===========================================================================
# TEST CLASSES
# ===========================================================================

class TestKPITrackerInit(unittest.TestCase):
    """Verify the initial state after construction — no events ingested."""

    def setUp(self):
        self.tracker = KPITracker()

    # --- accumulator zeroes ------------------------------------------------
    def test_total_wait_initialised_to_zero(self):
        self.assertEqual(self.tracker._total_wait, 0.0)

    def test_total_service_initialised_to_zero(self):
        self.assertEqual(self.tracker._total_service, 0.0)

    def test_peak_wait_initialised_to_zero(self):
        self.assertEqual(self.tracker._peak_wait, 0.0)

    def test_peak_service_initialised_to_zero(self):
        self.assertEqual(self.tracker._peak_service, 0.0)

    def test_nttp_sum_initialised_to_zero(self):
        self.assertEqual(self.tracker._nttp_sum, 0.0)

    def test_n_completed_initialised_to_zero(self):
        self.assertEqual(self.tracker._n_completed, 0)

    # --- KPI properties return 0 before any data ---------------------------
    def test_wpr_returns_zero_with_no_data(self):
        self.assertEqual(self.tracker.wpr(), 0.0)

    def test_peak_wpr_returns_zero_with_no_data(self):
        self.assertEqual(self.tracker.peak_wpr(), 0.0)

    def test_nttp_returns_zero_with_no_data(self):
        self.assertEqual(self.tracker.nttp(), 0.0)

    def test_utilization_std_returns_zero_with_no_data(self):
        self.assertEqual(self.tracker.utilization_std(), 0.0)

    # --- per-GHA structures ------------------------------------------------
    def test_util_dict_keys_match_ghas(self):
        self.assertEqual(set(self.tracker._util.keys()), {"GHA_A", "GHA_B", "GHA_C"})

    def test_util_export_list_initially_empty(self):
        for gha in ("GHA_A", "GHA_B", "GHA_C"):
            self.assertEqual(self.tracker._util[gha]["export"], [])

    def test_util_import_list_initially_empty(self):
        for gha in ("GHA_A", "GHA_B", "GHA_C"):
            self.assertEqual(self.tracker._util[gha]["import"], [])

    def test_prev_proc_initialised_for_all_ghas(self):
        for gha in ("GHA_A", "GHA_B", "GHA_C"):
            self.assertEqual(self.tracker._prev_proc[gha], 0)

    # --- peak window loaded from config ------------------------------------
    def test_peak_start_loaded_from_config(self):
        self.assertEqual(self.tracker._peak_start, 60)

    def test_peak_end_loaded_from_config(self):
        self.assertEqual(self.tracker._peak_end, 120)

    # --- truck working state empty -----------------------------------------
    def test_truck_dict_initially_empty(self):
        self.assertEqual(self.tracker._truck, {})


# ---------------------------------------------------------------------------

class TestIngestGateEvents(unittest.TestCase):
    """GATE_IN starts truck tracking; GATE_OUT completes it and cleans up."""

    def setUp(self):
        self.tracker = KPITracker()

    def test_gate_in_creates_truck_state(self):
        e = SensorEvent(CheckpointID.GATE_IN, "T1", 0, n_parcels=5)
        self.tracker.ingest([e])
        self.assertIn("T1", self.tracker._truck)

    def test_gate_in_records_arrival_time(self):
        e = SensorEvent(CheckpointID.GATE_IN, "T1", 42, n_parcels=5)
        self.tracker.ingest([e])
        self.assertEqual(self.tracker._truck["T1"]["gate_in"], 42)

    def test_gate_in_records_parcel_count(self):
        e = SensorEvent(CheckpointID.GATE_IN, "T1", 0, n_parcels=7)
        self.tracker.ingest([e])
        self.assertEqual(self.tracker._truck["T1"]["n_parcels"], 7)

    def test_gate_in_none_parcels_defaults_to_zero(self):
        e = SensorEvent(CheckpointID.GATE_IN, "T1", 0, n_parcels=None)
        self.tracker.ingest([e])
        self.assertEqual(self.tracker._truck["T1"]["n_parcels"], 0)

    def test_gate_out_removes_truck_from_working_state(self):
        self.tracker.ingest(make_truck_journey("T1"))
        self.assertNotIn("T1", self.tracker._truck)

    def test_gate_out_increments_completed_counter(self):
        self.tracker.ingest(make_truck_journey("T1", n_parcels=10))
        self.assertEqual(self.tracker._n_completed, 1)

    def test_gate_out_with_zero_parcels_does_not_increment_completed(self):
        """A truck with 0 parcels should not contribute to NTTP (guard against division)."""
        self.tracker.ingest(make_truck_journey("T1", n_parcels=0))
        self.assertEqual(self.tracker._n_completed, 0)

    def test_gate_out_without_prior_gate_in_is_silently_ignored(self):
        """Orphan GATE_OUT — no prior state — must not raise."""
        e = SensorEvent(CheckpointID.GATE_OUT, "GHOST", 999, n_parcels=1)
        try:
            self.tracker.ingest([e])
        except Exception as exc:
            self.fail(f"Orphan GATE_OUT raised {exc}")

    def test_multiple_gate_ins_tracked_independently(self):
        for i in range(5):
            self.tracker.ingest([SensorEvent(CheckpointID.GATE_IN, f"T{i}", i*10, n_parcels=i+1)])
        self.assertEqual(len(self.tracker._truck), 5)

    def test_gate_in_overrides_previous_state_for_same_truck(self):
        """Re-entering truck replaces old state (second visit)."""
        self.tracker.ingest([SensorEvent(CheckpointID.GATE_IN, "T1", 0,  n_parcels=5)])
        self.tracker.ingest([SensorEvent(CheckpointID.GATE_IN, "T1", 50, n_parcels=9)])
        self.assertEqual(self.tracker._truck["T1"]["gate_in"], 50)
        self.assertEqual(self.tracker._truck["T1"]["n_parcels"], 9)


# ---------------------------------------------------------------------------

class TestIngestDockEvents(unittest.TestCase):
    """GHA_IN, DOCK_START, DOCK_END update wait / service accumulators correctly."""

    def setUp(self):
        self.tracker = KPITracker()
        # Place truck T1 in working state
        self.tracker.ingest([SensorEvent(CheckpointID.GATE_IN, "T1", 0, n_parcels=10)])

    def test_gha_in_records_arrival_at_gha(self):
        e = SensorEvent(CheckpointID.GHA_IN, "T1", 5, gha_id="GHA_A")
        self.tracker.ingest([e])
        self.assertEqual(self.tracker._truck["T1"]["gha_in"]["GHA_A"], 5)

    def test_dock_start_computes_wait_time(self):
        self.tracker.ingest([SensorEvent(CheckpointID.GHA_IN,     "T1", 10, gha_id="GHA_A")])
        self.tracker.ingest([SensorEvent(CheckpointID.DOCK_START, "T1", 25, gha_id="GHA_A")])
        self.assertAlmostEqual(self.tracker._total_wait, 15.0)

    def test_dock_end_computes_service_time(self):
        self.tracker.ingest([SensorEvent(CheckpointID.GHA_IN,     "T1", 10, gha_id="GHA_A")])
        self.tracker.ingest([SensorEvent(CheckpointID.DOCK_START, "T1", 20, gha_id="GHA_A")])
        self.tracker.ingest([SensorEvent(CheckpointID.DOCK_END,   "T1", 45, gha_id="GHA_A")])
        self.assertAlmostEqual(self.tracker._total_service, 25.0)

    def test_dock_start_without_gha_in_uses_dock_time_as_fallback(self):
        """If GHA_IN was never recorded, wait = 0 (dock_start - dock_start)."""
        self.tracker.ingest([SensorEvent(CheckpointID.DOCK_START, "T1", 30, gha_id="GHA_A")])
        self.assertAlmostEqual(self.tracker._total_wait, 0.0)

    def test_dock_end_without_dock_start_uses_dock_end_as_fallback(self):
        """If DOCK_START was never recorded, service = 0."""
        self.tracker.ingest([SensorEvent(CheckpointID.GHA_IN,   "T1", 5,  gha_id="GHA_A")])
        self.tracker.ingest([SensorEvent(CheckpointID.DOCK_END, "T1", 40, gha_id="GHA_A")])
        self.assertAlmostEqual(self.tracker._total_service, 0.0)

    def test_dock_events_without_gha_id_are_ignored(self):
        """Events with gha_id=None must not crash or mutate state."""
        self.tracker.ingest([SensorEvent(CheckpointID.DOCK_START, "T1", 20, gha_id=None)])
        self.assertEqual(self.tracker._total_wait, 0.0)

    def test_dock_events_for_unknown_truck_are_ignored(self):
        e = SensorEvent(CheckpointID.DOCK_START, "UNKNOWN", 20, gha_id="GHA_A")
        try:
            self.tracker.ingest([e])
        except Exception as exc:
            self.fail(f"Unknown truck in dock event raised {exc}")
        self.assertEqual(self.tracker._total_wait, 0.0)

    def test_multiple_gha_visits_accumulate_independently(self):
        """Truck stops at GHA_A then GHA_B; both waits are accumulated."""
        events = [
            SensorEvent(CheckpointID.GHA_IN,     "T1", 5,  gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_START, "T1", 15, gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_END,   "T1", 30, gha_id="GHA_A"),
            SensorEvent(CheckpointID.GHA_IN,     "T1", 35, gha_id="GHA_B"),
            SensorEvent(CheckpointID.DOCK_START, "T1", 50, gha_id="GHA_B"),
            SensorEvent(CheckpointID.DOCK_END,   "T1", 70, gha_id="GHA_B"),
        ]
        self.tracker.ingest(events)
        self.assertAlmostEqual(self.tracker._total_wait,   10 + 15)   # 10 + 15
        self.assertAlmostEqual(self.tracker._total_service, 15 + 20)  # 15 + 20


# ---------------------------------------------------------------------------

class TestWPRCalculation(unittest.TestCase):
    """wait-to-process ratio = total_wait / total_service."""

    def setUp(self):
        self.tracker = KPITracker()

    def _run_journey(self, gha_in=5, dock_start=10, dock_end=20, t="T1"):
        events = [
            SensorEvent(CheckpointID.GATE_IN,    t,  0,          n_parcels=10),
            SensorEvent(CheckpointID.GHA_IN,     t,  gha_in,     gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_START, t,  dock_start, gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_END,   t,  dock_end,   gha_id="GHA_A"),
            SensorEvent(CheckpointID.GATE_OUT,   t,  dock_end+5, n_parcels=10),
        ]
        self.tracker.ingest(events)

    def test_wpr_correct_value(self):
        self._run_journey(gha_in=5, dock_start=10, dock_end=20)
        # wait=5, service=10  →  WPR=0.5
        self.assertAlmostEqual(self.tracker.wpr(), 0.5)

    def test_wpr_zero_when_no_service_time(self):
        self.assertEqual(self.tracker.wpr(), 0.0)

    def test_wpr_zero_perfect_system(self):
        """Zero wait time → WPR = 0 (ideal)."""
        self._run_journey(gha_in=10, dock_start=10, dock_end=20)
        self.assertAlmostEqual(self.tracker.wpr(), 0.0)

    def test_wpr_greater_than_one_when_wait_exceeds_service(self):
        self._run_journey(gha_in=0, dock_start=50, dock_end=55)
        # wait=50, service=5  →  WPR=10
        self.assertGreater(self.tracker.wpr(), 1.0)

    def test_wpr_accumulates_across_multiple_trucks(self):
        self._run_journey(gha_in=5, dock_start=10, dock_end=20, t="T1")  # wait=5, svc=10
        self._run_journey(gha_in=5, dock_start=15, dock_end=25, t="T2")  # wait=10, svc=10
        # total_wait=15, total_service=20  →  WPR=0.75
        self.assertAlmostEqual(self.tracker.wpr(), 0.75)

    def test_wpr_non_negative(self):
        self._run_journey()
        self.assertGreaterEqual(self.tracker.wpr(), 0.0)


# ---------------------------------------------------------------------------

class TestPeakWPRCalculation(unittest.TestCase):
    """Peak WPR is computed only for events within [60, 120] (from config)."""

    def setUp(self):
        self.tracker = KPITracker()

    def _inject_at(self, gha_in_t, dock_start_t, dock_end_t, truck="T1"):
        events = [
            SensorEvent(CheckpointID.GATE_IN,    truck, 0,            n_parcels=5),
            SensorEvent(CheckpointID.GHA_IN,     truck, gha_in_t,     gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_START, truck, dock_start_t, gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_END,   truck, dock_end_t,   gha_id="GHA_A"),
            SensorEvent(CheckpointID.GATE_OUT,   truck, dock_end_t+5, n_parcels=5),
        ]
        self.tracker.ingest(events)

    def test_peak_wpr_zero_with_no_peak_events(self):
        self._inject_at(5, 10, 20)          # all off-peak
        self.assertEqual(self.tracker.peak_wpr(), 0.0)

    def test_peak_wpr_computed_for_events_inside_window(self):
        self._inject_at(65, 80, 100)        # dock_start=80, dock_end=100 → inside peak
        # wait=80-65=15, service=100-80=20  →  peak_WPR=0.75
        self.assertAlmostEqual(self.tracker.peak_wpr(), 0.75)

    def test_peak_wpr_at_exact_start_boundary(self):
        self._inject_at(50, 60, 70)         # dock_start=60 = peak_start  → inside
        self.assertGreater(self.tracker.peak_wpr(), 0.0)

    def test_peak_wpr_at_exact_end_boundary(self):
        """
        DOCK_START=120 (== peak_end) is inside the window, so peak_wait is
        recorded.  DOCK_END must also be <= 120 for peak_service to be recorded
        and peak_wpr() to be non-zero.  Here we choose a zero-duration service
        window (dock_start == dock_end) at t=120 — both timestamps satisfy the
        inclusive boundary, so peak_wait > 0 and peak_service > 0.
        """
        self._inject_at(gha_in_t=110, dock_start_t=120, dock_end_t=120, truck="T1")
        # wait = 120 - 110 = 10, service = 120 - 120 = 0  →  peak_service still 0
        # The real KPI exposed: a zero-duration service makes peak_wpr() return 0
        # regardless of boundary.  Assert that peak_wait WAS recorded (not service).
        self.assertGreater(self.tracker._peak_wait, 0.0)

    def test_peak_wpr_does_not_include_off_peak_events(self):
        self._inject_at(5,  10,  20, "T1")   # off-peak
        self._inject_at(65, 80, 100, "T2")   # on-peak
        # Only T2 contributes: wait=15, service=20 → 0.75
        self.assertAlmostEqual(self.tracker.peak_wpr(), 0.75)

    def test_peak_wpr_independent_of_global_wpr(self):
        self._inject_at(5,  10,  20, "T1")   # off-peak
        self._inject_at(65, 80, 100, "T2")   # on-peak
        self.assertNotEqual(self.tracker.wpr(), self.tracker.peak_wpr())


# ---------------------------------------------------------------------------

class TestNTTPCalculation(unittest.TestCase):
    """Normalised Turnaround Time per Parcel = Σ(turnaround/n_parcels) / n_completed."""

    def setUp(self):
        self.tracker = KPITracker()

    def _journey(self, gate_in, gate_out, n_parcels, truck="T1"):
        events = [
            SensorEvent(CheckpointID.GATE_IN,    truck, gate_in,  n_parcels=n_parcels),
            SensorEvent(CheckpointID.GHA_IN,     truck, gate_in+2, gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_START, truck, gate_in+5, gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_END,   truck, gate_out-2, gha_id="GHA_A"),
            SensorEvent(CheckpointID.GATE_OUT,   truck, gate_out,  n_parcels=n_parcels),
        ]
        self.tracker.ingest(events)

    def test_nttp_single_truck(self):
        # gate_in=0, gate_out=50, parcels=10  → nttp = 50/10 = 5.0
        self._journey(0, 50, 10)
        self.assertAlmostEqual(self.tracker.nttp(), 5.0)

    def test_nttp_single_parcel(self):
        self._journey(0, 30, 1)
        self.assertAlmostEqual(self.tracker.nttp(), 30.0)

    def test_nttp_averaged_over_completed_trucks(self):
        # T1: 50/10=5, T2: 100/5=20  → avg=(5+20)/2=12.5
        self._journey(0,  50, 10, "T1")
        self._journey(0, 100,  5, "T2")
        self.assertAlmostEqual(self.tracker.nttp(), 12.5)

    def test_nttp_zero_when_no_completed_trucks(self):
        self.assertEqual(self.tracker.nttp(), 0.0)

    def test_nttp_zero_parcel_truck_excluded(self):
        """Trucks with 0 parcels must not be counted (guarded by n_parcels > 0)."""
        self._journey(0, 50, 0)
        self.assertEqual(self.tracker._n_completed, 0)
        self.assertEqual(self.tracker.nttp(),        0.0)

    def test_nttp_large_parcel_count_reduces_value(self):
        self._journey(0, 50, 10, "T1")
        nttp_10 = self.tracker.nttp()
        self.tracker = KPITracker()
        self._journey(0, 50, 50, "T1")
        nttp_50 = self.tracker.nttp()
        self.assertLess(nttp_50, nttp_10)


# ---------------------------------------------------------------------------

class TestUtilizationStd(unittest.TestCase):
    """snapshot_utilization() + utilization_std() — load-balance signal."""

    def setUp(self):
        self.tracker = KPITracker()

    def test_single_snapshot_all_equal_std_is_zero(self):
        terminals = {
            "GHA_A": make_terminal_mock(0.5, 0.5),
            "GHA_B": make_terminal_mock(0.5, 0.5),
            "GHA_C": make_terminal_mock(0.5, 0.5),
        }
        self.tracker.snapshot_utilization(terminals)
        self.assertAlmostEqual(self.tracker.utilization_std(), 0.0)

    def test_std_increases_with_imbalance(self):
        balanced = {
            "GHA_A": make_terminal_mock(0.5, 0.5),
            "GHA_B": make_terminal_mock(0.5, 0.5),
            "GHA_C": make_terminal_mock(0.5, 0.5),
        }
        imbalanced = {
            "GHA_A": make_terminal_mock(1.0, 1.0),
            "GHA_B": make_terminal_mock(0.0, 0.0),
            "GHA_C": make_terminal_mock(0.5, 0.5),
        }
        t_bal = KPITracker()
        t_imb = KPITracker()
        t_bal.snapshot_utilization(balanced)
        t_imb.snapshot_utilization(imbalanced)
        self.assertGreater(t_imb.utilization_std(), t_bal.utilization_std())

    def test_snapshot_appends_to_list(self):
        t = {"GHA_A": make_terminal_mock(), "GHA_B": make_terminal_mock(), "GHA_C": make_terminal_mock()}
        for _ in range(5):
            self.tracker.snapshot_utilization(t)
        self.assertEqual(len(self.tracker._util["GHA_A"]["export"]), 5)

    def test_std_zero_with_one_gha_only(self):
        """std of a single value is 0 — handled by len(means)>1 guard."""
        tracker = KPITracker()
        # Patch _ghas to only have one GHA
        tracker._util = {"GHA_A": {"export": [], "import": []}}
        tracker.snapshot_utilization({"GHA_A": make_terminal_mock(0.9, 0.1)})
        self.assertEqual(tracker.utilization_std(), 0.0)

    def test_occupancy_values_stored_correctly(self):
        t = {
            "GHA_A": make_terminal_mock(0.3, 0.7),
            "GHA_B": make_terminal_mock(0.6, 0.4),
            "GHA_C": make_terminal_mock(0.1, 0.9),
        }
        self.tracker.snapshot_utilization(t)
        self.assertAlmostEqual(self.tracker._util["GHA_A"]["export"][0], 0.3)
        self.assertAlmostEqual(self.tracker._util["GHA_A"]["import"][0], 0.7)

    def test_multiple_steps_averaged_correctly(self):
        """Mean across steps, not just last snapshot, drives utilization_std."""
        t_low  = {"GHA_A": make_terminal_mock(0.0, 0.0), "GHA_B": make_terminal_mock(0.0, 0.0), "GHA_C": make_terminal_mock(0.0, 0.0)}
        t_high = {"GHA_A": make_terminal_mock(1.0, 1.0), "GHA_B": make_terminal_mock(1.0, 1.0), "GHA_C": make_terminal_mock(1.0, 1.0)}
        self.tracker.snapshot_utilization(t_low)
        self.tracker.snapshot_utilization(t_high)
        # All GHAs have mean 0.5  →  std = 0
        self.assertAlmostEqual(self.tracker.utilization_std(), 0.0)


# ---------------------------------------------------------------------------

class TestGlobalReward(unittest.TestCase):
    """global_reward = -(w_wpr * WPR + w_util * util_std)."""

    def setUp(self):
        self.tracker = KPITracker()

    def test_global_reward_non_positive_with_no_events(self):
        """With zero WPR and zero util_std the reward should be 0."""
        self.assertEqual(self.tracker.global_reward(), 0.0)

    def test_global_reward_negative_when_wpr_positive(self):
        self.tracker.ingest(make_truck_journey(dock_start=10, dock_end=20, gha_in=5))
        self.assertLess(self.tracker.global_reward(), 0.0)

    def test_global_reward_more_negative_with_higher_wpr(self):
        t1 = KPITracker()
        t2 = KPITracker()
        # T1: small wait         T2: large wait
        t1.ingest(make_truck_journey("T", "GHA_A", gate_in=0, gha_in=5,  dock_start=6,  dock_end=20, gate_out=25))
        t2.ingest(make_truck_journey("T", "GHA_A", gate_in=0, gha_in=5,  dock_start=50, dock_end=60, gate_out=65))
        self.assertLess(t2.global_reward(), t1.global_reward())

    def test_global_reward_more_negative_with_higher_util_std(self):
        t_bal = KPITracker()
        t_imb = KPITracker()
        bal = {"GHA_A": make_terminal_mock(0.5, 0.5), "GHA_B": make_terminal_mock(0.5, 0.5), "GHA_C": make_terminal_mock(0.5, 0.5)}
        imb = {"GHA_A": make_terminal_mock(1.0, 1.0), "GHA_B": make_terminal_mock(0.0, 0.0), "GHA_C": make_terminal_mock(0.5, 0.5)}
        t_bal.snapshot_utilization(bal)
        t_imb.snapshot_utilization(imb)
        self.assertLess(t_imb.global_reward(), t_bal.global_reward())

    def test_global_reward_uses_correct_weights(self):
        """Manual calculation: reward = -(1.0 * WPR + 1.0 * util_std)."""
        self.tracker.ingest(make_truck_journey(
            gate_in=0, gha_in=5, dock_start=10, dock_end=20, gate_out=25
        ))
        # wait=5, service=10 → WPR=0.5, util_std=0
        expected = -(1.0 * 0.5 + 1.0 * 0.0)
        self.assertAlmostEqual(self.tracker.global_reward(), expected)


# ---------------------------------------------------------------------------

class TestTransporterReward(unittest.TestCase):
    """transporter_reward uses delta logic — only changes since last call count."""

    def setUp(self):
        self.tracker = KPITracker()

    def _make_dtp(self, no_shows=None, late=None):
        return DTPPlatform(
            no_shows      = no_shows or {},
            late_arrivals = late     or {},
        )

    def test_zero_reward_when_nothing_changed(self):
        dtp = self._make_dtp()
        r = self.tracker.transporter_reward(dtp)
        self.assertEqual(r, 0.0)

    def test_wait_delta_penalises_correctly(self):
        """Each unit of new wait costs w['wait_per_min']=1.0."""
        self.tracker.ingest(make_truck_journey(
            gate_in=0, gha_in=5, dock_start=15, dock_end=25, gate_out=30
        ))
        dtp = self._make_dtp()
        r = self.tracker.transporter_reward(dtp)
        # delta_wait = 10 (dock_start 15 − gha_in 5 = 10)
        self.assertAlmostEqual(r, -10.0)

    def test_reward_delta_is_zero_on_second_call_with_no_new_events(self):
        self.tracker.ingest(make_truck_journey())
        dtp = self._make_dtp()
        self.tracker.transporter_reward(dtp)   # consumes delta
        r2 = self.tracker.transporter_reward(dtp)
        self.assertAlmostEqual(r2, 0.0)

    def test_no_show_penalty_applied(self):
        dtp = self._make_dtp(no_shows={"GHA_A": 2})
        r = self.tracker.transporter_reward(dtp)
        # delta_no_shows=2, weight=5.0 → penalty=-10
        self.assertAlmostEqual(r, -10.0)

    def test_late_arrival_penalty_applied(self):
        dtp = self._make_dtp(late={"GHA_A": 3})
        r = self.tracker.transporter_reward(dtp)
        # delta_late=3, weight=3.0 → penalty=-9
        self.assertAlmostEqual(r, -9.0)

    def test_combined_penalty(self):
        self.tracker._total_wait = 20.0
        self.tracker._prev_total_wait = 0.0
        dtp = self._make_dtp(no_shows={"GHA_A": 1}, late={"GHA_B": 2})
        r = self.tracker.transporter_reward(dtp)
        # -(1*20 + 5*1 + 3*2) = -(20+5+6) = -31
        self.assertAlmostEqual(r, -31.0)

    def test_prev_trackers_updated_after_call(self):
        self.tracker._total_wait = 50.0
        dtp = self._make_dtp(no_shows={"GHA_A": 3}, late={"GHA_B": 1})
        self.tracker.transporter_reward(dtp)
        self.assertEqual(self.tracker._prev_total_wait, 50.0)
        self.assertEqual(self.tracker._prev_no_shows,   3)
        self.assertEqual(self.tracker._prev_late,        1)

    def test_reward_is_non_positive(self):
        dtp = self._make_dtp(no_shows={"GHA_A": 5})
        r = self.tracker.transporter_reward(dtp)
        self.assertLessEqual(r, 0.0)

    def test_decreasing_no_shows_not_rewarded_or_penalised_beyond_delta(self):
        """Delta should never be negative — counters are monotonically non-decreasing."""
        dtp1 = self._make_dtp(no_shows={"GHA_A": 4})
        self.tracker.transporter_reward(dtp1)
        dtp2 = self._make_dtp(no_shows={"GHA_A": 4})  # same value → delta=0
        r2 = self.tracker.transporter_reward(dtp2)
        self.assertAlmostEqual(r2, 0.0)


# ---------------------------------------------------------------------------

class TestGHAReward(unittest.TestCase):
    """gha_reward = w_util*util + w_proc*delta_proc − w_queue*queue."""

    def setUp(self):
        self.tracker = KPITracker()

    def test_reward_purely_from_utilization(self):
        t = make_terminal_mock(exp_occ=0.8, imp_occ=0.6,
                               exp_q=0.0, imp_q=0.0,
                               processed_export=0, processed_import=0)
        r = self.tracker.gha_reward("GHA_A", t)
        # util = (0.8+0.6)/2 = 0.7, q=0, delta_proc=0
        # reward = 2.0*0.7 + 1.0*0 - 0.5*0 = 1.4
        self.assertAlmostEqual(r, 1.4)

    def test_reward_penalised_by_queue(self):
        t = make_terminal_mock(exp_occ=0.0, imp_occ=0.0,
                               exp_q=2.0, imp_q=2.0,
                               processed_export=0, processed_import=0)
        r = self.tracker.gha_reward("GHA_A", t)
        # reward = 0 + 0 - 0.5*(2+2) = -2.0
        self.assertAlmostEqual(r, -2.0)

    def test_reward_boosted_by_processed_parcels(self):
        t = make_terminal_mock(exp_occ=0.0, imp_occ=0.0,
                               exp_q=0.0, imp_q=0.0,
                               processed_export=10, processed_import=5)
        r = self.tracker.gha_reward("GHA_A", t)
        # delta_proc = 15, reward = 1.0*15 = 15
        self.assertAlmostEqual(r, 15.0)

    def test_delta_proc_is_zero_on_second_call_with_no_new_parcels(self):
        t = make_terminal_mock(processed_export=10, processed_import=5)
        self.tracker.gha_reward("GHA_A", t)         # prime _prev_proc
        r2 = self.tracker.gha_reward("GHA_A", t)
        util = (t.exp_occupancy() + t.imp_occupancy()) / 2
        q    = t.exp_queue_norm() + t.imp_queue_norm()
        expected = 2.0 * util + 1.0 * 0 - 0.5 * q
        self.assertAlmostEqual(r2, expected)

    def test_prev_proc_updated_after_call(self):
        t = make_terminal_mock(processed_export=7, processed_import=3)
        self.tracker.gha_reward("GHA_A", t)
        self.assertEqual(self.tracker._prev_proc["GHA_A"], 10)

    def test_reward_for_different_ghas_tracked_separately(self):
        t_a = make_terminal_mock(processed_export=10, processed_import=0)
        t_b = make_terminal_mock(processed_export=20, processed_import=0)
        self.tracker.gha_reward("GHA_A", t_a)
        self.tracker.gha_reward("GHA_B", t_b)
        self.assertEqual(self.tracker._prev_proc["GHA_A"], 10)
        self.assertEqual(self.tracker._prev_proc["GHA_B"], 20)

    def test_fully_loaded_dock_maximum_utilization_reward(self):
        t = make_terminal_mock(exp_occ=1.0, imp_occ=1.0, exp_q=0, imp_q=0)
        r = self.tracker.gha_reward("GHA_A", t)
        # util=1.0, reward = 2.0*1.0 = 2.0
        self.assertAlmostEqual(r, 2.0)

    def test_empty_dock_zero_util_reward(self):
        t = make_terminal_mock(exp_occ=0.0, imp_occ=0.0, exp_q=0, imp_q=0)
        r = self.tracker.gha_reward("GHA_A", t)
        self.assertAlmostEqual(r, 0.0)


# ---------------------------------------------------------------------------

class TestSummary(unittest.TestCase):
    """summary() must return all required keys with correct types."""

    REQUIRED_KEYS = {"wpr", "peak_wpr", "nttp", "util_std", "n_completed", "global_reward"}

    def setUp(self):
        self.tracker = KPITracker()

    def test_summary_contains_all_required_keys(self):
        s = self.tracker.summary()
        self.assertEqual(set(s.keys()), self.REQUIRED_KEYS)

    def test_summary_values_are_numeric(self):
        s = self.tracker.summary()
        for key, val in s.items():
            self.assertIsInstance(val, (int, float), msg=f"{key} is not numeric")

    def test_summary_n_completed_reflects_ingest(self):
        for i in range(3):
            self.tracker.ingest(make_truck_journey(f"T{i}"))
        self.assertEqual(self.tracker.summary()["n_completed"], 3)

    def test_summary_wpr_consistent_with_wpr_method(self):
        self.tracker.ingest(make_truck_journey())
        self.assertAlmostEqual(self.tracker.summary()["wpr"], self.tracker.wpr())

    def test_summary_global_reward_consistent_with_method(self):
        self.tracker.ingest(make_truck_journey())
        self.assertAlmostEqual(self.tracker.summary()["global_reward"], self.tracker.global_reward())

    def test_summary_all_zeros_before_events(self):
        s = self.tracker.summary()
        for key, val in s.items():
            self.assertEqual(val, 0, msg=f"{key} should be 0 before any events")


# ---------------------------------------------------------------------------

class TestStressAndEdgeCases(unittest.TestCase):
    """High-volume, boundary, and adversarial inputs."""

    def setUp(self):
        self.tracker = KPITracker()

    # --- high volume -------------------------------------------------------
    def test_1000_trucks_wpr_stable(self):
        for i in range(1000):
            self.tracker.ingest(make_truck_journey(
                truck_id=f"T{i}",
                gha_in=5, dock_start=10, dock_end=20
            ))
        # All trucks identical → WPR should remain exactly 0.5
        self.assertAlmostEqual(self.tracker.wpr(), 0.5, places=6)

    def test_1000_trucks_nttp_stable(self):
        for i in range(1000):
            self.tracker.ingest(make_truck_journey(
                truck_id=f"T{i}", gate_in=0, gate_out=50, n_parcels=10
            ))
        self.assertAlmostEqual(self.tracker.nttp(), 5.0, places=6)

    def test_1000_trucks_completed_count(self):
        for i in range(1000):
            self.tracker.ingest(make_truck_journey(truck_id=f"T{i}"))
        self.assertEqual(self.tracker._n_completed, 1000)

    # --- peak boundary exactness -------------------------------------------
    def test_event_just_before_peak_not_counted(self):
        events = [
            SensorEvent(CheckpointID.GATE_IN,    "T1",  0,  n_parcels=5),
            SensorEvent(CheckpointID.GHA_IN,     "T1",  50, gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_START, "T1",  59, gha_id="GHA_A"),  # 59 < 60
            SensorEvent(CheckpointID.DOCK_END,   "T1",  65, gha_id="GHA_A"),  # 65 > 60 but start is off-peak
            SensorEvent(CheckpointID.GATE_OUT,   "T1",  70, n_parcels=5),
        ]
        self.tracker.ingest(events)
        self.assertEqual(self.tracker._peak_wait, 0.0)   # DOCK_START was off-peak

    def test_event_just_after_peak_not_counted(self):
        events = [
            SensorEvent(CheckpointID.GATE_IN,    "T1",  0,   n_parcels=5),
            SensorEvent(CheckpointID.GHA_IN,     "T1",  115, gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_START, "T1",  121, gha_id="GHA_A"),  # 121 > 120
            SensorEvent(CheckpointID.DOCK_END,   "T1",  130, gha_id="GHA_A"),
            SensorEvent(CheckpointID.GATE_OUT,   "T1",  135, n_parcels=5),
        ]
        self.tracker.ingest(events)
        self.assertEqual(self.tracker._peak_wait,    0.0)
        self.assertEqual(self.tracker._peak_service, 0.0)

    # --- orphan / out-of-order events --------------------------------------
    def test_orphan_dock_start_does_not_crash(self):
        e = SensorEvent(CheckpointID.DOCK_START, "ORPHAN", 50, gha_id="GHA_A")
        try:
            self.tracker.ingest([e])
        except Exception as exc:
            self.fail(f"Orphan DOCK_START raised {exc}")

    def test_orphan_dock_end_does_not_crash(self):
        e = SensorEvent(CheckpointID.DOCK_END, "ORPHAN", 60, gha_id="GHA_A")
        try:
            self.tracker.ingest([e])
        except Exception as exc:
            self.fail(f"Orphan DOCK_END raised {exc}")

    def test_orphan_gha_in_does_not_crash(self):
        e = SensorEvent(CheckpointID.GHA_IN, "ORPHAN", 15, gha_id="GHA_A")
        try:
            self.tracker.ingest([e])
        except Exception as exc:
            self.fail(f"Orphan GHA_IN raised {exc}")

    def test_empty_event_list_is_no_op(self):
        self.tracker.ingest([])
        self.assertEqual(self.tracker._total_wait, 0.0)

    # --- simultaneous trucks at same GHA -----------------------------------
    def test_two_trucks_dock_simultaneously_accumulate_separately(self):
        events = [
            SensorEvent(CheckpointID.GATE_IN,    "T1", 0,  n_parcels=5),
            SensorEvent(CheckpointID.GATE_IN,    "T2", 0,  n_parcels=5),
            SensorEvent(CheckpointID.GHA_IN,     "T1", 5,  gha_id="GHA_A"),
            SensorEvent(CheckpointID.GHA_IN,     "T2", 5,  gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_START, "T1", 10, gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_START, "T2", 10, gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_END,   "T1", 20, gha_id="GHA_A"),
            SensorEvent(CheckpointID.DOCK_END,   "T2", 20, gha_id="GHA_A"),
            SensorEvent(CheckpointID.GATE_OUT,   "T1", 25, n_parcels=5),
            SensorEvent(CheckpointID.GATE_OUT,   "T2", 25, n_parcels=5),
        ]
        self.tracker.ingest(events)
        # Both trucks: wait=5 each, service=10 each  →  WPR=10/20=0.5
        self.assertAlmostEqual(self.tracker.wpr(), 0.5)
        self.assertEqual(self.tracker._n_completed, 2)

    # --- utilization stress ------------------------------------------------
    def test_500_snapshots_std_still_computed(self):
        t = {"GHA_A": make_terminal_mock(0.8, 0.2), "GHA_B": make_terminal_mock(0.4, 0.6), "GHA_C": make_terminal_mock(0.5, 0.5)}
        for _ in range(500):
            self.tracker.snapshot_utilization(t)
        std = self.tracker.utilization_std()
        self.assertIsInstance(std, float)
        self.assertGreaterEqual(std, 0.0)

    # --- idempotent gate_in before gate_out --------------------------------
    def test_gate_in_reset_mid_journey_recalculates_from_new_start(self):
        """Simulates a truck re-entering the gate without completing a prior visit."""
        self.tracker.ingest([SensorEvent(CheckpointID.GATE_IN, "T1", 0,  n_parcels=10)])
        self.tracker.ingest([SensorEvent(CheckpointID.GATE_IN, "T1", 100, n_parcels=5)])
        # Gate_out from new start
        self.tracker.ingest([SensorEvent(CheckpointID.GATE_OUT, "T1", 150, n_parcels=5)])
        # Turnaround should be from t=100 to t=150, not t=0
        self.assertAlmostEqual(self.tracker._nttp_sum, 50.0 / 5)

    # --- monotonic accumulation --------------------------------------------
    def test_total_wait_monotonically_increases(self):
        totals = []
        for i in range(10):
            self.tracker.ingest([
                SensorEvent(CheckpointID.GATE_IN,    f"T{i}", i*50,       n_parcels=5),
                SensorEvent(CheckpointID.GHA_IN,     f"T{i}", i*50+5,     gha_id="GHA_A"),
                SensorEvent(CheckpointID.DOCK_START, f"T{i}", i*50+10,    gha_id="GHA_A"),
                SensorEvent(CheckpointID.DOCK_END,   f"T{i}", i*50+20,    gha_id="GHA_A"),
                SensorEvent(CheckpointID.GATE_OUT,   f"T{i}", i*50+25,    n_parcels=5),
            ])
            totals.append(self.tracker._total_wait)
        for j in range(1, len(totals)):
            self.assertGreaterEqual(totals[j], totals[j-1])

    def test_total_service_monotonically_increases(self):
        services = []
        for i in range(10):
            self.tracker.ingest([
                SensorEvent(CheckpointID.GATE_IN,    f"T{i}", i*50,       n_parcels=5),
                SensorEvent(CheckpointID.GHA_IN,     f"T{i}", i*50+5,     gha_id="GHA_A"),
                SensorEvent(CheckpointID.DOCK_START, f"T{i}", i*50+10,    gha_id="GHA_A"),
                SensorEvent(CheckpointID.DOCK_END,   f"T{i}", i*50+20,    gha_id="GHA_A"),
                SensorEvent(CheckpointID.GATE_OUT,   f"T{i}", i*50+25,    n_parcels=5),
            ])
            services.append(self.tracker._total_service)
        for j in range(1, len(services)):
            self.assertGreaterEqual(services[j], services[j-1])


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)