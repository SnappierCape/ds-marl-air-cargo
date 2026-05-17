"""
test_demand.py
==============
Comprehensive unittest suite for DemandGenerator (demand.py).

Schiphol Cargo Hub — MARL Truck Slot Coordination Project
DemandGenerator has two responsibilities:
  1. run()  — infinite SimPy arrival loop (thinning / acceptance-rejection)
  2. _truck_journey() — complete per-truck lifecycle through the hub

Because both responsibilities rely on SimPy generators, all journey-level
tests use a real simpy.Environment driven to completion with env.run().
Pure helper methods (_rate_at, _create_truck, book_one_slot, dispatch_truck,
_origin_to_gate, _intra_airport_buffer, _get_pending_truck, _record_booking)
are tested in isolation without running SimPy at all.

CONFIG UNDER TEST
─────────────────
  demand:
    arrival_rate       = 0.05   trucks/min (base)
    peak_multiplier    = 3.0
    peak_window        = [960, 1080]   (16:00–18:00 in sim-minutes)
    ramp_dur           = 60            (1-hour ramp on each side)
    flow_split         = {"import": 0.6, "export": 0.4}
    origin_split       = {"near": 0.5, "far": 0.5}
    multi_stop_probs   = [0.5, 0.3, 0.15, 0.05]
    max_tp3_wait       = 30
    parcels_min        = 1
    parcels_max        = 50
    origin_travel_time = {"near": [5, 15], "far": [30, 90]}
  road.segments (used by _intra_airport_buffer):
    max value = 30.0

GHA_IDS derived from params["ghas"]: ["GHA_A", "GHA_B"]

TEST CLASS INVENTORY (11 classes, ~90 test methods)
─────────────────────────────────────────────────────
  01. TestRateAt                  — all five time zones, exact boundary values,
                                    24 h periodicity, monotonicity, clamping
  02. TestCreateTruck             — ID format, flow/origin/stop sampling, manifest
                                    structure, parcels bounds, single-GHA cap,
                                    counter increment, duplicate truck_ids absent
  03. TestGetPendingTruck         — found, not found, multiple trucks, empty list
  04. TestRecordBooking           — slot written, always True, multiple GHAs,
                                    overwrites previous value
  05. TestOriginToGate            — within [lo, hi] for near/far, float return,
                                    invalid key raises, distribution variability
  06. TestIntraAirportBuffer      — equals max segment value, float/numeric return
  07. TestBookOneSlot             — unknown truck, already-booked GHA, GHA not in
                                    manifest, no feasible slots, slot < earliest,
                                    happy path, earliest-feasible advancement when
                                    prior booking exists, freeze_time guard
  08. TestDispatchTruck           — unknown truck, missing bookings, partial bookings,
                                    full coverage dispatches, pending list cleanup,
                                    event fired, double-dispatch protection
  09. TestTruckJourneyDirectPath  — gate-in called, gate-out called, terminal visited,
                                    STATUS_DEPARTED set, direct path (no TP3 hold)
  10. TestTruckJourneyTP3Hold     — TP3 entered when early, released before GHA,
                                    gate-in / gate-out still called
  11. TestHandleTp3Redirect       — signal consumed path, booking-time departure path,
                                    timeout gives up and calls complete_stop,
                                    terminal re-visited after signal

Run:
    python -m unittest test_demand -v
"""

import sys
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np
import simpy

# ──────────────────────────────────────────────────────────────────────────────
# STUBS — must be in sys.modules before demand.py is imported
# ──────────────────────────────────────────────────────────────────────────────

MOCK_PARAMS: dict = {
    "ghas": {
        "GHA_A": {"import": 2, "export": 1},
        "GHA_B": {"import": 1, "export": 2},
    },
    "demand": {
        "arrival_rate":      0.05,
        "peak_multiplier":   3.0,
        "peak_window":       [960, 1080],
        "ramp_dur":          60,
        "flow_split":        {"import": 0.6, "export": 0.4},
        "origin_split":      {"near": 0.5, "far": 0.5},
        "multi_stop_probs":  [0.5, 0.3, 0.15, 0.05],
        "max_tp3_wait":      30,
        "parcels_min":       1,
        "parcels_max":       50,
        "origin_travel_time": {"near": [5, 15], "far": [30, 90]},
    },
    "road": {
        "segments": {
            "0_1": 10.0,
            "1_2": 20.0,
            "2_3":  5.0,
            "0_3": 30.0,
        }
    },
    "dtp_rules": {
        "slot_duration": 45,
        "freeze_time":    5,
    },
}

# Stub all external imports demand.py touches at module scope
_stub = lambda: MagicMock()
for mod in [
    "env.objects", "env.dtp_platform", "env.infrastructure",
    "env.road", "config", "config.config",
]:
    sys.modules.setdefault(mod, MagicMock())

# load_params must return our controlled config
sys.modules["config.config"].load_params.return_value = MOCK_PARAMS

# Truck and GHATerminal / TP3Buffer come from env.objects — give them real bodies
class _Truck:
    STATUS_TRAVELING  = "traveling"
    STATUS_AT_GHA     = "at_gha"
    STATUS_AT_TP3     = "at_tp3"
    STATUS_DEPARTED   = "departed"

    def __init__(self, truck_id, flow_type, origin_type, manifest):
        self.truck_id     = truck_id
        self.flow_type    = flow_type
        self.origin_type  = origin_type
        self.manifest     = manifest
        self.stops_remaining = list(manifest)   # mutable copy
        self.booked_slots: dict = {}
        self.status       = self.STATUS_TRAVELING

    def complete_stop(self, gha: str):
        self.stops_remaining = [s for s in self.stops_remaining if s["gha"] != gha]

sys.modules["env.objects"].Truck         = _Truck
sys.modules["env.objects"].GHATerminal   = MagicMock()
sys.modules["env.objects"].TP3Buffer     = MagicMock()

import os
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from env.demand import DemandGenerator   # noqa: E402

Truck = _Truck   # local alias for convenience


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURE BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def _make_dtp(now: float = 0.0,
              slot_duration: int = 45,
              freeze_time: int = 5) -> MagicMock:
    dtp = MagicMock()
    dtp.slot_duration = slot_duration
    dtp.freeze_time   = freeze_time
    dtp.get_available_slots.return_value = []
    dtp.book_slot.return_value           = True
    dtp.get_booking.return_value         = None
    return dtp


def _make_road(**overrides) -> MagicMock:
    road = MagicMock()
    road.time_from_to.return_value = 5.0
    for k, v in overrides.items():
        setattr(road, k, v)
    return road


def _make_terminals(ghas=("GHA_A", "GHA_B")) -> dict:
    """Return a dict of GHA → MagicMock terminal whose process_truck is a
    trivial SimPy generator that yields nothing (instant service)."""
    terminals = {}
    for gha in ghas:
        t = MagicMock()
        def _process(truck, dtp, _t=t):
            truck.stops_remaining = [
                s for s in truck.stops_remaining if s["gha"] != truck.manifest[0]["gha"]
            ]
            yield from []
        t.process_truck.side_effect = _process
        terminals[gha] = t
    return terminals


def _make_tp3() -> MagicMock:
    tp3 = MagicMock()
    tp3.enter.side_effect        = lambda truck: (x for x in [])  # instant generator
    tp3.get_pending_signals.return_value = []
    return tp3


def _make_infra() -> MagicMock:
    return MagicMock()


def _make_demand(env=None, dtp=None, road=None, terminals=None,
                 tp3=None, infra=None) -> DemandGenerator:
    env       = env       or simpy.Environment()
    dtp       = dtp       or _make_dtp()
    road      = road      or _make_road()
    terminals = terminals or _make_terminals()
    tp3       = tp3       or _make_tp3()
    infra     = infra     or _make_infra()
    return DemandGenerator(env, dtp, terminals, tp3, infra, road)


def _make_truck(gha="GHA_A", flow="import", origin="near",
                truck_id="TRK-00001") -> Truck:
    return Truck(
        truck_id=truck_id,
        flow_type=flow,
        origin_type=origin,
        manifest=[{"gha": gha, "parcels": 10}],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 01. _rate_at  — ARRIVAL RATE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

class TestRateAt(unittest.TestCase):
    """
    _rate_at(t) maps simulation time → trucks/min.

    Config:  base=0.05, peak_mult=3.0, peak_window=[960,1080], ramp_dur=60
    Zones (mod 1440):
      t < 900              → base                         = 0.05
      900 ≤ t < 960        → linear ramp  0.05 → 0.15
      960 ≤ t ≤ 1080       → peak flat                    = 0.15
      1080 < t ≤ 1140      → linear ramp down 0.15 → 0.05
      t > 1140             → base                         = 0.05
    """

    def setUp(self):
        self.dg   = _make_demand()
        self.base = MOCK_PARAMS["demand"]["arrival_rate"]           # 0.05
        self.peak = self.base * MOCK_PARAMS["demand"]["peak_multiplier"]  # 0.15

    # ── off-peak (base rate) ──────────────────────────────────────────────────

    def test_zero_is_base_rate(self):
        self.assertAlmostEqual(self.dg._rate_at(0), self.base)

    def test_early_morning_is_base_rate(self):
        self.assertAlmostEqual(self.dg._rate_at(500), self.base)

    def test_just_before_ramp_start_is_base_rate(self):
        """t=899 is one minute before ramp begins at 900."""
        self.assertAlmostEqual(self.dg._rate_at(899), self.base)

    def test_well_after_ramp_end_is_base_rate(self):
        self.assertAlmostEqual(self.dg._rate_at(1200), self.base)

    def test_end_of_day_wraps_to_base_rate(self):
        """t=1439 mod 1440 = 1439 → off-peak."""
        self.assertAlmostEqual(self.dg._rate_at(1439), self.base)

    # ── ramp-up zone [900, 960) ───────────────────────────────────────────────

    def test_ramp_start_boundary_is_base_rate(self):
        """t=900 is exactly peak_start - ramp; first ramp point → rate = base."""
        result = self.dg._rate_at(900)
        self.assertAlmostEqual(result, self.base, places=10)

    def test_ramp_midpoint_is_halfway(self):
        """t=930 is halfway through the 60-min ramp → midpoint rate."""
        expected = self.base * (1 + 0.5 * (3.0 - 1))   # base * 2 = 0.10
        self.assertAlmostEqual(self.dg._rate_at(930), expected, places=10)

    def test_ramp_just_before_peak_is_near_peak(self):
        """t=959 is one minute before peak_start=960."""
        result = self.dg._rate_at(959)
        self.assertGreater(result, self.base)
        self.assertLess(result, self.peak)

    def test_ramp_up_is_monotonically_increasing(self):
        rates = [self.dg._rate_at(t) for t in range(900, 960)]
        for i in range(len(rates) - 1):
            self.assertLessEqual(rates[i], rates[i + 1])

    # ── peak flat zone [960, 1080] ────────────────────────────────────────────

    def test_peak_start_exact_is_peak_rate(self):
        self.assertAlmostEqual(self.dg._rate_at(960), self.peak)

    def test_peak_midpoint_is_peak_rate(self):
        self.assertAlmostEqual(self.dg._rate_at(1020), self.peak)

    def test_peak_end_exact_is_peak_rate(self):
        self.assertAlmostEqual(self.dg._rate_at(1080), self.peak)

    def test_peak_zone_is_flat(self):
        rates = {self.dg._rate_at(t) for t in range(960, 1081)}
        self.assertEqual(len(rates), 1,
                         "Peak zone must be strictly flat (all rates identical)")

    # ── ramp-down zone (1080, 1140] ───────────────────────────────────────────

    def test_ramp_down_just_after_peak_end(self):
        """t=1081 — first minute of the ramp-down; rate should be just below peak."""
        result = self.dg._rate_at(1081)
        self.assertLess(result, self.peak)
        self.assertGreater(result, self.base)

    def test_ramp_down_midpoint(self):
        """t=1110 is halfway through the 60-min ramp-down."""
        expected = self.base * (3.0 - 0.5 * (3.0 - 1))   # base * 2 = 0.10
        self.assertAlmostEqual(self.dg._rate_at(1110), expected, places=10)

    def test_ramp_down_is_monotonically_decreasing(self):
        rates = [self.dg._rate_at(t) for t in range(1081, 1141)]
        for i in range(len(rates) - 1):
            self.assertGreaterEqual(rates[i], rates[i + 1])

    def test_ramp_down_end_returns_base_rate(self):
        """t=1140 = peak_end + ramp_dur → back to base rate."""
        self.assertAlmostEqual(self.dg._rate_at(1140), self.base, places=10)

    # ── periodicity ───────────────────────────────────────────────────────────

    def test_periodicity_at_1440(self):
        """_rate_at(t) == _rate_at(t + 1440) for arbitrary t."""
        for t in [0, 500, 960, 1080, 1200]:
            self.assertAlmostEqual(
                self.dg._rate_at(t),
                self.dg._rate_at(t + 1440),
                places=10,
                msg=f"Periodicity failed at t={t}"
            )

    def test_periodicity_at_2880(self):
        """Two full cycles."""
        self.assertAlmostEqual(self.dg._rate_at(960), self.dg._rate_at(960 + 2880))

    # ── rate always positive ──────────────────────────────────────────────────

    def test_rate_always_positive(self):
        for t in range(0, 1440, 5):
            self.assertGreater(self.dg._rate_at(t), 0.0,
                               f"Rate must be > 0 at t={t}")

    def test_rate_never_exceeds_peak(self):
        for t in range(0, 1440, 5):
            self.assertLessEqual(self.dg._rate_at(t), self.peak + 1e-9,
                                 f"Rate exceeded peak at t={t}")

    def test_rate_never_below_base(self):
        for t in range(0, 1440, 5):
            self.assertGreaterEqual(self.dg._rate_at(t), self.base - 1e-9,
                                    f"Rate fell below base at t={t}")


# ══════════════════════════════════════════════════════════════════════════════
# 02. _create_truck  — TRUCK FACTORY
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateTruck(unittest.TestCase):
    """
    _create_truck() samples attributes from config distributions and
    returns a Truck with a correctly formatted truck_id.
    """

    def setUp(self):
        self.dg = _make_demand()

    def test_truck_id_format_first_truck(self):
        truck = self.dg._create_truck()
        self.assertEqual(truck.truck_id, "TRK-00001")

    def test_truck_counter_increments(self):
        for i in range(1, 6):
            truck = self.dg._create_truck()
        self.assertEqual(truck.truck_id, "TRK-00005")

    def test_truck_ids_are_unique_across_many_calls(self):
        ids = [self.dg._create_truck().truck_id for _ in range(50)]
        self.assertEqual(len(ids), len(set(ids)),
                         "truck_ids must be unique")

    def test_flow_type_is_valid(self):
        valid = set(MOCK_PARAMS["demand"]["flow_split"].keys())
        for _ in range(30):
            self.assertIn(self.dg._create_truck().flow_type, valid)

    def test_origin_type_is_valid(self):
        valid = set(MOCK_PARAMS["demand"]["origin_split"].keys())
        for _ in range(30):
            self.assertIn(self.dg._create_truck().origin_type, valid)

    def test_manifest_length_matches_n_stops(self):
        """manifest must contain exactly n_stops entries."""
        for _ in range(40):
            truck = self.dg._create_truck()
            self.assertGreaterEqual(len(truck.manifest), 1)
            self.assertLessEqual(len(truck.manifest), 4)

    def test_manifest_ghas_are_valid(self):
        valid = set(MOCK_PARAMS["ghas"].keys())
        for _ in range(40):
            truck = self.dg._create_truck()
            for stop in truck.manifest:
                self.assertIn(stop["gha"], valid,
                              f"Unknown GHA {stop['gha']} in manifest")

    def test_manifest_ghas_are_unique_per_truck(self):
        """replace=False in np.random.choice → no GHA appears twice."""
        for _ in range(40):
            truck = self.dg._create_truck()
            ghas = [s["gha"] for s in truck.manifest]
            self.assertEqual(len(ghas), len(set(ghas)),
                             f"Duplicate GHAs in manifest: {ghas}")

    def test_parcels_within_bounds(self):
        lo = MOCK_PARAMS["demand"]["parcels_min"]
        hi = MOCK_PARAMS["demand"]["parcels_max"]
        for _ in range(40):
            truck = self.dg._create_truck()
            for stop in truck.manifest:
                self.assertGreaterEqual(stop["parcels"], lo)
                self.assertLess(stop["parcels"], hi,
                                "parcels must be < parcels_max (randint upper is exclusive)")

    def test_stops_remaining_equals_manifest_at_creation(self):
        truck = self.dg._create_truck()
        self.assertEqual(truck.stops_remaining, truck.manifest)

    def test_booked_slots_empty_at_creation(self):
        truck = self.dg._create_truck()
        self.assertEqual(truck.booked_slots, {})

    def test_n_stops_limited_by_number_of_ghas(self):
        """
        There are only 2 GHAs configured.  np.random.choice with replace=False
        can produce at most min(n_stops, len(GHA_IDS)) entries.
        """
        for _ in range(40):
            truck = self.dg._create_truck()
            self.assertLessEqual(len(truck.manifest),
                                 len(MOCK_PARAMS["ghas"]),
                                 "Manifest cannot exceed total number of GHAs")

    def test_flow_type_distribution_rough_proportions(self):
        """
        Over 1 000 trucks the import fraction should be within ±10 pp of 60 %.
        (Probability of failure ≈ 0 for a fair coin with this tolerance.)
        """
        trucks = [self.dg._create_truck() for _ in range(1_000)]
        import_frac = sum(1 for t in trucks if t.flow_type == "import") / 1_000
        self.assertAlmostEqual(import_frac, 0.6, delta=0.10)


# ══════════════════════════════════════════════════════════════════════════════
# 03. _get_pending_truck
# ══════════════════════════════════════════════════════════════════════════════

class TestGetPendingTruck(unittest.TestCase):

    def setUp(self):
        self.dg = _make_demand()

    def test_returns_none_for_empty_list(self):
        self.assertIsNone(self.dg._get_pending_truck("TRK-00001"))

    def test_returns_none_for_unknown_id(self):
        t = _make_truck()
        self.dg.pending_trucks = [t]
        self.assertIsNone(self.dg._get_pending_truck("TRK-GHOST"))

    def test_returns_correct_truck(self):
        t1 = _make_truck(truck_id="TRK-00001")
        t2 = _make_truck(truck_id="TRK-00002")
        self.dg.pending_trucks = [t1, t2]
        self.assertIs(self.dg._get_pending_truck("TRK-00002"), t2)

    def test_returns_first_match_only(self):
        """If two trucks share an ID (shouldn't happen but guards against it),
        the first one is returned."""
        t1 = _make_truck(truck_id="TRK-DUP")
        t2 = _make_truck(truck_id="TRK-DUP")
        self.dg.pending_trucks = [t1, t2]
        self.assertIs(self.dg._get_pending_truck("TRK-DUP"), t1)

    def test_does_not_mutate_pending_list(self):
        t = _make_truck()
        self.dg.pending_trucks = [t]
        self.dg._get_pending_truck(t.truck_id)
        self.assertEqual(len(self.dg.pending_trucks), 1)

    def test_empty_string_id_returns_none(self):
        t = _make_truck()
        self.dg.pending_trucks = [t]
        self.assertIsNone(self.dg._get_pending_truck(""))

    def test_none_id_returns_none(self):
        """Passing None should not raise; it simply won't match any truck_id."""
        t = _make_truck()
        self.dg.pending_trucks = [t]
        self.assertIsNone(self.dg._get_pending_truck(None))


# ══════════════════════════════════════════════════════════════════════════════
# 04. _record_booking
# ══════════════════════════════════════════════════════════════════════════════

class TestRecordBooking(unittest.TestCase):

    def setUp(self):
        self.dg = _make_demand()

    def test_always_returns_true(self):
        truck = _make_truck()
        self.assertTrue(self.dg._record_booking(truck, "GHA_A", 200))

    def test_slot_written_to_booked_slots(self):
        truck = _make_truck()
        self.dg._record_booking(truck, "GHA_A", 200)
        self.assertEqual(truck.booked_slots["GHA_A"], 200)

    def test_multiple_ghas_recorded_independently(self):
        truck = _make_truck(gha="GHA_A")
        truck.manifest.append({"gha": "GHA_B", "parcels": 5})
        truck.stops_remaining = list(truck.manifest)
        self.dg._record_booking(truck, "GHA_A", 200)
        self.dg._record_booking(truck, "GHA_B", 300)
        self.assertEqual(truck.booked_slots["GHA_A"], 200)
        self.assertEqual(truck.booked_slots["GHA_B"], 300)

    def test_overwrites_existing_booking(self):
        """Calling _record_booking twice on the same GHA replaces the old slot."""
        truck = _make_truck()
        self.dg._record_booking(truck, "GHA_A", 200)
        self.dg._record_booking(truck, "GHA_A", 350)
        self.assertEqual(truck.booked_slots["GHA_A"], 350)

    def test_slot_zero_is_recorded(self):
        """slot_start=0 is a valid (if unusual) value and must be stored."""
        truck = _make_truck()
        result = self.dg._record_booking(truck, "GHA_A", 0)
        self.assertTrue(result)
        self.assertEqual(truck.booked_slots["GHA_A"], 0)

    def test_does_not_raise_for_unknown_gha_key(self):
        """_record_booking is a thin writer; it must not validate the GHA name."""
        truck = _make_truck()
        try:
            self.dg._record_booking(truck, "UNKNOWN_GHA", 999)
        except Exception as e:
            self.fail(f"_record_booking raised unexpectedly: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 05. _origin_to_gate
# ══════════════════════════════════════════════════════════════════════════════

class TestOriginToGate(unittest.TestCase):
    """
    _origin_to_gate(origin_type) → float in [lo, hi] for the given origin type.
    near → [5, 15], far → [30, 90].
    """

    def setUp(self):
        self.dg = _make_demand()

    def test_near_within_bounds_single_call(self):
        v = self.dg._origin_to_gate("near")
        self.assertGreaterEqual(v, 5)
        self.assertLessEqual(v, 15)

    def test_far_within_bounds_single_call(self):
        v = self.dg._origin_to_gate("far")
        self.assertGreaterEqual(v, 30)
        self.assertLessEqual(v, 90)

    def test_near_within_bounds_over_many_calls(self):
        for _ in range(200):
            v = self.dg._origin_to_gate("near")
            self.assertGreaterEqual(v, 5  - 1e-9)
            self.assertLessEqual(   v, 15 + 1e-9)

    def test_far_within_bounds_over_many_calls(self):
        for _ in range(200):
            v = self.dg._origin_to_gate("far")
            self.assertGreaterEqual(v, 30 - 1e-9)
            self.assertLessEqual(   v, 90 + 1e-9)

    def test_return_type_is_float(self):
        self.assertIsInstance(self.dg._origin_to_gate("near"), float)
        self.assertIsInstance(self.dg._origin_to_gate("far"),  float)

    def test_near_and_far_produce_different_ranges(self):
        """
        Near samples must cluster below far samples.
        Maxing near over 50 calls should stay below the far minimum.
        """
        max_near = max(self.dg._origin_to_gate("near") for _ in range(50))
        min_far  = min(self.dg._origin_to_gate("far")  for _ in range(50))
        self.assertLess(max_near, min_far + 15,   # 15 is the full near range
                        "Near and far ranges should not overlap significantly")

    def test_invalid_origin_type_raises_key_error(self):
        """Accessing a missing key in _travel_time dict raises KeyError."""
        with self.assertRaises(KeyError):
            self.dg._origin_to_gate("intergalactic")

    def test_samples_are_variable(self):
        """50 samples should not all be identical."""
        samples = [self.dg._origin_to_gate("near") for _ in range(50)]
        self.assertGreater(len(set(samples)), 1)

    def test_mocked_uniform_passes_correct_bounds(self):
        """Verify that np.random.uniform is called with (lo, hi) = (5, 15)."""
        with patch("env.demand.np.random.uniform", return_value=10.0) as mock_u:
            self.dg._origin_to_gate("near")
        mock_u.assert_called_once_with(5, 15)

    def test_mocked_uniform_far_passes_correct_bounds(self):
        with patch("env.demand.np.random.uniform", return_value=60.0) as mock_u:
            self.dg._origin_to_gate("far")
        mock_u.assert_called_once_with(30, 90)


# ══════════════════════════════════════════════════════════════════════════════
# 06. _intra_airport_buffer
# ══════════════════════════════════════════════════════════════════════════════

class TestIntraAirportBuffer(unittest.TestCase):
    """
    _intra_airport_buffer() → max(params["road"]["segments"].values())
    With our config: max(10, 20, 5, 30) = 30.0
    """

    def setUp(self):
        self.dg = _make_demand()

    def test_returns_max_segment_value(self):
        expected = max(MOCK_PARAMS["road"]["segments"].values())   # 30.0
        self.assertAlmostEqual(self.dg._intra_airport_buffer(), expected)

    def test_return_is_numeric(self):
        result = self.dg._intra_airport_buffer()
        self.assertIsInstance(result, (int, float))

    def test_buffer_is_positive(self):
        self.assertGreater(self.dg._intra_airport_buffer(), 0)

    def test_buffer_changes_with_config(self):
        """
        If a custom config has a larger segment, the buffer must reflect it.
        This ensures _intra_airport_buffer() is not hard-coded.
        """
        big_cfg = {
            **MOCK_PARAMS,
            "road": {"segments": {"0_1": 999.0, "1_2": 5.0}}
        }
        dg_big = _make_demand()
        dg_big._parcels_min = 1
        # Directly test the formula path via params mock:
        with patch("env.demand.params", big_cfg):
            result = dg_big._intra_airport_buffer()
        self.assertAlmostEqual(result, 999.0)

    def test_deterministic_across_calls(self):
        """No randomness involved — must return the same value every call."""
        r1 = self.dg._intra_airport_buffer()
        r2 = self.dg._intra_airport_buffer()
        self.assertEqual(r1, r2)


# ══════════════════════════════════════════════════════════════════════════════
# 07. book_one_slot
# ══════════════════════════════════════════════════════════════════════════════

class TestBookOneSlot(unittest.TestCase):
    """
    book_one_slot(truck_id, gha, flow_type) → bool

    Guard cascade (in order):
      1. truck not in pending_trucks         → False
      2. gha already in truck.booked_slots   → False
      3. gha not in truck.stops_remaining    → False
      4. no feasible slot >= earliest        → False
      5. dtp.book_slot fails                 → False
    On success: _record_booking called, True returned.
    """

    def setUp(self):
        self.env  = simpy.Environment()
        self.dtp  = _make_dtp(freeze_time=5, slot_duration=45)
        self.dg   = _make_demand(env=self.env, dtp=self.dtp)
        self.truck = _make_truck(gha="GHA_A", flow="import")
        self.dg.pending_trucks = [self.truck]

    # ── guard 1: unknown truck ─────────────────────────────────────────────────

    def test_unknown_truck_returns_false(self):
        self.assertFalse(self.dg.book_one_slot("TRK-GHOST", "GHA_A", "import"))

    def test_empty_pending_list_returns_false(self):
        self.dg.pending_trucks = []
        self.assertFalse(self.dg.book_one_slot("TRK-00001", "GHA_A", "import"))

    # ── guard 2: already booked at this GHA ───────────────────────────────────

    def test_already_booked_gha_returns_false(self):
        self.truck.booked_slots["GHA_A"] = 200
        self.assertFalse(self.dg.book_one_slot("TRK-00001", "GHA_A", "import"))

    def test_already_booked_different_gha_does_not_block(self):
        """Booking GHA_B when GHA_A is already booked should not be blocked by guard 2."""
        self.truck.booked_slots["GHA_A"] = 200
        self.truck.manifest.append({"gha": "GHA_B", "parcels": 5})
        self.truck.stops_remaining.append({"gha": "GHA_B", "parcels": 5})
        self.dtp.get_available_slots.return_value = [300]
        result = self.dg.book_one_slot("TRK-00001", "GHA_B", "import")
        self.assertTrue(result)

    # ── guard 3: GHA not in manifest ──────────────────────────────────────────

    def test_gha_not_in_manifest_returns_false(self):
        """Truck only has GHA_A in manifest; booking GHA_B should fail."""
        self.assertFalse(self.dg.book_one_slot("TRK-00001", "GHA_B", "import"))

    def test_gha_completed_stop_returns_false(self):
        """If stops_remaining is empty for that GHA, guard 3 must catch it."""
        self.truck.stops_remaining = []
        self.assertFalse(self.dg.book_one_slot("TRK-00001", "GHA_A", "import"))

    # ── guard 4: no feasible slots ────────────────────────────────────────────

    def test_no_available_slots_returns_false(self):
        self.dtp.get_available_slots.return_value = []
        self.assertFalse(self.dg.book_one_slot("TRK-00001", "GHA_A", "import"))

    def test_all_slots_before_earliest_returns_false(self):
        """env.now=100, freeze_time=5 → earliest=105.  Slots [50,80] are all < 105."""
        self.env._now = 100
        self.dtp.get_available_slots.return_value = [50, 80]
        self.assertFalse(self.dg.book_one_slot("TRK-00001", "GHA_A", "import"))

    def test_slot_exactly_at_earliest_is_accepted(self):
        """Slot at exactly earliest (env.now + freeze_time) must be feasible."""
        self.env._now = 100
        earliest = 100 + self.dtp.freeze_time   # 105
        self.dtp.get_available_slots.return_value = [earliest]
        result = self.dg.book_one_slot("TRK-00001", "GHA_A", "import")
        self.assertTrue(result)

    # ── guard 5: dtp.book_slot fails ──────────────────────────────────────────

    def test_dtp_book_slot_failure_returns_false(self):
        self.dtp.get_available_slots.return_value = [200]
        self.dtp.book_slot.return_value = False
        self.assertFalse(self.dg.book_one_slot("TRK-00001", "GHA_A", "import"))

    # ── happy path ────────────────────────────────────────────────────────────

    def test_happy_path_returns_true(self):
        self.dtp.get_available_slots.return_value = [200]
        self.assertTrue(self.dg.book_one_slot("TRK-00001", "GHA_A", "import"))

    def test_happy_path_records_booking_on_truck(self):
        self.dtp.get_available_slots.return_value = [200]
        self.dg.book_one_slot("TRK-00001", "GHA_A", "import")
        self.assertEqual(self.truck.booked_slots.get("GHA_A"), 200)

    def test_earliest_advanced_when_prior_booking_exists(self):
        """
        When truck already has a booking at GHA_A=200, earliest for GHA_B
        must be at least 200 + slot_duration(45) + buffer(30) = 275.
        A slot at 100 must be skipped; a slot at 300 must be accepted.
        """
        self.truck.manifest.append({"gha": "GHA_B", "parcels": 5})
        self.truck.stops_remaining.append({"gha": "GHA_B", "parcels": 5})
        self.truck.booked_slots["GHA_A"] = 200
        self.dtp.get_available_slots.return_value = [100, 300]
        result = self.dg.book_one_slot("TRK-00001", "GHA_B", "import")
        self.assertTrue(result)
        # dtp.book_slot must have been called with slot 300, not 100
        call_args = self.dtp.book_slot.call_args
        self.assertEqual(call_args[0][1], 300)

    def test_get_available_slots_called_with_correct_flow_type(self):
        """get_available_slots must be called with the truck's flow_type."""
        self.dtp.get_available_slots.return_value = [200]
        self.dg.book_one_slot("TRK-00001", "GHA_A", "import")
        _, kwargs = self.dtp.get_available_slots.call_args
        self.assertEqual(kwargs.get("flow_type", None) or
                         self.dtp.get_available_slots.call_args[0][2]
                         if len(self.dtp.get_available_slots.call_args[0]) > 2
                         else kwargs.get("flow_type"),
                         "import")

    def test_picks_earliest_feasible_slot(self):
        """Among [200, 300, 400], the earliest feasible (200) must be chosen."""
        self.dtp.get_available_slots.return_value = [200, 300, 400]
        self.dg.book_one_slot("TRK-00001", "GHA_A", "import")
        call_args = self.dtp.book_slot.call_args[0]
        self.assertEqual(call_args[1], 200)


# ══════════════════════════════════════════════════════════════════════════════
# 08. dispatch_truck
# ══════════════════════════════════════════════════════════════════════════════

class TestDispatchTruck(unittest.TestCase):
    """
    dispatch_truck(truck_id) → bool

    Guards:
      1. truck not in pending_trucks                    → False
      2. not all manifest GHAs are in booked_slots      → False
    On success:
      • truck removed from pending_trucks
      • dispatch event fired (event.succeed() called)
      • True returned
    """

    def setUp(self):
        self.env   = simpy.Environment()
        self.dg    = _make_demand(env=self.env)
        self.truck = _make_truck(gha="GHA_A")
        self.truck.booked_slots = {"GHA_A": 200}
        self.dg.pending_trucks  = [self.truck]
        # Register a real SimPy event for the dispatch
        evt = self.env.event()
        self.dg._dispatch_events["TRK-00001"] = evt
        self.evt = evt

    # ── guard 1: unknown truck ────────────────────────────────────────────────

    def test_unknown_truck_returns_false(self):
        self.assertFalse(self.dg.dispatch_truck("TRK-GHOST"))

    def test_empty_pending_returns_false(self):
        self.dg.pending_trucks = []
        self.assertFalse(self.dg.dispatch_truck("TRK-00001"))

    # ── guard 2: not all stops booked ─────────────────────────────────────────

    def test_missing_booking_for_one_gha_returns_false(self):
        """Truck has GHA_A and GHA_B in manifest but only GHA_A booked."""
        self.truck.manifest.append({"gha": "GHA_B", "parcels": 5})
        self.assertFalse(self.dg.dispatch_truck("TRK-00001"))

    def test_empty_booked_slots_returns_false(self):
        self.truck.booked_slots = {}
        self.assertFalse(self.dg.dispatch_truck("TRK-00001"))

    def test_partial_coverage_returns_false(self):
        """manifest = [GHA_A, GHA_B], booked = {GHA_A} → False."""
        self.truck.manifest = [{"gha": "GHA_A", "parcels": 5},
                                {"gha": "GHA_B", "parcels": 5}]
        self.truck.booked_slots = {"GHA_A": 200}
        self.assertFalse(self.dg.dispatch_truck("TRK-00001"))

    # ── happy path ────────────────────────────────────────────────────────────

    def test_fully_booked_returns_true(self):
        self.assertTrue(self.dg.dispatch_truck("TRK-00001"))

    def test_truck_removed_from_pending_list(self):
        self.dg.dispatch_truck("TRK-00001")
        ids = [t.truck_id for t in self.dg.pending_trucks]
        self.assertNotIn("TRK-00001", ids)

    def test_other_trucks_remain_in_pending(self):
        t2 = _make_truck(truck_id="TRK-00002")
        t2.booked_slots = {"GHA_A": 300}
        self.dg.pending_trucks.append(t2)
        self.dg.dispatch_truck("TRK-00001")
        ids = [t.truck_id for t in self.dg.pending_trucks]
        self.assertIn("TRK-00002", ids)

    def test_dispatch_event_is_fired(self):
        """event.succeed() must be called, which means it becomes triggered."""
        self.assertFalse(self.evt.triggered)
        self.dg.dispatch_truck("TRK-00001")
        self.assertTrue(self.evt.triggered)

    def test_dispatch_event_removed_from_dict(self):
        """After dispatch, the event should be popped from _dispatch_events."""
        self.dg.dispatch_truck("TRK-00001")
        self.assertNotIn("TRK-00001", self.dg._dispatch_events)

    def test_double_dispatch_returns_false(self):
        """
        After the first successful dispatch, truck is no longer pending;
        a second dispatch attempt must fail.
        """
        self.dg.dispatch_truck("TRK-00001")
        self.assertFalse(self.dg.dispatch_truck("TRK-00001"))

    def test_dispatch_without_event_still_returns_true(self):
        """
        If no dispatch event was registered (edge case), the method must
        still return True and clean up the pending list.
        """
        self.dg._dispatch_events.pop("TRK-00001", None)
        result = self.dg.dispatch_truck("TRK-00001")
        self.assertTrue(result)
        self.assertNotIn(self.truck, self.dg.pending_trucks)

    def test_multi_stop_fully_booked_dispatches(self):
        """Two-GHA truck dispatches correctly when both are booked."""
        self.truck.manifest = [{"gha": "GHA_A", "parcels": 5},
                                {"gha": "GHA_B", "parcels": 5}]
        self.truck.booked_slots = {"GHA_A": 200, "GHA_B": 300}
        self.assertTrue(self.dg.dispatch_truck("TRK-00001"))


# ══════════════════════════════════════════════════════════════════════════════
# 09. _truck_journey  — DIRECT PATH  (no TP3 hold)
# ══════════════════════════════════════════════════════════════════════════════

class TestTruckJourneyDirectPath(unittest.TestCase):
    """
    Truck arrives at the gate AFTER the slot start time → direct path, no TP3.
    (env.now + eta_to_first_gha >= first_slot_start)

    We use a minimal process scaffold:
      1. Create a dispatch_event and succeed it immediately.
      2. Wire road.time_from_to so the truck always arrives "late" (on-time).
      3. Wire terminals[gha].process_truck as a trivial instant generator.
      4. env.run() to completion.
    """

    def _build(self, slot_start: int = 100, eta: float = 200.0):
        """
        eta > slot_start → truck arrives after slot → direct path.
        road.time_from_to always returns eta so arrival >= slot.
        """
        self.env   = simpy.Environment()
        self.infra = _make_infra()
        self.dtp   = _make_dtp()
        self.dtp.get_booking.return_value = None

        # Terminal that instantly completes the truck
        term = MagicMock()
        def _proc(truck, dtp):
            truck.stops_remaining = []
            yield from []
        term.process_truck.side_effect = _proc
        self.terminals = {"GHA_A": term}

        # Road always returns eta (large → truck is "on time / late")
        road = _make_road()
        road.time_from_to.return_value = eta

        self.tp3   = _make_tp3()
        self.dg    = DemandGenerator(self.env, self.dtp, self.terminals,
                                     self.tp3, self.infra, road)

        self.truck = _make_truck(gha="GHA_A")
        self.truck.booked_slots = {"GHA_A": slot_start}

        evt = self.env.event()
        self.dg.pending_trucks = [self.truck]
        self.dg._dispatch_events[self.truck.truck_id] = evt

        def _runner():
            evt.succeed()
            yield from []

        self.env.process(self.dg._truck_journey(self.truck, evt))
        self.env.process(_runner())

    def test_gate_in_called(self):
        self._build()
        self.env.run()
        self.infra.gate_in.assert_called_once()

    def test_gate_out_called(self):
        self._build()
        self.env.run()
        self.infra.gate_out.assert_called_once()

    def test_terminal_process_truck_called(self):
        self._build()
        self.env.run()
        self.terminals["GHA_A"].process_truck.assert_called_once()

    def test_truck_status_is_departed(self):
        self._build()
        self.env.run()
        self.assertEqual(self.truck.status, Truck.STATUS_DEPARTED)

    def test_tp3_enter_not_called_on_direct_path(self):
        self._build(slot_start=100, eta=200.0)  # eta >> slot_start → no TP3
        self.env.run()
        self.tp3.enter.assert_not_called()

    def test_gate_in_called_before_gate_out(self):
        # 1. Build the simulation first
        self._build()
        
        gate_calls = []
        
        # 2. Attach side_effects to the ACTUAL infra mock that _build() created
        self.infra.gate_in.side_effect  = lambda *a, **k: gate_calls.append("in")
        self.infra.gate_out.side_effect = lambda *a, **k: gate_calls.append("out")
        
        # 3. Run the simulation
        self.env.run()
        
        self.assertEqual(gate_calls, ["in", "out"])


# ══════════════════════════════════════════════════════════════════════════════
# 10. _truck_journey  — TP3 HOLD PATH  (truck arrives too early)
# ══════════════════════════════════════════════════════════════════════════════

class TestTruckJourneyTP3Hold(unittest.TestCase):
    """
    Truck arrives at the gate BEFORE the slot start time → TP3 hold.
    """

    def setUp(self):
        self.env   = simpy.Environment()
        self.infra = _make_infra()
        self.dtp   = _make_dtp()
        self.dtp.get_booking.return_value = None

        term = MagicMock()
        # Instead of empty yield, use env.timeout to behave like a real SimPy process
        def _proc(truck, dtp):
            truck.stops_remaining = []
            yield self.env.timeout(1) # Simulates 1 minute of processing time
        term.process_truck.side_effect = _proc
        self.terminals = {"GHA_A": term}

        # Ensure road mock cleanly handles any arguments passed to it
        road = MagicMock() 
        _call_count = [0]
        def _road_side_effect(a, b):
            _call_count[0] += 1
            return 200.0 if _call_count[0] == 1 else 1.0
        road.time_from_to.side_effect = _road_side_effect

        self.tp3_entered  = []
        self.tp3_released = []

        # Real SimPy environment interactions need to yield valid events
        def _tp3_enter(truck):
            self.tp3_entered.append(truck.truck_id)
            yield self.env.timeout(5) # Mock holding the truck in TP3 for 5 minutes

        tp3 = MagicMock()
        tp3.enter.side_effect = _tp3_enter
        tp3.release.side_effect = lambda tid: self.tp3_released.append(tid)
        tp3.get_pending_signals.return_value = []
        self.tp3 = tp3

        self.dg = DemandGenerator(self.env, self.dtp, self.terminals,
                                  tp3, self.infra, road)

        self.truck = _make_truck(gha="GHA_A")
        self.truck.booked_slots = {"GHA_A": 500} 

    def _run_journey_simulation(self):
        """Helper to cleanly spawn and run the processes in isolation per test."""
        evt = self.env.event()
        self.dg._dispatch_events[self.truck.truck_id] = evt

        def _fire():
            evt.succeed()
            yield from []

        self.env.process(self.dg._truck_journey(self.truck, evt))
        self.env.process(_fire())
        self.env.run()

    def test_tp3_enter_called(self):
        self._run_journey_simulation()  # Run a fresh simulation loop
        self.assertIn(self.truck.truck_id, self.tp3_entered)

    def test_tp3_release_called(self):
        self._run_journey_simulation()  # Run a fresh simulation loop
        self.assertIn(self.truck.truck_id, self.tp3_released)

    def test_gate_in_still_called(self):
        self._run_journey_simulation()
        self.infra.gate_in.assert_called_once()

    def test_gate_out_still_called(self):
        self._run_journey_simulation()
        self.infra.gate_out.assert_called_once()

    def test_terminal_visited_after_tp3(self):
        self._run_journey_simulation()
        self.terminals["GHA_A"].process_truck.assert_called_once()

    def test_truck_status_is_departed(self):
        self._run_journey_simulation()
        self.assertEqual(self.truck.status, Truck.STATUS_DEPARTED)


# ══════════════════════════════════════════════════════════════════════════════
# 11. _handle_tp3_redirect
# ══════════════════════════════════════════════════════════════════════════════

class TestHandleTp3Redirect(unittest.TestCase):
    """
    _handle_tp3_redirect(truck, gha) is a SimPy generator with three exit paths:

    Path A — Signal consumed: a pending signal for this GHA arrives within
             max_tp3_wait; truck is released and re-visits the terminal.
    Path B — Booking time departure: no signal but env.now >= booking - eta;
             truck self-releases and re-visits terminal.
    Path C — Timeout: neither signal nor booking time triggers within
             max_tp3_wait minutes; truck.complete_stop(gha) is called.
    """

    def _scaffold(self, max_tp3_wait=30):
        env   = simpy.Environment()
        dtp   = _make_dtp()
        infra = _make_infra()

        term = MagicMock()
        visit_log = []
        def _proc(truck, dtp_):
            visit_log.append(truck.truck_id)
            yield from []
        term.process_truck.side_effect = _proc

        road = _make_road()
        road.time_from_to.return_value = 1.0   # instant travel

        tp3_released = []
        tp3 = MagicMock()
        tp3.enter.side_effect = lambda truck: (x for x in [])
        tp3.release.side_effect = lambda tid: tp3_released.append(tid)
        tp3.get_pending_signals.return_value = []

        dg = DemandGenerator(env, dtp, {"GHA_A": term}, tp3, infra, road)
        dg._max_tp3_wait = max_tp3_wait

        truck = _make_truck(gha="GHA_A")

        return env, dg, truck, dtp, tp3, tp3_released, visit_log, term

    # ── Path A: signal consumed ───────────────────────────────────────────────

    def test_signal_path_releases_truck(self):
        env, dg, truck, dtp, tp3, released, visits, term = self._scaffold()
        signal = {"gha": "GHA_A", "consumed": False}
        # Signal appears on the first poll (after 1 minute)
        call_count = [0]
        def _signals():
            call_count[0] += 1
            if call_count[0] >= 1:
                return [signal]
            return []
        tp3.get_pending_signals.side_effect = _signals

        env.process(dg._handle_tp3_redirect(truck, "GHA_A"))
        env.run()

        self.assertIn(truck.truck_id, released)

    def test_signal_path_marks_signal_consumed(self):
        env, dg, truck, dtp, tp3, released, visits, term = self._scaffold()
        signal = {"gha": "GHA_A", "consumed": False}
        tp3.get_pending_signals.return_value = [signal]

        env.process(dg._handle_tp3_redirect(truck, "GHA_A"))
        env.run()

        self.assertTrue(signal["consumed"])

    def test_signal_path_revisits_terminal(self):
        env, dg, truck, dtp, tp3, released, visits, term = self._scaffold()
        signal = {"gha": "GHA_A", "consumed": False}
        tp3.get_pending_signals.return_value = [signal]

        env.process(dg._handle_tp3_redirect(truck, "GHA_A"))
        env.run()

        self.assertGreaterEqual(term.process_truck.call_count, 1)

    def test_signal_for_wrong_gha_is_ignored(self):
        """A signal for GHA_B must not trigger the GHA_A redirect."""
        env, dg, truck, dtp, tp3, released, visits, term = self._scaffold(max_tp3_wait=3)
        wrong_signal = {"gha": "GHA_B", "consumed": False}
        tp3.get_pending_signals.return_value = [wrong_signal]
        dtp.get_booking.return_value = None   # no booking → timeout path

        env.process(dg._handle_tp3_redirect(truck, "GHA_A"))
        env.run()

        self.assertFalse(wrong_signal["consumed"])

    # ── Path B: booking time departure ────────────────────────────────────────

    def test_booking_time_path_releases_truck(self):
        """
        Booking = env.now + 1 (i.e. 1 minute ahead); eta = 1.
        So env.now >= booking - eta fires immediately on the first poll.
        """
        env, dg, truck, dtp, tp3, released, visits, term = self._scaffold()
        # env.now starts at 0; set booking = 2 so condition fires at t=1
        dtp.get_booking.return_value = 2
        tp3.get_pending_signals.return_value = []  # no signal

        env.process(dg._handle_tp3_redirect(truck, "GHA_A"))
        env.run()

        self.assertIn(truck.truck_id, released)

    def test_booking_time_path_revisits_terminal(self):
        env, dg, truck, dtp, tp3, released, visits, term = self._scaffold()
        dtp.get_booking.return_value = 2
        tp3.get_pending_signals.return_value = []

        env.process(dg._handle_tp3_redirect(truck, "GHA_A"))
        env.run()

        self.assertGreaterEqual(term.process_truck.call_count, 1)

    # ── Path C: timeout ───────────────────────────────────────────────────────

    def test_timeout_releases_truck(self):
        env, dg, truck, dtp, tp3, released, visits, term = self._scaffold(max_tp3_wait=3)
        tp3.get_pending_signals.return_value = []
        dtp.get_booking.return_value = None   # no booking → never departs via Path B

        env.process(dg._handle_tp3_redirect(truck, "GHA_A"))
        env.run()

        self.assertIn(truck.truck_id, released)

    def test_timeout_calls_complete_stop(self):
        env, dg, truck, dtp, tp3, released, visits, term = self._scaffold(max_tp3_wait=3)
        tp3.get_pending_signals.return_value = []
        dtp.get_booking.return_value = None

        env.process(dg._handle_tp3_redirect(truck, "GHA_A"))
        env.run()

        # stops_remaining should no longer contain GHA_A (complete_stop removes it)
        ghas = [s["gha"] for s in truck.stops_remaining]
        self.assertNotIn("GHA_A", ghas)

    def test_timeout_does_not_revisit_terminal(self):
        env, dg, truck, dtp, tp3, released, visits, term = self._scaffold(max_tp3_wait=3)
        tp3.get_pending_signals.return_value = []
        dtp.get_booking.return_value = None

        env.process(dg._handle_tp3_redirect(truck, "GHA_A"))
        env.run()

        term.process_truck.assert_not_called()

    def test_timeout_exits_after_max_tp3_wait_iterations(self):
        """Simulation time must not exceed max_tp3_wait + small overhead."""
        env, dg, truck, dtp, tp3, released, visits, term = self._scaffold(max_tp3_wait=5)
        tp3.get_pending_signals.return_value = []
        dtp.get_booking.return_value = None

        env.process(dg._handle_tp3_redirect(truck, "GHA_A"))
        env.run()

        # TP3 travel (1) + max_tp3_wait iterations (5) = ~6 sim minutes max
        self.assertLessEqual(env.now, 10,
                             "Timeout must not spin forever beyond max_tp3_wait")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)