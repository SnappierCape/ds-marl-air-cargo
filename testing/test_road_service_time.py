"""
test_road_service.py
====================
Comprehensive unittest suite for:
  • RoadNetwork    (road.py)       — stochastic travel time with lognormal noise
  • ServiceTimeModel (service_time.py) — dock service time sampler & analytical mean

Schiphol Cargo Hub — MARL Truck Slot Coordination Project
Both modules feed stochastic timing into the discrete-event simulation:
  • RoadNetwork determines when a truck physically reaches a GHA.
  • ServiceTimeModel determines how long a truck occupies a dock.

CONFIG UNDER TEST
─────────────────
  RoadNetwork:
    sigma  = 0.25          (lognormal shape parameter)
    lbound = 0.7           (clip floor: 70 % of base_time)
    ubound = 1.4           (clip ceiling: 140 % of base_time)
    nodes   = {A:0, B:1, C:2, D:3}
    segments = {
        "0_1": 10.0,   A↔B
        "1_2": 20.0,   B↔C
        "2_3":  5.0,   C↔D
        "0_3": 30.0,   A↔D
    }

  ServiceTimeModel:
    export: mu=3.0, sigma=0.3, bounds=[10.0, 60.0]
    import: mu=3.5, sigma=0.4, bounds=[15.0, 90.0]

TEST CLASS INVENTORY (8 classes, ~70 test methods)
───────────────────────────────────────────────────
  01. TestApplyNoiseValidation     — ValueError for zero / negative base_time
  02. TestApplyNoiseBehaviour      — clip bounds, median-preserving mu transform,
                                     float return, noise variability
  03. TestApplyNoiseEdgeCases      — very small / very large base_time, exact clip
                                     boundary injection via mock
  04. TestTimeFromToValidation     — unknown start node, unknown end node, both bad
  05. TestTimeFromToSegmentLookup  — canonical key construction, bidirectional
                                     symmetry, all four configured segments
  06. TestServiceSampleValidation  — ValueError for bad flow_type strings
  07. TestServiceSampleBehaviour   — output within bounds for export and import,
                                     float return, deterministic mock path
  08. TestServiceMean              — analytical formula exp(mu + σ²/2),
                                     export ≠ import, ValueError on bad flow_type

Run:
    python -m unittest test_road_service -v
"""

import math
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# MODULE BOOTSTRAP  — stubs for simpy and config must be in sys.modules BEFORE
# the modules under test are imported, because `params = config.load_params()`
# runs at module scope.
# ──────────────────────────────────────────────────────────────────────────────

MOCK_ROAD_CFG: dict = {
    "sigma":  0.25,
    "lbound": 0.7,
    "ubound": 1.4,
    "segments": {
        "0_1": 10.0,   # A ↔ B
        "1_2": 20.0,   # B ↔ C
        "2_3":  5.0,   # C ↔ D
        "0_3": 30.0,   # A ↔ D
    },
    "nodes": {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    },
    # ServiceTimeModel reads this sub-key
    "service_time": {
        "export": {"mu": 3.0, "sigma": 0.3, "bounds": [10.0, 60.0]},
        "import": {"mu": 3.5, "sigma": 0.4, "bounds": [15.0, 90.0]},
    },
}

_config_pkg = MagicMock()
_config_mod = MagicMock()
_config_mod.load_params.return_value = MOCK_ROAD_CFG
sys.modules.setdefault("config",        _config_pkg)
sys.modules.setdefault("config.config", _config_mod)

import os
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.road         import RoadNetwork      # noqa: E402
from env.service_time import ServiceTimeModel  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def make_road(cfg: dict = None) -> RoadNetwork:
    return RoadNetwork(cfg if cfg is not None else MOCK_ROAD_CFG)

def make_service(cfg: dict = None) -> ServiceTimeModel:
    return ServiceTimeModel(cfg if cfg is not None else MOCK_ROAD_CFG)

# Analytical mu for _apply_noise given a base_time and sigma
def _expected_mu(base_time: float, sigma: float) -> float:
    return math.log(base_time) - (sigma ** 2) / 2

# Analytical lognormal mean given mu and sigma
def _lognormal_mean(mu: float, sigma: float) -> float:
    return math.exp(mu + (sigma ** 2) / 2)


# ══════════════════════════════════════════════════════════════════════════════
# 01. _apply_noise — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyNoiseValidation(unittest.TestCase):
    """
    _apply_noise(base_time) must raise ValueError for any base_time <= 0.
    A lognormal is undefined for non-positive inputs, and the road model
    has no physical meaning for zero or negative travel time.
    """

    def setUp(self):
        self.road = make_road()

    def test_zero_base_time_raises(self):
        with self.assertRaises(ValueError):
            self.road._apply_noise(0)

    def test_negative_base_time_raises(self):
        with self.assertRaises(ValueError):
            self.road._apply_noise(-1.0)

    def test_large_negative_base_time_raises(self):
        with self.assertRaises(ValueError):
            self.road._apply_noise(-999.0)

    def test_near_zero_negative_raises(self):
        """Even an imperceptibly negative value must be rejected."""
        with self.assertRaises(ValueError):
            self.road._apply_noise(-1e-10)

    def test_positive_does_not_raise(self):
        """Any strictly positive base_time must not raise."""
        try:
            self.road._apply_noise(1.0)
        except ValueError:
            self.fail("_apply_noise raised ValueError for a positive base_time")

    def test_very_small_positive_does_not_raise(self):
        try:
            self.road._apply_noise(1e-6)
        except ValueError:
            self.fail("_apply_noise raised ValueError for a tiny positive base_time")

    def test_error_message_contains_input(self):
        """ValueError message should mention the bad value so logs are useful."""
        with self.assertRaises(ValueError) as ctx:
            self.road._apply_noise(-5.0)
        self.assertIn("-5", str(ctx.exception))


# ══════════════════════════════════════════════════════════════════════════════
# 02. _apply_noise — BEHAVIOUR  (deterministic via mock)
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyNoiseBehaviour(unittest.TestCase):
    """
    With a controlled lognormal sample we can verify:
      • The mu transformation: mu = log(base_time) - σ²/2
        (guarantees median of sampled_time equals base_time)
      • Output is clipped to [base_time * lbound, base_time * ubound]
      • Return type is float
      • Noise actually changes values across calls (non-deterministic in production)
    """

    def setUp(self):
        self.road = make_road()

    # ── mu transformation ──────────────────────────────────────────────────────

    def test_mu_passed_to_lognormal_is_median_preserving(self):
        """
        The transformation mu = log(base) - σ²/2 ensures E[X] = base_time.
        We verify the correct mu and sigma are forwarded to np.random.lognormal.
        """
        base_time = 15.0
        sigma     = self.road.sigma
        expected_mu = _expected_mu(base_time, sigma)

        with patch("env.road.np.random.lognormal", return_value=base_time) as mock_ln:
            self.road._apply_noise(base_time)
            args, kwargs = mock_ln.call_args
            # Accept both positional and keyword call styles
            actual_mu    = kwargs.get("mean",  args[0] if len(args) > 0 else None)
            actual_sigma = kwargs.get("sigma", args[1] if len(args) > 1 else None)
            self.assertAlmostEqual(actual_mu,    expected_mu, places=10)
            self.assertAlmostEqual(actual_sigma, sigma,       places=10)

    # ── clip floor ─────────────────────────────────────────────────────────────

    def test_sample_below_lbound_is_clipped_to_floor(self):
        """
        If the raw lognormal sample is below base_time * lbound (0.7 * 10 = 7),
        the output must be exactly 7.0.
        """
        base_time = 10.0
        floor     = base_time * self.road.lbound  # 7.0
        with patch("env.road.np.random.lognormal", return_value=1.0):   # 1.0 < floor
            result = self.road._apply_noise(base_time)
        self.assertAlmostEqual(result, floor)

    def test_sample_exactly_at_lbound_is_not_clipped(self):
        base_time = 10.0
        floor     = base_time * self.road.lbound   # 7.0
        with patch("env.road.np.random.lognormal", return_value=floor):
            result = self.road._apply_noise(base_time)
        self.assertAlmostEqual(result, floor)

    # ── clip ceiling ───────────────────────────────────────────────────────────

    def test_sample_above_ubound_is_clipped_to_ceiling(self):
        """
        If the raw sample exceeds base_time * ubound (1.4 * 10 = 14),
        the output must be exactly 14.0.
        """
        base_time = 10.0
        ceiling   = base_time * self.road.ubound  # 14.0
        with patch("env.road.np.random.lognormal", return_value=100.0):  # >> ceiling
            result = self.road._apply_noise(base_time)
        self.assertAlmostEqual(result, ceiling)

    def test_sample_exactly_at_ubound_is_not_clipped(self):
        base_time = 10.0
        ceiling   = base_time * self.road.ubound  # 14.0
        with patch("env.road.np.random.lognormal", return_value=ceiling):
            result = self.road._apply_noise(base_time)
        self.assertAlmostEqual(result, ceiling)

    # ── in-range pass-through ──────────────────────────────────────────────────

    def test_sample_within_bounds_passes_through_unchanged(self):
        """A sample perfectly equal to base_time falls inside [floor, ceiling]."""
        base_time = 20.0
        with patch("env.road.np.random.lognormal", return_value=base_time):
            result = self.road._apply_noise(base_time)
        self.assertAlmostEqual(result, base_time)

    # ── return type ───────────────────────────────────────────────────────────

    def test_return_type_is_float(self):
        result = self.road._apply_noise(10.0)
        self.assertIsInstance(result, float)

    # ── variability ───────────────────────────────────────────────────────────

    def test_repeated_calls_produce_different_values(self):
        """
        With the real RNG active, 50 samples should not all be identical.
        (Probability of failure ≈ 0 for any sane sigma > 0.)
        """
        samples = [self.road._apply_noise(20.0) for _ in range(50)]
        self.assertGreater(len(set(samples)), 1,
                           "50 calls should produce more than one unique value")

    def test_output_always_within_clip_bounds(self):
        """Over 200 real-RNG samples, every value must respect [lbound, ubound]."""
        base_time = 25.0
        lo = base_time * self.road.lbound
        hi = base_time * self.road.ubound
        for _ in range(200):
            v = self.road._apply_noise(base_time)
            self.assertGreaterEqual(v, lo - 1e-9)
            self.assertLessEqual(   v, hi + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# 03. _apply_noise — EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyNoiseEdgeCases(unittest.TestCase):
    """
    Correctness of scaling behaviour across very different magnitudes of
    base_time, and explicit verification that clip boundaries scale linearly
    with base_time.
    """

    def setUp(self):
        self.road = make_road()

    def test_clip_floor_scales_with_base_time(self):
        """Floor = base_time * lbound must scale proportionally."""
        for base_time in [1.0, 10.0, 100.0, 0.5]:
            expected_floor = base_time * self.road.lbound
            with patch("env.road.np.random.lognormal", return_value=0.0001):
                result = self.road._apply_noise(base_time)
            self.assertAlmostEqual(result, expected_floor,
                                   msg=f"Floor wrong for base_time={base_time}")

    def test_clip_ceiling_scales_with_base_time(self):
        """Ceiling = base_time * ubound must scale proportionally."""
        for base_time in [1.0, 10.0, 100.0, 0.5]:
            expected_ceiling = base_time * self.road.ubound
            with patch("env.road.np.random.lognormal", return_value=1e9):
                result = self.road._apply_noise(base_time)
            self.assertAlmostEqual(result, expected_ceiling,
                                   msg=f"Ceiling wrong for base_time={base_time}")

    def test_very_small_base_time_clips_correctly(self):
        """base_time = 0.001 should still obey lbound / ubound scaling."""
        base_time = 0.001
        with patch("env.road.np.random.lognormal", return_value=1e9):
            result = self.road._apply_noise(base_time)
        self.assertAlmostEqual(result, base_time * self.road.ubound, places=9)

    def test_large_base_time_clips_correctly(self):
        base_time = 10_000.0
        with patch("env.road.np.random.lognormal", return_value=0.0):
            result = self.road._apply_noise(base_time)
        self.assertAlmostEqual(result, base_time * self.road.lbound)

    def test_lognormal_called_exactly_once_per_invocation(self):
        """_apply_noise must not call the RNG more than once per call."""
        with patch("env.road.np.random.lognormal", return_value=10.0) as mock_ln:
            self.road._apply_noise(10.0)
        self.assertEqual(mock_ln.call_count, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 04. time_from_to — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeFromToValidation(unittest.TestCase):
    """
    time_from_to(start, end) raises ValueError when either node name is not
    in cfg["nodes"].  Both nodes must be independently validated.
    """

    def setUp(self):
        self.road = make_road()

    def test_invalid_start_node_raises(self):
        with self.assertRaises(ValueError):
            self.road.time_from_to("Z", "A")

    def test_invalid_end_node_raises(self):
        with self.assertRaises(ValueError):
            self.road.time_from_to("A", "Z")

    def test_both_invalid_nodes_raise(self):
        with self.assertRaises(ValueError):
            self.road.time_from_to("X", "Y")

    def test_empty_string_node_raises(self):
        with self.assertRaises(ValueError):
            self.road.time_from_to("", "A")

    def test_case_sensitive_node_name(self):
        """Node names are case-sensitive: 'a' ≠ 'A'."""
        with self.assertRaises(ValueError):
            self.road.time_from_to("a", "B")

    def test_valid_nodes_do_not_raise(self):
        try:
            self.road.time_from_to("A", "B")
        except ValueError:
            self.fail("time_from_to raised ValueError for valid nodes A, B")

    def test_error_message_contains_bad_node(self):
        """Error message should surface the offending node name."""
        with self.assertRaises(ValueError) as ctx:
            self.road.time_from_to("NOPE", "A")
        self.assertIn("NOPE", str(ctx.exception))


# ══════════════════════════════════════════════════════════════════════════════
# 05. time_from_to — SEGMENT LOOKUP  (deterministic via mock)
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeFromToSegmentLookup(unittest.TestCase):
    """
    Segment key construction: f'{min(node_id_start, node_id_end)}_{max(...)}'
    This guarantees the same key regardless of travel direction.

    Configured segments (node IDs in parentheses):
      A(0) ↔ B(1)  →  key "0_1",  base_time=10.0
      B(1) ↔ C(2)  →  key "1_2",  base_time=20.0
      C(2) ↔ D(3)  →  key "2_3",  base_time= 5.0
      A(0) ↔ D(3)  →  key "0_3",  base_time=30.0
    """

    def setUp(self):
        self.road = make_road()

    # ── bidirectional symmetry ─────────────────────────────────────────────────

    def test_a_to_b_and_b_to_a_use_same_segment(self):
        """
        A→B and B→A must resolve to the same segment key "0_1".
        We mock lognormal to a fixed value so both calls return identically.
        """
        with patch("env.road.np.random.lognormal", return_value=10.0):
            forward  = self.road.time_from_to("A", "B")
            backward = self.road.time_from_to("B", "A")
        self.assertAlmostEqual(forward, backward,
                               msg="A→B and B→A should use the same segment")

    def test_c_to_d_and_d_to_c_symmetric(self):
        with patch("env.road.np.random.lognormal", return_value=5.0):
            fwd = self.road.time_from_to("C", "D")
            bwd = self.road.time_from_to("D", "C")
        self.assertAlmostEqual(fwd, bwd)

    # ── correct base_time per segment ─────────────────────────────────────────

    def test_segment_ab_uses_base_time_10(self):
        """
        With lognormal mocked to the exact base_time, the output after clipping
        should still equal the base_time (no clip needed when sample == base).
        """
        with patch("env.road.np.random.lognormal", return_value=10.0) as mock_ln:
            self.road.time_from_to("A", "B")
            
        args, kwargs = mock_ln.call_args
        
        # Safely extract the mean whether it was passed via kwargs or args
        actual_mean = kwargs["mean"] if "mean" in kwargs else args[0]
        
        self.assertAlmostEqual(
            actual_mean,
            _expected_mu(10.0, self.road.sigma),
            places=10
        )

    def test_segment_bc_uses_base_time_20(self):
        with patch("env.road.np.random.lognormal", return_value=20.0) as mock_ln:
            self.road.time_from_to("B", "C")
            
        args, kwargs = mock_ln.call_args
        
        # Safely extract the mean whether it was passed via kwargs or args
        actual_mean = kwargs["mean"] if "mean" in kwargs else args[0]
        
        self.assertAlmostEqual(
            actual_mean,
            _expected_mu(20.0, self.road.sigma),
            places=10
        )

    def test_segment_cd_uses_base_time_5(self):
        with patch("env.road.np.random.lognormal", return_value=5.0) as mock_ln:
            self.road.time_from_to("C", "D")
            
        args, kwargs = mock_ln.call_args
        
        # Safely extract the mean whether it was passed via kwargs or args
        actual_mean = kwargs["mean"] if "mean" in kwargs else args[0]
        
        self.assertAlmostEqual(
            actual_mean,
            _expected_mu(5.0, self.road.sigma),
            places=10
        )

    def test_segment_ad_uses_base_time_30(self):
        with patch("env.road.np.random.lognormal", return_value=30.0) as mock_ln:
            self.road.time_from_to("A", "D")
            
        args, kwargs = mock_ln.call_args
        
        # Safely extract the mean whether it was passed via kwargs or args
        actual_mean = kwargs["mean"] if "mean" in kwargs else args[0]
        
        self.assertAlmostEqual(
            actual_mean,
            _expected_mu(30.0, self.road.sigma),
            places=10
        )

    # ── return type ───────────────────────────────────────────────────────────

    def test_return_type_is_float(self):
        result = self.road.time_from_to("A", "B")
        self.assertIsInstance(result, float)

    # ── clip bounds respected end-to-end ──────────────────────────────────────

    def test_result_within_clip_bounds_over_many_calls(self):
        """
        100 real-RNG calls for A→B (base_time=10) must all stay within
        [10 * lbound, 10 * ubound] = [7.0, 14.0].
        """
        lo = 10.0 * self.road.lbound
        hi = 10.0 * self.road.ubound
        for _ in range(100):
            v = self.road.time_from_to("A", "B")
            self.assertGreaterEqual(v, lo - 1e-9)
            self.assertLessEqual(   v, hi + 1e-9)

    def test_same_node_to_itself_raises_or_invalid_key(self):
        """
        A self-loop A→A produces segment key "0_0", which is NOT in cfg["segments"].
        Depending on implementation, this either raises KeyError or ValueError.
        We just verify it does not silently succeed.
        """
        with self.assertRaises(Exception):
            self.road.time_from_to("A", "A")


# ══════════════════════════════════════════════════════════════════════════════
# 06. ServiceTimeModel.sample — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestServiceSampleValidation(unittest.TestCase):
    """
    sample(flow_type) raises ValueError for any string not in {"export", "import"}.
    """

    def setUp(self):
        self.svc = make_service()

    def test_invalid_flow_type_raises(self):
        with self.assertRaises(ValueError):
            self.svc.sample("transit")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            self.svc.sample("")

    def test_numeric_string_raises(self):
        with self.assertRaises(ValueError):
            self.svc.sample("123")

    def test_uppercase_export_raises(self):
        """Flow type matching is case-sensitive; 'Export' is not 'export'."""
        with self.assertRaises(ValueError):
            self.svc.sample("Export")

    def test_uppercase_import_raises(self):
        with self.assertRaises(ValueError):
            self.svc.sample("IMPORT")

    def test_export_does_not_raise(self):
        try:
            self.svc.sample("export")
        except ValueError:
            self.fail("sample('export') raised ValueError unexpectedly")

    def test_import_does_not_raise(self):
        try:
            self.svc.sample("import")
        except ValueError:
            self.fail("sample('import') raised ValueError unexpectedly")

    def test_error_message_contains_bad_flow_type(self):
        with self.assertRaises(ValueError) as ctx:
            self.svc.sample("cabotage")
        self.assertIn("cabotage", str(ctx.exception))


# ══════════════════════════════════════════════════════════════════════════════
# 07. ServiceTimeModel.sample — BEHAVIOUR
# ══════════════════════════════════════════════════════════════════════════════

class TestServiceSampleBehaviour(unittest.TestCase):
    """
    sample(flow_type) must:
      • Read the correct sub-config (export vs import use different mu/sigma/bounds)
      • Clip the raw lognormal draw to [lo, hi] of the respective bounds
      • Return a float
      • Produce variability across calls
    """

    def setUp(self):
        self.svc    = make_service()
        self.exp_lo, self.exp_hi = MOCK_ROAD_CFG["service_time"]["export"]["bounds"]
        self.imp_lo, self.imp_hi = MOCK_ROAD_CFG["service_time"]["import"]["bounds"]

    # ── clip floor ─────────────────────────────────────────────────────────────

    def test_export_sample_clipped_to_floor(self):
        with patch("env.service_time.np.random.lognormal", return_value=0.0001):
            result = self.svc.sample("export")
        self.assertAlmostEqual(result, self.exp_lo)

    def test_import_sample_clipped_to_floor(self):
        with patch("env.service_time.np.random.lognormal", return_value=0.0001):
            result = self.svc.sample("import")
        self.assertAlmostEqual(result, self.imp_lo)

    # ── clip ceiling ───────────────────────────────────────────────────────────

    def test_export_sample_clipped_to_ceiling(self):
        with patch("env.service_time.np.random.lognormal", return_value=1e9):
            result = self.svc.sample("export")
        self.assertAlmostEqual(result, self.exp_hi)

    def test_import_sample_clipped_to_ceiling(self):
        with patch("env.service_time.np.random.lognormal", return_value=1e9):
            result = self.svc.sample("import")
        self.assertAlmostEqual(result, self.imp_hi)

    # ── pass-through inside bounds ─────────────────────────────────────────────

    def test_export_in_range_sample_unchanged(self):
        mid = (self.exp_lo + self.exp_hi) / 2   # 35.0
        with patch("env.service_time.np.random.lognormal", return_value=mid):
            result = self.svc.sample("export")
        self.assertAlmostEqual(result, mid)

    def test_import_in_range_sample_unchanged(self):
        mid = (self.imp_lo + self.imp_hi) / 2   # 52.5
        with patch("env.service_time.np.random.lognormal", return_value=mid):
            result = self.svc.sample("import")
        self.assertAlmostEqual(result, mid)

    # ── correct config routing ─────────────────────────────────────────────────

    def test_export_uses_export_mu_sigma(self):
        """
        The lognormal call for 'export' must receive export mu and sigma,
        not import's.
        """
        exp_cfg = MOCK_ROAD_CFG["service_time"]["export"]
        with patch("env.service_time.np.random.lognormal", return_value=20.0) as mock_ln:
            self.svc.sample("export")
            
        args, kwargs = mock_ln.call_args
        
        # Safely extract values using short-circuit logic
        actual_mu    = kwargs["mean"] if "mean" in kwargs else args[0]
        actual_sigma = kwargs["sigma"] if "sigma" in kwargs else args[1]
        
        self.assertAlmostEqual(actual_mu,    exp_cfg["mu"],    places=10)
        self.assertAlmostEqual(actual_sigma, exp_cfg["sigma"], places=10)

    def test_import_uses_import_mu_sigma(self):
        imp_cfg = MOCK_ROAD_CFG["service_time"]["import"]
        with patch("env.service_time.np.random.lognormal", return_value=20.0) as mock_ln:
            self.svc.sample("import")
            
        args, kwargs = mock_ln.call_args
        
        # Safely extract values using short-circuit logic
        actual_mu    = kwargs["mean"] if "mean" in kwargs else args[0]
        actual_sigma = kwargs["sigma"] if "sigma" in kwargs else args[1]
        
        self.assertAlmostEqual(actual_mu,    imp_cfg["mu"],    places=10)
        self.assertAlmostEqual(actual_sigma, imp_cfg["sigma"], places=10)

    def test_export_and_import_use_different_bounds(self):
        """
        Import and export have different bounds; mock to a value that sits
        inside import's bounds but below export's floor to verify routing.
        Export bounds [10, 60], import bounds [15, 90].
        A raw sample of 0.001 → export clips to 10.0, import clips to 15.0.
        """
        with patch("env.service_time.np.random.lognormal", return_value=0.001):
            exp_result = self.svc.sample("export")
            imp_result = self.svc.sample("import")
        self.assertAlmostEqual(exp_result, self.exp_lo)   # 10.0
        self.assertAlmostEqual(imp_result, self.imp_lo)   # 15.0
        self.assertNotAlmostEqual(exp_result, imp_result,
                                  msg="Export and import floors must differ")

    # ── return type ───────────────────────────────────────────────────────────

    def test_export_return_type_is_float(self):
        self.assertIsInstance(self.svc.sample("export"), float)

    def test_import_return_type_is_float(self):
        self.assertIsInstance(self.svc.sample("import"), float)

    # ── real-RNG bounds check over many draws ─────────────────────────────────

    def test_export_samples_always_within_bounds(self):
        for _ in range(200):
            v = self.svc.sample("export")
            self.assertGreaterEqual(v, self.exp_lo - 1e-9)
            self.assertLessEqual(   v, self.exp_hi + 1e-9)

    def test_import_samples_always_within_bounds(self):
        for _ in range(200):
            v = self.svc.sample("import")
            self.assertGreaterEqual(v, self.imp_lo - 1e-9)
            self.assertLessEqual(   v, self.imp_hi + 1e-9)

    # ── variability ───────────────────────────────────────────────────────────

    def test_export_samples_are_not_all_identical(self):
        samples = [self.svc.sample("export") for _ in range(50)]
        self.assertGreater(len(set(samples)), 1)

    def test_import_samples_are_not_all_identical(self):
        samples = [self.svc.sample("import") for _ in range(50)]
        self.assertGreater(len(set(samples)), 1)

    def test_lognormal_called_exactly_once_per_sample_call(self):
        with patch("env.service_time.np.random.lognormal", return_value=20.0) as mock_ln:
            self.svc.sample("export")
        self.assertEqual(mock_ln.call_count, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 08. ServiceTimeModel.mean — ANALYTICAL MEAN
# ══════════════════════════════════════════════════════════════════════════════

class TestServiceMean(unittest.TestCase):
    """
    mean(flow_type) → exp(mu + σ²/2)

    This is the analytical expected value of a lognormal distribution.
    No RNG is involved; the result is fully deterministic given mu and sigma.
    """

    def setUp(self):
        self.svc    = make_service()
        self.exp_mu = MOCK_ROAD_CFG["service_time"]["export"]["mu"]
        self.exp_sig = MOCK_ROAD_CFG["service_time"]["export"]["sigma"]
        self.imp_mu = MOCK_ROAD_CFG["service_time"]["import"]["mu"]
        self.imp_sig = MOCK_ROAD_CFG["service_time"]["import"]["sigma"]

    def test_export_mean_matches_formula(self):
        expected = _lognormal_mean(self.exp_mu, self.exp_sig)
        self.assertAlmostEqual(self.svc.mean("export"), expected, places=10)

    def test_import_mean_matches_formula(self):
        expected = _lognormal_mean(self.imp_mu, self.imp_sig)
        self.assertAlmostEqual(self.svc.mean("import"), expected, places=10)

    def test_export_and_import_means_differ(self):
        """
        Different mu and sigma for export vs import must yield different means.
        This validates that mean() routes to the correct sub-config.
        """
        self.assertNotAlmostEqual(
            self.svc.mean("export"),
            self.svc.mean("import"),
            places=5,
            msg="Export and import analytical means should differ"
        )

    def test_export_mean_is_positive(self):
        """Lognormal mean is always strictly positive."""
        self.assertGreater(self.svc.mean("export"), 0.0)

    def test_import_mean_is_positive(self):
        self.assertGreater(self.svc.mean("import"), 0.0)

    def test_mean_is_deterministic(self):
        """No RNG involved — two consecutive calls must return identical values."""
        self.assertEqual(self.svc.mean("export"), self.svc.mean("export"))
        self.assertEqual(self.svc.mean("import"), self.svc.mean("import"))

    def test_mean_return_type(self):
        """np.exp returns numpy float; both types satisfy numeric contract."""
        result = self.svc.mean("export")
        self.assertTrue(
            isinstance(result, (float, np.floating)),
            f"Expected float or np.floating, got {type(result)}"
        )

    def test_mean_invalid_flow_type_raises(self):
        with self.assertRaises(ValueError):
            self.svc.mean("transit")

    def test_mean_empty_string_raises(self):
        with self.assertRaises(ValueError):
            self.svc.mean("")

    def test_mean_case_sensitive(self):
        with self.assertRaises(ValueError):
            self.svc.mean("Export")

    def test_mean_error_message_contains_bad_type(self):
        with self.assertRaises(ValueError) as ctx:
            self.svc.mean("wrong_type")
        self.assertIn("wrong_type", str(ctx.exception))

    def test_mean_numerically_larger_for_higher_mu(self):
        """
        Import has higher mu (3.5) than export (3.0).
        Holding sigma approximately equal, import mean should be larger.
        """
        self.assertGreater(self.svc.mean("import"), self.svc.mean("export"))

    def test_mean_consistent_with_empirical_samples(self):
        """
        Over 50 000 draws the sample mean should be within ±5 % of the
        analytical mean (law of large numbers).  This cross-validates that
        sample() and mean() agree on the same distribution parameters.
        """
        n = 50_000
        analytical = self.svc.mean("export")
        empirical  = sum(self.svc.sample("export") for _ in range(n)) / n
        tolerance  = 0.05 * analytical   # 5 %
        self.assertAlmostEqual(
            empirical, analytical, delta=tolerance,
            msg=(f"Empirical mean {empirical:.4f} deviates >5 % "
                 f"from analytical mean {analytical:.4f}")
        )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)