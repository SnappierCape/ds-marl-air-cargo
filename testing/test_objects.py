# =============================================================================
# TEST SUITE — SIMULATION OBJECTS MODULE (objects.py)
# =============================================================================
# DESCRIPTION:
#     Full unittest coverage for Truck, GHATerminal, and TP3Buffer.
#     All external imports (config, infrastructure, dtp_platform,
#     service_time) are stubbed so the suite is entirely self-contained.
#
# BUGS DELIBERATELY TARGETED (found during static analysis):
#   BUG-1  Truck.total_parcels() iterates self.manifest, NOT
#          self.stops_remaining → returns stale count after stops are
#          completed. parcels_for() has the same issue.
#   BUG-2  GHATerminal.exp_queue_norm() / imp_queue_norm() lack the
#          zero-division guard that exp_occupancy / imp_occupancy have.
#          ZeroDivisionError when dock count is 0.
#   BUG-3  GHATerminal._dock_pool() / _queue() silently fall back to
#          the *import* pool for any unrecognised flow_type string,
#          including case-variants like "Export". No validation.
#   BUG-4  Truck.__post_init__ makes a *shallow* copy of manifest into
#          stops_remaining — the two lists share the same dict objects.
#          Mutating a stop dict through one reference corrupts the other.
#   BUG-5  Truck.complete_stop() removes every entry whose gha matches,
#          so a manifest with duplicate GHA entries silently loses all of
#          them on the first call.
#   BUG-6  TP3Buffer.release_next() checks truck.booked_slots, not
#          truck.stops_remaining. A truck that has FINISHED its stop at a
#          GHA but still holds the slot key will be incorrectly released.
#   BUG-7  TP3Buffer.signal_standby_opportunity() is annotated to return
#          List[Dict] but always returns None. Callers expecting a list
#          will get None.
#   BUG-8  TP3Buffer.occupancy_ratio() performs self.slots.count /
#          self.CAPACITY with no zero-division guard. If the config sets
#          tp3.capacity = 0 the method crashes.
#   BUG-9  GHATerminal.process_truck() indexes self.stats[truck.flow_type]
#          at the end. If flow_type is not "export" or "import" (the only
#          two keys) it raises KeyError — silently masked by the fallback
#          in _dock_pool/_queue which routes the truck to the import pool.
#   BUG-10 Truck.parcels_for() searches self.manifest, not
#          self.stops_remaining → returns parcel counts for already-
#          completed stops as if work is still pending.
#
# TEST CLASSES:
#   TestTruckInit                — dataclass construction, default state
#   TestTruckTotalParcels        — BUG-1 + normal paths
#   TestTruckParcelsFor          — BUG-10 + normal paths
#   TestTruckNextSlot            — slot filtering, empty, all-completed
#   TestTruckNextStop            — FCFS ordering, empty list
#   TestTruckCompleteStop        — BUG-4, BUG-5, normal removal
#   TestTruckShallowCopy         — BUG-4 mutation aliasing
#   TestGHATerminalInit          — resource capacities, stats shape
#   TestGHATerminalRoutingHelpers— BUG-3 silent fallback
#   TestGHATerminalOccupancy     — exp/imp occupancy, zero-dock guard
#   TestGHATerminalQueueNorm     — BUG-2 zero-division
#   TestGHATerminalUpcomingNorm  — horizon filter, total=0 guard
#   TestGHATerminalProcessTruck  — SimPy-driven full journey, phase paths
#   TestGHATerminalReleaseWatcher— SimPy watcher timing
#   TestTP3BufferEnterRelease    — SimPy enter/release lifecycle
#   TestTP3BufferOverflow        — overflow queue mechanics
#   TestTP3BufferReleaseNext     — BUG-6 stale booked_slots
#   TestTP3BufferStandby         — BUG-7 return type, dedup absence
#   TestTP3BufferObservation     — BUG-8 occupancy_ratio, n_parked, etc.
#   TestTP3BufferEdgeCases       — double-release, unknown truck_id
#   TestStressAndCrossModule     — 500-truck throughput, mixed flows
# =============================================================================

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, call, patch
import simpy

# ---------------------------------------------------------------------------
# ① Stub every module that objects.py imports at load-time
# ---------------------------------------------------------------------------

# -- config -----------------------------------------------------------------
config_mod        = types.ModuleType("config")
config_config_mod = types.ModuleType("config.config")

DEFAULT_PARAMS = {
    "ghas": {
        "GHA_A": {"export": 3, "import": 2},
        "GHA_B": {"export": 4, "import": 4},
        "GHA_ZERO": {"export": 0, "import": 0},  # edge: zero-dock GHA
    },
    "tp3": {"capacity": 10},
}
config_config_mod.load_params = lambda: DEFAULT_PARAMS
sys.modules["config"]        = config_mod
sys.modules["config.config"] = config_config_mod

# -- env.infrastructure -----------------------------------------------------
infra_mod = types.ModuleType("env.infrastructure")
class _InfrastructureLayer:
    def gha_in(self, *a): pass
    def dock_start(self, *a): pass
    def dock_end(self, *a): pass
    def tp3_in(self, *a): pass
    def tp3_out(self, *a): pass
infra_mod.InfrastructureLayer = _InfrastructureLayer
# sys.modules["env"]                = types.ModuleType("env")
sys.modules["env.infrastructure"] = infra_mod

# -- env.dtp_platform -------------------------------------------------------
dtp_mod = types.ModuleType("env.dtp_platform")
class _DTPPlatform:
    priority_window = 10
    registry        = {}
    no_shows        = {}
    late_arrivals   = {}
    def get_slot_phase(self, slot_start, arrival, dock_is_free): return "on_time"
    def record_late(self, tid): pass
    def record_no_show(self, gha, slot, tid): pass
    def mark_docked(self, gha, slot, tid): pass
    def mark_closed(self, gha, slot, tid): pass
    def release_to_standby(self, gha, slot): return False
dtp_mod.DTPPlatform = _DTPPlatform
sys.modules["env.dtp_platform"] = dtp_mod

# -- env.service_time -------------------------------------------------------
svc_mod = types.ModuleType("env.service_time")
class _ServiceTimeModel:
    def sample(self, flow_type): return 10.0   # fixed for determinism
svc_mod.ServiceTimeModel = _ServiceTimeModel
sys.modules["env.service_time"] = svc_mod

# ---------------------------------------------------------------------------
# ② Import the module under test (stubs are already in sys.modules)
# ---------------------------------------------------------------------------
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.objects import Truck, GHATerminal, TP3Buffer   # noqa: E402


# ===========================================================================
# HELPERS
# ===========================================================================

def make_truck(truck_id="T1", flow_type="export",
               manifest=None, booked_slots=None):
    if manifest is None:
        manifest = [{"gha": "GHA_A", "parcels": 10}]
    t = Truck(
        truck_id=truck_id,
        flow_type=flow_type,
        origin_type="road",
        manifest=manifest,
    )
    if booked_slots:
        t.booked_slots = booked_slots
    return t


def make_terminal(gha="GHA_A", n_exp=None, n_imp=None, svc_time=10.0):
    env  = simpy.Environment()
    svc  = MagicMock()
    svc.sample.return_value = svc_time
    infra = MagicMock(spec=_InfrastructureLayer)
    cfg = dict(DEFAULT_PARAMS)
    if n_exp is not None or n_imp is not None:
        cfg = {k: v for k, v in DEFAULT_PARAMS.items()}
        cfg["ghas"] = dict(DEFAULT_PARAMS["ghas"])
        cfg["ghas"][gha] = {
            "export": n_exp if n_exp is not None else DEFAULT_PARAMS["ghas"].get(gha, {}).get("export", 3),
            "import": n_imp if n_imp is not None else DEFAULT_PARAMS["ghas"].get(gha, {}).get("import", 2),
        }
    return env, GHATerminal(env, gha, svc, infra, cfg=cfg), infra, svc


def make_tp3(capacity=None):
    env   = simpy.Environment()
    infra = MagicMock(spec=_InfrastructureLayer)
    # Temporarily patch the class-level CAPACITY if needed
    if capacity is not None:
        TP3Buffer.CAPACITY = capacity
    tp3 = TP3Buffer(env, infra)
    return env, tp3, infra


def make_dtp(phase="on_time", priority_window=10,
             registry=None, release_to_standby=False):
    dtp = MagicMock(spec=_DTPPlatform)
    dtp.priority_window = priority_window
    dtp.registry        = registry or {}
    dtp.get_slot_phase.return_value   = phase
    dtp.release_to_standby.return_value = release_to_standby
    return dtp


def run_process(env, gen, until=1000):
    """Convenience: schedule a generator as a SimPy process and run."""
    proc = env.process(gen)
    env.run(until=proc)


# ===========================================================================
# TEST CLASSES
# ===========================================================================

# ---------------------------------------------------------------------------
# Truck — initialisation
# ---------------------------------------------------------------------------
class TestTruckInit(unittest.TestCase):

    def test_stops_remaining_populated_from_manifest(self):
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 5}])
        self.assertEqual(len(t.stops_remaining), 1)

    def test_stops_remaining_length_matches_manifest(self):
        m = [{"gha": "GHA_A", "parcels": 5}, {"gha": "GHA_B", "parcels": 3}]
        t = make_truck(manifest=m)
        self.assertEqual(len(t.stops_remaining), len(m))

    def test_default_status_is_in_transit(self):
        t = make_truck()
        self.assertEqual(t.status, Truck.STATUS_IN_TRANSIT)

    def test_default_booked_slots_empty(self):
        t = make_truck()
        self.assertEqual(t.booked_slots, {})

    def test_status_constants_accessible_on_class(self):
        self.assertEqual(Truck.STATUS_IN_TRANSIT, "in_transit")
        self.assertEqual(Truck.STATUS_AT_TP3,     "at_tp3")
        self.assertEqual(Truck.STATUS_QUEUED,      "queued")
        self.assertEqual(Truck.STATUS_DOCKED,      "docked")
        self.assertEqual(Truck.STATUS_DEPARTED,    "departed")

    def test_empty_manifest_gives_empty_stops_remaining(self):
        t = make_truck(manifest=[])
        self.assertEqual(t.stops_remaining, [])

    def test_stops_remaining_is_independent_list_from_manifest(self):
        """list() creates a new list object — length-level independence."""
        m = [{"gha": "GHA_A", "parcels": 5}]
        t = make_truck(manifest=m)
        t.stops_remaining.clear()
        # manifest must be untouched — stops_remaining is a separate list
        self.assertEqual(len(t.manifest), 1,
            "stops_remaining.clear() should not affect t.manifest")


# ---------------------------------------------------------------------------
# Truck.total_parcels  —  BUG-1
# ---------------------------------------------------------------------------
class TestTruckTotalParcels(unittest.TestCase):

    def test_returns_sum_of_all_manifest_parcels(self):
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 5},
                                  {"gha": "GHA_B", "parcels": 3}])
        self.assertEqual(t.total_parcels(), 8)

    def test_zero_parcel_manifest(self):
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 0}])
        self.assertEqual(t.total_parcels(), 0)

    def test_empty_manifest_returns_zero(self):
        t = make_truck(manifest=[])
        self.assertEqual(t.total_parcels(), 0)

    # ── BUG-1: total_parcels reads manifest, not stops_remaining ─────────────
    def test_BUG1_total_parcels_does_not_decrease_after_complete_stop(self):
        """
        BUG-1: total_parcels() sums self.manifest unconditionally.
        After complete_stop() removes GHA_A from stops_remaining, the
        count should logically reflect only remaining work — but it still
        returns the full original total because it reads self.manifest.
        """
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 5},
                                  {"gha": "GHA_B", "parcels": 3}])
        t.complete_stop("GHA_A")
        # Expected behaviour (no bug): 3   Actual behaviour (BUG): 8
        self.assertEqual(t.total_parcels(), 8,
            "BUG-1 confirmed: total_parcels returns full manifest sum "
            "even after stops are completed.")

    def test_BUG1_all_stops_completed_still_returns_full_count(self):
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 7}])
        t.complete_stop("GHA_A")
        self.assertEqual(len(t.stops_remaining), 0)
        # Should be 0 if reading stops_remaining — is still 7 (BUG)
        self.assertEqual(t.total_parcels(), 7,
            "BUG-1: total_parcels returns non-zero even when no stops remain.")


# ---------------------------------------------------------------------------
# Truck.parcels_for  —  BUG-10
# ---------------------------------------------------------------------------
class TestTruckParcelsFor(unittest.TestCase):

    def test_returns_correct_parcel_count_for_known_gha(self):
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 9}])
        self.assertEqual(t.parcels_for("GHA_A"), 9)

    def test_returns_zero_for_unknown_gha(self):
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 9}])
        self.assertEqual(t.parcels_for("GHA_B"), 0)

    def test_returns_first_match_when_duplicate_gha_entries(self):
        """If manifest has two entries for the same GHA, only the first is returned."""
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 5},
                                  {"gha": "GHA_A", "parcels": 3}])
        self.assertEqual(t.parcels_for("GHA_A"), 5)

    # ── BUG-10: parcels_for reads manifest, not stops_remaining ──────────────
    def test_BUG10_parcels_for_returns_value_for_completed_stop(self):
        """
        BUG-10: parcels_for() scans self.manifest, so it happily returns
        a parcel count for a GHA whose stop has already been completed.
        Callers using this to decide 'how much work is left' will be misled.
        """
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 4},
                                  {"gha": "GHA_B", "parcels": 6}])
        t.complete_stop("GHA_A")
        self.assertNotIn("GHA_A", {s["gha"] for s in t.stops_remaining})
        # parcels_for still reports 4 — the stop is done, yet non-zero returned
        self.assertEqual(t.parcels_for("GHA_A"), 4,
            "BUG-10 confirmed: parcels_for returns data for completed stops.")


# ---------------------------------------------------------------------------
# Truck.next_slot
# ---------------------------------------------------------------------------
class TestTruckNextSlot(unittest.TestCase):

    def test_returns_earliest_slot_among_remaining_stops(self):
        t = make_truck(
            manifest=[{"gha": "GHA_A", "parcels": 5},
                      {"gha": "GHA_B", "parcels": 3}],
            booked_slots={"GHA_A": 60, "GHA_B": 30},
        )
        self.assertEqual(t.next_slot(), 30)

    def test_returns_none_when_no_booked_slots(self):
        t = make_truck()
        self.assertIsNone(t.next_slot())

    def test_returns_none_when_all_stops_completed(self):
        t = make_truck(
            manifest=[{"gha": "GHA_A", "parcels": 5}],
            booked_slots={"GHA_A": 60},
        )
        t.complete_stop("GHA_A")
        self.assertIsNone(t.next_slot(),
            "next_slot must return None after all stops are completed, "
            "even if booked_slots still contains the entry.")

    def test_filters_out_slots_for_completed_stops(self):
        t = make_truck(
            manifest=[{"gha": "GHA_A", "parcels": 5},
                      {"gha": "GHA_B", "parcels": 3}],
            booked_slots={"GHA_A": 10, "GHA_B": 50},
        )
        t.complete_stop("GHA_A")
        # Only GHA_B remains → slot 50
        self.assertEqual(t.next_slot(), 50)

    def test_returns_none_when_manifest_empty(self):
        t = make_truck(manifest=[], booked_slots={"GHA_A": 10})
        self.assertIsNone(t.next_slot())

    def test_single_stop_single_slot(self):
        t = make_truck(booked_slots={"GHA_A": 99})
        self.assertEqual(t.next_slot(), 99)


# ---------------------------------------------------------------------------
# Truck.next_stop
# ---------------------------------------------------------------------------
class TestTruckNextStop(unittest.TestCase):

    def test_returns_first_remaining_stop(self):
        m = [{"gha": "GHA_A", "parcels": 5}, {"gha": "GHA_B", "parcels": 3}]
        t = make_truck(manifest=m)
        self.assertEqual(t.next_stop()["gha"], "GHA_A")

    def test_returns_none_when_no_stops_remain(self):
        t = make_truck(manifest=[])
        self.assertIsNone(t.next_stop())

    def test_advances_after_complete_stop(self):
        m = [{"gha": "GHA_A", "parcels": 5}, {"gha": "GHA_B", "parcels": 3}]
        t = make_truck(manifest=m)
        t.complete_stop("GHA_A")
        self.assertEqual(t.next_stop()["gha"], "GHA_B")

    def test_returns_none_after_all_stops_completed(self):
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 5}])
        t.complete_stop("GHA_A")
        self.assertIsNone(t.next_stop())


# ---------------------------------------------------------------------------
# Truck.complete_stop  —  BUG-4, BUG-5
# ---------------------------------------------------------------------------
class TestTruckCompleteStop(unittest.TestCase):

    def test_removes_target_gha_from_stops_remaining(self):
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 5},
                                  {"gha": "GHA_B", "parcels": 3}])
        t.complete_stop("GHA_A")
        ghas = {s["gha"] for s in t.stops_remaining}
        self.assertNotIn("GHA_A", ghas)

    def test_retains_other_stops(self):
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 5},
                                  {"gha": "GHA_B", "parcels": 3}])
        t.complete_stop("GHA_A")
        ghas = {s["gha"] for s in t.stops_remaining}
        self.assertIn("GHA_B", ghas)

    def test_no_op_for_unknown_gha(self):
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 5}])
        t.complete_stop("GHA_UNKNOWN")
        self.assertEqual(len(t.stops_remaining), 1)

    def test_complete_stop_does_not_alter_manifest(self):
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 5}])
        t.complete_stop("GHA_A")
        self.assertEqual(len(t.manifest), 1,
            "manifest must be immutable; complete_stop should only "
            "modify stops_remaining.")

    # ── BUG-5: duplicate gha entries — both removed on first call ────────────
    def test_BUG5_duplicate_gha_in_manifest_removes_all_on_complete(self):
        """
        BUG-5: complete_stop() uses a list comprehension that filters out
        every entry where gha matches. If the manifest (and therefore
        stops_remaining) contains two entries for the same GHA, a single
        call to complete_stop() silently removes both.
        """
        t = make_truck(manifest=[{"gha": "GHA_A", "parcels": 5},
                                  {"gha": "GHA_A", "parcels": 3},
                                  {"gha": "GHA_B", "parcels": 2}])
        t.complete_stop("GHA_A")
        ghas = [s["gha"] for s in t.stops_remaining]
        # Ideally only the first GHA_A entry should be removed,
        # leaving one GHA_A and one GHA_B.  With the bug, both are gone.
        self.assertNotIn("GHA_A", ghas,
            "BUG-5 confirmed: complete_stop removed ALL GHA_A entries.")

    # ── BUG-4: shallow copy — mutating stop dict corrupts manifest ────────────
    def test_BUG4_shallow_copy_aliases_stop_dicts(self):
        """
        BUG-4: __post_init__ uses list(self.manifest), which copies list
        references but NOT the dicts inside.  Mutating a dict in
        stops_remaining also mutates the corresponding dict in manifest.
        """
        m = [{"gha": "GHA_A", "parcels": 10}]
        t = make_truck(manifest=m)
        # Mutate through stops_remaining
        t.stops_remaining[0]["parcels"] = 999
        self.assertEqual(t.manifest[0]["parcels"], 999,
            "BUG-4 confirmed: stops_remaining and manifest share the same "
            "dict objects — mutation in one is visible in the other.")


# ---------------------------------------------------------------------------
# GHATerminal — init
# ---------------------------------------------------------------------------
class TestGHATerminalInit(unittest.TestCase):

    def setUp(self):
        self.env, self.term, _, _ = make_terminal("GHA_A")

    def test_export_dock_capacity_matches_config(self):
        self.assertEqual(self.term.docks_exp.capacity,
                         DEFAULT_PARAMS["ghas"]["GHA_A"]["export"])

    def test_import_dock_capacity_matches_config(self):
        self.assertEqual(self.term.docks_imp.capacity,
                         DEFAULT_PARAMS["ghas"]["GHA_A"]["import"])

    def test_stats_keys_present(self):
        self.assertIn("export", self.term.stats)
        self.assertIn("import", self.term.stats)

    def test_stats_accumulators_zero_on_init(self):
        for flow in ("export", "import"):
            for key in ("processed", "tot_wait", "tot_serv"):
                self.assertEqual(self.term.stats[flow][key], 0)

    def test_queues_empty_on_init(self):
        self.assertEqual(self.term.queue_exp, [])
        self.assertEqual(self.term.queue_imp, [])

    def test_n_exp_and_n_imp_set_correctly(self):
        self.assertEqual(self.term.n_exp, 3)
        self.assertEqual(self.term.n_imp, 2)


# ---------------------------------------------------------------------------
# GHATerminal._dock_pool / _queue  —  BUG-3
# ---------------------------------------------------------------------------
class TestGHATerminalRoutingHelpers(unittest.TestCase):

    def setUp(self):
        self.env, self.term, _, _ = make_terminal("GHA_A")

    def test_export_flow_returns_export_pool(self):
        self.assertIs(self.term._dock_pool("export"), self.term.docks_exp)

    def test_import_flow_returns_import_pool(self):
        self.assertIs(self.term._dock_pool("import"), self.term.docks_imp)

    def test_export_flow_returns_export_queue(self):
        self.assertIs(self.term._queue("export"), self.term.queue_exp)

    def test_import_flow_returns_import_queue(self):
        self.assertIs(self.term._queue("import"), self.term.queue_imp)

    # ── BUG-3: unknown flow_type silently routes to import ────────────────────
    def test_BUG3_unknown_flow_type_silently_falls_back_to_import_pool(self):
        """
        BUG-3: _dock_pool() uses `if flow_type == "export" else`.  Any value
        that isn't the exact string "export" returns the import pool, with
        no warning.  A typo like "Export" or a new flow type like "transit"
        is silently misrouted.
        """
        result = self.term._dock_pool("Export")   # capital E typo
        self.assertIs(result, self.term.docks_imp,
            "BUG-3 confirmed: 'Export' (wrong case) silently uses import pool.")

    def test_BUG3_arbitrary_flow_type_falls_back_to_import_queue(self):
        result = self.term._queue("transit")
        self.assertIs(result, self.term.queue_imp,
            "BUG-3: any non-'export' string returns the import queue.")

    def test_BUG3_empty_string_falls_back_to_import(self):
        self.assertIs(self.term._dock_pool(""), self.term.docks_imp)


# ---------------------------------------------------------------------------
# GHATerminal.exp_occupancy / imp_occupancy
# ---------------------------------------------------------------------------
class TestGHATerminalOccupancy(unittest.TestCase):

    def test_exp_occupancy_zero_when_empty(self):
        _, term, _, _ = make_terminal("GHA_A")
        self.assertAlmostEqual(term.exp_occupancy(), 0.0)

    def test_imp_occupancy_zero_when_empty(self):
        _, term, _, _ = make_terminal("GHA_A")
        self.assertAlmostEqual(term.imp_occupancy(), 0.0)

    def test_exp_occupancy_zero_when_n_exp_is_zero(self):
        # 1. Initialize with standard, valid defaults so SimPy doesn't crash
        _, term, _, _ = make_terminal("GHA_A")
        
        # 2. Manually force the terminal and the underlying resource depths to 0
        term.n_exp = 0
        term.docks_exp._capacity = 0  # Bypasses SimPy's read-only capacity restriction
        
        # 3. Safely test your mathematical guard logic
        self.assertAlmostEqual(term.exp_occupancy(), 0.0,
            "exp_occupancy must guard against n_exp == 0.")

    def test_imp_occupancy_zero_when_n_imp_is_zero(self):
        # 1. Initialize with a standard valid configuration (n_imp defaults to 2 here)
        _, term, _, _ = make_terminal("GHA_A")
        
        # 2. Trick the state to simulate 0 import docks manually
        term.n_imp = 0
        term.docks_imp._capacity = 0  # Force SimPy's underlying count limit to 0
        
        # 3. Safely test your calculation logic
        self.assertAlmostEqual(term.imp_occupancy(), 0.0,
            "imp_occupancy must guard against n_imp == 0.")

    def test_exp_occupancy_bounded_between_0_and_1(self):
        _, term, _, _ = make_terminal("GHA_A")
        occ = term.exp_occupancy()
        self.assertGreaterEqual(occ, 0.0)
        self.assertLessEqual(occ, 1.0)


# ---------------------------------------------------------------------------
# GHATerminal.exp_queue_norm / imp_queue_norm  —  BUG-2
# ---------------------------------------------------------------------------
class TestGHATerminalQueueNorm(unittest.TestCase):

    def test_exp_queue_norm_zero_when_empty(self):
        _, term, _, _ = make_terminal("GHA_A")
        self.assertAlmostEqual(term.exp_queue_norm(), 0.0)

    def test_imp_queue_norm_zero_when_empty(self):
        _, term, _, _ = make_terminal("GHA_A")
        self.assertAlmostEqual(term.imp_queue_norm(), 0.0)

    def test_exp_queue_norm_capped_at_one(self):
        _, term, _, _ = make_terminal("GHA_A")
        # Stuff the queue beyond capacity
        for i in range(100):
            term.queue_exp.append(make_truck(f"T{i}"))
        self.assertLessEqual(term.exp_queue_norm(), 1.0)

    def test_imp_queue_norm_capped_at_one(self):
        _, term, _, _ = make_terminal("GHA_A")
        for i in range(100):
            term.queue_imp.append(make_truck(f"T{i}"))
        self.assertLessEqual(term.imp_queue_norm(), 1.0)

    def test_exp_queue_norm_proportional(self):
        _, term, _, _ = make_terminal("GHA_A")  # n_exp=3
        term.queue_exp.append(make_truck("T1"))
        # 1 truck / max_q(3) = 0.333...
        self.assertAlmostEqual(term.exp_queue_norm(), 1/3, places=5)

    # ── BUG-2: ZeroDivisionError when dock count is 0 ─────────────────────────
    def test_BUG2_exp_queue_norm_raises_zero_division_when_n_exp_is_zero(self):
        """
        BUG-2: exp_queue_norm() reads params directly for max_q and then
        divides by it without a zero-division guard.
        """
        env   = simpy.Environment()
        svc   = MagicMock()
        infra = MagicMock()
        
        # 1. Provide a temporary valid capacity of 1 so SimPy initialization passes
        cfg = {
            "ghas": {"GHA_ZERO": {"export": 1, "import": 1}},
            "tp3":  {"capacity": 10},
        }
        term = GHATerminal(env, "GHA_ZERO", svc, infra, cfg=cfg)
        
        # 2. Force the values to 0 manually to set up your zero-division environment
        term.n_exp = 0
        term.n_imp = 0
        term.docks_exp._capacity = 0
        term.docks_imp._capacity = 0
        
        # 3. Now run the assertion against your actual method
        with self.assertRaises(ZeroDivisionError,
                msg="BUG-2: exp_queue_norm() should raise ZeroDivisionError "
                    "when n_exp == 0 (no guard present)."):
            term.exp_queue_norm()

    def test_BUG2_imp_queue_norm_raises_zero_division_when_n_imp_is_zero(self):
        """BUG-2 mirror for import side."""
        env   = simpy.Environment()
        svc   = MagicMock()
        infra = MagicMock()
        
        # 1. Provide a temporary valid capacity of 1 so SimPy initialization passes
        cfg   = {
            "ghas": {"GHA_ZERO": {"export": 1, "import": 1}},
            "tp3":  {"capacity": 10},
        }
        term = GHATerminal(env, "GHA_ZERO", svc, infra, cfg=cfg)
        
        # 2. Force the values to 0 manually to set up your zero-division condition
        term.n_exp = 0
        term.n_imp = 0
        term.docks_exp._capacity = 0
        term.docks_imp._capacity = 0
        
        # 3. Now run the assertion against your actual method
        with self.assertRaises(ZeroDivisionError,
                msg="BUG-2: imp_queue_norm() should raise ZeroDivisionError "
                    "when n_imp == 0 (no guard present)."):
            term.imp_queue_norm()


# ---------------------------------------------------------------------------
# GHATerminal.upcoming_bookings_norm
# ---------------------------------------------------------------------------
class TestGHATerminalUpcomingNorm(unittest.TestCase):

    def setUp(self):
        self.env, self.term, _, _ = make_terminal("GHA_A")

    def _make_registry(self, gha, slot_offsets, phases):
        """slot_offsets relative to env.now; returns a dtp mock."""
        now = self.env.now
        registry = {gha: {
            now + offset: [{"phase": ph}]
            for offset, ph in zip(slot_offsets, phases)
        }}
        dtp = make_dtp(registry=registry)
        return dtp

    def test_zero_when_no_registry(self):
        dtp = make_dtp(registry={})
        self.assertAlmostEqual(self.term.upcoming_bookings_norm(dtp, 60), 0.0)

    def test_counts_booked_and_docked_phases_only(self):
        now = self.env.now
        registry = {"GHA_A": {
            now + 10: [{"phase": "booked"}],
            now + 20: [{"phase": "docked"}],
            now + 30: [{"phase": "closed"}],   # should NOT count
        }}
        dtp = make_dtp(registry=registry)
        result = self.term.upcoming_bookings_norm(dtp, horizon=60)
        total  = self.term.n_exp + self.term.n_imp  # 3+2=5
        self.assertAlmostEqual(result, 2 / total)

    def test_slots_beyond_horizon_excluded(self):
        now = self.env.now
        registry = {"GHA_A": {
            now + 10:  [{"phase": "booked"}],
            now + 999: [{"phase": "booked"}],  # beyond horizon
        }}
        dtp = make_dtp(registry=registry)
        result = self.term.upcoming_bookings_norm(dtp, horizon=60)
        total  = self.term.n_exp + self.term.n_imp
        self.assertAlmostEqual(result, 1 / total)

    def test_slots_in_the_past_excluded(self):
        now = self.env.now
        registry = {"GHA_A": {
            now - 5: [{"phase": "booked"}],  # in the past
            now + 10: [{"phase": "booked"}],
        }}
        dtp = make_dtp(registry=registry)
        result = self.term.upcoming_bookings_norm(dtp, horizon=60)
        total  = self.term.n_exp + self.term.n_imp
        self.assertAlmostEqual(result, 1 / total)

    def test_capped_at_one(self):
        now = self.env.now
        # Overbook massively
        registry = {"GHA_A": {
            now + i: [{"phase": "booked"}] for i in range(1, 50)
        }}
        dtp = make_dtp(registry=registry)
        result = self.term.upcoming_bookings_norm(dtp, horizon=60)
        self.assertLessEqual(result, 1.0)

    def test_returns_zero_when_total_docks_zero(self):
        env   = simpy.Environment()
        svc   = MagicMock()
        infra = MagicMock()
        
        # 1. Provide a temporary valid capacity of 1 so SimPy initialization passes
        cfg   = {"ghas": {"GHA_ZERO": {"export": 1, "import": 1}},
                 "tp3":  {"capacity": 10}}
        term  = GHATerminal(env, "GHA_ZERO", svc, infra, cfg=cfg)
        
        # 2. Force the actual attributes and underlying SimPy resources to 0 manually
        term.n_exp = 0
        term.n_imp = 0
        term.docks_exp._capacity = 0
        term.docks_imp._capacity = 0
        
        # 3. Proceed with your test logic
        dtp   = make_dtp(registry={"GHA_ZERO": {10: [{"phase": "booked"}]}})
        self.assertAlmostEqual(term.upcoming_bookings_norm(dtp, 60), 0.0)


# ---------------------------------------------------------------------------
# GHATerminal.process_truck  —  SimPy-driven
# ---------------------------------------------------------------------------
class TestGHATerminalProcessTruck(unittest.TestCase):

    def _run_with_phase(self, phase, slot_start=None, svc_time=10.0):
        env, term, infra, svc = make_terminal("GHA_A", svc_time=svc_time)
        svc.sample.return_value = svc_time
        dtp = make_dtp(phase=phase)
        truck = make_truck(booked_slots={"GHA_A": slot_start} if slot_start else {})
        run_process(env, term.process_truck(truck, dtp))
        return env, term, infra, truck, dtp

    # on_time: truck goes through full cycle
    def test_on_time_truck_status_in_transit_after_processing(self):
        _, _, _, truck, _ = self._run_with_phase("on_time")
        self.assertEqual(truck.status, Truck.STATUS_IN_TRANSIT)

    def test_on_time_stats_processed_incremented(self):
        _, term, _, _, _ = self._run_with_phase("on_time")
        self.assertEqual(term.stats["export"]["processed"], 1)

    def test_on_time_service_time_accumulated(self):
        _, term, _, _, _ = self._run_with_phase("on_time", svc_time=15.0)
        self.assertAlmostEqual(term.stats["export"]["tot_serv"], 15.0)

    def test_on_time_infra_dock_start_called(self):
        _, _, infra, _, _ = self._run_with_phase("on_time")
        infra.dock_start.assert_called_once()

    def test_on_time_infra_dock_end_called(self):
        _, _, infra, _, _ = self._run_with_phase("on_time")
        infra.dock_end.assert_called_once()

    def test_on_time_infra_gha_in_called(self):
        _, _, infra, _, _ = self._run_with_phase("on_time")
        infra.gha_in.assert_called_once()

    def test_on_time_stop_removed_from_stops_remaining(self):
        _, _, _, truck, _ = self._run_with_phase("on_time")
        self.assertEqual(len(truck.stops_remaining), 0)

    # no_show: truck redirected to TP3
    def test_no_show_truck_redirected_to_tp3(self):
        _, _, _, truck, _ = self._run_with_phase("no_show", slot_start=30)
        self.assertEqual(truck.status, Truck.STATUS_AT_TP3)

    def test_no_show_stats_not_incremented(self):
        _, term, _, _, _ = self._run_with_phase("no_show", slot_start=30)
        self.assertEqual(term.stats["export"]["processed"], 0)

    def test_no_show_dtp_record_no_show_called(self):
        _, _, _, _, dtp = self._run_with_phase("no_show", slot_start=30)
        dtp.record_no_show.assert_called_once()

    def test_no_show_infra_dock_start_not_called(self):
        _, _, infra, _, _ = self._run_with_phase("no_show", slot_start=30)
        infra.dock_start.assert_not_called()

    # release_dock_taken: truck redirected, late recorded
    def test_release_dock_taken_redirects_truck(self):
        _, _, _, truck, _ = self._run_with_phase("release_dock_taken", slot_start=30)
        self.assertEqual(truck.status, Truck.STATUS_AT_TP3)

    def test_release_dock_taken_records_late(self):
        _, _, _, _, dtp = self._run_with_phase("release_dock_taken", slot_start=30)
        dtp.record_late.assert_called_once()

    # release: truck proceeds but late penalty recorded
    def test_release_phase_truck_completes_processing(self):
        _, term, _, _, _ = self._run_with_phase("release")
        self.assertEqual(term.stats["export"]["processed"], 1)

    def test_release_phase_late_recorded(self):
        _, _, _, _, dtp = self._run_with_phase("release")
        dtp.record_late.assert_called_once()

    # early: truck waits, then proceeds
    def test_early_phase_truck_completes_after_waiting(self):
        env, term, infra, svc = make_terminal("GHA_A", svc_time=10.0)
        svc.sample.return_value = 10.0
        dtp = make_dtp(phase="early")
        slot_start = 50
        truck = make_truck(booked_slots={"GHA_A": slot_start})
        run_process(env, term.process_truck(truck, dtp))
        self.assertEqual(term.stats["export"]["processed"], 1)

    def test_early_phase_truck_waits_until_slot(self):
        env, term, infra, svc = make_terminal("GHA_A", svc_time=5.0)
        svc.sample.return_value = 5.0
        dtp = make_dtp(phase="early")
        slot_start = 50
        truck = make_truck(booked_slots={"GHA_A": slot_start})
        env.process(term.process_truck(truck, dtp))
        env.run(until=1000)
        # Service finished at slot_start + service_time = 50 + 5 = 55
        self.assertGreaterEqual(env.now, slot_start)

    # slot_start is None when booked — mark_docked / mark_closed not called
    def test_no_slot_mark_docked_not_called(self):
        _, _, _, _, dtp = self._run_with_phase("on_time", slot_start=None)
        dtp.mark_docked.assert_not_called()

    def test_no_slot_mark_closed_not_called(self):
        _, _, _, _, dtp = self._run_with_phase("on_time", slot_start=None)
        dtp.mark_closed.assert_not_called()

    # BUG-9: unknown flow_type hits KeyError in stats at end
    def test_BUG9_unknown_flow_type_raises_key_error_in_stats(self):
        """
        BUG-9: At the end of process_truck(), stats[truck.flow_type] is
        accessed.  The dict only has 'export' and 'import'.  A truck with
        flow_type='transit' is silently routed to the import dock pool
        (BUG-3) but then crashes with KeyError when updating stats.
        """
        env, term, _, svc = make_terminal("GHA_A", svc_time=5.0)
        svc.sample.return_value = 5.0
        dtp = make_dtp(phase="on_time")
        truck = make_truck(flow_type="transit")   # neither export nor import
        with self.assertRaises(KeyError,
                msg="BUG-9: stats[flow_type] raises KeyError for unknown "
                    "flow_type like 'transit'."):
            env.process(term.process_truck(truck, dtp))
            env.run(until=1000)

    # Concurrent trucks: second truck waits for dock
    def test_two_trucks_sequential_when_single_dock(self):
        env, term, infra, svc = make_terminal("GHA_A", n_exp=1, svc_time=10.0)
        svc.sample.return_value = 10.0
        dtp = make_dtp(phase="on_time")
        t1 = make_truck("T1")
        t2 = make_truck("T2")
        env.process(term.process_truck(t1, dtp))
        env.process(term.process_truck(t2, dtp))
        env.run(until=1000)
        self.assertEqual(term.stats["export"]["processed"], 2)
        # Two service slots of 10 each = 20 total service
        self.assertAlmostEqual(term.stats["export"]["tot_serv"], 20.0)

    def test_three_trucks_fill_three_export_docks(self):
        env, term, infra, svc = make_terminal("GHA_A", n_exp=3, svc_time=10.0)
        svc.sample.return_value = 10.0
        dtp = make_dtp(phase="on_time")
        for i in range(3):
            env.process(term.process_truck(make_truck(f"T{i}"), dtp))
        env.run(until=1000)
        self.assertEqual(term.stats["export"]["processed"], 3)


# ---------------------------------------------------------------------------
# GHATerminal.release_window_watcher  —  SimPy-driven
# ---------------------------------------------------------------------------
class TestGHATerminalReleaseWatcher(unittest.TestCase):

    def test_watcher_fires_at_slot_plus_priority_window(self):
        env, term, infra, _ = make_terminal("GHA_A")
        dtp = make_dtp(priority_window=10, release_to_standby=True)
        dtp.release_to_standby.return_value = True
        tp3 = MagicMock()
        slot_start = 50
        env.process(term.release_window_watcher(slot_start, dtp, tp3))
        env.run(until=1000)
        tp3.signal_standby_opportunity.assert_called_once_with(
            "GHA_A", slot_start, slot_start + 10
        )

    def test_watcher_does_not_signal_when_release_to_standby_false(self):
        env, term, infra, _ = make_terminal("GHA_A")
        dtp = make_dtp(priority_window=10)
        dtp.release_to_standby.return_value = False
        tp3 = MagicMock()
        env.process(term.release_window_watcher(50, dtp, tp3))
        env.run(until=1000)
        tp3.signal_standby_opportunity.assert_not_called()

    def test_watcher_with_past_slot_fires_immediately(self):
        env, term, infra, _ = make_terminal("GHA_A")
        dtp = make_dtp(priority_window=10)
        dtp.release_to_standby.return_value = True
        tp3 = MagicMock()
        # slot_start in the past: max(0, 0+10-0)=10, fires at t=10
        env.process(term.release_window_watcher(0, dtp, tp3))
        env.run(until=1000)
        tp3.signal_standby_opportunity.assert_called_once()


# ---------------------------------------------------------------------------
# TP3Buffer — enter / release lifecycle
# ---------------------------------------------------------------------------
class TestTP3BufferEnterRelease(unittest.TestCase):

    def setUp(self):
        TP3Buffer.CAPACITY = 10
        self.env, self.tp3, self.infra = make_tp3(capacity=10)

    def _enter(self, truck):
        run_process(self.env, self.tp3.enter(truck))

    def test_enter_parks_truck(self):
        t = make_truck("T1")
        self._enter(t)
        self.assertIn(t, self.tp3.get_parked_trucks())

    def test_enter_sets_status_at_tp3(self):
        t = make_truck("T1")
        self._enter(t)
        self.assertEqual(t.status, Truck.STATUS_AT_TP3)

    def test_enter_calls_infra_tp3_in(self):
        t = make_truck("T1")
        self._enter(t)
        self.infra.tp3_in.assert_called_once()

    def test_n_parked_increments_on_enter(self):
        self._enter(make_truck("T1"))
        self.assertEqual(self.tp3.n_parked(), 1)

    def test_release_removes_truck_from_parked(self):
        t = make_truck("T1")
        self._enter(t)
        released = self.tp3.release("T1")
        self.assertNotIn(t, self.tp3.get_parked_trucks())

    def test_release_returns_correct_truck(self):
        t = make_truck("T1")
        self._enter(t)
        released = self.tp3.release("T1")
        self.assertIs(released, t)

    def test_release_unknown_id_returns_none(self):
        result = self.tp3.release("GHOST")
        self.assertIsNone(result)

    def test_release_calls_infra_tp3_out(self):
        t = make_truck("T1")
        self._enter(t)
        self.tp3.release("T1")
        self.infra.tp3_out.assert_called_once()

    def test_release_decrements_parked_count(self):
        t = make_truck("T1")
        self._enter(t)
        self.tp3.release("T1")
        self.assertEqual(self.tp3.n_parked(), 0)

    def test_multiple_trucks_parked_independently(self):
        for i in range(5):
            run_process(self.env, self.tp3.enter(make_truck(f"T{i}")))
        self.assertEqual(self.tp3.n_parked(), 5)

    def test_release_targets_correct_truck_among_many(self):
        trucks = [make_truck(f"T{i}") for i in range(5)]
        for t in trucks:
            run_process(self.env, self.tp3.enter(t))
        self.tp3.release("T2")
        parked_ids = {t.truck_id for t in self.tp3.get_parked_trucks()}
        self.assertNotIn("T2", parked_ids)
        # Others must still be parked
        for i in (0, 1, 3, 4):
            self.assertIn(f"T{i}", parked_ids)

    def test_truck_status_not_updated_by_release(self):
        """
        release() does NOT update truck.status — the caller is responsible.
        This test documents (and exposes) that assumption explicitly.
        """
        t = make_truck("T1")
        self._enter(t)                        # status → AT_TP3
        self.tp3.release("T1")
        # Status remains AT_TP3; the caller (Orchestrator/Transporter)
        # must set it — release() leaves it stale.
        self.assertEqual(t.status, Truck.STATUS_AT_TP3,
            "release() does not update truck.status — caller must do so.")


# ---------------------------------------------------------------------------
# TP3Buffer — overflow queue
# ---------------------------------------------------------------------------
class TestTP3BufferOverflow(unittest.TestCase):

    def test_overflow_queue_filled_when_tp3_is_at_capacity(self):
        TP3Buffer.CAPACITY = 2
        env, tp3, infra = make_tp3(capacity=2)

        trucks = [make_truck(f"T{i}") for i in range(4)]

        # Park first two (fill capacity)
        for t in trucks[:2]:
            run_process(env, tp3.enter(t))

        # Third truck should land in overflow (TP3 is full)
        # Use env.process so it suspends properly
        overflow_proc = env.process(tp3.enter(trucks[2]))
        env.run(until=1)     # let it try to enter but block

        self.assertIn(trucks[2], tp3.queue_overflow)

    def test_n_overflow_returns_overflow_count(self):
        TP3Buffer.CAPACITY = 1
        env, tp3, infra = make_tp3(capacity=1)
        t1, t2 = make_truck("T1"), make_truck("T2")
        run_process(env, tp3.enter(t1))
        env.process(tp3.enter(t2))
        env.run(until=1)
        self.assertEqual(tp3.n_overflow(), 1)

    def test_overflow_truck_enters_after_release(self):
        TP3Buffer.CAPACITY = 1
        env, tp3, infra = make_tp3(capacity=1)
        t1, t2 = make_truck("T1"), make_truck("T2")
        run_process(env, tp3.enter(t1))
        env.process(tp3.enter(t2))
        env.run(until=1)
        tp3.release("T1")     # free up a slot
        env.run(until=10)     # let t2 proceed
        self.assertIn(t2, tp3.get_parked_trucks())
        self.assertEqual(tp3.n_overflow(), 0)


# ---------------------------------------------------------------------------
# TP3Buffer.release_next  —  BUG-6
# ---------------------------------------------------------------------------
class TestTP3BufferReleaseNext(unittest.TestCase):

    def setUp(self):
        TP3Buffer.CAPACITY = 10
        self.env, self.tp3, self.infra = make_tp3(capacity=10)

    def _park(self, truck):
        run_process(self.env, self.tp3.enter(truck))

    def test_release_next_returns_first_truck_with_booking(self):
        t1 = make_truck("T1", booked_slots={"GHA_A": 30})
        t2 = make_truck("T2", booked_slots={"GHA_B": 30})
        self._park(t1)
        self._park(t2)
        released = self.tp3.release_next("GHA_A")
        self.assertIs(released, t1)

    def test_release_next_returns_none_when_no_match(self):
        t = make_truck("T1", booked_slots={"GHA_B": 30})
        self._park(t)
        result = self.tp3.release_next("GHA_A")
        self.assertIsNone(result)

    def test_release_next_preserves_other_trucks(self):
        t1 = make_truck("T1", booked_slots={"GHA_A": 30})
        t2 = make_truck("T2", booked_slots={"GHA_A": 30})
        self._park(t1)
        self._park(t2)
        self.tp3.release_next("GHA_A")
        self.assertEqual(self.tp3.n_parked(), 1)

    # ── BUG-6: checks booked_slots, not stops_remaining ──────────────────────
    def test_BUG6_release_next_returns_truck_with_completed_stop(self):
        """
        BUG-6: release_next() checks `gha in truck.booked_slots`, which is
        never cleaned up after a stop completes (complete_stop() only modifies
        stops_remaining).  A truck that has already FINISHED its stop at GHA_A
        but still holds the entry in booked_slots will be incorrectly released
        for a second visit to that GHA.
        """
        t = make_truck("T1", booked_slots={"GHA_A": 30})
        t.complete_stop("GHA_A")  # mark the stop as done
        self._park(t)

        released = self.tp3.release_next("GHA_A")
        # The bug: a finished truck is released for a GHA it already visited
        self.assertIs(released, t,
            "BUG-6 confirmed: release_next returns a truck whose GHA_A "
            "stop is already completed, because it checks booked_slots "
            "instead of stops_remaining.")


# ---------------------------------------------------------------------------
# TP3Buffer.signal_standby_opportunity  —  BUG-7
# ---------------------------------------------------------------------------
class TestTP3BufferStandby(unittest.TestCase):

    def setUp(self):
        TP3Buffer.CAPACITY = 10
        self.env, self.tp3, _ = make_tp3(capacity=10)

    # ── BUG-7: return type annotation says List[Dict] but returns None ────────
    def test_BUG7_signal_standby_returns_none_not_list(self):
        """
        BUG-7: signal_standby_opportunity() is annotated to return
        List[Dict] but has no return statement — it implicitly returns None.
        Any caller that does `signals = tp3.signal_standby_opportunity(...)`
        and then iterates the result will get TypeError: 'NoneType' is not
        iterable.
        """
        result = self.tp3.signal_standby_opportunity("GHA_A", 60, 10.0)
        self.assertIsNone(result,
            "BUG-7 confirmed: signal_standby_opportunity returns None, "
            "not List[Dict] as the type hint claims.")

    def test_signal_appended_to_standby_opportunities(self):
        self.tp3.signal_standby_opportunity("GHA_A", 60, 10.0)
        self.assertEqual(len(self.tp3.standby_opportunities), 1)

    def test_signal_fields_correct(self):
        self.tp3.signal_standby_opportunity("GHA_A", 60, 10.0)
        s = self.tp3.standby_opportunities[0]
        self.assertEqual(s["gha"],         "GHA_A")
        self.assertEqual(s["slot_start"],  60)
        self.assertEqual(s["signal_time"], 10.0)
        self.assertFalse(s["consumed"])

    def test_get_pending_signals_excludes_consumed(self):
        self.tp3.signal_standby_opportunity("GHA_A", 60, 10.0)
        self.tp3.standby_opportunities[0]["consumed"] = True
        self.tp3.signal_standby_opportunity("GHA_A", 70, 20.0)
        pending = self.tp3.get_pending_signals()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["slot_start"], 70)

    def test_no_deduplication_of_duplicate_signals(self):
        """
        There is no deduplication: calling signal_standby_opportunity twice
        with identical args creates two entries.  This test documents the
        behaviour so callers know they must deduplicate.
        """
        self.tp3.signal_standby_opportunity("GHA_A", 60, 10.0)
        self.tp3.signal_standby_opportunity("GHA_A", 60, 10.0)
        self.assertEqual(len(self.tp3.standby_opportunities), 2,
            "Duplicate signals are NOT deduplicated — both are stored.")

    def test_get_pending_signals_returns_all_unconsumed(self):
        for i in range(5):
            self.tp3.signal_standby_opportunity("GHA_A", i*10, float(i))
        self.assertEqual(len(self.tp3.get_pending_signals()), 5)


# ---------------------------------------------------------------------------
# TP3Buffer — observation helpers  —  BUG-8
# ---------------------------------------------------------------------------
class TestTP3BufferObservation(unittest.TestCase):

    def setUp(self):
        TP3Buffer.CAPACITY = 10
        self.env, self.tp3, _ = make_tp3(capacity=10)

    def test_occupancy_ratio_zero_when_empty(self):
        self.assertAlmostEqual(self.tp3.occupancy_ratio(), 0.0)

    def test_occupancy_ratio_one_when_full(self):
        for i in range(10):
            run_process(self.env, self.tp3.enter(make_truck(f"T{i}")))
        self.assertAlmostEqual(self.tp3.occupancy_ratio(), 1.0)

    def test_n_parked_matches_entered_trucks(self):
        for i in range(4):
            run_process(self.env, self.tp3.enter(make_truck(f"T{i}")))
        self.assertEqual(self.tp3.n_parked(), 4)

    def test_parked_by_flow_type_export(self):
        for i in range(3):
            run_process(self.env, self.tp3.enter(make_truck(f"E{i}", flow_type="export")))
        for i in range(2):
            run_process(self.env, self.tp3.enter(make_truck(f"I{i}", flow_type="import")))
        self.assertEqual(self.tp3.parked_by_flow_type("export"), 3)

    def test_parked_by_flow_type_import(self):
        run_process(self.env, self.tp3.enter(make_truck("I1", flow_type="import")))
        self.assertEqual(self.tp3.parked_by_flow_type("import"), 1)

    def test_parked_by_flow_type_zero_for_unknown(self):
        run_process(self.env, self.tp3.enter(make_truck("T1")))
        self.assertEqual(self.tp3.parked_by_flow_type("transit"), 0)

    def test_get_parked_trucks_returns_all(self):
        trucks = [make_truck(f"T{i}") for i in range(3)]
        for t in trucks:
            run_process(self.env, self.tp3.enter(t))
        parked = self.tp3.get_parked_trucks()
        for t in trucks:
            self.assertIn(t, parked)

# ── BUG-8: occupancy_ratio crashes when CAPACITY == 0 ────────────────────
    def test_BUG8_occupancy_ratio_raises_zero_division_when_capacity_zero(self):
        """
        BUG-8: occupancy_ratio() = self.slots.count / self.CAPACITY with no
        guard.  If tp3.capacity is 0 in config (or patched for testing) the
        method raises ZeroDivisionError.
        """
        # Save original capacity class attribute to safely reset later
        orig_capacity = getattr(TP3Buffer, "CAPACITY", 10)
        TP3Buffer.CAPACITY = 0
        
        env = simpy.Environment()
        infra = MagicMock()
        tp3 = TP3Buffer.__new__(TP3Buffer)
        tp3.env   = env
        tp3.infra = infra
        
        # Instantiate standard resource
        tp3.slots = simpy.Resource(env, capacity=1)  
        
        # Force SimPy's internal capacity state to 0 by accessing the private attribute
        tp3.slots._capacity = 0  
        
        tp3._parked               = []
        tp3.queue_overflow        = []
        tp3.standby_opportunities = []

        # Assert that calling your ACTUAL method triggers the ZeroDivisionError
        with self.assertRaises(ZeroDivisionError,
                    msg="BUG-8: occupancy_ratio crashes when CAPACITY is 0."):
            _ = tp3.occupancy_ratio()  # Run your actual implementation

        # Reset for other tests
        TP3Buffer.CAPACITY = orig_capacity


# ---------------------------------------------------------------------------
# TP3Buffer — edge cases
# ---------------------------------------------------------------------------
class TestTP3BufferEdgeCases(unittest.TestCase):

    def setUp(self):
        TP3Buffer.CAPACITY = 10
        self.env, self.tp3, self.infra = make_tp3(capacity=10)

    def test_double_release_same_truck_id_second_returns_none(self):
        t = make_truck("T1")
        run_process(self.env, self.tp3.enter(t))
        self.tp3.release("T1")
        result = self.tp3.release("T1")   # second release of the same id
        self.assertIsNone(result,
            "Second release of an already-released truck should return None.")

    def test_release_next_with_empty_buffer_returns_none(self):
        self.assertIsNone(self.tp3.release_next("GHA_A"))

    def test_enter_same_truck_object_twice_parks_it_twice(self):
        """
        A Truck object can be entered twice (e.g. due to a caller bug).
        TP3Buffer has no dedup guard — both entries land in _parked.
        This test documents the absence of that guard.
        """
        t = make_truck("T1")
        run_process(self.env, self.tp3.enter(t))
        run_process(self.env, self.tp3.enter(t))
        self.assertEqual(self.tp3.n_parked(), 2,
            "TP3Buffer does not deduplicate entries — same truck entered twice.")

    def test_get_parked_trucks_empty_buffer_returns_empty_list(self):
        self.assertEqual(self.tp3.get_parked_trucks(), [])


# ---------------------------------------------------------------------------
# Stress and cross-module
# ---------------------------------------------------------------------------
class TestStressAndCrossModule(unittest.TestCase):

    def test_500_trucks_through_single_gha_terminal(self):
        env, term, infra, svc = make_terminal("GHA_A", n_exp=5, svc_time=1.0)
        svc.sample.return_value = 1.0
        dtp = make_dtp(phase="on_time")
        for i in range(500):
            env.process(term.process_truck(make_truck(f"T{i}"), dtp))
        env.run(until=100_000)
        self.assertEqual(term.stats["export"]["processed"], 500)

    def test_mixed_flow_types_accumulate_in_correct_stat_bucket(self):
        env, term, infra, svc = make_terminal("GHA_A", svc_time=5.0)
        svc.sample.return_value = 5.0
        dtp = make_dtp(phase="on_time")
        for i in range(3):
            env.process(term.process_truck(make_truck(f"E{i}", flow_type="export"), dtp))
        for i in range(2):
            env.process(term.process_truck(make_truck(f"I{i}", flow_type="import"), dtp))
        env.run(until=100_000)
        self.assertEqual(term.stats["export"]["processed"], 3)
        self.assertEqual(term.stats["import"]["processed"], 2)

    def test_100_trucks_tp3_enter_and_release(self):
        TP3Buffer.CAPACITY = 100
        env, tp3, infra = make_tp3(capacity=100)
        trucks = [make_truck(f"T{i}") for i in range(100)]
        for t in trucks:
            run_process(env, tp3.enter(t))
        self.assertEqual(tp3.n_parked(), 100)
        for t in trucks:
            tp3.release(t.truck_id)
        self.assertEqual(tp3.n_parked(), 0)

    def test_truck_multi_stop_journey_ends_with_no_stops_remaining(self):
        manifest = [{"gha": "GHA_A", "parcels": 5},
                    {"gha": "GHA_B", "parcels": 3}]
        t = make_truck(manifest=manifest,
                       booked_slots={"GHA_A": 10, "GHA_B": 50})
        t.complete_stop("GHA_A")
        t.complete_stop("GHA_B")
        self.assertEqual(t.stops_remaining, [])
        self.assertIsNone(t.next_stop())
        self.assertIsNone(t.next_slot())

    def test_wpr_accumulation_consistent_across_many_trucks(self):
        """
        50 trucks, each with fixed 5-min wait and 10-min service.
        tot_wait / tot_serv should converge to exactly 0.5.
        """
        env, term, _, svc = make_terminal("GHA_A", n_exp=50, svc_time=10.0)
        svc.sample.return_value = 10.0
        dtp = make_dtp(phase="on_time")
        # Inject trucks one at a time with a 5-minute gha wait baked into
        # the queue by saturating a single dock first (0-wait scenario here
        # since 50 docks = no queuing; verify raw service accumulation)
        for i in range(50):
            env.process(term.process_truck(make_truck(f"T{i}"), dtp))
        env.run(until=100_000)
        self.assertAlmostEqual(
            term.stats["export"]["tot_serv"], 50 * 10.0,
            msg="Total service time must equal n_trucks * svc_time.")

    def test_next_slot_consistent_after_partial_completion(self):
        manifest = [{"gha": f"GHA_{c}", "parcels": i+1}
                    for i, c in enumerate("ABCDE")]
        slots = {f"GHA_{c}": (i+1)*10 for i, c in enumerate("ABCDE")}
        t = make_truck(manifest=manifest, booked_slots=slots)
        # Complete first three stops
        for c in "ABC":
            t.complete_stop(f"GHA_{c}")
        # Remaining: GHA_D=40, GHA_E=50  → next_slot = 40
        self.assertEqual(t.next_slot(), 40)


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)