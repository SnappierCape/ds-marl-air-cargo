# =============================================================================
# TEST SUITE — infrastructure.py
# =============================================================================
# Tests every class and method in isolation and in full-project context.
# Enforces the slot flow_type contract: every dock event must carry a
# flow_type that matches the truck's flow_type, and the infrastructure layer
# must faithfully record it so that KPITracker and downstream consumers can
# distinguish export from import traffic.
# =============================================================================
import sys
import os
import unittest

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from env.infrastructure import InfrastructureLayer, CheckpointID, SensorEvent


# =============================================================================
# MOCK TRUCK
# Minimal duck-typed truck that satisfies every attribute InfrastructureLayer
# reads. All fields are explicit so tests can verify exact values in events.
# =============================================================================
class MockTruck:
    def __init__(
        self,
        truck_id: str = "TRK-001",
        flow_type: str = "export",
        parcels: int = 10,
        booked_slots: dict = None,
    ):
        self.truck_id     = truck_id
        self.flow_type    = flow_type
        self.timestamps   = {}
        self.booked_slots = booked_slots or {}
        self._parcels     = parcels

    def total_parcels(self) -> int:
        return self._parcels

    def parcels_for(self, gha_id: str) -> int:
        return self._parcels

    def next_slot(self):
        return min(self.booked_slots.values()) if self.booked_slots else None


# =============================================================================
# TEST CLASS 1 — CheckpointID
# =============================================================================
class TestCheckpointID(unittest.TestCase):
    """Enum values are contracts consumed by KPITracker and schiphol_env.
    Any rename silently breaks downstream string comparisons."""

    def test_all_seven_checkpoints_exist(self):
        names = {cp.name for cp in CheckpointID}
        self.assertEqual(
            names,
            {"GATE_IN", "GATE_OUT", "TP3_IN", "TP3_OUT", "GHA_IN", "DOCK_START", "DOCK_END"},
        )

    def test_string_values_are_lowercase_snake_case(self):
        for cp in CheckpointID:
            self.assertEqual(cp.value, cp.value.lower())
            self.assertNotIn(" ", cp.value)

    def test_gate_in_value(self):
        self.assertEqual(CheckpointID.GATE_IN.value, "gate_in")

    def test_dock_start_value(self):
        self.assertEqual(CheckpointID.DOCK_START.value, "dock_start")

    def test_enum_members_are_unique(self):
        values = [cp.value for cp in CheckpointID]
        self.assertEqual(len(values), len(set(values)))


# =============================================================================
# TEST CLASS 2 — SensorEvent dataclass
# =============================================================================
class TestSensorEvent(unittest.TestCase):
    """SensorEvent is a plain dataclass. Tests guard field presence and types
    so that KPITracker attribute access never raises AttributeError."""

    def _make_event(self, **kwargs):
        defaults = dict(
            sim_time=100.0,
            checkpoint=CheckpointID.GATE_IN,
            truck_id="TRK-001",
            flow_type="export",
            gha_id=None,
            dock_id=None,
            n_parcels=10,
            slot_window=480.0,
        )
        defaults.update(kwargs)
        return SensorEvent(**defaults)

    def test_fields_accessible_by_name(self):
        e = self._make_event()
        self.assertEqual(e.sim_time, 100.0)
        self.assertEqual(e.checkpoint, CheckpointID.GATE_IN)
        self.assertEqual(e.truck_id, "TRK-001")
        self.assertEqual(e.flow_type, "export")
        self.assertIsNone(e.gha_id)
        self.assertIsNone(e.dock_id)
        self.assertEqual(e.n_parcels, 10)
        self.assertEqual(e.slot_window, 480.0)

    def test_optional_fields_accept_none(self):
        e = self._make_event(gha_id=None, dock_id=None, n_parcels=None, slot_window=None)
        self.assertIsNone(e.gha_id)
        self.assertIsNone(e.dock_id)
        self.assertIsNone(e.n_parcels)
        self.assertIsNone(e.slot_window)

    def test_flow_type_stored_verbatim(self):
        for ft in ("export", "import"):
            e = self._make_event(flow_type=ft)
            self.assertEqual(e.flow_type, ft)

    def test_import_flow_type_distinct_from_export(self):
        e_exp = self._make_event(flow_type="export")
        e_imp = self._make_event(flow_type="import")
        self.assertNotEqual(e_exp.flow_type, e_imp.flow_type)

    def test_dataclass_equality(self):
        e1 = self._make_event(sim_time=50.0)
        e2 = self._make_event(sim_time=50.0)
        self.assertEqual(e1, e2)

    def test_dataclass_inequality_on_sim_time(self):
        e1 = self._make_event(sim_time=10.0)
        e2 = self._make_event(sim_time=20.0)
        self.assertNotEqual(e1, e2)

    def test_dock_event_can_have_none_parcels(self):
        # dock_end never carries parcels — already logged at dock_start
        e = self._make_event(checkpoint=CheckpointID.DOCK_END, n_parcels=None)
        self.assertIsNone(e.n_parcels)


# =============================================================================
# TEST CLASS 3 — InfrastructureLayer initialisation
# =============================================================================
class TestInfrastructureLayerInit(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def test_event_log_starts_empty(self):
        self.assertEqual(len(self.infra.event_log), 0)

    def test_step_buffer_starts_empty(self):
        self.assertEqual(len(self.infra.step_buffer), 0)

    def test_event_log_is_list(self):
        self.assertIsInstance(self.infra.event_log, list)

    def test_step_buffer_is_list(self):
        self.assertIsInstance(self.infra.step_buffer, list)

    def test_two_instances_do_not_share_state(self):
        infra2 = InfrastructureLayer()
        truck = MockTruck()
        self.infra.gate_in(0.0, truck)
        self.assertEqual(len(infra2.get_all_events()), 0)


# =============================================================================
# TEST CLASS 4 — _log (internal method)
# =============================================================================
class TestInternalLog(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def _make_event(self):
        return SensorEvent(
            sim_time=1.0,
            checkpoint=CheckpointID.GATE_IN,
            truck_id="TRK-X",
            flow_type="export",
            gha_id=None,
            dock_id=None,
            n_parcels=5,
            slot_window=None,
        )

    def test_log_appends_to_event_log(self):
        e = self._make_event()
        self.infra._log(e)
        self.assertEqual(len(self.infra.event_log), 1)
        self.assertIs(self.infra.event_log[0], e)

    def test_log_appends_to_step_buffer(self):
        e = self._make_event()
        self.infra._log(e)
        self.assertEqual(len(self.infra.step_buffer), 1)
        self.assertIs(self.infra.step_buffer[0], e)

    def test_log_same_object_in_both_containers(self):
        e = self._make_event()
        self.infra._log(e)
        self.assertIs(self.infra.event_log[0], self.infra.step_buffer[0])

    def test_log_multiple_events_preserves_order(self):
        for i in range(5):
            e = SensorEvent(float(i), CheckpointID.GATE_IN, f"TRK-{i}",
                            "export", None, None, None, None)
            self.infra._log(e)
        times = [e.sim_time for e in self.infra.event_log]
        self.assertEqual(times, [0.0, 1.0, 2.0, 3.0, 4.0])


# =============================================================================
# TEST CLASS 5 — flush_step_buffer
# =============================================================================
class TestFlushStepBuffer(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def test_flush_returns_all_buffered_events(self):
        trucks = [MockTruck(f"TRK-{i}") for i in range(3)]
        for t in trucks:
            self.infra.gate_in(float(t.truck_id[-1]), t)
        flushed = self.infra.flush_step_buffer()
        self.assertEqual(len(flushed), 3)

    def test_flush_clears_step_buffer(self):
        truck = MockTruck()
        self.infra.gate_in(0.0, truck)
        self.infra.flush_step_buffer()
        self.assertEqual(len(self.infra.step_buffer), 0)

    def test_flush_does_not_affect_event_log(self):
        truck = MockTruck()
        self.infra.gate_in(0.0, truck)
        self.infra.flush_step_buffer()
        self.assertEqual(len(self.infra.event_log), 1)

    def test_flush_on_empty_buffer_returns_empty_list(self):
        result = self.infra.flush_step_buffer()
        self.assertEqual(result, [])

    def test_second_flush_returns_only_new_events(self):
        t1 = MockTruck("TRK-A")
        t2 = MockTruck("TRK-B")
        self.infra.gate_in(0.0, t1)
        self.infra.flush_step_buffer()          # consume first event
        self.infra.gate_in(10.0, t2)
        flushed = self.infra.flush_step_buffer()
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0].truck_id, "TRK-B")

    def test_flush_returns_copy_not_reference(self):
        truck = MockTruck()
        self.infra.gate_in(0.0, truck)
        flushed = self.infra.flush_step_buffer()
        # mutating the returned list must not affect the internal buffer
        flushed.append("garbage")
        self.assertEqual(len(self.infra.step_buffer), 0)

    def test_events_across_multiple_steps_are_isolated(self):
        """Simulates 3 MARL steps — each flush must return only that step's events."""
        trucks = [MockTruck(f"TRK-{i}") for i in range(6)]
        counts = []
        for step in range(3):
            self.infra.gate_in(float(step * 10), trucks[step * 2])
            self.infra.gate_in(float(step * 10 + 1), trucks[step * 2 + 1])
            flushed = self.infra.flush_step_buffer()
            counts.append(len(flushed))
        self.assertEqual(counts, [2, 2, 2])
        self.assertEqual(len(self.infra.event_log), 6)


# =============================================================================
# TEST CLASS 6 — get_all_events
# =============================================================================
class TestGetAllEvents(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def test_returns_list(self):
        self.assertIsInstance(self.infra.get_all_events(), list)

    def test_returns_all_events_including_flushed(self):
        truck = MockTruck()
        self.infra.gate_in(0.0, truck)
        self.infra.flush_step_buffer()
        self.infra.gate_out(100.0, truck)
        self.assertEqual(len(self.infra.get_all_events()), 2)

    def test_events_in_insertion_order(self):
        t = MockTruck()
        self.infra.gate_in(10.0, t)
        self.infra.tp3_in(20.0, t)
        self.infra.tp3_out(30.0, t)
        events = self.infra.get_all_events()
        checkpoints = [e.checkpoint for e in events]
        self.assertEqual(
            checkpoints,
            [CheckpointID.GATE_IN, CheckpointID.TP3_IN, CheckpointID.TP3_OUT],
        )

    def test_returns_reference_to_internal_log(self):
        """get_all_events exposes the live list — callers should not mutate it."""
        events = self.infra.get_all_events()
        truck = MockTruck()
        self.infra.gate_in(0.0, truck)
        # the returned list reflects the new event because it is the same object
        self.assertEqual(len(events), 1)


# =============================================================================
# TEST CLASS 7 — gate_in
# =============================================================================
class TestGateIn(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def test_creates_exactly_one_event(self):
        self.infra.gate_in(100.0, MockTruck())
        self.assertEqual(len(self.infra.event_log), 1)

    def test_checkpoint_is_gate_in(self):
        self.infra.gate_in(0.0, MockTruck())
        self.assertEqual(self.infra.event_log[0].checkpoint, CheckpointID.GATE_IN)

    def test_sim_time_recorded_correctly(self):
        self.infra.gate_in(123.45, MockTruck())
        self.assertEqual(self.infra.event_log[0].sim_time, 123.45)

    def test_truck_id_recorded(self):
        self.infra.gate_in(0.0, MockTruck("TRK-XYZ"))
        self.assertEqual(self.infra.event_log[0].truck_id, "TRK-XYZ")

    def test_export_flow_type_recorded(self):
        self.infra.gate_in(0.0, MockTruck(flow_type="export"))
        self.assertEqual(self.infra.event_log[0].flow_type, "export")

    def test_import_flow_type_recorded(self):
        self.infra.gate_in(0.0, MockTruck(flow_type="import"))
        self.assertEqual(self.infra.event_log[0].flow_type, "import")

    def test_flow_type_not_coerced(self):
        """Infrastructure must not silently transform the flow_type string."""
        truck = MockTruck(flow_type="export")
        self.infra.gate_in(0.0, truck)
        self.assertNotEqual(self.infra.event_log[0].flow_type, "import")

    def test_n_parcels_equals_total_parcels(self):
        self.infra.gate_in(0.0, MockTruck(parcels=17))
        self.assertEqual(self.infra.event_log[0].n_parcels, 17)

    def test_slot_window_uses_next_slot(self):
        truck = MockTruck(booked_slots={"dnata": 480, "klm": 600})
        self.infra.gate_in(0.0, truck)
        self.assertEqual(self.infra.event_log[0].slot_window, 480)

    def test_slot_window_none_when_no_bookings(self):
        self.infra.gate_in(0.0, MockTruck(booked_slots={}))
        self.assertIsNone(self.infra.event_log[0].slot_window)

    def test_gha_id_is_none(self):
        self.infra.gate_in(0.0, MockTruck())
        self.assertIsNone(self.infra.event_log[0].gha_id)

    def test_dock_id_is_none(self):
        self.infra.gate_in(0.0, MockTruck())
        self.assertIsNone(self.infra.event_log[0].dock_id)

    def test_timestamps_dict_updated(self):
        truck = MockTruck()
        self.infra.gate_in(55.5, truck)
        self.assertEqual(truck.timestamps["gate_in"], 55.5)

    def test_timestamps_not_overwritten_by_other_events(self):
        truck = MockTruck()
        self.infra.gate_in(10.0, truck)
        self.infra.gate_out(200.0, truck)
        self.assertEqual(truck.timestamps["gate_in"], 10.0)

    def test_multiple_trucks_each_get_own_event(self):
        for i in range(5):
            self.infra.gate_in(float(i), MockTruck(f"TRK-{i:03d}"))
        ids = [e.truck_id for e in self.infra.event_log]
        self.assertEqual(len(set(ids)), 5)

    def test_export_and_import_trucks_recorded_separately(self):
        self.infra.gate_in(0.0, MockTruck("EXP", flow_type="export"))
        self.infra.gate_in(1.0, MockTruck("IMP", flow_type="import"))
        flow_types = [e.flow_type for e in self.infra.event_log]
        self.assertIn("export", flow_types)
        self.assertIn("import", flow_types)


# =============================================================================
# TEST CLASS 8 — gate_out
# =============================================================================
class TestGateOut(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def test_checkpoint_is_gate_out(self):
        truck = MockTruck()
        self.infra.gate_in(0.0, truck)
        self.infra.gate_out(100.0, truck)
        self.assertEqual(self.infra.event_log[1].checkpoint, CheckpointID.GATE_OUT)

    def test_n_parcels_is_none(self):
        truck = MockTruck(parcels=20)
        self.infra.gate_out(100.0, truck)
        self.assertIsNone(self.infra.event_log[0].n_parcels)

    def test_slot_window_is_none(self):
        truck = MockTruck(booked_slots={"dnata": 480})
        self.infra.gate_out(100.0, truck)
        self.assertIsNone(self.infra.event_log[0].slot_window)

    def test_flow_type_preserved_on_gate_out(self):
        for ft in ("export", "import"):
            infra = InfrastructureLayer()
            truck = MockTruck(flow_type=ft)
            infra.gate_out(100.0, truck)
            self.assertEqual(infra.event_log[0].flow_type, ft)

    def test_timestamps_gate_out_set(self):
        truck = MockTruck()
        self.infra.gate_out(99.0, truck)
        self.assertEqual(truck.timestamps["gate_out"], 99.0)

    def test_gate_out_after_gate_in_both_in_log(self):
        truck = MockTruck()
        self.infra.gate_in(0.0, truck)
        self.infra.gate_out(100.0, truck)
        checkpoints = [e.checkpoint for e in self.infra.event_log]
        self.assertIn(CheckpointID.GATE_IN, checkpoints)
        self.assertIn(CheckpointID.GATE_OUT, checkpoints)

    def test_gate_out_time_greater_than_gate_in_time(self):
        truck = MockTruck()
        self.infra.gate_in(10.0, truck)
        self.infra.gate_out(200.0, truck)
        t_in  = self.infra.event_log[0].sim_time
        t_out = self.infra.event_log[1].sim_time
        self.assertLess(t_in, t_out)


# =============================================================================
# TEST CLASS 9 — tp3_in
# =============================================================================
class TestTp3In(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def test_checkpoint_is_tp3_in(self):
        self.infra.tp3_in(0.0, MockTruck())
        self.assertEqual(self.infra.event_log[0].checkpoint, CheckpointID.TP3_IN)

    def test_n_parcels_is_none(self):
        self.infra.tp3_in(0.0, MockTruck(parcels=50))
        self.assertIsNone(self.infra.event_log[0].n_parcels)

    def test_slot_window_is_none(self):
        self.infra.tp3_in(0.0, MockTruck(booked_slots={"klm": 300}))
        self.assertIsNone(self.infra.event_log[0].slot_window)

    def test_gha_id_is_none(self):
        self.infra.tp3_in(0.0, MockTruck())
        self.assertIsNone(self.infra.event_log[0].gha_id)

    def test_flow_type_export_preserved(self):
        self.infra.tp3_in(0.0, MockTruck(flow_type="export"))
        self.assertEqual(self.infra.event_log[0].flow_type, "export")

    def test_flow_type_import_preserved(self):
        self.infra.tp3_in(0.0, MockTruck(flow_type="import"))
        self.assertEqual(self.infra.event_log[0].flow_type, "import")

    def test_timestamps_tp3_in_set(self):
        truck = MockTruck()
        self.infra.tp3_in(77.0, truck)
        self.assertEqual(truck.timestamps["tp3_in"], 77.0)

    def test_tp3_in_appears_in_step_buffer(self):
        self.infra.tp3_in(0.0, MockTruck())
        self.assertEqual(len(self.infra.step_buffer), 1)


# =============================================================================
# TEST CLASS 10 — tp3_out
# =============================================================================
class TestTp3Out(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def test_checkpoint_is_tp3_out(self):
        self.infra.tp3_out(0.0, MockTruck())
        self.assertEqual(self.infra.event_log[0].checkpoint, CheckpointID.TP3_OUT)

    def test_n_parcels_is_none(self):
        self.infra.tp3_out(0.0, MockTruck(parcels=30))
        self.assertIsNone(self.infra.event_log[0].n_parcels)

    def test_flow_type_export_preserved(self):
        self.infra.tp3_out(0.0, MockTruck(flow_type="export"))
        self.assertEqual(self.infra.event_log[0].flow_type, "export")

    def test_flow_type_import_preserved(self):
        self.infra.tp3_out(0.0, MockTruck(flow_type="import"))
        self.assertEqual(self.infra.event_log[0].flow_type, "import")

    def test_timestamps_tp3_out_set(self):
        truck = MockTruck()
        self.infra.tp3_out(88.0, truck)
        self.assertEqual(truck.timestamps["tp3_out"], 88.0)

    def test_tp3_in_before_tp3_out_ordering(self):
        truck = MockTruck()
        self.infra.tp3_in(10.0, truck)
        self.infra.tp3_out(50.0, truck)
        t_in  = self.infra.event_log[0].sim_time
        t_out = self.infra.event_log[1].sim_time
        self.assertLess(t_in, t_out)


# =============================================================================
# TEST CLASS 11 — gha_in
# =============================================================================
class TestGhaIn(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def test_checkpoint_is_gha_in(self):
        self.infra.gha_in(0.0, MockTruck(), "dnata")
        self.assertEqual(self.infra.event_log[0].checkpoint, CheckpointID.GHA_IN)

    def test_gha_id_recorded(self):
        self.infra.gha_in(0.0, MockTruck(), "klm")
        self.assertEqual(self.infra.event_log[0].gha_id, "klm")

    def test_n_parcels_from_parcels_for(self):
        truck = MockTruck(parcels=12)
        self.infra.gha_in(0.0, truck, "wfs")
        self.assertEqual(self.infra.event_log[0].n_parcels, 12)

    def test_slot_window_from_booked_slots(self):
        truck = MockTruck(booked_slots={"dnata": 480})
        self.infra.gha_in(0.0, truck, "dnata")
        self.assertEqual(self.infra.event_log[0].slot_window, 480)

    def test_slot_window_none_when_gha_not_booked(self):
        truck = MockTruck(booked_slots={"klm": 300})
        self.infra.gha_in(0.0, truck, "dnata")   # dnata not in booked_slots
        self.assertIsNone(self.infra.event_log[0].slot_window)

    def test_flow_type_export_at_gha(self):
        """Export truck at GHA: event must carry 'export' so KPITracker can
        bucket the wait time against the correct dock pool."""
        truck = MockTruck(flow_type="export", booked_slots={"dnata": 480})
        self.infra.gha_in(0.0, truck, "dnata")
        self.assertEqual(self.infra.event_log[0].flow_type, "export")

    def test_flow_type_import_at_gha(self):
        truck = MockTruck(flow_type="import", booked_slots={"dnata": 480})
        self.infra.gha_in(0.0, truck, "dnata")
        self.assertEqual(self.infra.event_log[0].flow_type, "import")

    def test_flow_type_matches_truck_not_gha(self):
        """The slot at a GHA is now typed, but the event's flow_type must
        always come from the truck, not from the slot.  This validates that
        the infrastructure layer is neutral — it records facts, not rules."""
        truck = MockTruck(flow_type="import", booked_slots={"swiss": 600})
        self.infra.gha_in(0.0, truck, "swiss")
        event = self.infra.event_log[0]
        self.assertEqual(event.flow_type, "import")
        self.assertEqual(event.gha_id, "swiss")

    def test_timestamps_gha_in_keyed_by_gha(self):
        truck = MockTruck()
        self.infra.gha_in(55.0, truck, "klm")
        self.assertEqual(truck.timestamps["gha_in_klm"], 55.0)

    def test_multi_stop_each_gha_gets_own_timestamp(self):
        truck = MockTruck(booked_slots={"dnata": 480, "klm": 540, "wfs": 600})
        self.infra.gha_in(10.0, truck, "dnata")
        self.infra.gha_in(80.0, truck, "klm")
        self.infra.gha_in(150.0, truck, "wfs")
        self.assertEqual(truck.timestamps["gha_in_dnata"], 10.0)
        self.assertEqual(truck.timestamps["gha_in_klm"],   80.0)
        self.assertEqual(truck.timestamps["gha_in_wfs"],  150.0)

    def test_multi_stop_all_four_ghas(self):
        truck = MockTruck(booked_slots={"dnata": 480, "klm": 540, "wfs": 600, "swiss": 660})
        for gha in ["dnata", "klm", "wfs", "swiss"]:
            self.infra.gha_in(0.0, truck, gha)
        gha_ids = [e.gha_id for e in self.infra.event_log]
        self.assertCountEqual(gha_ids, ["dnata", "klm", "wfs", "swiss"])

    def test_dock_id_is_none_at_gha_in(self):
        self.infra.gha_in(0.0, MockTruck(), "dnata")
        self.assertIsNone(self.infra.event_log[0].dock_id)


# =============================================================================
# TEST CLASS 12 — dock_start
# =============================================================================
class TestDockStart(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def test_checkpoint_is_dock_start(self):
        self.infra.dock_start(0.0, MockTruck(), "dnata", 0)
        self.assertEqual(self.infra.event_log[0].checkpoint, CheckpointID.DOCK_START)

    def test_dock_id_recorded(self):
        self.infra.dock_start(0.0, MockTruck(), "dnata", 7)
        self.assertEqual(self.infra.event_log[0].dock_id, 7)

    def test_gha_id_recorded(self):
        self.infra.dock_start(0.0, MockTruck(), "wfs", 2)
        self.assertEqual(self.infra.event_log[0].gha_id, "wfs")

    def test_n_parcels_recorded(self):
        self.infra.dock_start(0.0, MockTruck(parcels=9), "klm", 1)
        self.assertEqual(self.infra.event_log[0].n_parcels, 9)

    def test_slot_window_from_booked_slots(self):
        truck = MockTruck(booked_slots={"dnata": 480})
        self.infra.dock_start(0.0, truck, "dnata", 3)
        self.assertEqual(self.infra.event_log[0].slot_window, 480)

    def test_export_flow_type_at_dock_start(self):
        """This is the critical flow_type test: the dock_start event must carry
        'export' so that KPITracker can compute per-flow-type service stats and
        the new slot-to-flow-type constraint can be audited post-hoc."""
        truck = MockTruck(flow_type="export", booked_slots={"dnata": 480})
        self.infra.dock_start(0.0, truck, "dnata", 0)
        self.assertEqual(self.infra.event_log[0].flow_type, "export")

    def test_import_flow_type_at_dock_start(self):
        truck = MockTruck(flow_type="import", booked_slots={"klm": 540})
        self.infra.dock_start(0.0, truck, "klm", 1)
        self.assertEqual(self.infra.event_log[0].flow_type, "import")

    def test_export_and_import_dock_events_distinguishable(self):
        """Both flow types docking simultaneously — their events must differ."""
        exp_truck = MockTruck("EXP", flow_type="export", booked_slots={"dnata": 480})
        imp_truck = MockTruck("IMP", flow_type="import", booked_slots={"dnata": 480})
        self.infra.dock_start(10.0, exp_truck, "dnata", 0)
        self.infra.dock_start(10.0, imp_truck, "dnata", 25)   # import dock starts at 25
        flow_types = [e.flow_type for e in self.infra.event_log]
        self.assertIn("export", flow_types)
        self.assertIn("import", flow_types)

    def test_dock_ids_across_multiple_trucks_all_recorded(self):
        for dock_id in range(10):
            truck = MockTruck(f"TRK-{dock_id}", booked_slots={"dnata": 480})
            self.infra.dock_start(float(dock_id), truck, "dnata", dock_id)
        recorded_dock_ids = [e.dock_id for e in self.infra.event_log]
        self.assertEqual(recorded_dock_ids, list(range(10)))

    def test_timestamps_dock_start_keyed_by_gha(self):
        truck = MockTruck()
        self.infra.dock_start(33.0, truck, "wfs", 0)
        self.assertEqual(truck.timestamps["dock_start_wfs"], 33.0)

    def test_timestamps_for_multi_stop_dock_starts(self):
        truck = MockTruck(booked_slots={"dnata": 480, "klm": 540})
        self.infra.dock_start(100.0, truck, "dnata", 0)
        self.infra.dock_start(200.0, truck, "klm", 0)
        self.assertEqual(truck.timestamps["dock_start_dnata"], 100.0)
        self.assertEqual(truck.timestamps["dock_start_klm"],   200.0)


# =============================================================================
# TEST CLASS 13 — dock_end
# =============================================================================
class TestDockEnd(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def test_checkpoint_is_dock_end(self):
        self.infra.dock_end(0.0, MockTruck(), "dnata", 0)
        self.assertEqual(self.infra.event_log[0].checkpoint, CheckpointID.DOCK_END)

    def test_n_parcels_is_none(self):
        """Parcel count is already logged at dock_start — dock_end carries None."""
        self.infra.dock_end(0.0, MockTruck(parcels=15), "dnata", 0)
        self.assertIsNone(self.infra.event_log[0].n_parcels)

    def test_gha_id_recorded(self):
        self.infra.dock_end(0.0, MockTruck(), "swiss", 3)
        self.assertEqual(self.infra.event_log[0].gha_id, "swiss")

    def test_dock_id_recorded(self):
        self.infra.dock_end(0.0, MockTruck(), "dnata", 12)
        self.assertEqual(self.infra.event_log[0].dock_id, 12)

    def test_slot_window_from_booked_slots(self):
        truck = MockTruck(booked_slots={"wfs": 600})
        self.infra.dock_end(0.0, truck, "wfs", 0)
        self.assertEqual(self.infra.event_log[0].slot_window, 600)

    def test_export_flow_type_at_dock_end(self):
        truck = MockTruck(flow_type="export")
        self.infra.dock_end(0.0, truck, "dnata", 0)
        self.assertEqual(self.infra.event_log[0].flow_type, "export")

    def test_import_flow_type_at_dock_end(self):
        truck = MockTruck(flow_type="import")
        self.infra.dock_end(0.0, truck, "dnata", 0)
        self.assertEqual(self.infra.event_log[0].flow_type, "import")

    def test_dock_end_after_dock_start_time_ordering(self):
        truck = MockTruck()
        self.infra.dock_start(100.0, truck, "dnata", 0)
        self.infra.dock_end(135.0, truck, "dnata", 0)
        t_start = self.infra.event_log[0].sim_time
        t_end   = self.infra.event_log[1].sim_time
        self.assertLess(t_start, t_end)

    def test_service_duration_computable_from_events(self):
        """KPITracker computes service time as dock_end.sim_time - dock_start.sim_time.
        Verify both events carry enough info to compute a meaningful duration."""
        truck = MockTruck()
        self.infra.dock_start(100.0, truck, "klm", 0)
        self.infra.dock_end(145.0, truck, "klm", 0)
        start_event = next(e for e in self.infra.event_log if e.checkpoint == CheckpointID.DOCK_START)
        end_event   = next(e for e in self.infra.event_log if e.checkpoint == CheckpointID.DOCK_END)
        service_time = end_event.sim_time - start_event.sim_time
        self.assertEqual(service_time, 45.0)
        self.assertEqual(start_event.gha_id, end_event.gha_id)
        self.assertEqual(start_event.truck_id, end_event.truck_id)

    def test_timestamps_dock_end_keyed_by_gha(self):
        truck = MockTruck()
        self.infra.dock_end(180.0, truck, "klm", 2)
        self.assertEqual(truck.timestamps["dock_end_klm"], 180.0)


# =============================================================================
# TEST CLASS 14 — flow_type contract across the full event lifecycle
# This is the core test class for the slot flow_type feature.
# It verifies that export and import trucks leave distinguishable traces
# in the event log so that auditing tools and KPITracker can enforce
# the flow_type-to-slot binding post hoc.
# =============================================================================
class TestFlowTypeContract(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def test_export_truck_all_events_carry_export(self):
        truck = MockTruck("EXP-001", flow_type="export",
                          parcels=10, booked_slots={"dnata": 480})
        self.infra.gate_in(0.0, truck)
        self.infra.gha_in(10.0, truck, "dnata")
        self.infra.dock_start(20.0, truck, "dnata", 0)
        self.infra.dock_end(60.0, truck, "dnata", 0)
        self.infra.gate_out(70.0, truck)
        for event in self.infra.event_log:
            self.assertEqual(
                event.flow_type, "export",
                f"Expected 'export' at {event.checkpoint} but got '{event.flow_type}'"
            )

    def test_import_truck_all_events_carry_import(self):
        truck = MockTruck("IMP-001", flow_type="import",
                          parcels=8, booked_slots={"klm": 540})
        self.infra.gate_in(0.0, truck)
        self.infra.gha_in(10.0, truck, "klm")
        self.infra.dock_start(20.0, truck, "klm", 25)
        self.infra.dock_end(55.0, truck, "klm", 25)
        self.infra.gate_out(65.0, truck)
        for event in self.infra.event_log:
            self.assertEqual(
                event.flow_type, "import",
                f"Expected 'import' at {event.checkpoint} but got '{event.flow_type}'"
            )

    def test_mixed_fleet_flow_types_never_cross_contaminate(self):
        """Simulates two trucks — one export, one import — going through the
        same GHA concurrently.  Their events must never share each other's
        flow_type."""
        exp_truck = MockTruck("EXP-002", flow_type="export", booked_slots={"wfs": 480})
        imp_truck = MockTruck("IMP-002", flow_type="import", booked_slots={"wfs": 480})

        self.infra.gate_in(0.0, exp_truck)
        self.infra.gate_in(1.0, imp_truck)
        self.infra.gha_in(10.0, exp_truck, "wfs")
        self.infra.gha_in(11.0, imp_truck, "wfs")
        self.infra.dock_start(15.0, exp_truck, "wfs", 0)   # export dock 0
        self.infra.dock_start(15.0, imp_truck, "wfs", 15)  # import dock 15
        self.infra.dock_end(55.0, exp_truck, "wfs", 0)
        self.infra.dock_end(50.0, imp_truck, "wfs", 15)

        for event in self.infra.event_log:
            if event.truck_id == "EXP-002":
                self.assertEqual(event.flow_type, "export",
                    f"EXP truck has wrong flow_type at {event.checkpoint}")
            else:
                self.assertEqual(event.flow_type, "import",
                    f"IMP truck has wrong flow_type at {event.checkpoint}")

    def test_flow_type_consistent_across_dock_start_and_dock_end(self):
        """dock_start and dock_end for the same truck/gha must carry the same
        flow_type so KPITracker can join them without ambiguity."""
        for ft in ("export", "import"):
            infra = InfrastructureLayer()
            truck = MockTruck(flow_type=ft, booked_slots={"dnata": 480})
            infra.dock_start(10.0, truck, "dnata", 0)
            infra.dock_end(50.0, truck, "dnata", 0)
            start_ft = infra.event_log[0].flow_type
            end_ft   = infra.event_log[1].flow_type
            self.assertEqual(start_ft, end_ft,
                f"flow_type mismatch: dock_start={start_ft}, dock_end={end_ft}")

    def test_flow_type_consistent_gha_in_through_dock_end(self):
        """Complete GHA visit chain: gha_in -> dock_start -> dock_end must
        all carry the same flow_type for a given truck."""
        for ft in ("export", "import"):
            infra = InfrastructureLayer()
            truck = MockTruck(flow_type=ft, booked_slots={"swiss": 600})
            infra.gha_in(0.0, truck, "swiss")
            infra.dock_start(5.0, truck, "swiss", 0)
            infra.dock_end(40.0, truck, "swiss", 0)
            flow_types_in_chain = [e.flow_type for e in infra.event_log]
            self.assertTrue(all(f == ft for f in flow_types_in_chain),
                f"Inconsistent flow_types in chain: {flow_types_in_chain}")

    def test_kpi_tracker_can_separate_export_import_wait_times(self):
        """Simulate what KPITracker does: group dock events by flow_type to
        compute per-type wait times.  Verifies the event data supports this."""
        trucks = [
            MockTruck("E1", "export", booked_slots={"dnata": 480}),
            MockTruck("E2", "export", booked_slots={"dnata": 480}),
            MockTruck("I1", "import", booked_slots={"dnata": 480}),
        ]
        gha_in_times  = {"E1": 10.0, "E2": 12.0, "I1": 11.0}
        dock_in_times = {"E1": 25.0, "E2": 30.0, "I1": 20.0}

        for truck in trucks:
            self.infra.gha_in(gha_in_times[truck.truck_id], truck, "dnata")
            self.infra.dock_start(dock_in_times[truck.truck_id], truck, "dnata", 0)

        gha_in_events    = {e.truck_id: e for e in self.infra.event_log
                            if e.checkpoint == CheckpointID.GHA_IN}
        dock_start_events = {e.truck_id: e for e in self.infra.event_log
                             if e.checkpoint == CheckpointID.DOCK_START}

        export_waits = []
        import_waits = []
        for tid in dock_start_events:
            wait = dock_start_events[tid].sim_time - gha_in_events[tid].sim_time
            if dock_start_events[tid].flow_type == "export":
                export_waits.append(wait)
            else:
                import_waits.append(wait)

        self.assertEqual(sorted(export_waits), [15.0, 18.0])
        self.assertEqual(import_waits, [9.0])


# =============================================================================
# TEST CLASS 15 — full truck journey integration
# Tests the module in the context of the whole project by verifying that
# the event sequence produced by a complete truck lifecycle is correct,
# in order, and carries the right metadata for every downstream consumer.
# =============================================================================
class TestFullJourneyIntegration(unittest.TestCase):

    def setUp(self):
        self.infra = InfrastructureLayer()

    def _run_single_stop_journey(self, flow_type: str, gha: str,
                                  slot: int, parcels: int, dock_id: int):
        truck = MockTruck("TRK-J1", flow_type=flow_type,
                          parcels=parcels, booked_slots={gha: slot})
        self.infra.gate_in(0.0, truck)
        self.infra.gha_in(10.0, truck, gha)
        self.infra.dock_start(20.0, truck, gha, dock_id)
        self.infra.dock_end(60.0, truck, gha, dock_id)
        self.infra.gate_out(70.0, truck)
        return truck

    def test_single_stop_export_journey_produces_five_events(self):
        self._run_single_stop_journey("export", "dnata", 480, 10, 0)
        self.assertEqual(len(self.infra.event_log), 5)

    def test_single_stop_import_journey_produces_five_events(self):
        self._run_single_stop_journey("import", "klm", 540, 8, 25)
        self.assertEqual(len(self.infra.event_log), 5)

    def test_checkpoint_order_for_full_journey(self):
        self._run_single_stop_journey("export", "dnata", 480, 10, 0)
        expected = [
            CheckpointID.GATE_IN,
            CheckpointID.GHA_IN,
            CheckpointID.DOCK_START,
            CheckpointID.DOCK_END,
            CheckpointID.GATE_OUT,
        ]
        actual = [e.checkpoint for e in self.infra.event_log]
        self.assertEqual(actual, expected)

    def test_all_events_same_truck_id(self):
        self._run_single_stop_journey("export", "dnata", 480, 10, 0)
        truck_ids = [e.truck_id for e in self.infra.event_log]
        self.assertTrue(all(tid == "TRK-J1" for tid in truck_ids))

    def test_all_events_same_flow_type_export(self):
        self._run_single_stop_journey("export", "dnata", 480, 10, 0)
        flow_types = [e.flow_type for e in self.infra.event_log]
        self.assertTrue(all(ft == "export" for ft in flow_types))

    def test_all_events_same_flow_type_import(self):
        self._run_single_stop_journey("import", "klm", 540, 8, 25)
        flow_types = [e.flow_type for e in self.infra.event_log]
        self.assertTrue(all(ft == "import" for ft in flow_types))

    def test_timestamps_complete_after_journey(self):
        truck = self._run_single_stop_journey("export", "dnata", 480, 10, 0)
        self.assertIn("gate_in",        truck.timestamps)
        self.assertIn("gha_in_dnata",   truck.timestamps)
        self.assertIn("dock_start_dnata", truck.timestamps)
        self.assertIn("dock_end_dnata", truck.timestamps)
        self.assertIn("gate_out",       truck.timestamps)

    def test_timestamps_are_monotonically_increasing(self):
        truck = self._run_single_stop_journey("export", "dnata", 480, 10, 0)
        times = [
            truck.timestamps["gate_in"],
            truck.timestamps["gha_in_dnata"],
            truck.timestamps["dock_start_dnata"],
            truck.timestamps["dock_end_dnata"],
            truck.timestamps["gate_out"],
        ]
        self.assertEqual(times, sorted(times))

    def test_multi_stop_journey_three_ghas(self):
        truck = MockTruck(
            "TRK-MULTI", flow_type="export", parcels=15,
            booked_slots={"dnata": 480, "klm": 540, "wfs": 600}
        )
        self.infra.gate_in(0.0, truck)
        for i, (gha, slot) in enumerate([("dnata", 480), ("klm", 540), ("wfs", 600)]):
            base = (i + 1) * 60
            self.infra.gha_in(float(base), truck, gha)
            self.infra.dock_start(float(base + 10), truck, gha, 0)
            self.infra.dock_end(float(base + 40), truck, gha, 0)
        self.infra.gate_out(250.0, truck)

        # 1 gate_in + 3*(gha_in + dock_start + dock_end) + 1 gate_out = 11
        self.assertEqual(len(self.infra.event_log), 11)

        # all events carry the truck's flow_type
        for event in self.infra.event_log:
            self.assertEqual(event.flow_type, "export")

        # three distinct gha_ids in gha_in events
        gha_in_ids = [e.gha_id for e in self.infra.event_log
                      if e.checkpoint == CheckpointID.GHA_IN]
        self.assertCountEqual(gha_in_ids, ["dnata", "klm", "wfs"])

    def test_tp3_redirect_journey(self):
        """Truck goes gate_in -> tp3_in -> tp3_out -> gha_in -> dock cycle -> gate_out."""
        truck = MockTruck("TRK-TP3", flow_type="import",
                          parcels=6, booked_slots={"swiss": 600})
        self.infra.gate_in(0.0, truck)
        self.infra.tp3_in(5.0, truck)
        self.infra.tp3_out(55.0, truck)
        self.infra.gha_in(60.0, truck, "swiss")
        self.infra.dock_start(65.0, truck, "swiss", 14)
        self.infra.dock_end(100.0, truck, "swiss", 14)
        self.infra.gate_out(110.0, truck)

        checkpoints = [e.checkpoint for e in self.infra.event_log]
        self.assertEqual(checkpoints, [
            CheckpointID.GATE_IN,
            CheckpointID.TP3_IN,
            CheckpointID.TP3_OUT,
            CheckpointID.GHA_IN,
            CheckpointID.DOCK_START,
            CheckpointID.DOCK_END,
            CheckpointID.GATE_OUT,
        ])
        for event in self.infra.event_log:
            self.assertEqual(event.flow_type, "import")

    def test_concurrent_fleet_step_buffer_isolation(self):
        """Mimics a MARL step: 4 trucks arrive, buffer is flushed, then 2 more
        dock. The second flush must contain exactly 2 dock events."""
        fleet = [MockTruck(f"TRK-{i}", flow_type="export") for i in range(4)]
        for truck in fleet:
            self.infra.gate_in(0.0, truck)
        step1 = self.infra.flush_step_buffer()
        self.assertEqual(len(step1), 4)

        dockers = fleet[:2]
        for truck in dockers:
            self.infra.dock_start(10.0, truck, "dnata", 0)
        step2 = self.infra.flush_step_buffer()
        self.assertEqual(len(step2), 2)
        self.assertTrue(all(e.checkpoint == CheckpointID.DOCK_START for e in step2))

        # full log has all 6 events
        self.assertEqual(len(self.infra.get_all_events()), 6)

    def test_event_log_survives_multiple_flush_cycles(self):
        truck = MockTruck(booked_slots={"dnata": 480})
        events_expected = [
            ("gate_in",    0.0),
            ("gha_in",    10.0),
            ("dock_start", 20.0),
            ("dock_end",   60.0),
            ("gate_out",   70.0),
        ]
        methods = {
            "gate_in":    lambda t: self.infra.gate_in(t, truck),
            "gha_in":     lambda t: self.infra.gha_in(t, truck, "dnata"),
            "dock_start": lambda t: self.infra.dock_start(t, truck, "dnata", 0),
            "dock_end":   lambda t: self.infra.dock_end(t, truck, "dnata", 0),
            "gate_out":   lambda t: self.infra.gate_out(t, truck),
        }
        for name, sim_time in events_expected:
            methods[name](sim_time)
            self.infra.flush_step_buffer()   # flush after each event

        # event_log must still contain all 5 events
        self.assertEqual(len(self.infra.get_all_events()), 5)
        self.assertEqual(len(self.infra.step_buffer), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)