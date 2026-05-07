# =============================================================================
# INFRASTRUCTURE MODULE
# =============================================================================
# DESCRIPTION:
#     Simulates the ANPR cameras and dock sensors in the cargo area.
#     Every time a truck passes a physical checkpoint, a SensorEvent is written
#     to two places:
#       - event_log: full episode history, read by KPITracker at episode end
#       - step_buffer: events since last MARL step, flushed by schiphol_env.py
#
#     This module has no dependencies on SimPy, params, or other custom modules.
#     It only knows about trucks via duck typing (it reads truck attributes).
# =============================================================================
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env.objects import Truck

# =============================================================================
# CHECKPOINT IDENTIFIERS
# =============================================================================
class CheckpointID(Enum):
    GATE_IN = "gate_in"
    GATE_OUT = "gate_out"
    TP3_IN = "tp3_in"
    TP3_OUT = "tp3_out"
    GHA_IN = "gha_in"
    DOCK_START = "dock_start"
    DOCK_END = "dock_end"

# =============================================================================
# SENSOR EVENT
# =============================================================================
@dataclass
class SensorEvent:
    """
    One row in the event log. Mirrors what a real ANPR or dock sensor writes.

    Fields:
        sim_time    : simulation time in minutes when the event fired
        checkpoint  : which sensor fired (gate, tp3, gha, dock)
        truck_id    : license plate
        flow_type   : "export" or "import"
        gha_id      : which GHA (None for gate and TP3 events)
        dock_id     : which dock door (None for non-dock events)
        n_parcels   : parcel count at this stop (None at dock_end, already logged at dock_start)
        slot_window : booked slot start time (for on-time/late classification)
    """
    sim_time: float
    checkpoint: CheckpointID
    truck_id: str
    flow_type: str
    gha_id: Optional[str]
    dock_id: Optional[int]
    n_parcels: Optional[int]
    slot_window: Optional[float]

# =============================================================================
# INFRASTRUCTURE LAYER
# =============================================================================
class InfrastructureLayer:
    def __init__(self):
        self.event_log: List[SensorEvent] = []    # full episode log
        self.step_buffer: List[SensorEvent] = []    # last stel log

    # ─────────────────────────────────────────────────────────────────────────
    # Internal logger
    # ─────────────────────────────────────────────────────────────────────────
    def _log(self, event: SensorEvent) -> None:
        """Writes one event to both the full log and the current step buffer."""
        self.event_log.append(event)
        self.step_buffer.append(event)

    # ─────────────────────────────────────────────────────────────────────────
    # Buffer management — called by schiphol_env.py
    # ─────────────────────────────────────────────────────────────────────────
    def flush_step_buffer(self) -> List[SensorEvent]:
        events = self.step_buffer.copy()
        self.step_buffer.clear()
        return events

    def get_all_events(self) -> List[SensorEvent]:
        return self.event_log

    # ─────────────────────────────────────────────────────────────────────────
    # Gate checkpoints — called by demand.py
    # ─────────────────────────────────────────────────────────────────────────
    def gate_in(self, sim_time: float, truck: Truck) -> None:
        self._log(SensorEvent(
            sim_time=sim_time, checkpoint=CheckpointID.GATE_IN,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=None, dock_id=None,
            n_parcels=truck.total_parcels(),
            slot_window=truck.next_slot()
        ))
        truck.timestamps["gate_in"] = sim_time

    def gate_out(self, sim_time: float, truck: Truck) -> None:
        self._log(SensorEvent(
            sim_time=sim_time, checkpoint=CheckpointID.GATE_OUT,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=None, dock_id=None,
            n_parcels=None, slot_window=None
        ))
        truck.timestamps["gate_out"] = sim_time

    # ─────────────────────────────────────────────────────────────────────────
    # TP3 checkpoints — called by objects.py (TP3Buffer)
    # ─────────────────────────────────────────────────────────────────────────
    def tp3_in(self, sim_time: float, truck: Truck) -> None:
        self._log(SensorEvent(
            sim_time=sim_time, checkpoint=CheckpointID.TP3_IN,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=None, dock_id=None,
            n_parcels=None, slot_window=None
        ))
        truck.timestamps["tp3_in"] = sim_time

    def tp3_out(self, sim_time: float, truck: Truck) -> None:
        self._log(SensorEvent(
            sim_time=sim_time, checkpoint=CheckpointID.TP3_OUT,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=None, dock_id=None,
            n_parcels=None, slot_window=None
        ))
        truck.timestamps["tp3_out"] = sim_time

    # ─────────────────────────────────────────────────────────────────────────
    # GHA checkpoints — called by objects.py (GHATerminal)
    # ─────────────────────────────────────────────────────────────────────────
    def gha_in(self, sim_time: float, truck: Truck, gha_id: str) -> None:
        self._log(SensorEvent(
            sim_time=sim_time, checkpoint=CheckpointID.GHA_IN,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=gha_id, dock_id=None,
            n_parcels=truck.parcels_for(gha_id),
            slot_window=truck.booked_slots.get(gha_id)
        ))
        truck.timestamps[f"gha_in_{gha_id}"] = sim_time

    def dock_start(self, sim_time: float, truck: Truck, gha_id: str, dock_id: int) -> None:
        self._log(SensorEvent(
            sim_time=sim_time, checkpoint=CheckpointID.DOCK_START,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=gha_id, dock_id=dock_id,
            n_parcels=truck.parcels_for(gha_id),
            slot_window=truck.booked_slots.get(gha_id)
        ))
        truck.timestamps[f"dock_start_{gha_id}"] = sim_time

    def dock_end(self, sim_time: float, truck: Truck, gha_id: str, dock_id: int) -> None:
        self._log(SensorEvent(
            sim_time=sim_time, checkpoint=CheckpointID.DOCK_END,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=gha_id, dock_id=dock_id,
            n_parcels=None,
            slot_window=truck.booked_slots.get(gha_id)
        ))
        truck.timestamps[f"dock_end_{gha_id}"] = sim_time