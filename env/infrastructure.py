# =============================================================================
# INFRASTRUCTURE MODULE
# =============================================================================
# DESCRIPTION:
#     Infrastructure module that contains all the classes for the checkpoint
#     tracking at the Airport.
# =============================================================================
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

# =============================================================================
# CHECKPOINTS
# =============================================================================
class CheckpointID(Enum):
    GATE_IN        = "gate_in"
    GATE_OUT       = "gate_out"
    TP3_IN         = "tp3_in"
    TP3_OUT        = "tp3_out"
    GHA_IN_DNATA   = "gha_in_dnata"
    GHA_IN_KLM     = "gha_in_klm"
    GHA_IN_SWISS   = "gha_in_swissport"
    GHA_IN_MENZ    = "gha_in_menzies_wfs"
    GHA_OUT_DNATA  = "gha_out_dnata"
    GHA_OUT_KLM    = "gha_out_klm"
    GHA_OUT_SWISS  = "gha_out_swissport"
    GHA_OUT_MENZ   = "gha_out_menzies_wfs"
    DOCK_START     = "dock_start"
    DOCK_END       = "dock_end"

# =============================================================================
# SENSORS
# =============================================================================
@dataclass
class SensorEvent:
    """
    Atomic event emitted by any checkpoint sensor.
    This is the unit of data that KPITracker and agents consume.
    In the real system, these are rows in the eLink/ANPR database.
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
# INFRASTRUCTURE
# =============================================================================
class InfrastructureLayer:
    """
    Manages all sensor checkpoints in the simulation.
    Called by SimPy processes when trucks pass checkpoints.
    Writes SensorEvents to the event buffer consumed by:
      - KPITracker  (for WPR, NTTP, peak resilience)
      - Agent obs   (for recent_events in observation vectors)
    """
    def __init__(self):
        self.event_log: List[SensorEvent] = []
        self.step_buffer: List[SensorEvent] = []

    def log(self, event: SensorEvent):
        self.event_log.append(event)
        self.step_buffer.append(event)

    def flush_step_buffer(self) -> List[SensorEvent]:
        """Called by PettingZoo wrapper at the start of each step."""
        events = self.step_buffer.copy()
        self.step_buffer.clear()
        return events

    def gate_in(self, sim_time, truck):
        self.log(SensorEvent(
            sim_time=sim_time, checkpoint=CheckpointID.GATE_IN,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=None, dock_id=None,
            n_parcels=truck.total_parcels(),
            slot_window=truck.next_slot_window()
        ))
        truck.timestamps["gate_in"] = sim_time

    def tp3_in(self, sim_time, truck):
        self.log(SensorEvent(
            sim_time=sim_time, checkpoint=CheckpointID.TP3_IN,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=None, dock_id=None, n_parcels=None, slot_window=None
        ))
        truck.timestamps["tp3_in"] = sim_time

    def tp3_out(self, sim_time, truck):
        self.log(SensorEvent(sim_time=sim_time, checkpoint=CheckpointID.TP3_OUT,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=None, dock_id=None, n_parcels=None, slot_window=None))
        truck.timestamps["tp3_out"] = sim_time

    def gha_in(self, sim_time, truck, gha_id):
        cp = CheckpointID[f"GHA_IN_{gha_id.upper()[:5]}"]
        self.log(SensorEvent(sim_time=sim_time, checkpoint=cp,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=gha_id, dock_id=None,
            n_parcels=truck.parcels_for(gha_id),
            slot_window=truck.booked_slots.get(gha_id)))
        truck.timestamps[f"gha_in_{gha_id}"] = sim_time

    def dock_start(self, sim_time, truck, gha_id, dock_id):
        self.log(SensorEvent(sim_time=sim_time, checkpoint=CheckpointID.DOCK_START,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=gha_id, dock_id=dock_id,
            n_parcels=truck.parcels_for(gha_id),
            slot_window=truck.booked_slots.get(gha_id)))
        truck.timestamps[f"dock_start_{gha_id}"] = sim_time

    def dock_end(self, sim_time, truck, gha_id, dock_id):
        self.log(SensorEvent(sim_time=sim_time, checkpoint=CheckpointID.DOCK_END,
            truck_id=truck.truck_id, flow_type=truck.flow_type,
            gha_id=gha_id, dock_id=dock_id, n_parcels=None,
            slot_window=truck.booked_slots.get(gha_id)))
        truck.timestamps[f"dock_end_{gha_id}"] = sim_time
    
    def get_all_events(self) -> List[SensorEvent]:
        return self.event_log