# =============================================================================
# SIMULATION OBJECTS MODULE
# =============================================================================
# DESCRIPTION:
#     Core SimPy objects: Truck dataclass, GHATerminal (with import/export
#     dock split), and TP3Buffer (140-slot constrained buffer).
#     This module owns all physical logistics logic.
#     It knows nothing about MARL, rewards, DTP rules or policies.
# =============================================================================
import sys
import os
from typing import Dict, List, Optional

import simpy
import numpy as np
from dataclasses import dataclass, field

# Setting base path for local imports
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0: -2]))
import config.config
from env.infrastructure import InfrastructureLayer
from env.dtp_platform import DTPPlatform
from env.service_time import ServiceTimeModel

# =============================================================================
# PARAMETERS IMPORT
# =============================================================================
params = config.load_params()

# =============================================================================
# TRUCK
# =============================================================================
@dataclass
class Truck:
    # ── Immutable attributes ─────────────────────────────────────────────────
    truck_id: str
    flow_type: str
    origin_type: str
    manifest: List[Dict]    # [{"gha": "dnata", "parcels": 5}, {"gha": "wfs", "parcels": 12}, ...]
    
    # ── Mutable attributes ───────────────────────────────────────────────────
    status: str = "in_transit"
    current_node: str = "origin"
    booked_slots: Dict = field(default_factory=dict)    # {"gha": slot_start}
    timestamps: Dict = field(default_factory=dict)
    stops_remaining: List[Dict] = field(default_factory=list)
    
    # ── Status constants ─────────────────────────────────────────────────────
    STATUS_IN_TRANSIT = "in_transit"
    STATUS_AT_TP3 = "at_tp3"
    STATUS_QUEUED = "queued"
    STATUS_DOCKED = "docked"
    STATUS_DEPARTED = "departed"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Methods
    # ─────────────────────────────────────────────────────────────────────────
    def __post_init__(self):
        self.stops_remaining = list(self.manifest)
    
    def total_parcels(self) -> int:
        return sum(stop["parcels"] for stop in self.manifest)
    
    def parcels_for(self, gha: str) -> int:
        for stop in self.manifest:
            if stop["gha"] == gha:
                return stop["parcels"]
        return 0
    
    def next_slot(self) -> Optional[int]:
        if not self.booked_slots:
            return None
        
        remaining_ghas = {stop["gha"] for stop in self.stops_remaining}    # check remaining ghas
        remaining_slots = {
            gha: slot_start for gha, slot_start in self.booked_slots.item()    # check remaining slots
            if gha in remaining_ghas
        }
        return min(remaining_slots.values()) if remaining_slots else None
    
    def next_stop(self) -> Optional[Dict]:
        return self.stops_remaining[0] if self.stops_remaining else None
    
    def complete_stop(self, gha: str):
        self.stops_remaining = [
            stop for stop in self.stops_remaining if stop["gha"] != gha
        ]
        