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
        
# =============================================================================
# GHA TERMINAL
# =============================================================================
class GHATerminal:
    """
    Models one Ground Handling Agent as a pair of SimPy Resources.
    Export docks and import docks are independent — a truck's flow_type
    determines which pool it enters.

    The terminal knows nothing about agent policies. It provides:
      - SimPy processes for truck service
      - Observation helper methods for the MARL layer
    """
    def __init__(
        self,
        env: simpy.Environment,
        gha: str,
        svc_tm: "ServiceTimeModel",
        infra: "InfrastructureLayer",
        cfg: Dict = params
    ):
        self.env = env
        self.gha = gha
        self.svc_tm = svc_tm
        self.infra = infra

        # ── Dock counts ──────────────────────────────────────────────────────
        dock_cfg = cfg["gha"][gha]    # we dont need to pass the whole cfg to self
        self.n_exp = dock_cfg["export"]
        self.n_imp = dock_cfg["import"]
        
        # ── Create SimPy docks ───────────────────────────────────────────────
        self.docks_imp = simpy.Resource(env, capacity=self.n_imp)
        self.docks_exp = simpy.Resource(env, capacity=self.n_exp)
        
        # ── Create queues ────────────────────────────────────────────────────
        self.queue_imp: List[Truck] = []
        self.queue_exp: List[Truck] = []
        
        # ── Track KPIs ───────────────────────────────────────────────────────
        self.gha_stats = {
            "exp": {"processed": 0, "tot_wait": 0, "tot_serv": 0},
            "imp": {"processed": 0, "tot_wait": 0, "tot_serv": 0}
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Resource routing
    # ─────────────────────────────────────────────────────────────────────────
    def _route_dock_pool(self, flow_type: str) -> simpy.Resource:
        return self.docks_exp if flow_type == "export" else self.docks_imp
    
    def _route_queue(self, flow_type: str) -> List[Truck]:
        return self.queue_exp if flow_type == "export" else self.queue_imp
    
    # ─────────────────────────────────────────────────────────────────────────
    # Core SimPy truck processing logic
    # ─────────────────────────────────────────────────────────────────────────
    def process_truck(self, truck: Truck, dtp: "DTPPlatform"):
        """
        SimPy generator: truck arrives at GHA → waits for dock → served → departs.
        Implements R8 slot phase logic via DTPPlatform.
        """
        arrival_time = self.env.now
        flow_type = truck.flow_type
        dock_pool = self._route_dock_pool(flow_type)
        queue = self._route_queue(flow_type)
        
        # ANPR camera recognizes truck
        self.infra.gha_in(arrival_time, truck, self.gha)
        
        # Check slot phase
        slot_start = truck.booked_slots.get(self.gha)
        dock_is_free = dock_pool < dock_pool.capacity    # NOTE: this works only if dock_pool returns the number of occupied docks
        phase = dtp.get_slot_phase(slot_start, arrival_time, dock_is_free)
        
        # NOTE: I might have to tweak the the logic behind an early arrival, because at the moment a truck waits in the queue forever
        if phase == "early":
            # Should have beed sent to tp3, now wait until priority windoe opens
            if slot_start is not None:
                wait = max(0, slot_start - arrival_time)
                yield self.env.timeout(wait)    # yield only stops this specific truck
                
        elif phase == "release":
            dtp.record_late(truck.truck_id)
        
        elif phase == "release_dock_taken":
            dtp.record_late(truck.truck_id)
            truck.status = Truck.STATUS_AT_TP3
            return    # the caller handles the truct redirection at tp3, here we only care about internal gha logistics
        
        elif phase == "no_show":
            dtp.record_no_show(self.gha, slot_start, truck.truck_id)
            truck.status = Truck.STATUS_AT_TP3
            return
        
        # If got to this point it meas that the slot_phase is "priority" or "release", so the truck joins the queue
        truck.status = Truck.STATUS_QUEUED
        queue.append(truck)    # this becomes a list of trucks from the Truck class with all their attributes
        
        # NOTE: Priority window check: if in priority window, we do NOT allow
        # standby trucks to jump ahead. The pool.request() handles this
        # naturally via FIFO ordering in SimPy.
        with dock_pool.request() as req:
            yield req    # wait until a dock is free
            
            queue_time = self.env.now - arrival_time    # kpi logging
            if truck in queue:
                queue.remove(truck)
                
            truck.status = Truck.STATUS_DOCKED
            
            # Dock sensor fires
            dock_id = dock_pool.count    # NOTE: replace with real mapping
            self.infra.dock_start(self.env.now, truck, self.gha, dock_id)
            
            if slot_start is not None:
                dtp.mark_docked(self.gha, slot_start, truck.truck_id)
                
            # Sample service time from distribution
            service_time = self.svc_tm.sample(flow_type)
            yield self.env.timeout(service_time)
            
            # Dock sensor fires (simpy releases automatically at the end of the with block)
            self.infra.dock_end(self.env.now, truck, self.gha, dock_id)
            
            if slot_start is not None:
                dtp.mark_closed(self.gha, slot_start, truck.truck_id)
                
            # Update stats
            stats = self.gha_stats[flow_type]    # NOTE: why not updating original?
            stats["processed"] += 1
            stats["tot_wait"] += queue_time
            stats["tot_serv"] += service_time