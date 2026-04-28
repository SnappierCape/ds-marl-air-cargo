# =============================================================================
# TEST 1 - BASE LOGIC
# =============================================================================
import sys
import os
import numpy as np

# Setting base path for local imports.
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from env.service_time import ServiceTimeModel
from env.road import RoadNetwork
from env.dtp_platform import DTPPlatform
from env.infrastructure import InfrastructureLayer

# ─────────────────────────────────────────────────────────────────────────────
# Create mock config file
# ─────────────────────────────────────────────────────────────────────────────
mock_cfg = {
    "service_time": {
        "export": {
            "family": "lognormal",
            "params": {"mu": 3.4, "sigma": 0.2}
        },
        "import": {
            "family": "lognormal",
            "params": {"mu": 2.8, "sigma": 0.15}
        }
    },
    "travel_time": {
        "sigma": 0.15,
        "segments": {
            "N0_N1": 2.0, "N0_N2": 1.0, "N1_N2": 1.0, "N2_N3": 1.5,
            "N2_N4": 2.0, "N2_N5": 2.5, "N2_N6": 1.0
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Create mock truck
# ─────────────────────────────────────────────────────────────────────────────
class MockTruck:
    def __init__(self):
        self.truck_id = "TRK-999"
        self.flow_type = "export"
        self.timestamps = {}
        self.booked_slots = {"swissport": 480}    # 480 would be 08:00
    def parcels_for(self, gha_id: str):
        return 5
    def total_parcels(self):
        return 15
    def next_slot_window(self):
        return list(self.booked_slots.values())[0]

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

# ── Test 1.1: Service Time Module ────────────────────────────────────────────
print(f'\n--- Starting test 1.1: Service Time Module ---')

model = ServiceTimeModel(mock_cfg["service_time"])

exp_time = [model.sample("export") for _ in range(200)]
imp_time = [model.sample("import") for _ in range(200)]

print(f'Export time --> Min: {min(exp_time):.2f}m | Max: {max(exp_time):.2f}m | Mean: {np.mean(exp_time):.2f}')
print(f'Import time --> Min: {min(imp_time):.2f}m | Max: {max(imp_time):.2f}m | Mean: {np.mean(imp_time):.2f}')

# ── Test 1.2: Road Infrastructure ────────────────────────────────────────────
print(f'\n--- Starting test 1.2: Road Infrastructure ---')
road = RoadNetwork(mock_cfg)

print("Itinerary: Main Gate --> Dnata")
for i in range(20):
    time = road.time_gate_to_gha("dnata")
    print(f'    Trip {i+1:>2}, {time:.2f}m')

print("Itinerary: Menzies --> TP3")
for i in range(20):
    time = road.time_tp3_to_gha("dnata")
    print(f'    Trip {i+1:>2}, {time:.2f}m')
    
# ── Test 1.3: DTP Platform ───────────────────────────────────────────────────
print("\n--- Starting test 1.3: DTP Platform ---")
dtp = DTPPlatform(env = None)

# Publish a slot.
print("Publishing 2 slots at 08:00 for Swissport:")
dtp.publish_slot(gha_id="swissport", time_min=480, capacity=2)

# Try to book slots.
print(f'    Booking slot 1/2: {dtp.book_slot("swissport", 480)}')
print(f'    Booking slot 2/2: {dtp.book_slot("swissport", 480)}')
print(f'    Booking slot 3/2 (Overbooking): {dtp.book_slot("swissport", 480)}')

# Test truck arrival logic.
print("Testing truck arrival logic:")
print(f'    Phase at 07:40: {dtp.get_slot_phase("swissport", 480, 460)}')
print(f'    Phase at 08:05: {dtp.get_slot_phase("swissport", 480, 485)}')
print(f'    Phase at 08:25: {dtp.get_slot_phase("swissport", 480, 505)}')
print(f'    Phase at 09:00: {dtp.get_slot_phase("swissport", 480, 540)}')

# ── Test 1.4: System logging ─────────────────────────────────────────────────
print("\n--- Starting test 1.4: System Logging ---")
logger = InfrastructureLayer()
truck = MockTruck()

# Truck enters main gate and goes to gha.
logger.gate_in(465, truck)
logger.gha_in(470, truck, "swissport")

events = logger.get_all_events()
for e in events:
    print(f'Time: {e.sim_time:.2f} | Checkpoint: {e.checkpoint.value} | Truck: {e.truck_id}')

print(f'Checking if the truck timestamps updated: {truck.timestamps}')