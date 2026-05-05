# =============================================================================
# ROAD NETWORK MODULE
# =============================================================================
# DESCRIPTION:
#     Calculates stochastic travel times between nodes in the Schiphol cargo
#     area based on the params file.
# =============================================================================
import sys
import os
import numpy as np
from typing import Dict

# Setting base path for local imports
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
import config.config

# =============================================================================
# PARAMETERS IMPORT
# =============================================================================
params = config.load_params()

# =============================================================================
# ROAD MODEL
# =============================================================================
class RoadNetwork:
    """Generates stochastic travel times between nodes."""
    def __init__(self, cfg: Dict):
        self.cfg = cfg["road"]
        self.sigma = self.cfg["sigma"]
        self.segments = self.cfg["segments"]
        self.ghas = self.cfg["nodes"]

    def _apply_noise(self, base_time: float) -> float:
        """Applies lognormal noise to a base travel time to simulate traffic variance."""
        if base_time <= 0:
            return 0.0
        
        mu = np.log(base_time) - (self.sigma**2) / 2
        sampled_time = np.random.lognormal(mean=mu, sigma=self.sigma)
        
        return float(np.clip(sampled_time, base_time * 0.5, base_time * 3.0))    # Hardcoded.

    def gate_to_gha(self, gha: str) -> float:
        target_node = self.ghas.get(gha)
        if not target_node:
            raise ValueError(f"Unknown GHA ID: {gha}")
            
        base = self.segments["N0_N2"] + self.segments[f"N2_{target_node}"]
        return self._apply_noise(base)

    def gate_to_tp3(self) -> float:
        base = self.segments["N0_N1"]
        return self._apply_noise(base)

    def tp3_to_gha(self, gha: str) -> float:
        target_node = self.ghas.get(gha)
        if not target_node:
            raise ValueError(f"Unknown GHA ID: {gha}")
            
        base = self.segments["N1_N2"] + self.segments[f"N2_{target_node}"]
        return self._apply_noise(base)