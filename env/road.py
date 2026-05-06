# =============================================================================
# ROAD NETWORK MODULE
# =============================================================================
# DESCRIPTION:
#     This module creates travel times between each node of the Schiphol
#     landside area, and applies a stochastic noise to those values.
#     It is used by other modules to sample a randomized travel time for a
#     specific segment.
#
# NOISE DISTRIBUTION:
#     The chosen distribution for the noise is a lognormal distribution.
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
    def __init__(self, cfg: Dict = params):
        self.cfg = cfg["road"]
        self.sigma = self.cfg["sigma"]
        self.segments = self.cfg["segments"]
        self.ghas = self.cfg["nodes"]
        self.lbound = self.cfg["lbound"]
        self.ubound = self.cfg["ubound"]

    def _apply_noise(self, base_time: float) -> float:
        """Applies lognormal noise."""
        if base_time <= 0:
            raise ValueError(f'Input base_time: {base_time}. Please input positive base_time.')
        
        mu = np.log(base_time) - (self.sigma**2) / 2
        sampled_time = np.random.lognormal(mean=mu, sigma=self.sigma)
        
        return float(np.clip(sampled_time, base_time * self.lbound, base_time * self.ubound))

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