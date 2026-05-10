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
import config.config as config

params = config.load_params()

# =============================================================================
# ROAD MODEL
# =============================================================================
class RoadNetwork:
    def __init__(self, cfg: Dict = params):
        self.sigma = cfg["sigma"]
        self.lbound = cfg["lbound"]
        self.ubound = cfg["ubound"]
        self.segments = cfg["segments"]
        self.nodes = cfg["nodes"]

    def _apply_noise(self, base_time: float) -> float:
        """Applies lognormal noise."""
        if base_time <= 0:
            raise ValueError(f'Input base_time: {base_time}. Please input positive base_time.')
        
        mu = np.log(base_time) - (self.sigma**2) / 2    # NOTE: i need to understand this transformation
        sampled_time = np.random.lognormal(mean=mu, sigma=self.sigma)
        
        return float(np.clip(sampled_time, base_time * self.lbound, base_time * self.ubound))
    
    def time_from_to(self, start: str, end: str) -> float:
        """Calculates base travel time between 2 nodes."""
        if start not in self.nodes.keys() or end not in self.nodes.keys():
            raise ValueError(f'Invalid input nodes: {start}, {end}. Please input valid nodes.')
        
        start_node = self.nodes[start]
        end_node = self.nodes[end]
        
        segment = f'{min(start_node, end_node)}_{max(start_node, end_node)}'
        
        return self._apply_noise(self.segments[segment])