# =============================================================================
# ROAD NETWORK MODULE
# =============================================================================
# DESCRIPTION:
#     Calculates stochastic travel times between nodes in the Schiphol cargo
#     area based on the sim_params.yaml configuration.
# =============================================================================
import numpy as np
from typing import Dict

# =============================================================================
# ROAD MODEL
# =============================================================================
class RoadNetwork:
    """
    Simulates physical travel times across the airport infrastructure.
    Nodes:
      N0: Main Gate / Perimeter entry
      N1: TP3 Buffer Zone
      N2: Main road split
      N3: dnata
      N4: wfs
      N5: swiss
      N6: klm
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg.get("travel_time", {})
        self.sigma = self.cfg.get("sigma", 0.20)    # Hardcoded.
        
        self.segments = self.cfg.get("segments", {
            "N0_N1": 2.0,
            "N0_N2": 1.0,
            "N1_N2": 1.0,
            "N2_N3": 1.5,
            "N2_N4": 2.0,
            "N2_N5": 2.5,
            "N2_N6": 1.0
        })
        
        self.gha_nodes = {
            "dnata": "N3",
            "wfs": "N4",
            "swiss": "N5",
            "klm": "N6"
        }

    def _apply_stochastic_noise(self, base_time: float) -> float:
        """
        Applies lognormal noise to a base travel time to simulate traffic variance.
        Base time acts as the expected value (mean).
        """
        if base_time <= 0:
            return 0.0
            
        # Mathematical conversion: mean of lognormal = exp(mu + sigma^2 / 2)
        # Therefore, mu = ln(mean) - sigma^2 / 2
        mu = np.log(base_time) - (self.sigma**2) / 2
        sampled_time = np.random.lognormal(mean=mu, sigma=self.sigma)
        
        return float(np.clip(sampled_time, base_time * 0.5, base_time * 3.0))    # Hardcoded.

    def time_gate_to_gha(self, gha_id: str) -> float:
        """Calculate travel time from Main Gate (N0) straight to a GHA."""
        target_node = self.gha_nodes.get(gha_id.lower())
        if not target_node:
            raise ValueError(f"Unknown GHA ID: {gha_id}")
            
        base = self.segments["N0_N2"] + self.segments[f"N2_{target_node}"]
        return self._apply_stochastic_noise(base)

    def time_gate_to_tp3(self) -> float:
        """Calculate travel time from Main Gate (N0) to the TP3 buffer (N1)."""
        base = self.segments["N0_N1"]
        return self._apply_stochastic_noise(base)

    def time_tp3_to_gha(self, gha_id: str) -> float:
        """Calculate travel time from the TP3 buffer (N1) to a specific GHA."""
        target_node = self.gha_nodes.get(gha_id.lower())
        if not target_node:
            raise ValueError(f"Unknown GHA ID: {gha_id}")
            
        base = self.segments["N1_N2"] + self.segments[f"N2_{target_node}"]
        return self._apply_stochastic_noise(base)