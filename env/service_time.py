# =============================================================================
# SERVICE TIME SAMPLER
# =============================================================================
# DESCRIPTION:
#     This module contains the utilities to extract a sample dock service time
#     following a lognormal distribution.
# =============================================================================
from typing import Dict

import numpy as np

from config.config import load_params
params = load_params()

# =============================================================================
# SERVICE TIME MODEL
# =============================================================================
class ServiceTimeModel:
    def __init__(self, cfg: Dict = params):
        self.exp = cfg["service_time"]["export"]
        self.imp = cfg["service_time"]["import"]
    
    def sample(self, flow_type: str) -> float:
        """Samples one service time from the distribution."""
        if flow_type not in ("export", "import"):
            raise ValueError(f'Flow type {flow_type} is not valid.')
        
        cfg = self.exp if flow_type == "export" else self.imp
        raw = np.random.lognormal(mean=cfg["mu"], sigma=cfg["sigma"])
        lo, hi = cfg["bounds"]
        return float(np.clip(raw, lo, hi))
    
    def mean(self, flow_type: str) -> float:
        """Analytical mean of the lognormal distribution."""
        if flow_type not in ("export", "import"):
            raise ValueError(f'Flow type {flow_type} is not valid.')
        
        cfg = self.exp if flow_type == "export" else self.imp
        return np.exp(cfg["mu"] + cfg["sigma"] ** 2 / 2)