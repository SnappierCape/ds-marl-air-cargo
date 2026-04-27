# =============================================================================
# SERVICE TIME SAMPLER
# =============================================================================
# DESCRIPTION:
#     This module contains the utilities to extract a sample service time from
#     the statistical distribution present in /config/sim_params.yaml.
#     This allows to decouple the distribution from the sampler.
# =============================================================================
import numpy as np
from typing import Dict

# =============================================================================
# SERVICE TIME MODEL
# =============================================================================
class ServiceTimeModel:
    """
    Config-driven service time sampler.

    Supported families:
      "uniform"   → params: {low, high}
      "lognormal" → params: {mu, sigma}
      "gamma"     → params: {shape, scale}
      "empirical" → params: {data_path}

    Usage:
      model = ServiceTimeModel(cfg["service_time"])
      t = model.sample("export")    # → float, minutes
      t = model.sample("import")    # → float, minutes
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self._validate(cfg)

    def _validate(self, cfg):
        for flow in ["export", "import"]:
            assert flow in cfg, f"Missing service time config for flow: {flow}"
            assert "family" in cfg[flow] and "params" in cfg[flow]

    def sample(self, flow_type: str) -> float:
        """
        Draw one service time sample for the given flow type.
        
        Parameters:
        -----------
        flow_type : str
            Either "Import" or "Export".
        
        Returns:
        --------
        : float
            Simulation time sample in minutes.
        """
        spec = self.cfg[flow_type]
        family = spec["family"]
        p = spec["params"]

        if family == "uniform":
            return np.random.uniform(p["low"], p["high"])

        elif family == "lognormal":
            raw = np.random.lognormal(mean=p["mu"], sigma=p["sigma"])
            bounds = {"export": (15, 60), "import": (10, 30)}    # Hardcoded
            lo, hi = bounds[flow_type]
            return float(np.clip(raw, lo, hi))

        elif family == "gamma":
            raw = np.random.gamma(shape=p["shape"], scale=p["scale"])
            bounds = {"export": (15, 60), "import": (10, 30)}
            lo, hi = bounds[flow_type]
            return float(np.clip(raw, lo, hi))

    def mean(self, flow_type: str) -> float:
        """Analytical mean, used to initialize reward normalization."""
        spec = self.cfg[flow_type]
        p = spec["params"]
        if spec["family"] == "uniform":
            return (p["low"] + p["high"]) / 2
        elif spec["family"] == "lognormal":
            return np.exp(p["mu"] + p["sigma"]**2 / 2)
        return None