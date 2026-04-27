# =============================================================================
# PETTINGZOO FINAL WRAPPER
# =============================================================================
# DESCRIPTION:
#     To do...
# =============================================================================

import yaml, simpy, numpy as np
from pettingzoo import ParallelEnv
import gymnasium as gym

from .simulation import GHATerminal, TP3Buffer, Truck
from .dtp_platform import DTPPlatform
from .infrastructure import InfrastructureLayer
from .service_time  import ServiceTimeModel
from .demand import DemandGenerator
from .kpi_tracker import KPITracker
from agents.transporter import TransporterAgent
from agents.gha import GHAAgent
from agents.orchestrator import OrchestratorAgent

# =============================================================================
# INITIALIZATION
# =============================================================================
GHA_IDS = ["dnata", "klm", "swissport", "menzies_wfs"]

# =============================================================================
# MAIN WRAPPER
# =============================================================================
class SchipholCargoEnv(ParallelEnv):
    metadata = {"name": "schiphol_cargo_v0"}

    def __init__(self, sim_params_path: str, scenario_cfg: dict):
        # Load ALL numerical parameters from YAML
        with open(sim_params_path) as f:
            self.cfg = yaml.safe_load(f)

        self.with_orchestrator = scenario_cfg.get("with_orchestrator", False)
        self.alpha  = self.cfg["marl"]["alpha"]
        self.step_m = self.cfg["marl"]["step_minutes"]

        self.possible_agents = ["transporter"] + [f"gha_{g}" for g in GHA_IDS]
        if self.with_orchestrator:
            self.possible_agents.append("orchestrator")
        self.agents = self.possible_agents[:]

        # Agent logic instances (stateless — only compute obs/reward/mask)
        self._t_agent = TransporterAgent(self.cfg)
        self._g_agents = {g: GHAAgent(self.cfg) for g in GHA_IDS}
        self._o_agent  = OrchestratorAgent(self.cfg) if self.with_orchestrator else None

    def observation_space(self, agent):
        dims = {
            "transporter":     32,
            "gha_dnata":       21,  "gha_klm":         21,
            "gha_swissport":   21,  "gha_menzies_wfs": 21,
            "orchestrator":    119,
        }
        return gym.spaces.Box(0.0, 1.0, shape=(dims[agent],), dtype=np.float32)

    def action_space(self, agent):
        dims = {
            "transporter":     50,
            "gha_dnata":       3,   "gha_klm":         3,
            "gha_swissport":   3,   "gha_menzies_wfs": 3,
            "orchestrator":    20,
        }
        return gym.spaces.Discrete(dims[agent])

    def reset(self, seed=None, options=None):
        self.sim = simpy.Environment()

        # Infrastructure layer — created first, passed to everything
        self.infra = InfrastructureLayer()

        # Service time model — reads distribution config
        self.svc   = ServiceTimeModel(self.cfg["service_time"])

        # GHA terminals — each gets infra + svc model
        self.terminals = {
            g: GHATerminal(self.sim, g, self.cfg, self.svc, self.infra)
            for g in GHA_IDS
        }

        # TP3 — 140 slots, constrained
        self.tp3 = TP3Buffer(self.sim, self.infra)

        # DTP platform — rule engine
        self.dtp = DTPPlatform(self.sim)

        # KPI tracker — reads from infra.event_log
        self.kpi = KPITracker(self.infra)

        self.agents = self.possible_agents[:]

        # Pre-publish slots for the episode
        ep_start = self.cfg["demand"]["episode_start_min"]
        ep_end   = self.cfg["demand"]["episode_end_min"]
        for g in GHA_IDS:
            for t in range(ep_start, ep_end, 30):
                self.dtp.publish_slot(g, float(t))

        # Start demand generator
        self.sim.process(
            DemandGenerator(self.sim, self.cfg, self.dtp,
                             self.terminals, self.tp3,
                             self.infra).run()
        )

        obs   = {a: self._obs(a)  for a in self.agents}
        infos = {a: {"action_mask": self._mask(a)} for a in self.agents}
        return obs, infos

    def step(self, actions: dict):
        # 1. Apply actions
        for agent, action in actions.items():
            self._apply(agent, action)

        # 2. Advance simulation
        self.sim.run(until=self.sim.now + self.step_m)

        # 3. Ingest new sensor events into KPI tracker
        new_events = self.infra.flush_step_buffer()
        self.kpi.ingest(new_events)

        # 4. Compute global reward
        r_global = self.kpi.global_reward(self.cfg["marl"]["reward_weights"])

        obs   = {a: self._obs(a)  for a in self.agents}
        rews  = {a: self._rew(a, r_global) for a in self.agents}
        dones = {a: self.sim.now >= self.cfg["demand"]["episode_end_min"]
                  for a in self.agents}
        trunc = {a: False for a in self.agents}
        info  = {a: {"action_mask": self._mask(a)} for a in self.agents}

        if all(dones.values()):
            self.agents = []
        return obs, rews, dones, trunc, info

    def _rew(self, agent, r_global):
        r_priv = self._private_reward(agent)
        if agent == "orchestrator":
            return r_global  # Orchestrator: global only
        return (1 - self.alpha) * r_priv + self.alpha * r_global