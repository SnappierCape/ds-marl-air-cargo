# =============================================================================
# BENCHMARL TASK MODULE
# =============================================================================
# DESCRIPTION:
#     Registers the Schiphol simulation as a BenchMARL Task.
#     BenchMARL wraps PettingZoo env through TorchRL's PettingZooWrapper.
#
# STRUCTURE:
#     SchipholTask (Enum)  — one entry per scenario (M and MO)
#     SchipholConfig       — task-level parameters (dataclass)
#
# HOW BENCHMARL USES THIS:
#     1. Experiment calls task.get_env_fun() to get a factory function
#     2. Factory returns a TorchRL-compatible env built from PettingZoo env
#     3. BenchMARL runs MAPPO rollouts against that env
#     4. All logging, checkpointing, and evaluation handled automatically
#
# HETEROGENEOUS AGENTS:
#     Because Transporter, GHAs, and Orchestrator have different observation 
#     and action shapes, they MUST be separated into different "groups". 
#     BenchMARL will instantiate separate neural networks (policies/critics) 
#     for each group automatically.
# =============================================================================
from dataclasses import dataclass, MISSING
from typing import Callable, Dict, List, Optional

from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs import EnvBase, TransformedEnv, RewardSum, StepCounter

from benchmarl.environments import Task

from env.schiphol_env import SchipholCargoEnv

from config.config import load_params
params = load_params()

# =============================================================================
# TASK CONFIGURATION DATACLASS
# =============================================================================
@dataclass
class SchipholConfig:
    """
    Task-level parameters exposed to BenchMARL's config system.
    These can be overridden from the command line via Hydra.
    """
    max_steps: int = MISSING
    with_orchestrator: bool = MISSING

# =============================================================================
# TASK ENUM
# =============================================================================
class SchipholTaskImplementation:
    """
    BenchMARL Task enum. One entry per scenario.
    The string values act as the identifier for Hydra YAML configs.
    """
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = SchipholConfig(**config)
    
    @staticmethod
    def env_name() -> str:
        return "schiphol"

    # ── BenchMARL required: environment factory ──────────────────────────────
    def get_env_fun(self, seed: Optional[int], device: str, continuous_actions: bool, num_envs: int) -> Callable[[], EnvBase]:
        """
        Returns a callable that BenchMARL uses to create environment instances.
        BenchMARL may call this multiple times for parallel rollout workers.
        """
        task_cfg: SchipholConfig = self.config
        with_orch = task_cfg.with_orchestrator
        if with_orch is MISSING:
            with_orch = (self.name == "SCENARIO_MO")

        def make_env() -> EnvBase:
            pz_env = SchipholCargoEnv(with_orchestrator=with_orch)    # pettingzoo env
            group_mapping = {
                "transporter": ["transporter"],
                "ghas": [a for a in pz_env.possible_agents if a in params["ghas"].keys()]
            }
            if with_orch:
                group_mapping["orchestrator"] = ["orchestrator"]
                
            # Transform into torchrl language
            torchrl_env = PettingZooWrapper(
                env=pz_env,
                group_map=group_mapping,
                use_mask=True,    # pass action masks through to MAPPO
                categorical_actions=True,    # action space is discrete
                device=device,
                seed=seed,
            )
            torchrl_env = TransformedEnv(torchrl_env)
            torchrl_env.append_transform(StepCounter(task_cfg.max_steps))
            
            return torchrl_env

        return make_env

    # ── BenchMARL required: capability flags ─────────────────────────────────
    def supports_continuous_actions(self) -> bool:
        return False
    
    def supports_discrete_actions(self) -> bool:
        return True
    
    def has_render(self, env: EnvBase) -> bool:
        return False
    
    def max_steps(self, env: EnvBase) -> int:
        return self.config.max_steps
    
    def get_env_transforms(self, env: EnvBase) -> List:
        """
        Returns a sequence of TorchRL transforms to apply to the environment.
        We remove the agent liveness 'mask' key from each group because it has 
        a scalar shape [1] that lacks a trailing feature dimension, which 
        causes BenchMARL's MLP feature validator to crash.
        """
        from torchrl.envs.transforms import ExcludeTransform
        
        group_map = self.group_map(env)
        keys_to_exclude = [(group, "mask") for group in group_map.keys()]
        
        return [ExcludeTransform(*keys_to_exclude)]
    
    def info_spec(self, env: EnvBase):
        return None
    
    def get_reward_sum_transform(self, env: EnvBase):
        """
        Returns an official RewardSum transform instance. 
        BenchMARL uses this to track episodic performance metrics.
        """
        group_map = self.group_map(env)
        reward_keys = [(group, "reward") for group in group_map.keys()]
        ep_reward_keys = [(group, "episode_reward") for group in group_map.keys()]
        
        return RewardSum(in_keys=reward_keys, out_keys=ep_reward_keys)
    
    # ── BenchMARL required: agent grouping ───────────────────────────────────
    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.base_env.group_map
    
    # ── BenchMARL required: tensor specs ─────────────────────────────────────
    def observation_spec(self, env: EnvBase):
        return env.observation_spec
    
    def action_spec(self, env: EnvBase):
        return env.full_action_spec
    
    def state_spec(self, env: EnvBase):
        return None    # this tells BenchMARL to concatenate the agents' obs
    
    def action_mask_spec(self, env: EnvBase):
        return None
    
    def get_replay_buffer_transforms(self, env: EnvBase, group: str) -> List:
        return []
    
    def log_info(self, batch):
        return {}
    
# ── Config Linking ───────────────────────────────────────────────────────
class SchipholTask(Task):
    SCENARIO_M = "scenario_m"
    SCENARIO_MO = "scenario_mo"
    
    @staticmethod
    def associated_class():
        return SchipholTaskImplementation