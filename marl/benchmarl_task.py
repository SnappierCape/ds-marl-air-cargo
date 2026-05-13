# =============================================================================
# BENCHMARL TASK MODULE
# =============================================================================
# DESCRIPTION:
#     Registers the Schiphol simulation as a BenchMARL Task.
#     BenchMARL wraps your PettingZoo env through TorchRL's PettingZooWrapper.
#
# STRUCTURE:
#     SchipholTask (Enum)  — one entry per scenario (M and MO)
#     SchipholConfig       — task-level parameters (dataclass)
#
# HOW BENCHMARL USES THIS:
#     1. Experiment calls task.get_env_fun() to get a factory function
#     2. Factory returns a TorchRL-compatible env built from your PettingZoo env
#     3. BenchMARL runs MAPPO rollouts against that env
#     4. All logging, checkpointing, and evaluation handled automatically
#
# USAGE:
#     from benchmarl_task import SchipholTask
#     task = SchipholTask.SCENARIO_M.get_from_yaml()
# =============================================================================
import sys
import os
from dataclasses import dataclass, MISSING
from typing import Callable, Dict, List, Optional

from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs import EnvBase, TransformedEnv, RewardSum
from benchmarl.environments import Task

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
import config.config as config
from env.schiphol_env import SchipholCargoEnv

params = config.load_params()

# =============================================================================
# TASK CONFIGURATION DATACLASS
# =============================================================================
@dataclass
class SchipholConfig:
    """
    Task-level parameters exposed to BenchMARL's config system.
    These can be overridden from the command line via Hydra.
    """
    # Episode length in steps — controls when BenchMARL resets the env
    max_steps: int = MISSING

    # Whether to use the Orchestrator agent (Scenario MO)
    with_orchestrator: bool = MISSING

# =============================================================================
# TASK ENUM
# =============================================================================
class SchipholTask(Task):
    """
    BenchMARL Task enum. One entry per scenario.

    Usage:
        task = SchipholTask.SCENARIO_M.get_from_yaml()
        task = SchipholTask.SCENARIO_MO.get_from_yaml()
    """
    SCENARIO_M  = None    # filled by enum machinery, see __new__
    SCENARIO_MO = None

    # ── BenchMARL required: environment factory ───────────────────────────────
    def get_env_fun(
        self,
        seed: Optional[int],
        device: str,
    ) -> Callable[[], EnvBase]:
        """
        Returns a callable that BenchMARL uses to create environment instances.
        BenchMARL may call this multiple times for parallel rollout workers.
        """
        task_cfg: SchipholConfig = self.config
        with_orchestrator = task_cfg.with_orchestrator

        def make_env() -> EnvBase:
            # Create your PettingZoo env
            pz_env = SchipholCargoEnv(with_orchestrator=with_orchestrator)

            # Wrap in TorchRL's PettingZoo adapter
            # categorical_actions=True because our action spaces are Discrete
            torchrl_env = PettingZooWrapper(
                env=pz_env,
                use_mask=True,           # pass action masks through to MAPPO
                categorical_actions=True,
                device=device,
                seed=seed,
            )

            # Add standard transforms:
            # RewardSum: accumulates episode return per agent group
            # StepCounter: tracks step count for episode termination
            torchrl_env = TransformedEnv(
                torchrl_env,
                RewardSum(
                    in_keys=[torchrl_env.reward_key],
                    out_keys=[("agents", "episode_reward")]
                ),
            )
            return torchrl_env

        return make_env

    # ── BenchMARL required: capability flags ──────────────────────────────────
    def supports_continuous_actions(self) -> bool:
        return False    # all action spaces are Discrete

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return False    # no visual rendering

    def max_steps(self, env: EnvBase) -> int:
        """Episode length in steps."""
        return self.config.max_steps

    # ── BenchMARL required: agent grouping ────────────────────────────────────
    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        """
        Maps group names to lists of agent names.

        BenchMARL uses groups to determine which agents share policy networks,
        losses, and replay buffers. We put all agents in one group ("agents")
        so they share the same MAPPO policy — this is the standard setup for
        cooperative MARL and for mixed settings where you want a single
        unified policy to emerge.

        If you wanted the Orchestrator to have a completely separate policy
        architecture, you would put it in its own group here. For now, one
        group keeps things simple and comparable across scenarios.
        """
        return {"agents": env.possible_agents}

    # ── BenchMARL required: observation spec ──────────────────────────────────
    def observation_spec(self, env: EnvBase):
        """
        Returns the observation spec as seen by BenchMARL.
        TorchRL/PettingZooWrapper builds this automatically from your
        PettingZoo observation_space() declarations — no manual work needed.
        """
        return env.observation_spec

    # ── BenchMARL required: action spec ───────────────────────────────────────
    def action_spec(self, env: EnvBase):
        """
        Returns the action spec. Again built automatically by PettingZooWrapper.
        """
        return env.full_action_spec

    # ── BenchMARL required: state spec (for centralized critic) ───────────────
    def state_spec(self, env: EnvBase):
        """
        Global state spec used by MAPPO's centralized critic.
        We return None here and let BenchMARL concatenate agent observations
        into the global state automatically — the standard CTDE approach.
        """
        return None

    # ── BenchMARL required: config class ──────────────────────────────────────
    @staticmethod
    def associated_class():
        """Links this Task to its configuration dataclass."""
        return SchipholConfig