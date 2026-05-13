# scripts/train.py
import sys
import os
import argparse

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from benchmarl.hydra_config import load_experiment_from_yaml
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from marl.benchmarl_task import SchipholTask
from scripts.register_task import *    # trigger registration

def train(scenario: str, seed: int = 0):
    print(f"Training {scenario} | seed {seed}")

    # Load task from YAML
    if scenario == "scenario_m":
        task = SchipholTask.SCENARIO_M.get_from_yaml()
    else:
        task = SchipholTask.SCENARIO_MO.get_from_yaml()

    # MAPPO with GRU (handles partial observability)
    algorithm_config = MappoConfig.get_from_yaml()

    # Actor: GRU to give agents memory across steps
    # Critic: MLP on concatenated observations (CTDE)
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Experiment-level hyperparameters
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_frames = 5_000_000
    experiment_config.on_policy_collected_frames_per_batch = 1440 * 4
    experiment_config.evaluation_interval = 50_000
    experiment_config.loggers = ["wandb"]
    experiment_config.project_name = "schiphol-marl-thesis"
    experiment_config.experiment_name = f"{scenario}_seed{seed}"
    experiment_config.checkpoint_interval = 100_000

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
    )
    experiment.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["scenario_m", "scenario_mo"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.scenario, args.seed)