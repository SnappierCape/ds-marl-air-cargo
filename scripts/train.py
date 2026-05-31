# =============================================================================
# BENCHMARL TRAINING EXECUTION SCRIPT
# =============================================================================
# DESCRIPTION:
#     Programmatically executes a BenchMARL training experiment.
#     Uses Multi-Agent PPO (MAPPO) to train policies for the Schiphol simulation.
# =============================================================================
from pathlib import Path

import torch

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig

from marl.benchmarl_task import SchipholTask

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "experiments"

# =============================================================================
# TRAINING LOGIC
# =============================================================================
def main():
    # ── Retrieve custom task ────────────────────────────────────────────────
    task = SchipholTask.SCENARIO_MO.get_task(
        config={"max_steps": 1440, "with_orchestrator": True}
    )
    
    print(f'\nLoaded Task:\n{task.name}')
    print(f'\nTask Configuration:\n{task.config}\n')
    
    # ── Experiment hyperparameters ───────────────────────────────────────────
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device="cuda" if torch.cuda.is_available() else "cpu"
    experiment_config.train_device="cuda" if torch.cuda.is_available() else "cpu"
    experiment_config.max_n_iters=None
    experiment_config.max_n_frames=2_211_840
    experiment_config.on_policy_collected_frames_per_batch=34_560
    experiment_config.on_policy_n_minibatch_iters=8
    experiment_config.on_policy_minibatch_size=8640
    experiment_config.lr=5e-4
    experiment_config.parallel_collection=True
    experiment_config.on_policy_n_envs_per_worker=4
    experiment_config.evaluation_interval=276_480
    experiment_config.clip_grad_norm=True
    experiment_config.clip_grad_val=0.5
    experiment_config.save_folder=EXPERIMENT_DIR
    experiment_config.gamma=0.999
    
    # ── MAPPO config ─────────────────────────────────────────────────────────
    algorithm_config = MappoConfig(
        share_param_critic=False,    # groups have distinct obs vectors
        clip_epsilon=0.2,
        entropy_coef=0.01,
        critic_coef=0.5,
        loss_critic_type="l2",
        lmbda=0.99,
        scale_mapping="biased_softplus",
        use_tanh_normal=True,
        minibatch_advantage=True
    )
    
    # ── Neural Network architecture ──────────────────────────────────────────
    policy_model_config = MlpConfig(
        num_cells=[128, 128],
        activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear
    )
    
    critic_model_config = MlpConfig(
    num_cells=[128, 128],
    activation_class=torch.nn.Tanh,
        layer_class=torch.nn.Linear
    )
    
    # ── Assemble the experiment ──────────────────────────────────────────────
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=policy_model_config,
        critic_model_config=critic_model_config,
        seed=42,
        config=experiment_config
    )
    
    # ── Run ──────────────────────────────────────────────────────────────────
    print("\nStarting BenchMARL training loop...")
    experiment.run()
    print("\nTraining finished successfully!")
    
if __name__ == "__main__":
    main()