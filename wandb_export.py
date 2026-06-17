# =============================================================================
# WANDB DATA EXTRACTION MODULE
# =============================================================================

import wandb
import pandas as pd

api = wandb.Api()

# Change the run id to the specific run you wanna extract data for
run = api.run("/ulriconava-main-none/benchmarl/runs/mappo_scenario_mo_mlp__e91654ce_26_06_09-15_32_50")

# Save to csv
df = run.history()
df = pd.DataFrame(df)

cols_to_keep = [
    "train/transporter/explained_variance",
    "train/ghas/explained_variance",
    "train/orchestrator/explained_variance",
    "train/transporter/entropy",
    "train/ghas/entropy",
    "train/orchestrator/entropy",
    "collection/transporter/reward/episode_reward_mean",
    "collection/ghas/reward/episode_reward_mean",
    "collection/orchestrator/reward/episode_reward_mean",
    "eval/reward/episode_reward_mean"
]

# df = df[cols_to_keep]

df["num_epoch"] = df.index + 1
df = df[df["num_epoch"] % 5 == 0] 

df.to_csv("wandb_export.csv", index=False)