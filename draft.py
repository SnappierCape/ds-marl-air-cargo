import wandb
api = wandb.Api()
run = api.run("/ulriconava-main-none/benchmarl/runs/mappo_scenario_mo_mlp__38e34ebc_26_06_01-19_05_13")
df = run.history()
output_csv = "wandb_results.csv"
df.to_csv(output_csv, index=False)