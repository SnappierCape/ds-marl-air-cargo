import wandb
api = wandb.Api()
run = api.run("/ulriconava-main-none/benchmarl/runs/mappo_scenario_mo_mlp__19da1222_26_05_31-16_37_15")
df = run.history()
output_csv = "wandb_results.csv"
df.to_csv(output_csv, index=False)