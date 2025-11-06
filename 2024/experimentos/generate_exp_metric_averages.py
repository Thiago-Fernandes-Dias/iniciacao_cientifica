import os

import pandas as pd

from lib.repositories.results_repository_factory import results_repository_factory

repo = results_repository_factory()

results_dir = "./results"

exp_average_bacc = {}
exp_average_frr = {}
exp_average_far = {}

for exp in os.listdir(results_dir):
    exp_results = repo.read_results(exp)
    if exp_results is None or len(exp_results.model_predictions_per_seed) == 0:
        continue
    metrics_per_user = exp_results.get_metrics_per_user()
    total_bacc = sum(m.getBAcc() for m in metrics_per_user.values())
    total_frr = sum(m.frr for m in metrics_per_user.values())
    total_far = sum(m.far for m in metrics_per_user.values())
    num_users = len(metrics_per_user)
    exp_average_bacc[exp] = total_bacc / num_users if num_users > 0 else 0
    exp_average_frr[exp] = total_frr / num_users if num_users > 0 else 0
    exp_average_far[exp] = total_far / num_users if num_users > 0 else 0

data = {"BACC": exp_average_bacc, "FRR": exp_average_frr, "FAR": exp_average_far}

df = pd.DataFrame(data).T
df.columns = list(exp_average_bacc.keys())
df.index = ["BACC", "FRR", "FAR"]

df.to_csv(f"{results_dir}/exp_metric_averages.csv")
