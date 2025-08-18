from os import listdir
from lib.experiment_results import ExperimentResults
from lib.repositories.results_repository_factory import results_repository_factory
import os
import pandas as pd

from lib.utils import create_dir_if_not_exists

def exp_name_to_key(exp_name: str) -> str:
    if "Random" in exp_name:
        return "RF"
    elif "SVM" in exp_name:
        return "SVM"
    elif "Magalhães" in exp_name:
        return "ST"
    return "unknown"

def generate_hp_comparison_table(global_hp_cmu: dict[str, ExperimentResults], user_hp_cmu: dict[str, ExperimentResults]) -> pd.DataFrame:
    table_series = []
    for exp in global_hp_cmu.keys():
        beneficiados_ajuste_global, beneficiados_ajuste_por_usuario = 0, 0
        ghp_metrics_per_user = global_hp_cmu[exp].get_metrics_per_user()
        uhp_metrics_per_user = user_hp_cmu[exp].get_metrics_per_user()
        for user_id in ghp_metrics_per_user.keys():
            if user_id not in uhp_metrics_per_user:
                continue
            ghp_bacc = ghp_metrics_per_user[user_id].getBAcc()
            uhp_bacc = uhp_metrics_per_user[user_id].getBAcc()
            if ghp_bacc > uhp_bacc:
                beneficiados_ajuste_global += 1
            else:
                beneficiados_ajuste_por_usuario += 1
        serie = pd.Series({"Algorítmo": exp,
                           "Beneficiados pelo ajuste global": beneficiados_ajuste_global,
                           "Beneficiados pelo ajuste por usuário": beneficiados_ajuste_por_usuario})
        table_series.append(serie)
    return pd.DataFrame(table_series)

global_hp_cmu: dict[str, ExperimentResults] = {}
global_hp_keyrecs: dict[str, ExperimentResults] = {}
user_hp_cmu: dict[str, ExperimentResults] = {}
user_hp_keyrecs: dict[str, ExperimentResults] = {}

repo = results_repository_factory()

results_directory_path = "./results"

experiments = []
for f in os.listdir(results_directory_path):
    dir_path = os.path.join(results_directory_path, f)
    if os.path.isdir(dir_path):
        experiments.append(f)

for exp in experiments:
    result = repo.read_results(exp)
    if result is None:
        continue
    if "global" in exp:
        if "CMU" in exp:
            global_hp_cmu[exp_name_to_key(exp)] = result
        else:
            global_hp_keyrecs[exp_name_to_key(exp)] = result
    else:
        if "CMU" in exp:
            user_hp_cmu[exp_name_to_key(exp)] = result
        else:
            user_hp_keyrecs[exp_name_to_key(exp)] = result

cmu_table = generate_hp_comparison_table(global_hp_cmu, user_hp_cmu)
keyrecs_table = generate_hp_comparison_table(global_hp_keyrecs, user_hp_keyrecs)

results_directory_path = "./hp_comparison_tables"
create_dir_if_not_exists(results_directory_path)

cmu_table.to_csv(f"{results_directory_path}/cmu_hp_comparison.csv", index=False)
keyrecs_table.to_csv(f"{results_directory_path}/keyrecs_hp_comparison.csv", index=False)