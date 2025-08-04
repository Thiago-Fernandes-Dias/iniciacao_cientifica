from lib.constants import GENUINE_LABEL, IMPOSTOR_LABEL
import matplotlib.pyplot as plt
import itertools
from lib.user_model_metrics import UserModelMetrics
from lib.utils import create_dir_if_not_exists, seeds_range
from lib.repositories.results_repository_factory import results_repository_factory
from os.path import join, isdir
from os import listdir
import numpy as np

repo = results_repository_factory()

directory_path = "./results"

experiments = []
for f in listdir(directory_path):
    dir_path = join(directory_path, f)
    if isdir(dir_path) and "hp" in f:
        experiments.append(f)

experiments_metrics_per_user = {}

create_dir_if_not_exists('./grafics')

for exp in experiments:
    result = repo.read_results(exp)
    if result is None:
        continue
    else:
        experiments_metrics_per_user[exp] = {}
        user_keys = result.model_predictions["user_id"].drop_duplicates().tolist()
        for user_key, seed in itertools.product(user_keys, seeds_range):
            if user_key not in experiments_metrics_per_user[exp]:
                experiments_metrics_per_user[exp][user_key] = []
            user_predictions_df = result.model_predictions[(result.model_predictions["user_id"] == user_key) & (result.model_predictions["seed"] == seed)]
            total_impostor_attempts = len(user_predictions_df[user_predictions_df["expected"] == IMPOSTOR_LABEL])
            accepted_impostor_attempts = len(user_predictions_df[(user_predictions_df["expected"] == IMPOSTOR_LABEL) & (user_predictions_df["predicted"] == GENUINE_LABEL)])
            total_genuine_attempts = len(user_predictions_df[user_predictions_df["expected"] == GENUINE_LABEL])
            rejected_genuine_attempts = len(user_predictions_df[(user_predictions_df["expected"] == GENUINE_LABEL) & (user_predictions_df["predicted"] == IMPOSTOR_LABEL)])
            user_model_metrics = UserModelMetrics(
                frr=rejected_genuine_attempts / total_genuine_attempts if total_genuine_attempts > 0 else 0,
                far=accepted_impostor_attempts / total_impostor_attempts if total_impostor_attempts > 0 else 0
            )
            experiments_metrics_per_user[exp][user_key].append(user_model_metrics)

for exp, users_metrics in experiments_metrics_per_user.items():
    user_ids = []
    mean_frr = []
    mean_far = []
    for user_id, metrics_list in users_metrics.items():
        user_ids.append(user_id)
        frr_values = [m.frr for m in metrics_list]
        far_values = [m.far for m in metrics_list]
        mean_frr.append(np.mean(frr_values))
        mean_far.append(np.mean(far_values))
        x = np.arange(len(user_ids))
        width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, mean_frr, width, label='FRR', alpha=0.7)
    plt.bar(x + width/2, mean_far, width, label='FAR', alpha=0.7)
    plt.xlabel('User ID')
    plt.ylabel('Rate')
    plt.title(f'FRR and FAR per User for Experiment {exp}')
    plt.legend()
    plt.xticks(x, user_ids, rotation=90)
    plt.tight_layout()
    plt.savefig(f'./grafics/{exp}_frr_far_per_user.png')
    # plt.figure(figsize=(10, 5))
    # plt.bar(user_ids, mean_frr, alpha=0.7, label='FRR')
    # plt.bar(user_ids, mean_far, alpha=0.7, label='FAR', bottom=mean_frr)
    # plt.xlabel('User ID')
    # plt.ylabel('Rate')
    # plt.title(f'FRR and FAR per User for Experiment {exp}')
    # plt.legend()
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.savefig(f'./grafics/{exp}_frr_far_per_user.png')
        