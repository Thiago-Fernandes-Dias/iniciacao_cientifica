from os import listdir
from os.path import join, isdir

import matplotlib.pyplot as plt
import numpy as np

from lib.repositories.results_repository_factory import results_repository_factory
from lib.utils import create_dir_if_not_exists

repo = results_repository_factory()

directory_path = "./results"

experiments = []
for f in listdir(directory_path):
    dir_path = join(directory_path, f)
    if isdir(dir_path):
        experiments.append(f)

create_dir_if_not_exists('./charts')

for exp in experiments:
    result = repo.read_results(exp)
    if result is None or len(result.model_predictions_per_seed) == 0:
        continue
    experiments_metrics_per_user = result.get_metrics_per_user()
    user_ids = []
    mean_frr = []
    mean_far = []
    mean_ba = []
    for user_id, metrics in experiments_metrics_per_user.items():
        user_ids.append(user_id)
        mean_frr.append(metrics.frr)
        mean_far.append(metrics.far)
        mean_ba.append(metrics.getBAcc())
    x = np.arange(len(user_ids))
    width = 0.4
    if "Keyrecs" in exp:
        split_index = 50

        # Data splitting
        user_ids_part1, user_ids_part2 = user_ids[:split_index], user_ids[split_index:]
        mean_frr_part1, mean_frr_part2 = mean_frr[:split_index], mean_frr[split_index:]
        mean_far_part1, mean_far_part2 = mean_far[:split_index], mean_far[split_index:]
        mean_ba_part1, mean_ba_part2 = mean_ba[:split_index], mean_ba[split_index:]

        y_part1 = np.arange(len(user_ids_part1))
        y_part2 = np.arange(len(user_ids_part2))

        # --- FMR/FNMR Combined Chart ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        
        # Part 1
        ax1.barh(y_part1 - width / 2, mean_frr_part1, width, label='FNMR', alpha=0.7)
        ax1.barh(y_part1 + width / 2, mean_far_part1, width, label='FMR', alpha=0.7)
        ax1.set_ylabel('Usuário')
        ax1.set_xlabel('FMR/FNMR')
        ax1.set_title(f'FNMR e FMR por usuário - Parte 1')
        ax1.legend()
        ax1.set_yticks(y_part1)
        ax1.set_yticklabels(user_ids_part1)
        ax1.invert_yaxis()  # Para mostrar o primeiro usuário no topo
        
        # Part 2
        ax2.barh(y_part2 - width / 2, mean_frr_part2, width, label='FNMR', alpha=0.7)
        ax2.barh(y_part2 + width / 2, mean_far_part2, width, label='FMR', alpha=0.7)
        ax2.set_ylabel('Usuário')
        ax2.set_xlabel('FMR/FNMR')
        ax2.set_title(f'FNMR e FMR por usuário - Parte 2')
        ax2.legend()
        ax2.set_yticks(y_part2)
        ax2.set_yticklabels(user_ids_part2)
        ax2.invert_yaxis()  # Para mostrar o primeiro usuário no topo
        
        plt.suptitle(f'FNMR e FMR por usuário para o experimento "{exp}"', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'./charts/{exp} - FMR e FNMR.png')
        plt.close()

        # --- BAcc Combined Chart ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        
        # Part 1
        ax1.barh(y_part1, mean_ba_part1, width, label='Acurácia balanceada', alpha=0.7, color='green')
        ax1.set_ylabel('Usuário')
        ax1.set_xlabel('BAcc')
        ax1.set_title(f'Acurácia balanceada por usuário - Parte 1')
        ax1.legend()
        ax1.grid(axis='x')
        ax1.set_yticks(y_part1)
        ax1.set_yticklabels(user_ids_part1)
        ax1.invert_yaxis()  # Para mostrar o primeiro usuário no topo
        
        # Part 2
        ax2.barh(y_part2, mean_ba_part2, width, label='Acurácia balanceada', alpha=0.7, color='green')
        ax2.set_ylabel('Usuário')
        ax2.set_xlabel('BAcc')
        ax2.set_title(f'Acurácia balanceada por usuário - Parte 2')
        ax2.legend()
        ax2.grid(axis='x')
        ax2.set_yticks(y_part2)
        ax2.set_yticklabels(user_ids_part2)
        ax2.invert_yaxis()  # Para mostrar o primeiro usuário no topo
        
        plt.suptitle(f'Acurácia balanceada por usuário para o experimento "{exp}"', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'./charts/{exp} - BAcc.png')
        plt.close()
    else:
        # FMR/FNMR
        plt.figure(figsize=(10, 4))
        plt.bar(x - width / 2, mean_frr, width, label='FNMR', alpha=0.7)
        plt.bar(x + width / 2, mean_far, width, label='FMR', alpha=0.7)
        plt.xlabel('Usuário')
        plt.ylabel('FMR/FNMR')
        plt.title(f'FMR e FRR por usuário para o experimento "{exp}"')
        plt.legend()
        plt.grid()
        plt.xticks(x, user_ids, rotation=90)
        plt.tight_layout()
        plt.savefig(join('./charts', f'{exp} - FMR e FNMR.png'))
        plt.close()

        # BAcc
        plt.figure(figsize=(10, 4))
        plt.bar(x, mean_ba, width, label='BAcc', alpha=0.7, color='green')
        plt.xlabel('Usuário')
        plt.ylabel('BAcc')
        plt.title(f'Acurácia balanceada por usuário para o experimento "{exp}"')
        plt.legend()
        plt.xticks(x, user_ids, rotation=90)
        plt.tight_layout()
        plt.savefig(join('./charts', f'{exp} - BAcc.png'))
