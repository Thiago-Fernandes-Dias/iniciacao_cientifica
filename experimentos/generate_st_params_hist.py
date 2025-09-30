from lib.repositories.results_repository_factory import results_repository_factory
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt

repo = results_repository_factory()

st_results_per_user = repo.read_results("Magalhães com HPO por usuário (CMU)")
st_results_global = repo.read_results("Magalhães com HPO global (CMU)")

if st_results_global is None or st_results_per_user is None:
    print("Resultados não encontrados.")
    exit(1)
ic(st_results_global.model_predictions_per_seed[0])

ts_per_user: dict[str, list[float]] = {}

if st_results_per_user is not None:
    for i, hp in enumerate(st_results_per_user.hp_per_seed):
        for user_id, params in hp.items():
            if user_id not in ts_per_user:
                ts_per_user[user_id] = []
            ts_per_user[user_id].append(params['threshold'])


avg_t_per_user = {user_id: np.mean(thresholds) for user_id, thresholds in ts_per_user.items()}

ts_global_per_seed = [hp['threshold'] for hp in st_results_global.hp_per_seed[:5] if 'threshold' in hp]
ts_global_avg = np.mean(ts_global_per_seed).item()

plt.figure(figsize=(10, 6))
plt.tight_layout()
plt.axvline(ts_global_avg, color='red', linestyle='dashed', linewidth=2, label=f'Threshold global: {ts_global_avg:.2f}')
plt.legend()
plt.hist(list(avg_t_per_user.values()), bins=20, edgecolor='black')
plt.xlabel('Threshold médio por usuário')
plt.ylabel('Quantidade de usuários')
plt.title('Histograma de Thresholds Médios por Usuário')
plt.grid(True)
plt.savefig('histograma_thresholds_medio_cmu.png')

st_results_per_userk = repo.read_results("Magalhães com HPO por usuário (Keyrecs)")
st_results_globalk = repo.read_results("Magalhães com HPO global (Keyrecs)")

if st_results_globalk is None or st_results_per_userk is None:
    print("Resultados não encontrados.")
    exit(1)

ts_per_user: dict[str, list[float]] = {}

if st_results_per_userk is not None:
    for i, hp in enumerate(st_results_per_userk.hp_per_seed):
        for user_id, params in hp.items():
            if user_id not in ts_per_user:
                ts_per_user[user_id] = []
            ts_per_user[user_id].append(params['threshold'])


avg_t_per_user = {user_id: np.mean(thresholds) for user_id, thresholds in ts_per_user.items()}

ts_global_per_seed = [hp['threshold'] for hp in st_results_globalk.hp_per_seed[:5] if 'threshold' in hp]
ts_global_avg = np.mean(ts_global_per_seed).item()

plt.figure(figsize=(10, 6))
plt.tight_layout()
plt.axvline(ts_global_avg, color='red', linestyle='dashed', linewidth=2, label=f'Threshold global: {ts_global_avg:.2f}')
plt.legend()
plt.hist(list(avg_t_per_user.values()), bins=20, edgecolor='black')
plt.xlabel('Threshold médio por usuário')
plt.ylabel('Quantidade de usuários')
plt.title('Histograma de Thresholds Médios por Usuário')
plt.grid(True)
plt.savefig('histograma_thresholds_medio_keyrecs.png')