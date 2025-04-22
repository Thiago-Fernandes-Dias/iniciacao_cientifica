from lib.repositories.mongo_results_repository import results_repository_factory
from lib.utils import select

repo = results_repository_factory()

# lw_g_c = repo.get_one_class_results("one_class_lw_global_threshold_tuning_cmu")[0]
# lw_u_c = repo.get_one_class_results("one_class_lw_user_threshold_tuning_cmu")[0]
lw_g_k = repo.get_one_class_results("one_class_lw_global_threshold_tuning_keyrecs")[0]
lw_u_k = repo.get_one_class_results("one_class_lw_user_threshold_tuning_keyrecs")[0]

# print("CMU:")
# for user in select(lw_g_c.frr, lambda x: x.user_id):
#     g_bacc = ((1 - next(x for x in lw_g_c.far if x.user_id == user).value) + (1 - next(x for x in lw_g_c.frr if x.user_id == user).value)) / 2
#     u_bacc = ((1 - next(x for x in lw_u_c.far if x.user_id == user).value) + (1 - next(x for x in lw_u_c.frr if x.user_id == user).value)) / 2
#     if (g_bacc > u_bacc):
#         print(f"O usuário {user} foi beneficiado com o ajuste global do limiar de corte")

print("KeyRecs:")
for user in select(lw_g_k.frr, lambda x: x.user_id):
    g_bacc = ((1 - next(x for x in lw_g_k.far if x.user_id == user).value) + (1 - next(x for x in lw_g_k.frr if x.user_id == user).value)) / 2
    u_bacc = ((1 - next(x for x in lw_u_k.far if x.user_id == user).value) + (1 - next(x for x in lw_u_k.frr if x.user_id == user).value)) / 2
    if (g_bacc > u_bacc):
        print(f"O usuário {user} foi beneficiado com o ajuste global do limiar de corte")
