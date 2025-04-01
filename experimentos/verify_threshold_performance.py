import json


lw_global_first_session = "results\one_class_lw_global_threshold_tuning.py_first_session_split_2025-04-01_13-59-07.json"
lw_global_two_sessions = "results\one_class_lw_global_threshold_tuning.py_two_session_split_2025-04-01_13-59-54.json"
lw_user_first_session = "results\one_class_lw_user_threshold_tuning.py_first_session_split_2025-04-01_14-02-39.json"
lw_user_two_sessions = "results\one_class_lw_user_threshold_tuning.py_two_session_split_2025-04-01_14-01-31.json"

with open(lw_global_first_session) as lw_g_f, \
     open(lw_global_two_sessions) as lw_g_t, \
     open(lw_user_first_session) as lw_u_f, \
     open(lw_user_two_sessions) as lw_u_t:
    
    lw_g_f = json.load(lw_g_f)
    lw_g_t = json.load(lw_g_t)
    lw_u_f = json.load(lw_u_f)
    lw_u_t = json.load(lw_u_t)

    for user in lw_g_f["user_model_frr"].keys():
        g_bacc = ((1 - lw_g_f["user_model_frr"][user]) + (1 - lw_g_f["user_model_far"][user])) / 2
        u_bacc = ((1 - lw_u_f["user_model_frr"][user]) + (1 - lw_u_f["user_model_far"][user])) / 2
        if (g_bacc > u_bacc):
            print(f"O usuário {user} foi beneficiado com o ajuste global do limiar de corte")
