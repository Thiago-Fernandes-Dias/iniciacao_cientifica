import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from cmu import *

cmu_database = CMUDatabase('datasets/cmu/DSL-StrongPasswordData.csv')

X_training: dict[str, pd.DataFrame] = {}
X_test: dict[str, pd.DataFrame] = {}
y_training: dict[str, list[int]] = {}
y_test: dict[str, list[int]] = {}

for uk in cmu_database.user_keys():
    X_training[uk], y_training[uk] = cmu_database.one_vs_one_training_rows(uk)
    X_test[uk], y_test[uk] = cmu_database.one_vs_one_test_rows(uk)

one_class_estimators_map: dict[str, OneClassSVM] = {}

for uk in cmu_database.user_keys():
    one_class_estimators_map[uk] = OneClassSVM().fit(X_training[uk], y_training[uk])

user_model_acc_on_genuine_samples_map: dict[str, float] = {}

for uk in cmu_database.user_keys():
    predictions = one_class_estimators_map[uk].predict(X_test[uk]).flatten().tolist()
    user_model_acc_on_genuine_samples_map[uk] = accuracy_score(y_test[uk], predictions)

average_acc = np.average(list(user_model_acc_on_genuine_samples_map.values()))

print(f"Acurácia dos modelos One-Vs-One: {average_acc}")

# Ataques aos modelos One-Vs-One

user_model_far_on_attack_samples_map: dict[str, float] = {}

for uk in cmu_database.user_keys():
    X_attacks, y_attacks = cmu_database.one_vs_one_attacks_rows(uk)
    predictions = one_class_estimators_map[uk].predict(X_attacks).flatten().tolist()
    user_model_far_on_attack_samples_map[uk] = accuracy_score(y_attacks, predictions)

average_far = np.average(list(user_model_far_on_attack_samples_map.values()))
print(f"FAR dos modelos One-Vs-One: {average_far}")

# Criação de um modelo One-vs-Rest para cada usuário
# Divisão dos dados: (80:20), sendo que em cada conjunto 50% dos dados são do 
# próprio usuário e 50% são registros aleatórios de outros usuários 

for uk in cmu_database.user_keys():
    X_training[uk], y_training[uk] = cmu_database.one_vs_rest_training_rows(uk)
    X_test[uk], y_test[uk] = cmu_database.one_vs_rest_test_rows(uk)

two_class_estimators_map: dict[str, RandomForestClassifier] = {}
two_class_acc_map: dict[str, float] = {}

for uk in cmu_database.user_keys():
    two_class_estimators_map[uk] = RandomForestClassifier().fit(X_training[uk], y_training[uk])

for uk in cmu_database.user_keys():
    predictions = two_class_estimators_map[uk].predict(X_test[uk])
    two_class_acc_map[uk] = balanced_accuracy_score(y_test[uk], predictions)

average_acc = np.average(list(two_class_acc_map.values()))

print(f"Acurácia média dos modelos One-Vs-Rest: {average_acc}")