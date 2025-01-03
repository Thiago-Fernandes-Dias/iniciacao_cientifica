import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.svm import OneClassSVM
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score
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

params_grid = [
    {
        'kernel': ['poly'],
        'degree': range(1, 4),
        'gamma': ['scale', 'auto', .1, .01, .001],
        'coef0': float_range(0, 5, .25),
        # Causando falhas em alguns fits -> 'nu': [.1, .25, .5, .75, 1], 
        'cache_size': [4096]
    },
    {
        'kernel': ['linear'],
        'cache_size': [4096],
    },
    {
        'kernel': ['sigmoid'],
        'gamma': ['scale', 'auto', .1, .01, .001],
        'coef0': float_range(0, 5, .25),
        # 'nu': [.1, .25, .5, .75, 1], 
        'cache_size': [4096]
    },
    {
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto', .1, .01, .001],
        # 'nu': [.1, .25, .5, .75, 1], 
        'cache_size': [4096]
    }
]

one_vs_one_cv = KFold(n_splits=5)
one_vs_one_gs = GridSearchCV(OneClassSVM(), params_grid, 
                             scoring='accuracy', cv=one_vs_one_cv, n_jobs=-1)

for uk in cmu_database.user_keys():
    fitted_gs = one_vs_one_gs.fit(X_training[uk], y_training[uk])
    one_class_estimators_map[uk] = fitted_gs.best_estimator_

user_model_acc_on_genuine_samples_map: dict[str, float] = {}
user_model_recall_map: dict[str, float] = {}

for uk in cmu_database.user_keys():
    predictions = one_class_estimators_map[uk].predict(X_test[uk]).flatten().tolist()
    user_model_acc_on_genuine_samples_map[uk] = accuracy_score(y_test[uk], predictions)
    user_model_recall_map[uk] = recall_score(y_test[uk], predictions, average='micro')

average_acc = np.average(list(user_model_acc_on_genuine_samples_map.values()))
average_recall = np.average(list(user_model_recall_map.values()))

print(f"Acurácia dos modelos One-Vs-One: {average_acc}")
print(f"Recall dos modelos One-Vs-One: {average_recall}")

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
one_vs_rest_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rf_params_grid = [
    {
        'n_estimators': [100, 200, 300, 400, 500],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_features': ['sqrt', 'log2', None, 10, 20, 31],
        'bootstrap': [True, False],
        'n_jobs': [-1],
        'random_state': [RANDOM_STATE],
        'warm_start': [True, False]
    }
]
one_vs_rest_gs = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_params_grid, 
                              cv=one_vs_rest_cv, n_jobs=-1, scoring='accuracy')

for uk in cmu_database.user_keys():
    fitted_gs = one_vs_rest_gs.fit(X_training[uk], y_training[uk])
    two_class_estimators_map[uk] = fitted_gs.best_estimator_

for uk in cmu_database.user_keys():
    predictions = two_class_estimators_map[uk].predict(X_test[uk])
    two_class_acc_map[uk] = balanced_accuracy_score(y_test[uk], predictions)

average_acc = np.average(list(two_class_acc_map.values()))

print(f"Acurácia média dos modelos One-Vs-Rest: {average_acc}")