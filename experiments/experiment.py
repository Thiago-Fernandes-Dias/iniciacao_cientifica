import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import balanced_accuracy_score, accuracy_score

RANDOM_STATE=42

cmu: pd.DataFrame = pd.read_csv('datasets/cmu/DSL-StrongPasswordData.csv')
cmu_training_df = cmu[cmu['sessionIndex'] == 1]
cmu_test_df = cmu[cmu['sessionIndex'] != 1]
drop_columns = ['subject', 'sessionIndex', 'rep']
user_keys: set[str] = set(cmu["subject"].drop_duplicates().tolist())

X_training: dict[str, pd.DataFrame] = {}
X_test: dict[str, pd.DataFrame] = {}
y_training: dict[str, list[str]] = {}
y_test: dict[str, list[str]] = {}

for uk in user_keys:
    X_training[uk] = cmu_training_df[cmu_training_df['subject'] == uk].drop(columns=drop_columns)
    y_training[uk] = [uk] * X_training[uk].shape[0]
    X_test[uk] = cmu_test_df[cmu_test_df['subject'] == uk].drop(columns=drop_columns)
    y_test[uk] = [uk] * X_test[uk].shape[0]

one_class_estimators_map: dict[str, OneClassSVM] = {}

for uk in user_keys:
    one_class_estimators_map[uk] = OneClassSVM().fit(X_training[uk], y_training[uk])

user_model_acc_on_genuine_samples_map: dict[str, float] = {}

for uk in user_keys:
    predictions = one_class_estimators_map[uk].predict(X_test[uk]).flatten().tolist()
    user_model_acc_on_genuine_samples_map[uk] = accuracy_score(y_test[uk], predictions)

average_acc = np.average(list(user_model_acc_on_genuine_samples_map.values()))

print(f"Acurácia dos modelos One-Vs-One: {average_acc}")

# Ataques aos modelos One-Vs-One

user_model_far_on_attack_samples_map: dict[str, float] = {}

for uk in user_keys:
    attack_vecs = cmu_test_df[cmu_test_df['subject'] != uk].drop(columns=drop_columns)
    y_true = [-1] * attack_vecs.shape[0]
    predictions = one_class_estimators_map[uk].predict(attack_vecs).flatten().tolist()
    user_model_far_on_attack_samples_map[uk] = accuracy_score(y_true, predictions)

average_far = np.average(list(user_model_far_on_attack_samples_map.values()))
print(f"FAR dos modelos One-Vs-One: {average_far}")

# Criação de um modelo One-vs-Rest para cada usuário
# Divisão dos dados: (80:20), sendo que em cada conjunto 50% dos dados são do 
# próprio usuário e 50% são registros aleatórios de outros usuários 

X_user_training: dict[str, pd.DataFrame] = {}
X_user_test: dict[str, pd.DataFrame] = {}
y_user_training: dict[str, list[int]] = {}
y_user_test: dict[str, list[int]] = {}
X_other_training: dict[str, pd.DataFrame] = {}
X_other_test: dict[str, pd.DataFrame] = {}
y_other_training: dict[str, list[int]] = {}
y_other_test: dict[str, list[int]] = {}

for uk in user_keys:
    other_keys = user_keys - {uk}
    X_user_training[uk] = cmu_training_df[cmu_training_df['subject'] == uk].drop(columns=drop_columns)
    X_other_training[uk] = cmu_training_df[(cmu_training_df['subject'] != uk)].sample(n=X_user_training[uk].shape[0], random_state=RANDOM_STATE).drop(columns=drop_columns) 
    y_other_training[uk] = [0] * X_other_training[uk].shape[0]
    y_user_training[uk] = [1] * X_user_training[uk].shape[0]
    X_user_test[uk] =  cmu_test_df[cmu_test_df['subject'] == uk].drop(columns=drop_columns)
    X_other_test[uk] = cmu_test_df[cmu_test_df['subject'] != uk].drop(columns=drop_columns)
    y_user_test[uk] = [1] * X_user_test[uk].shape[0]
    y_other_test[uk] = [0] * X_other_test[uk].shape[0]

two_class_estimators_map: dict[str, RandomForestClassifier] = {}
two_class_acc_map: dict[str, float] = {}

for uk in user_keys:
    X = pd.concat([X_user_training[uk], X_other_training[uk]])
    y = y_user_training[uk] + y_other_training[uk]
    two_class_estimators_map[uk] = RandomForestClassifier().fit(X, y)

for uk in user_keys:
    X_test = pd.concat([X_user_test[uk], X_other_test[uk]])
    predictions = two_class_estimators_map[uk].predict(X_test)
    y_true = y_user_test[uk] + y_other_test[uk]
    two_class_acc_map[uk] = balanced_accuracy_score(y_true, predictions)

average_acc = np.average(list(two_class_acc_map.values()))

print(f"Acurácia média dos modelos One-Vs-Rest: {average_acc}")