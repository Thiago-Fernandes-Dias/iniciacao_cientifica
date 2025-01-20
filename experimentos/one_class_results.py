import numpy as np

from utils import *

class OneClassResults:
    user_model_acc_on_genuine_samples_map: dict[str, float]
    user_model_recall_map: dict[str, float]
    user_model_tn_rate_on_attack_samples_map: dict[str, float]
    prediction_on_user_samples_map: dict[str, list[int]]
    prediction_on_impostor_samples_map: dict[str, list[int]]

    def __init__(self, user_model_acc_on_genuine_samples_map: dict[str, float], 
                 user_model_recall_map: dict[str, float], 
                 user_model_tn_rate_on_attack_samples_map: dict[str, float], 
                 predictions_on_user_samples_map: dict[str, list[int]],
                 prediction_on_impostor_samples_map: dict[str, list[int]]) -> None:
        self.user_model_acc_on_genuine_samples_map = user_model_acc_on_genuine_samples_map
        self.user_model_recall_map = user_model_recall_map
        self.user_model_tn_rate_on_attack_samples_map = user_model_tn_rate_on_attack_samples_map
        self.prediction_on_user_samples_map = predictions_on_user_samples_map
        self.prediction_on_impostor_samples_map = prediction_on_impostor_samples_map

    def get_average_acc_on_genuine_samples(self) -> float:
        return np.average(list(self.user_model_acc_on_genuine_samples_map.values()))

    def get_average_recall(self) -> float:
        return np.average(list(self.user_model_recall_map.values()))
    
    def get_average_tn_rate_on_attack_samples(self) -> float:
        return np.average(list(self.user_model_tn_rate_on_attack_samples_map.values()))
    
    def get_acc_on_genuine_samples(self, user_key: str) -> float:
        return self.user_model_acc_on_genuine_samples_map[user_key]
    
    def get_recall(self, user_key: str) -> float:
        return self.user_model_recall_map[user_key]
    
    def get_tn_rate_on_attack_samples(self, user_key: str) -> float:
        return self.user_model_tn_rate_on_attack_samples_map[user_key]
    
    def get_prediction_on_user_samples(self, user_key: str) -> list[int]:
        return self.prediction_on_user_samples_map[user_key]

    def get_prediction_on_impostor_samples(self, user_key: str) -> list[int]:
        return self.prediction_on_impostor_samples_map[user_key]
    
    def get_best_accuracy(self) -> tuple[str, float]:
        return item_with_max_value(self.user_model_acc_on_genuine_samples_map)
    
    def get_best_recall(self) -> tuple[str, float]:
        return item_with_max_value(self.user_model_recall_map)
    
    def get_best_tn_rate_on_attack_samples(self) -> tuple[str, float]:
        return item_with_max_value(self.user_model_tn_rate_on_attack_samples_map)

    def print_results(self) -> None:
        print(f"- Average accuracy on genuine samples: {self.get_average_acc_on_genuine_samples()}")
        print(f"- Average TN rate on attack samples: {self.get_average_tn_rate_on_attack_samples()}")
        print(f"- Average recall: {self.get_average_recall()}")
        print(f"- User model accuracy on genuine samples:")
        for (uk, acc) in self.user_model_acc_on_genuine_samples_map.items():
            print(f"    - {uk}: {acc}")
        print(f"- User model TN rate:")
        for (uk, tn_rate) in self.user_model_tn_rate_on_attack_samples_map.items():
            print(f"    - {uk}: {tn_rate}")
        print(f"- User model recall:")
        for (uk, recall) in self.user_model_recall_map.items():
            print(f"    - {uk}: {recall}")
        user_sub, best_acc = self.get_best_accuracy()
        print(f"Model with highest accuracy: user {user_sub} with accuracy {best_acc}")
        user_sub, best_recall = self.get_best_recall()
        print(f"Model with highest recall: user {user_sub} with recall {best_recall}")
        user_sub, best_tn_rate = self.get_best_tn_rate_on_attack_samples()
        print(f"Model with highest TN rate: user {user_sub} with TN rate {best_tn_rate}")
