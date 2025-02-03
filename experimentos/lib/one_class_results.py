from lib.utils import *

class OneClassResults:
    user_model_acc_map: dict[str, float]
    user_model_recall_map: dict[str, float]
    user_model_far_map: dict[str, float]
    predictions_on_user_samples_map: dict[str, list[int]]
    predictions_on_impostor_samples_map: dict[str, list[int]]
    hp: dict[str, dict[str, object]] | None

    def __init__(self, *,
                 user_model_acc_on_genuine_samples_map: dict[str, float], 
                 user_model_recall_map: dict[str, float], 
                 user_model_far_map: dict[str, float], 
                 predictions_on_user_samples_map: dict[str, list[int]],
                 predictions_on_impostor_samples_map: dict[str, list[int]],
                 hp: dict[str, dict[str, object]] | None) -> None:
        self.user_model_acc_map = user_model_acc_on_genuine_samples_map
        self.user_model_recall_map = user_model_recall_map
        self.user_model_far_map = user_model_far_map
        self.predictions_on_user_samples_map = predictions_on_user_samples_map
        self.predictions_on_impostor_samples_map = predictions_on_impostor_samples_map
        self.hp = hp

    def get_average_acc(self) -> float:
        return dict_values_average(self.user_model_acc_map)

    def get_average_recall(self) -> float:
        return dict_values_average(self.user_model_recall_map)
    
    def get_average_far(self) -> float:
        return dict_values_average(self.user_model_far_map)
    
    def get_acc_on_genuine_samples(self, user_key: str) -> float:
        return self.user_model_acc_map[user_key]
    
    def get_recall(self, user_key: str) -> float:
        return self.user_model_recall_map[user_key]
    
    def get_tn_rate_on_attack_samples(self, user_key: str) -> float:
        return self.user_model_far_map[user_key]
    
    def get_prediction_on_user_samples(self, user_key: str) -> list[int]:
        return self.predictions_on_user_samples_map[user_key]

    def get_prediction_on_impostor_samples(self, user_key: str) -> list[int]:
        return self.predictions_on_impostor_samples_map[user_key]
    
    def get_best_accuracy(self) -> tuple[str, float]:
        return item_with_max_value(self.user_model_acc_map, bigger_comp)
    
    def get_best_recall(self) -> tuple[str, float]:
        return item_with_max_value(self.user_model_recall_map, bigger_comp)
    
    def get_best_far(self) -> tuple[str, float]:
        return item_with_max_value(self.user_model_far_map, lower_comp)

    def print_results(self) -> None:
        print(f"- Average accuracy on genuine samples: {self.get_average_acc()}")
        print(f"- Average TN rate on attack samples: {self.get_average_far()}")
        print(f"- Average recall: {self.get_average_recall()}")
        print(f"- User model accuracy on genuine samples:")
        for (uk, acc) in self.user_model_acc_map.items():
            print(f"    - {uk}: {acc}")
        print(f"- User model TN rate:")
        for (uk, tn_rate) in self.user_model_far_map.items():
            print(f"    - {uk}: {tn_rate}")
        print(f"- User model recall:")
        for (uk, recall) in self.user_model_recall_map.items():
            print(f"    - {uk}: {recall}")
        user_sub, best_acc = self.get_best_accuracy()
        print(f"Model with highest accuracy: user {user_sub} with accuracy {best_acc}")
        user_sub, best_recall = self.get_best_recall()
        print(f"Model with highest recall: user {user_sub} with recall {best_recall}")
        user_sub, best_tn_rate = self.get_best_far()
        print(f"Model with highest TN rate: user {user_sub} with TN rate {best_tn_rate}")
    
    def to_dict(self) -> dict[str, object]:
        return {
            "hp": self.hp,
            "average_acc": self.get_average_acc(),
            "average_recall": self.get_average_recall(),
            "average_far": self.get_average_far(),
            "user_model_acc": self.user_model_acc_map,
            "user_model_recall": self.user_model_recall_map,
            "user_model_far": self.user_model_far_map,
            "predictions_on_user_samples": self.predictions_on_user_samples_map,
            "predictions_on_impostor_samples": self.predictions_on_impostor_samples_map,
            "best_acc": self.get_best_accuracy(),
            "best_recall": self.get_best_recall(),
            "best_far": self.get_best_far()
        }
