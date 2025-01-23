from utils import *

class TwoClassResults:
    two_class_bacc_map: dict[str, float]
    two_class_recall_map: dict[str, float]
    predictions_on_user_samples_map: dict[str, list[int]]
    predictions_on_impostor_samples_map: dict[str, list[int]]
    hp: dict[str, dict[str,object]] | None
    
    def __init__(self, *, two_class_bacc_map: dict[str, float], 
                 two_class_recall_map: dict[str, float], 
                 predictions_on_user_samples_map: dict[str, list[int]],
                 predictions_on_impostor_samples_map: dict[str, list[int]], 
                 hp: dict[str, dict[ str,object ]] | None) -> None:
        self.two_class_bacc_map = two_class_bacc_map
        self.two_class_recall_map = two_class_recall_map
        self.predictions_on_user_samples_map = predictions_on_user_samples_map
        self.predictions_on_impostor_samples_map = predictions_on_impostor_samples_map
        self.hp = hp
    
    def get_average_bacc(self) -> float:
        return dict_values_average(self.two_class_bacc_map)

    def get_average_recall(self) -> float:
        return dict_values_average(self.two_class_recall_map)
    
    def get_bacc(self, user_key: str) -> float:
        return self.two_class_bacc_map[user_key]
    
    def get_recall(self, user_key: str) -> float:
        return self.two_class_recall_map[user_key]

    def get_predictions_on_user_samples(self, user_key: str) -> list[int]:
        return self.predictions_on_user_samples_map[user_key]

    def get_predictions_on_impostor_samples(self, user_key: str) -> list[int]:
        return self.predictions_on_impostor_samples_map[user_key]

    def get_best_bacc(self) -> tuple[str, float]:
        return item_with_max_value(self.two_class_bacc_map)
    
    def get_best_recall(self) -> tuple[str, float]:
        return item_with_max_value(self.two_class_recall_map)
    
    def print_results(self) -> None:
        print(f"- Average balanced accuracy: {self.get_average_bacc()}")
        print(f"- Average recall: {self.get_average_recall()}")
        print(f"- User model balanced accuracy:")
        for uk, bacc in self.two_class_bacc_map.items():
            print(f"    - {uk}: {bacc}")
        print(f"- User model recall:")
        for uk, recall in self.two_class_recall_map.items():
            print(f"    - {uk}: {recall}")
        user_sub, best_bacc = self.get_best_bacc()
        print(f"Model with highest balanced accuracy: user {user_sub} with accuracy {best_bacc}")
        user_sub, best_recall = self.get_best_recall()
        print(f"Model with highest recall: user {user_sub} with recall {best_recall}")

    def to_dict(self) -> dict[str, object]:
        return {
            "hp": self.hp,
            "average_bacc": self.get_average_bacc(),
            "average_recall": self.get_average_recall(),
            "user_model_bacc": self.two_class_bacc_map,
            "user_model_recall": self.two_class_recall_map,
            "predictions_on_user_samples": self.predictions_on_user_samples_map,
            "predictions_on_impostor_samples": self.predictions_on_impostor_samples_map
        }
