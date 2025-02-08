from lib.utils import *

class OneClassResults:
    tp_rate_map: dict[str, float]
    tn_rate_map: dict[str, float]
    predictions_on_user_samples_map: dict[str, list[int]]
    predictions_on_impostor_samples_map: dict[str, list[int]]
    hp: dict[str, dict[str, object]] | None

    def __init__(self, *,
                 tp_rate_map: dict[str, float],
                 tn_rate_map: dict[str, float],
                 predictions_on_user_samples_map: dict[str, list[int]],
                 predictions_on_impostor_samples_map: dict[str, list[int]],
                 hp: dict[str, dict[str, object]] | None) -> None:
        self.tp_rate_map = tp_rate_map
        self.tn_rate_map = tn_rate_map
        self.predictions_on_user_samples_map = predictions_on_user_samples_map
        self.predictions_on_impostor_samples_map = predictions_on_impostor_samples_map
        self.hp = hp

    def get_average_tp_rate(self) -> float:
        return dict_values_average(self.tp_rate_map)

    def get_average_tn_rate(self) -> float:
        return dict_values_average(self.tn_rate_map)
    
    def get_tp_rate(self, user_key: str) -> float:
        return self.tp_rate_map[user_key]
    
    def get_tn_rate(self, user_key: str) -> float:
        return self.tn_rate_map[user_key]
    
    def get_prediction_on_user_samples(self, user_key: str) -> list[int]:
        return self.predictions_on_user_samples_map[user_key]

    def get_prediction_on_impostor_samples(self, user_key: str) -> list[int]:
        return self.predictions_on_impostor_samples_map[user_key]
    
    def get_best_tp_rate(self) -> tuple[str, float]:
        return item_with_max_value(self.tp_rate_map, bigger_comp)
    
    def get_best_tn_rate(self) -> tuple[str, float]:
        return item_with_max_value(self.tn_rate_map, lower_comp)
    
    def to_dict(self) -> dict[str, object]:
        return {
            "hp": self.hp,
            "averate_tp_rate": self.get_average_tp_rate(),
            "average_tn_rate": self.get_average_tn_rate(),
            "user_model_tp_rate": self.tp_rate_map,
            "user_model_tn_rate": self.tn_rate_map,
            "predictions_on_user_samples": self.predictions_on_user_samples_map,
            "predictions_on_impostor_samples": self.predictions_on_impostor_samples_map,
            "best_tp_rate": self.get_best_tp_rate(),
            "best_tn_rate": self.get_best_tn_rate()
        }
