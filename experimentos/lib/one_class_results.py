from lib.utils import *


class OneClassResults:
    frr_map: dict[str, float]
    far_map: dict[str, float]
    hp: dict[str, dict[str, object]] | None

    def __init__(self, *,
                 frr_map: dict[str, float],
                 far_map: dict[str, float],
                 hp: dict[str, dict[str, object]] | None) -> None:
        self.frr_map = frr_map
        self.far_map = far_map
        self.hp = hp

    def get_average_frr(self) -> float:
        return dict_values_average(self.frr_map)

    def get_average_far(self) -> float:
        return dict_values_average(self.far_map)

    def get_frr(self, user_key: str) -> float:
        return self.frr_map[user_key]

    def get_far(self, user_key: str) -> float:
        return self.far_map[user_key]

    def get_best_frr(self) -> tuple[str, float]:
        return item_with_max_value(self.frr_map, lower_comp)

    def get_best_far(self) -> tuple[str, float]:
        return item_with_max_value(self.far_map, lower_comp)

    def to_dict(self) -> dict[str, object]:
        return {
            "hp": self.hp,
            "averate_frr": self.get_average_frr(),
            "average_far": self.get_average_far(),
            "user_model_frr": self.frr_map,
            "user_model_far": self.far_map,
            "best_frr": self.get_best_frr(),
            "best_far": self.get_best_far()
        }
