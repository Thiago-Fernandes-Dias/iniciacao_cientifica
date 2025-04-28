from datetime import datetime
from lib.user_model_metrics import UserModelMetrics
import numpy as np
from lib.utils import *

class OneClassResults:
    date: datetime
    user_model_metrics: list[UserModelMetrics]
    hp: dict[str, dict[str, object]] | None

    def __init__(self, *,
                 user_model_metrics: list[UserModelMetrics],
                 hp: dict[str, dict[str, object]] | None,
                 date: datetime | None) -> None:
        self.user_model_metrics = user_model_metrics
        self.hp = hp
        self.date = date if date else datetime.now()

    def get_average_frr(self) -> float:
        return np.array(select(self.user_model_metrics, lambda x: x.frr)).mean().item()

    def get_average_far(self) -> float:
        return np.array(select(self.user_model_metrics, lambda x: x.far)).mean().item()

    def get_frr(self, user_key: str) -> float | None:
        try:
            return next(x for x in self.user_model_metrics if x.user_id == user_key).frr
        except StopIteration:
            return None

    def get_far(self, user_key: str) -> float:
        try:
            return next(x for x in self.user_model_metrics if x.user_id == user_key).far
        except StopIteration:
            return None

    def get_best_frr(self) -> UserModelMetrics:
        return min(self.user_model_metrics, key=lambda x: x.frr)

    def get_best_far(self) -> UserModelMetrics:
        return min(self.user_model_metrics, key=lambda x: x.far)

    def to_dict_with_stats(self) -> dict[str, object]:
        return {
            "hp": self.hp,
            "average_frr": self.get_average_frr(),
            "average_far": self.get_average_far(),
            "date": self.date.isoformat(),
            "user_model_metrics": select(self.user_model_metrics, lambda x: x.to_dict()),
            "best_frr": self.get_best_frr().to_dict(),
            "best_far": self.get_best_far().to_dict(),
        }
    
    def to_dict(self) -> dict[str, object]:
        return {
            "hp": self.hp,
            "date": self.date.isoformat(),
            "user_model_metrics": select(self.user_model_metrics, lambda x: x.to_dict()),
        }
    
    @staticmethod
    def from_dict(data: dict[str, object]) -> "OneClassResults":
        user_model_metrics = select(data["user_model_metrics"], lambda x: UserModelMetrics.from_dict(x))
        hp = data["hp"]
        date = datetime.fromisoformat(data["date"])
        return OneClassResults(
            user_model_metrics=user_model_metrics,
            hp=hp,
            date=date)