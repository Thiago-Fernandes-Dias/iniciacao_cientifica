from datetime import datetime
from lib.user_metric import UserModelMetric
import numpy as np
from lib.utils import *

class OneClassResults:
    date: datetime
    frr: list[UserModelMetric]
    far: list[UserModelMetric]
    hp: dict[str, dict[str, object]] | None

    def __init__(self, *,
                 frr: list[UserModelMetric],
                 far: list[UserModelMetric],
                 hp: dict[str, dict[str, object]] | None,
                 date: datetime | None) -> None:
        self.frr = frr
        self.far = far
        self.hp = hp
        self.date = date if date else datetime.now()

    def get_average_frr(self) -> float:
        return np.array(select(self.frr, lambda x: x.value)).mean().item()

    def get_average_far(self) -> float:
        return np.array(select(self.frr, lambda x: x.value)).mean().item()

    def get_frr(self, user_key: str) -> float | None:
        try:
            return next(x for x in self.frr if x.user_id == user_key).value
        except StopIteration:
            return None

    def get_far(self, user_key: str) -> float:
        try:
            return next(x for x in self.frr if x.user_id == user_key).value
        except StopIteration:
            return None

    def get_best_frr(self) -> UserModelMetric:
        return min(self.frr, key=lambda x: x.value)

    def get_best_far(self) -> UserModelMetric:
        return min(self.far, key=lambda x: x.value)

    def to_dict_with_stats(self) -> dict[str, object]:
        return {
            "hp": self.hp,
            "average_frr": self.get_average_frr(),
            "average_far": self.get_average_far(),
            "date": self.date.isoformat(),
            "user_model_frr": list(map(lambda x: x.to_dict(), self.frr)),
            "user_model_far": list(map(lambda x: x.to_dict(), self.far)),
            "best_frr": self.get_best_frr().to_dict(),
            "best_far": self.get_best_far().to_dict(),
        }
    
    def to_dict(self) -> dict[str, object]:
        return {
            "hp": self.hp,
            "date": self.date.isoformat(),
            "user_model_frr": list(map(lambda x: x.to_dict(), self.frr)),
            "user_model_far": list(map(lambda x: x.to_dict(), self.far)),
        }
    
    @staticmethod
    def from_dict(data: dict[str, object]) -> "OneClassResults":
        frr = list(map(lambda x: UserModelMetric.from_dict(x), data["user_model_frr"]))
        far = list(map(lambda x: UserModelMetric.from_dict(x), data["user_model_far"]))
        hp = data["hp"]
        date = datetime.fromisoformat(data["date"])
        return OneClassResults(
            frr=frr,
            far=far,
            hp=hp,
            date=date)