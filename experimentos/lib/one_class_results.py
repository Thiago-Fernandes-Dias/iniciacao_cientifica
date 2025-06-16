from datetime import datetime
from lib.user_model_prediction import UserModelPrediction
from lib.utils import *

class OneClassResults:
    date: datetime
    user_model_predictions: list[UserModelPrediction]
    hp: dict[str, dict[str, object]] | None

    def __init__(self, *,
                 user_model_predictions: list[UserModelPrediction],
                 hp: dict[str, dict[str, object]] | None,
                 date: datetime | None) -> None:
        self.user_model_predictions = user_model_predictions
        self.hp = hp
        self.date = date if date else datetime.now()

    def to_dict(self) -> dict[str, object]:
        return {
            "hp": self.hp,
            "date": self.date.isoformat(),
            "user_model_predictions": select(self.user_model_predictions, lambda x: x.to_dict()),
        }
    
    @staticmethod
    def from_dict(data: dict[str, object]) -> "OneClassResults":
        user_model_predictions = select(data["user_model_predictions"], lambda x: UserModelPrediction.from_dict(x))
        hp = data["hp"]
        date = datetime.fromisoformat(data["date"])
        return OneClassResults(
            user_model_predictions=user_model_predictions,
            hp=hp,
            date=date)