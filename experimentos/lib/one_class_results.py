from datetime import datetime
from lib.user_model_prediction import UserModelPrediction
from lib.utils import *

class OneClassResults:
    date: datetime
    user_model_predictions: pd.DataFrame
    hp: dict[str, list[dict[str, object]]]

    def __init__(self, *,
                 user_model_predictions: pd.DataFrame,
                 hp: dict[str, list[dict[str, object]]],
                 date: datetime | None) -> None:
        self.user_model_predictions = user_model_predictions
        self.hp = hp
        self.date = date if date else datetime.now()
