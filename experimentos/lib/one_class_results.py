from datetime import datetime
from lib.user_model_prediction import UserModelPrediction
from lib.utils import *

class OneClassResults:
    date: datetime
    user_model_predictions: pd.DataFrame
    hp: pd.DataFrame

    def __init__(self, *,
                 user_model_predictions: pd.DataFrame,
                 hp: pd.DataFrame,
                 date: datetime | None) -> None:
        self.user_model_predictions = user_model_predictions
        self.hp = hp
        self.date = date if date else datetime.now()
