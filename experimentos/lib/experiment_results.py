from lib.user_model_metrics import UserModelMetrics
from lib.utils import *


class ExperimentResults:
    date: datetime
    exp_name: str
    model_predictions_per_seed: list[pd.DataFrame]
    hp_per_seed: list[dict[str, dict[str, object]]]

    def __init__(self, *,
                 model_predictions_per_seed: list[pd.DataFrame],
                 hp_per_seed: list[dict[str, dict[str, object]]],
                 date: datetime | None) -> None:
        self.model_predictions_per_seed = model_predictions_per_seed
        self.hp_per_seed = hp_per_seed
        self.date = date if date else datetime.now()

    def get_metrics_per_user(self) -> dict[str, UserModelMetrics]:
        experiments_metrics_per_user: dict[str, UserModelMetrics] = {}
        user_keys = self.model_predictions_per_seed[0]["user_id"].drop_duplicates().tolist()
        for user_key in user_keys:
            user_metrics_list: list[UserModelMetrics] = []
            # TODO: Descomentar a linha abaixo e remover o comentário da linha seguinte, para considerar todas as seeds
            # for seed in range(len(self.model_predictions_per_seed)):
            for seed in range(5): # Considerando apenas as 5 primeiras seeds 
                predictions_df = self.model_predictions_per_seed[seed]
                user_predictions_df = predictions_df[(predictions_df["user_id"] == user_key)]
                total_impostor_attempts = len(user_predictions_df[user_predictions_df["expected"] == IMPOSTOR_LABEL])
                accepted_impostor_attempts = len(user_predictions_df[
                                                     (user_predictions_df["expected"] == IMPOSTOR_LABEL) & (
                                                             user_predictions_df["predicted"] == GENUINE_LABEL)])
                total_genuine_attempts = len(user_predictions_df[user_predictions_df["expected"] == GENUINE_LABEL])
                rejected_genuine_attempts = len(user_predictions_df[
                                                    (user_predictions_df["expected"] == GENUINE_LABEL) & (
                                                            user_predictions_df["predicted"] == IMPOSTOR_LABEL)])
                metrics = UserModelMetrics(
                    frr=rejected_genuine_attempts / total_genuine_attempts if total_genuine_attempts > 0 else 0,
                    far=accepted_impostor_attempts / total_impostor_attempts if total_impostor_attempts > 0 else 0
                )
                user_metrics_list.append(metrics)
            experiments_metrics_per_user[user_key] = UserModelMetrics(
                frr=sum(m.frr for m in user_metrics_list) / len(user_metrics_list),
                far=sum(m.far for m in user_metrics_list) / len(user_metrics_list)
            )
        return experiments_metrics_per_user
