class OneClassResults:
    average_acc: float
    average_recall: float
    average_far: float
    user_model_acc_on_genuine_samples_map: dict[str, float]
    user_model_recall_map: dict[str, float]
    user_model_far_on_attack_samples_map: dict[str, float]

    def __init__(self, average_acc: float, average_recall: float, average_far: float, user_model_acc_on_genuine_samples_map: dict[str, float], user_model_recall_map: dict[str, float], user_model_far_on_attack_samples_map: dict[str, float]) -> None:
        self.average_acc = average_acc
        self.average_recall = average_recall
        self.average_far = average_far
        self.user_model_acc_on_genuine_samples_map = user_model_acc_on_genuine_samples_map
        self.user_model_recall_map = user_model_recall_map
        self.user_model_far_on_attack_samples_map = user_model_far_on_attack_samples_map
