import os

from sklearn.model_selection import KFold

from lib.cmu_dataset import CMUDataset
from lib.constants import RANDOM_STATE
from lib.global_threshold_search import GlobalThresholdTuning
from lib.lightweight_alg import LightWeightAlg
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl
from lib.utils import save_results, float_range, first_session_split, two_session_split


def exec(dataset: CMUDataset, file_suffix: str):
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    threshold_tuning = GlobalThresholdTuning(
        dataset = dataset,
        thresholds = float_range(0.2, 0.9, 0.05),
        estimator_factory = lambda: LightWeightAlg(),
        cv=cv
    )
    best_threshold = threshold_tuning.search()
    one_class_lw_experiment = OneClassExperimentRunnerImpl(
        dataset=dataset, estimator=LightWeightAlg(threshold=best_threshold)
    )
    results = one_class_lw_experiment.exec()
    save_results(f"{os.path.basename(__file__)}_{file_suffix}", results.to_dict())

def main() -> None:
    cmu1 = CMUDataset("datasets/cmu/DSL-StrongPasswordData.csv", first_session_split)
    exec(cmu1, "first_session_split")
    cmu2 = CMUDataset("datasets/cmu/DSL-StrongPasswordData.csv", two_session_split)
    exec(cmu2, "two_session_split")

if __name__ == "__main__":
    main()
