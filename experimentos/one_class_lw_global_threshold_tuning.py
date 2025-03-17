import os

from lib.cmu_dataset import CMUDataset
from lib.global_threshold_search import GlobalThresholdTuning
from lib.lightweight_alg import LightWeightAlg
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl
from lib.utils import save_results, float_range, first_session_split


def main() -> None:
    cmu_database = CMUDataset("datasets/cmu/DSL-StrongPasswordData.csv", first_session_split)
    threshold_tuning = GlobalThresholdTuning(
        dataset = cmu_database,
        thresholds = float_range(0.2, 0.9, 0.05),
        estimator_factory = lambda: LightWeightAlg()
    )
    best_threshold = threshold_tuning.search()
    one_class_lw_experiment = OneClassExperimentRunnerImpl(
        dataset=cmu_database, estimator=LightWeightAlg(threshold=best_threshold)
    )
    results = one_class_lw_experiment.exec()
    save_results(os.path.basename(__file__), results.to_dict())


if __name__ == "__main__":
    main()
