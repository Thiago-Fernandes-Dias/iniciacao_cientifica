import os

from sklearn.model_selection import KFold

from lib.cmu_dataset import CMUDataset
from lib.dataset import Dataset
from lib.constants import RANDOM_STATE
from lib.global_threshold_search import GlobalThresholdTuning
from lib.keyrecs_dataset import KeyrecsDataset
from lib.lightweight_alg import LightWeightAlg
from lib.repositories.mongo_results_repository import results_repository_factory
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl
from lib.utils import float_range, cmu_first_session_split, keyrecs_split, two_session_split


def exec(dataset: Dataset, file_suffix: str):
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
    repo = results_repository_factory()
    repo.add_one_class_result(results, os.path.basename(__file__).replace(".py", "") + "_" + file_suffix)

def main() -> None:
    # cmu = CMUDataset("datasets/cmu/DSL-StrongPasswordData.csv", cmu_first_session_split)
    # exec(cmu, "cmu")
    keyrecs = KeyrecsDataset("datasets/keyrecs/fixed-text.csv", keyrecs_split)
    exec(keyrecs, "keyrecs")

if __name__ == "__main__":
    main()
