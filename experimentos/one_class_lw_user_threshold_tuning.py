import os

from sklearn.model_selection import KFold

from lib.datasets.cmu_dataset import CMUDataset
from lib.datasets.dataset import Dataset
from lib.constants import RANDOM_STATE
from lib.datasets.keyrecs_dataset import KeyrecsDataset
from lib.experiment_executor import ExperimentExecutor
from lib.lightweight_alg import LightWeightAlg
from lib.one_class_threshold_search_cv import OneClassThresholdSearchCV
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.one_class_experiment_with_threshold_search_cv import OneClassExperimentWithThresholdSearchCV
from lib.utils import float_range


def main() -> None:
    one_class_lw_grid_cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    one_class_lw_gs = OneClassThresholdSearchCV(
        estimator=LightWeightAlg(),
        cv=one_class_lw_grid_cv,
        thresholds=float_range(0.2, 0.9, 0.05),
    )
    executor = ExperimentExecutor(
        name=os.path.basename(__file__).replace(".py", ""),
        results_repo=results_repository_factory(),
        runner_factory=lambda ds: OneClassExperimentWithThresholdSearchCV(
            dataset=ds, estimator=one_class_lw_gs
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
