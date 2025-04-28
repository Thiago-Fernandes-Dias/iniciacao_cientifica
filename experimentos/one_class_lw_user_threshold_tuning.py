import os

from sklearn.model_selection import KFold

from lib.cmu_dataset import CMUDataset
from lib.dataset import Dataset
from lib.constants import RANDOM_STATE
from lib.keyrecs_dataset import KeyrecsDataset
from lib.lightweight_alg import LightWeightAlg
from lib.one_class_threshold_search_cv import OneClassThresholdSearchCV
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.one_class_experiment_with_threshold_search_cv import OneClassExperimentWithThresholdSearchCV
from lib.utils import  cmu_first_session_split, float_range, keyrecs_split

def exec_with_dataset(dataset: Dataset, file_suffix: str) -> None:
    one_class_lw_grid_cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    one_class_lw_gs = OneClassThresholdSearchCV(
        estimator=LightWeightAlg(),
        cv=one_class_lw_grid_cv,
        thresholds=float_range(0.2, 0.9, 0.05),
    )
    one_class_lw_with_hpo_experiment = OneClassExperimentWithThresholdSearchCV(
        dataset=dataset, estimator=one_class_lw_gs
    )
    results = one_class_lw_with_hpo_experiment.exec()
    repo = results_repository_factory()
    repo.add_one_class_result(results, os.path.basename(__file__).replace(".py", "") + "_" + file_suffix)

def main() -> None:
    cmu = CMUDataset("datasets/cmu/DSL-StrongPasswordData.csv", cmu_first_session_split)
    exec_with_dataset(cmu, "cmu")
    keyrecs = KeyrecsDataset("datasets/keyrecs/fixed-text.csv", keyrecs_split)
    exec_with_dataset(keyrecs, "keyrecs")


if __name__ == "__main__":
    main()
