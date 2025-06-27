import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier

from lib.datasets.dataset import Dataset
from lib.constants import RANDOM_STATE, N_JOBS
from lib.runners.multi_class_experiment_with_search_cv_runner import (
    MultiClassExperimentWithSearchCVRunner,
)
from lib.utils import cmu_split, save_results
from lib.hp_grids import mlp_params_grid


def main() -> None:
    cmu_database = Dataset(
        "datasets/cmu/DSL-StrongPasswordData.csv", cmu_split
    )
    multi_class_mlp_grid_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    multi_class_mlp_gs = GridSearchCV(
        estimator=MLPClassifier(),
        param_grid=mlp_params_grid,
        cv=multi_class_mlp_grid_cv,
        n_jobs=N_JOBS,
        scoring="accuracy",
    )
    multi_class_mlp_with_hpo_experiment = MultiClassExperimentWithSearchCVRunner(
        dataset=cmu_database, estimator=multi_class_mlp_gs
    )
    results = multi_class_mlp_with_hpo_experiment.exec().to_dict()
    save_results(os.path.basename(__file__).replace(".py", ""), results)


if __name__ == "__main__":
    main()
