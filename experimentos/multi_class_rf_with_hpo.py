import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from lib.datasets.dataset import Dataset
from lib.runners.multi_class_experiment_with_search_cv_runner import (
    MultiClassExperimentWithSearchCVRunner,
)
from lib.constants import RANDOM_STATE, N_JOBS
from lib.hp_grids import rf_params_grid
from lib.utils import cmu_first_session_split, save_results


def main() -> None:
    cmu_database = Dataset(
        "datasets/cmu/DSL-StrongPasswordData.csv", cmu_first_session_split
    )
    multi_class_rf_grid_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    multi_class_rf_gs = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=rf_params_grid,
        cv=multi_class_rf_grid_cv,
        n_jobs=N_JOBS,
        scoring="accuracy",
    )
    multi_class_rf_with_hpo = MultiClassExperimentWithSearchCVRunner(
        dataset=cmu_database, estimator=multi_class_rf_gs
    )
    results = multi_class_rf_with_hpo.exec()
    save_results(os.path.basename(__file__).replace(".py", ""), results.to_dict())


if __name__ == "__main__":
    main()
