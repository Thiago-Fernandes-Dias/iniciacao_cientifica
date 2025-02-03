import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from lib.runners.one_vs_rest_experiment_with_search_cv_runner import OneVsRestExperimentWithSearchCVRunner
from lib.cmu_dataset import CMUDataset
from lib.constants import RANDOM_STATE
from lib.hp_grids import mlp_params_grid
from lib.utils import first_session_split, save_results

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    one_vs_rest_mlp_grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    one_vs_rest_mlp_gs = GridSearchCV(estimator=MLPClassifier(), param_grid=mlp_params_grid, 
                        cv=one_vs_rest_mlp_grid_cv, n_jobs=-1, scoring='accuracy')
    one_vs_rest_mlp_with_hpo = OneVsRestExperimentWithSearchCVRunner(cmu_database=cmu_database, estimator=one_vs_rest_mlp_gs)
    results = one_vs_rest_mlp_with_hpo.exec()
    save_results(os.path.basename(__file__), results.to_dict())

if __name__ == "__main__":
    main()