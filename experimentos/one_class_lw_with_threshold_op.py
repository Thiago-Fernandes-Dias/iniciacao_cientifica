import os
from sklearn.model_selection import GridSearchCV, KFold

from lib.cmu_dataset import CMUDataset
from lib.runners.single_class_experiment_with_search_cv_runner import SingleClassExperimentWithSearchCVRunner
from lib.utils import lw_split, save_results, far_score
from lib.hp_grids import lw_params_gri_var_t
from lib.lightweight_alg import LightWeightAlg

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', lw_split, "DD.*")
    one_class_lw_grid_cv = KFold(n_splits=5)
    one_class_lw_gs = GridSearchCV(LightWeightAlg(), lw_params_gri_var_t, 
                                    scoring=far_score, cv=one_class_lw_grid_cv, n_jobs=-1)
    one_class_lw_with_hpo_experiment = SingleClassExperimentWithSearchCVRunner(dataset=cmu_database, 
                                                                                estimator=one_class_lw_gs)
    results = one_class_lw_with_hpo_experiment.exec()
    save_results(os.path.basename(__file__), results.to_dict())

if __name__ == "__main__":
    main()