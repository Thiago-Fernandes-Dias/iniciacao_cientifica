import os
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM

from lib.cmu_dataset import CMUDataset
from lib.runners.single_class_experiment_with_search_cv_runner import SingleClassExperimentWithSearchCVRunner
from lib.utils import first_session_split, save_results
from lib.hp_grids import mlp_params_grid

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    one_class_mlp_grid_cv = KFold(n_splits=5)
    one_class_mlp_gs = GridSearchCV(MLPClassifier(), mlp_params_grid, 
                                    scoring='accuracy', cv=one_class_mlp_grid_cv, n_jobs=-1)
    one_class_mlp_with_hpo_experiment = SingleClassExperimentWithSearchCVRunner(dataset=cmu_database, 
                                                                                estimator=one_class_mlp_gs)
    results = one_class_mlp_with_hpo_experiment.exec()
    save_results(os.path.basename(__file__), results.to_dict())

if __name__ == "__main__":
    main()