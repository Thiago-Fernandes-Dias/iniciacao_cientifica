from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from cmu_dataset import CMUDataset
from constants import RANDOM_STATE
from runners.multi_class_experiment_with_search_cv_runner import MultiClassExperimentWithSearchCVRunner
from utils import first_session_split, save_results
from hp_grids import mlp_params_grid

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    multi_class_mlp_grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    multi_class_mlp_gs = GridSearchCV(estimator=MLPClassifier(), param_grid=mlp_params_grid, 
                        cv=multi_class_mlp_grid_cv, n_jobs=-1, scoring='accuracy')
    multi_class_mlp_with_hpo_experiment = MultiClassExperimentWithSearchCVRunner(dataset=cmu_database, estimator=multi_class_mlp_gs)
    results = multi_class_mlp_with_hpo_experiment.exec().to_dict()
    save_results("multi_class_with_hpo", results)

if __name__ == "__main__":
    main()