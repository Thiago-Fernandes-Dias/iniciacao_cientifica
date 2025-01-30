from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from cmu_dataset import CMUDataset
from runners.multi_class_experiment_runner import MultiClassExperimentRunner
from constants import RANDOM_STATE
from hp_grids import rf_params_grid
from utils import first_session_split, save_results

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    multi_class_rf_grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    multi_class_rf_gs = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_params_grid, 
                        cv=multi_class_rf_grid_cv, n_jobs=-1, scoring='accuracy')
    multi_class_rf_with_hpo = MultiClassExperimentRunner(dataset=cmu_database, estimator=multi_class_rf_gs)
    results = multi_class_rf_with_hpo.exec()
    save_results("multi_class_rf_with_hpo", results.to_dict())

if __name__ == "__main__":
    main()