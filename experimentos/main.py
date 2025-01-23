import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier

from multi_class_experiment import *
from cmu_dataset import *

from cmu_dataset import *
from experiments.single_class_experiment import *
from two_class_experiment import *
from hp_grids import *

def save_results(name: str, experiment_results: dict[str, object]) -> None:
    json_string = json.dumps(experiment_results, indent=4)
    create_dir_if_not_exists("results")
    with open(f"results/exp_{name}_{get_datetime()}.json", 'w+') as file:
        file.write(json_string)

def run_experiments(dataset: CMUDataset, name: str, include_hpo: bool) -> None:
    experiment_results: dict[str, object] = {}

    one_class_svm_experiment = SingleClassExperiment(dataset=dataset, estimator=OneClassSVM())
    experiment_results['one_class_svm'] = one_class_svm_experiment.exec().to_dict()

    # two_class_rf_experiment = TwoClassExperiment(dataset=dataset, estimator_factory=lambda: RandomForestClassifier())
    # experiment_results['two_class_rf'] = two_class_rf_experiment.exec().to_dict()

    # multi_class_mlp_experiment = MultiClassExperiment(dataset=dataset, estimator=MLPClassifier(max_iter=1000000000))
    # experiment_results['multi_class_mlp'] = multi_class_mlp_experiment.exec().to_dict()

    if (include_hpo):
        one_class_svm_grid_cv = KFold(n_splits=5)
        one_class_svm_gs = GridSearchCV(OneClassSVM(), one_class_svm_params_grid, 
                                    scoring='accuracy', cv=one_class_svm_grid_cv, n_jobs=-1)
        one_class_svm_with_hpo_experiment = SingleClassExperiment(dataset=dataset, estimator=one_class_svm_gs)
        experiment_results['one_class_svm_with_hpo'] = one_class_svm_with_hpo_experiment.exec().to_dict()

        # two_class_rf_grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        # two_class_rf_gs_factory = lambda: GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_params_grid, 
        #                             cv=two_class_rf_grid_cv, n_jobs=-1, scoring='accuracy')
        # two_class_rf_with_hpo_experiment = TwoClassExperiment(dataset=dataset, estimator_factory=two_class_rf_gs_factory)
        # experiment_results['two_class_rf_with_hpo'] = two_class_rf_with_hpo_experiment.exec().to_dict()
        
        # multi_class_mlp_grid_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        # multi_class_mlp_gs = GridSearchCV(estimator=MLPClassifier(), param_grid=mlp_params_grid, 
        #                     cv=multi_class_mlp_grid_cv, n_jobs=-1, scoring='accuracy')
        # multi_class_mlp_with_hpo_experiment = MultiClassExperiment(dataset=dataset, estimator=multi_class_mlp_gs)
        # experiment_results["multi_class_mlp_with_hpo"] = multi_class_mlp_with_hpo_experiment.exec().to_dict()

    save_results(name, experiment_results)

def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    first_session_split: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]] = \
        lambda df: (df[df['sessionIndex'] == 1], df[df['sessionIndex'] != 1])
    cmu_database_1 = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    run_experiments(cmu_database_1, "first_session_split", True)

    second_session_split: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]] = \
        lambda df: (df[df['sessionIndex'] != 8], df[df['sessionIndex'] == 8])
    cmu_database_2 = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', second_session_split)
    run_experiments(cmu_database_2, "last_session_split", True)

if __name__ == '__main__':
    main()