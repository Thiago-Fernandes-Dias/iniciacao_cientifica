from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.svm import OneClassSVM

import os

from cmu import *
from one_class_experiment import *
from two_class_experiment import *
from hp_grids import *

def run_experiments(cmu_database: CMUDatabase) -> None:
    one_class_svm_experiment = OneClassExperiment(cmu_database=cmu_database, estimator_factory=lambda: OneClassSVM())
    one_class_svm_results = one_class_svm_experiment.exec()
    
    one_vs_one_cv = KFold(n_splits=5)
    one_vs_one_gs_factory = lambda: GridSearchCV(OneClassSVM(), one_class_svm_params_grid, 
                                scoring='accuracy', cv=one_vs_one_cv, n_jobs=-1)
    one_class_svm_with_hpo_experiment = OneClassExperiment(cmu_database=cmu_database, estimator_factory=one_vs_one_gs_factory)
    one_class_svm_with_hpo_results = one_class_svm_with_hpo_experiment.exec()

    two_class_experiment = TwoClassExperiment(cmu_database=cmu_database, estimator_factory=lambda: RandomForestClassifier())
    two_class_rf_results = two_class_experiment.exec()

    one_vs_rest_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    one_vs_rest_gs_factory = lambda: GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_params_grid, 
                                cv=one_vs_rest_cv, n_jobs=-1, scoring='accuracy')
    two_class_experiment = TwoClassExperiment(cmu_database=cmu_database, estimator_factory=one_vs_rest_gs_factory)
    two_class_rf_with_hpo_results = two_class_experiment.exec()

    print("\n" + "-" * 20 + " Experiment results with First data split" + "-" * 20)
    print("** One Class SVM results **")
    one_class_svm_results.print_results()
    print("** One Class SVM with HPO results ** ")
    one_class_svm_with_hpo_results.print_results()
    print("** Two Class RF results **")
    two_class_rf_results.print_results()
    print("** Two Class RF with HPO results ** ")
    two_class_rf_with_hpo_results.print_results()

def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    print("\n" + "-" * 20 + " CMU: First data split " + "-" * 20)
    print("The first session of each user is used for training and the other sessions are used for testing.")
    
    first_session_split = lambda df: (df[df['sessionIndex'] == 1], df[df['sessionIndex'] != 1])
    cmu_database_1 = CMUDatabase('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    run_experiments(cmu_database_1)

    print("\n" + "-" * 20 + " CMU: Second data split " + "-" * 20)
    print("The last session of each user is used for testing and the other sessions are used for training.")

    second_session_split = lambda df: (df[df['sessionIndex'] != 8], df[df['sessionIndex'] == 8])
    cmu_database_2 = CMUDatabase('datasets/cmu/DSL-StrongPasswordData.csv', second_session_split)
    run_experiments(cmu_database_2)

if __name__ == '__main__':
    main()