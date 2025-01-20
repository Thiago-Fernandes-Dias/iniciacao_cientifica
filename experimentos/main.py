from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.svm import OneClassSVM
import os


from cmu import *
from one_class_experiment import *
from two_class_experiment import *

def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    cmu_database = CMUDatabase('datasets/cmu/DSL-StrongPasswordData.csv')
    
    one_class_svm_experiment = OneClassExperiment(cmu_database=cmu_database, estimator_factory=lambda: OneClassSVM())
    one_class_svm_results = one_class_svm_experiment.exec()
    
    one_class_svm_params_grid = [
        {
            'kernel': ['poly'],
            'degree': range(1, 4),
            'gamma': ['scale', 'auto', .1, .01, .001],
            'coef0': float_range(0, 5, .25),
            # Causando falhas em alguns fits -> 'nu': [.1, .25, .5, .75, 1], 
            'cache_size': [4096]
        },
        {
            'kernel': ['linear'],
            'cache_size': [4096],
        },
        {
            'kernel': ['sigmoid'],
            'gamma': ['scale', 'auto', .1, .01, .001],
            'coef0': float_range(0, 5, .25),
            # 'nu': [.1, .25, .5, .75, 1], 
            'cache_size': [4096]
        },
        {
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto', .1, .01, .001],
            # 'nu': [.1, .25, .5, .75, 1], 
            'cache_size': [4096]
        }
    ]
    one_vs_one_cv = KFold(n_splits=5)
    one_vs_one_gs_factory = lambda: GridSearchCV(OneClassSVM(), one_class_svm_params_grid, 
                                scoring='accuracy', cv=one_vs_one_cv, n_jobs=-1)
    one_class_svm_with_hpo_experiment = OneClassExperiment(cmu_database=cmu_database, estimator_factory=one_vs_one_gs_factory)
    one_class_svm_with_hpo_results = one_class_svm_with_hpo_experiment.exec()

    two_class_experiment = TwoClassExperiment(cmu_database=cmu_database, estimator_factory=lambda: RandomForestClassifier())
    two_class_rf_results = two_class_experiment.exec()

    one_vs_rest_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_params_grid = [
        {
            'n_estimators': [50, 100, 150],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': ['sqrt', 'log2', None, 10, 20, 31],
            'bootstrap': [True, False],
            'n_jobs': [-1],
            'random_state': [RANDOM_STATE],
            'warm_start': [True, False]
        }
    ]
    one_vs_rest_gs_factory = lambda: GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_params_grid, 
                                cv=one_vs_rest_cv, n_jobs=-1, scoring='accuracy')
    two_class_experiment = TwoClassExperiment(cmu_database=cmu_database, estimator_factory=one_vs_rest_gs_factory)
    two_class_rf_with_hpo_results = two_class_experiment.exec()

    print("\n" + "-" * 20 + " Experiment results " + "-" * 20)
    print("** One Class SVM results **")
    one_class_svm_results.print_results()
    print("** One Class SVM with HPO results ** ")
    one_class_svm_with_hpo_results.print_results()
    print("** Two Class RF results **")
    two_class_rf_results.print_results()
    print("** Two Class RF with HPO results ** ")
    two_class_rf_with_hpo_results.print_results()

if __name__ == '__main__':
    main()