from utils import float_range
from constants import *

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