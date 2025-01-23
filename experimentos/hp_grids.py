import numpy as np

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

mlp_params_grid = [
    {
        'hidden_layer_sizes': [(100,), (100, 100), (50,), (50,50,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['adam', 'sgd', 'lbfgs'],
        # 'alpha': np.logspace(-5, 3, 5),
        # 'batch_size': ['auto', 32, 64, 128, 256],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        # 'learning_rate_init': np.logspace(-5, 3, 5),
        # 'power_t': [0.5, 0.33, 0.25],
        'max_iter': [200, 500, 800],
        'shuffle': [True, False],
        'random_state': [RANDOM_STATE],
        'warm_start': [True, False],
        'momentum': np.linspace(0, 1, 10),
        # 'nesterovs_momentum': [True, False],
        # 'beta_1': np.linspace(0, 1, 10, endpoint=False),
        # 'beta_1': np.linspace(0, 1, 10, endpoint=False),
    }
]