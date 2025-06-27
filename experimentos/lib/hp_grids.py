from lib.constants import *
from lib.utils import float_range

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
        'n_jobs': [N_JOBS],
        'random_state': list(range(1, 31)),
        'warm_start': [True, False]
    }
]

st_params_grid = {
    'threshold': [0.7, 0.6, 0.8, 0.9]
}

mlp_params_grid = [
    {
        'hidden_layer_sizes': [(100,), (100, 100), (50,), (50, 50,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd'],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        'max_iter': [200, 500, 800],
        'shuffle': [True, False],
        'random_state': list(range(1, 31)),
        'momentum': float_range(0, 1, 0.25) + [0.9],
    },
    {

        'hidden_layer_sizes': [(100,), (100, 100), (50,), (50, 50,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['adam'],
        'max_iter': [200, 500, 800],
        'shuffle': [True, False],
        'random_state': list(range(1, 31)),
    },
    {
        'hidden_layer_sizes': [(100,), (100, 100), (50,), (50, 50,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs'],
        'max_iter': [200, 500, 800],
        'random_state': list(range(1, 31)),
        'warm_start': [True, False],
    }
]
