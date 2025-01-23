from sklearn.model_selection import BaseSearchCV

class BaseSearchCVAdapter(BaseSearchCV):
    wrapped_estimator: BaseSearchCV

    def __init__(self, wrapped_estimator: BaseSearchCV) -> None:
        self.wrapped_estimator = wrapped_estimator
    
    def get_params(self, deep: bool =True):
        if self.wrapped_estimator.best_params_ is not None:
            return best_params_
        return self.wrapped_estimator.get_params(deep)
