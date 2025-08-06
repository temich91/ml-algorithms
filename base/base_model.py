from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def set_params(self):
        pass

    @abstractmethod
    def get_params(self):
        pass
