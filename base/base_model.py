from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass

    @abstractmethod
    def get_score(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def set_params(self, params: dict):
        pass

    @abstractmethod
    def get_params(self):
        pass
