import numpy as np
from base import BaseModel

class LinearRegression(BaseModel):
    def __init__(self):
        super().__init__()
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Computes weights vector using analytical solution.

        Pseudo-inverting of matrix is applied so the solution is suitable for singular matrix as well.
        w = (X.T @ X).pinv @ X.T @ y

        Args:
            X: Features matrix.
            y: Target matrix.

        Returns:
            Fitted instance of LinearRegression.
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray):
        """Predicts target by multiplying the feature matrix by the calculated weights.

        Args:
            X: Feature matrix.

        Returns:
            Predicted target matrix.
        """
        if not self._is_fitted:
            raise Exception("Model was not fitted.")
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return X @ self.w

    def get_score(self, X: np.ndarray, y: np.ndarray):
        if not self._is_fitted:
            raise Exception("Model was not fitted.")
        pass

    def set_params(self, params: dict):
        """Sets parameters to model.

        If there is unknown parameter the exception is thrown.
        Args:
            params: Dictionary of parameters to be set.

        Returns:
            Model with updated parameters.
        """
        valid_params = self.get_params()
        for param_name, value in params.items():
            if param_name not in valid_params:
                raise ValueError(f"Unknown parameter `{param_name}`.")
            self.__setattr__(param_name, value)


    def get_params(self) -> dict:
        return {"w": self.w}
