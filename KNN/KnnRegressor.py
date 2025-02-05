import matplotlib.pyplot as plt
import numpy as np
from distance_functions import DISTANCE_FUNCTIONS


class KnnRegressor:
    def __init__(self, k, dist="euclidean"):
        self.k = k
        self.X = None
        self.y = None
        self.dist_function = dist
        # self.colors = np.array(["black", "r", "g", "b"])

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def single_predict(self, x, verbose=False):
        if (self.X is None) or (self.y is None):
            raise Exception("Model has not been fitted yet.")

        distances = list(map(lambda v: DISTANCE_FUNCTIONS[self.dist_function](x, v), self.X))
        neighbours = np.argsort(distances)
        knn_idx = neighbours[:self.k]
        nearest_neighbours = self.y[knn_idx]
        prediction = nearest_neighbours.mean()

        if verbose:
            print(f"================================\n"
                  f"k={self.k}\n"
                  f"x={x}\n"
                  f"values={self.y[knn_idx]}\n"
                  f"nearest_neighbours={self.X[knn_idx]}\n"
                  f"prediction={prediction}\n"
                  f"================================\n")

        return prediction

    def predict(self, X, plot=False, verbose=False):
        predictions = []
        for x in X:
            prediction = self.single_predict(x, verbose)
            predictions.append(prediction)
            if plot:
                self.plot(x, prediction)
        if plot:
            plt.show()

        return predictions

    def plot(self, x, y):
        pass
