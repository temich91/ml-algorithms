import matplotlib.pyplot as plt
import numpy as np
from distance_functions import DISTANCE_FUNCTIONS

plt.set_cmap("Set1")
cmap = plt.get_cmap("Set1")
COLORS = np.linspace(0, 1, 20)


class KnnClassifier:
    def __init__(self, k, dist="euclidean"):
        self.k = k
        self.X = None
        self.y = None
        self.dist_function = dist

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
        prediction = np.bincount(nearest_neighbours).argmax()


        if verbose:
            print(f"================================\n"
                  f"k={self.k}\n"
                  f"x={x}\n"
                  f"classes={self.y[knn_idx]}\n"
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

        return predictions # predicted class numbers

    def plot(self, x, y):
        plt.scatter(self.X.T[0], self.X.T[1], c=COLORS[self.y])
        plt.scatter(x[0], x[1], marker="x", s=100, c=COLORS[y])