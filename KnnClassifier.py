import matplotlib.pyplot as plt
import numpy as np
from distance_functions import DISTANCE_FUNCTIONS


class KnnClassifier:
    def __init__(self, k, dist="euclidean"):
        self.k = k
        self.X = None
        self.y = None
        self.pred_result = None
        self.distances = None
        self.dist = dist
        self.colors = np.array(["black", "r", "g", "b"])

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def single_predict(self, x, verbose=False):
        if (self.X is None) or (self.y is None):
            raise Exception("Model has not been fitted yet.")

        self.distances = list(map(lambda v: DISTANCE_FUNCTIONS[self.dist](x, v), self.X))
        neighbours = np.argsort(self.distances)
        knn_idx = neighbours[:self.k]
        result = self.y[[knn_idx]].mean()
        prediction = int(np.rint(result))

        if verbose:
            print(f"================================\n"
                  f"k={self.k}\n"
                  f"x={x}\n"
                  f"mean class= {result}\n"
                  f"classes={self.colors[self.y[knn_idx]]}\n"
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

        return self.colors[predictions] # return predictions for class numbers

    def plot(self, x, y):
        plt.scatter(self.X.T[0], self.X.T[1], color=self.colors[self.y])
        plt.scatter(x[0], x[1], marker="x", color=self.colors[y], s=100)
