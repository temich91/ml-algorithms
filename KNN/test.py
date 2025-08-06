import numpy as np
from KnnClassifier import KnnClassifier
from KnnRegressor import KnnRegressor

CLF_DATA = {
    "x_train": np.array([[-2, 0], [1, 3], [3, 0], [4, -1], [1, -1], [1, 1], [-1, -1],
                        [4, 3], [2, -1], [-2, 4], [-1, 2], [5, 4], [5, 3], [3, 5]]),
    "y_train": np.array([1, 1, 2, 2, 2, 2, 1, 3, 2, 1, 1, 3, 3, 3]),
    "x_test": np.array([[2.5, 5], [1, 3], [0, 1]])
}

REG_DATA = {
    "x_train": np.arange(-3, 5, 0.5),
    "y_train": 2 * np.arange(-3, 5, 0.5) + np.random.normal(0, 1, 16),
    "x_test": np.array([2.5, 0.9, -0.4])
}


def classification():
    x_train = CLF_DATA["x_train"]
    y_train = CLF_DATA["y_train"]
    x_test = CLF_DATA["x_test"]

    clf = KnnClassifier(1, "cosine")
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test, plot=True, verbose=True)
    print(prediction)


def regression():
    x_train = REG_DATA["x_train"]
    y_train = REG_DATA["y_train"]
    x_test = REG_DATA["x_test"]

    reg = KnnRegressor(4)
    reg.fit(x_train, y_train)
    prediction = reg.predict(x_test, plot=True)
    print(prediction)


regression()
# classification()