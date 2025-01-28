import numpy as np
from KnnClassifier import KnnClassifier

x_train = np.array([[-2, 0], [1, 3], [3, 0], [4, -1], [1, -1], [1, 1], [-1, -1],
                    [4, 3], [2, -1], [-2, 4], [-1, 2], [5, 4], [5, 3], [3, 5]])
y_train = np.array([1, 1, 2, 2, 2, 2, 1, 3, 2, 1, 1, 3, 3, 3])
x_test = np.array([[2.5, 5], [1, 3], [0, 0]])

clf = KnnClassifier(1, "cosine")
clf.fit(x_train, y_train)
prediction = clf.predict(x_test, plot=True, verbose=True)
print(prediction)
