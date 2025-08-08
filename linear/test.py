from sklearn.datasets import make_regression
from linear_regression import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def plot_regression(X, y, noise, ax):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    # Predictions for extreme points to plot a line running through the entire data
    y_0 = lin_reg.predict(np.array([X.min()]))
    y_1 = lin_reg.predict(np.array([X.max()]))

    ax.scatter(X, y, s=5, c="green")
    ax.plot([X.min(), X.max()], [y_0, y_1], c="black")
    ax.set_title(f"noise={noise}")

noises = [10.0, 50.0, 150.0]
fig, axes = plt.subplots(1, len(noises), figsize=(15, 5), sharey=True)

for i in range(len(noises)):
    # Mock data
    X, y = make_regression(n_samples=200,
                           n_features=1,
                           n_informative=1,
                           n_targets=1,
                           noise=noises[i])
    plot_regression(X, y, noises[i], axes[i])

plt.show()
