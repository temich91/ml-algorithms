import numpy as np


def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def minkowski_distance(vec1, vec2, p):
    return np.power(np.sum((vec1 - vec2) ** p), 1/p)


def cosine_distance(vec1, vec2):
    return np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


DISTANCE_FUNCTIONS = {"euclidean": euclidean_distance,
                      "minkowski": minkowski_distance,
                      "cosine": cosine_distance}
