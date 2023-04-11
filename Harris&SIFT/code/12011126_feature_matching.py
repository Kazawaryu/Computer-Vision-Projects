import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    matches = []
    confidences = []

    for i in range(features1.shape[0]):
        dist = np.sqrt(np.square(np.subtract(features1[i, :], features2)).sum(axis=1))
        idx = np.argsort(dist)

        if dist[idx[0]] / dist[idx[1]] < 0.8:
            matches.append([i, idx[0]])
            confidences.append(1.0 - dist[idx[0]] / dist[idx[1]])

    return np.asarray(matches), np.asarray(confidences)
