import numpy as np
from skimage import filters


def get_features(image, x, y, feature_width, scales=None):
    H, W = image.shape
    features = np.zeros((len(x), 4, 4, 8))
    filtered_image = filters.gaussian(image, sigma=0.1)

    G = np.sqrt(np.add(np.square(filters.sobel_v(filtered_image)), np.square(filters.sobel_h(filtered_image))))
    D = np.arctan2(filters.sobel_h(filtered_image), filters.sobel_v(filtered_image))
    D[D < 0] += 2 * np.pi

    for cnt in range(len(x)):
        r_range = (y[cnt] - int(feature_width / 2), y[cnt] + int(feature_width / 2) + 1)
        c_range = (x[cnt] - int(feature_width / 2), x[cnt] + int(feature_width / 2) + 1)
        num_p = filters.gaussian(G[r_range[0]:r_range[1], c_range[0]:c_range[1]], sigma=0.4)
        dir_p = filters.gaussian(D[r_range[0]:r_range[1], c_range[0]:c_range[1]], sigma=0.4)

        for i in range(int(feature_width / 4)):
            for j in range(int(feature_width / 4)):
                now_num_p = num_p[i * int(feature_width / 4): (i + 1) * int(feature_width / 4),
                            j * int(feature_width / 4):(j + 1) * int(feature_width / 4)]
                now_dir_p = dir_p[i * int(feature_width / 4): (i + 1) * int(feature_width / 4),
                            j * int(feature_width / 4):(j + 1) * int(feature_width / 4)]
                features[cnt, i, j] = np.histogram(now_dir_p.reshape(-1), bins=8, range=(0, 2 * np.pi),
                                                   weights=now_num_p.reshape(-1))[0]

    features = features.reshape((len(x), -1,))
    features = features / np.sqrt(np.square(features).sum(axis=1)).reshape(-1, 1)
    features[features >= 0.2] = 0.2
    features = features / np.sqrt(np.square(features).sum(axis=1)).reshape(-1, 1)

    return features
