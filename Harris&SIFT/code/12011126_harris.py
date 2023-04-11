import math
import cv2
import numpy as np
from numpy import Infinity


def get_interest_points(image, feature_width):
    confidences, scales, orientations = None, None, None
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    ix2 = Ix ** 2
    iy2 = Iy ** 2
    ixy = Ix * Iy
    Ix2 = cv2.GaussianBlur(ix2, (3, 3), 0, 0)
    Iy2 = cv2.GaussianBlur(iy2, (3, 3), 0, 0)
    Ixy = cv2.GaussianBlur(ixy, (3, 3), 0, 0)
    R = (Ix2 * Iy2 - Ixy ** 2) - 0.04 * ((Ix2 + Iy2) ** 2)

    rank = []
    H, W = image.shape
    jud = R.max()
    for y in range(feature_width, H - feature_width - 1):
        for x in range(feature_width, W - feature_width - 1):
            if R[y, x] > 0.01 * jud:
                rank.append((R[y, x], (y, x)))

    rank = sorted(rank, key=lambda x: x[0])
    lens = 1
    choices = [(-Infinity, rank[0][1])]
    maxs = rank[0]
    while lens < len(rank):
        if rank[lens][0] > 0.9 * maxs[0]:
            choices.append((-Infinity, rank[lens][1]))
            lens += 1
        else:
            nows = rank[lens]
            lens += 1
            while lens < len(rank):
                if rank[lens][0] > 0.9 * nows[0]:
                    dis = math.sqrt((rank[lens][1][0] - nows[1][0]) ** 2 + (rank[lens][1][1] - nows[1][1]) ** 2)
                    choices.append((-dis, rank[lens][1]))
                    lens += 1
                    break
                else:
                    lens += 1

    choices = sorted(choices, key=lambda x: x[0])

    x = []
    y = []
    for i in range(2000):
        if i == len(choices): break
        y.append(choices[i][1][0])
        x.append(choices[i][1][1])

    return np.array(x), np.array(y), confidences, scales, orientations


def ANMS(x, y, r, maximum):
    i = 0
    j = 0
    NewList = []
    while i < len(x):
        minimum = 1000000000000

        X, Y = x[i], y[i]
        while j < len(x):
            CX, CY = x[j], y[j]
            if (X != CX and Y != CY) and r[i] < r[j]:
                distance = math.sqrt((CX - X) ** 2 + (CY - Y) ** 2)
                if distance < minimum:
                    minimum = distance
            j = j + 1
        NewList.append([X, Y, minimum])
        i = i + 1
        j = 0
    NewList.sort(key=lambda t: t[2])
    NewList = NewList[len(NewList) - maximum:len(NewList)]
    return NewList
