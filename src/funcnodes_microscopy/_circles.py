import numpy as np
import cv2


def calculateMaxCircleDiameters(
    conts: list, cents: list, pixel_size: float = 1.0
) -> list:
    """
    Calculate the maximum incircle diameters and the corresponding points and store
    them as a list in thes maxCircleDiameters and maxCirclePoints. The points are stored in a way so
    that they can easily be drawn with the cv2.drawContours() method. It also returns both lists.
    :return: List of maximum incircle diameters and the coordinates of the corresponding points
    """
    max_circle_d = []
    max_circle_x = []
    max_circle_y = []
    max_circle_r = []

    number = len(conts)
    for i in range(0, number):
        contour = conts[i]
        center = cents[i]
        Y = center[0]
        X = center[1]
        c = np.array((Y, X), dtype=float)
        dist = []
        for j in range(len(contour)):
            b = contour[j][0]
            dist.append(np.linalg.norm(c - b))
        radius = np.min(dist)
        if radius == 0:
            radius = 1
        max_circle_x.append(Y)
        max_circle_y.append(X)
        max_circle_r.append(radius)
        max_circle_d.append(radius * 2 * pixel_size)

    return max_circle_x, max_circle_y, max_circle_r, max_circle_d


def calculateMinCircleDiameters(conts: list, pixel_size: float = 1.0) -> list:
    """
    Calculate the minimum enclosed circle diameters and the corresponding points and store
    them as a list. The points are stored in a way so
    that they can easily be drawn with the cv2.drawContours() method. It also returns both lists.
    :return: List of minimum enclosed circle diameters and the coordinates of the corresponding points
    """
    min_circle_d = []
    min_circle_x = []
    min_circle_y = []
    min_circle_r = []

    number = len(conts)
    for i in range(0, number):
        contour = conts[i]
        (x, y), radius = cv2.minEnclosingCircle(contour)
        min_circle_x.append(x)
        min_circle_y.append(y)
        min_circle_r.append(radius)
        min_circle_d.append(radius * 2 * pixel_size)

    return min_circle_x, min_circle_y, min_circle_r, min_circle_d


def calculateCircularityScores(conts: list) -> list:
    """
    Calculates the circularity scores for each contour.
    Circularity is defined as (4 * π * Area) / Perimeter^2.
    :return: List of circularity scores for each contour
    """
    circularity_score = []
    for contour in conts:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            circularity_score.append(0)
        else:
            circularity_score.append((4 * np.pi * area) / (perimeter**2))
    return circularity_score
