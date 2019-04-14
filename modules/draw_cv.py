import cv2
import numpy as np


def dotline(img, pt1, pt2, color, thickness=1, gap=20):
    dist = ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i/dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    for p in pts:
        cv2.circle(img, p, thickness, color, -1)


def polydotline(img, pts, color, thickness=1):
    for p in pts:
        cv2.circle(img, p, thickness, color, -1)
