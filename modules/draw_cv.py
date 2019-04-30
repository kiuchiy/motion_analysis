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
        cv2.circle(img, (p[0], p[1]), thickness, color, -1)


# def plot_graph(img, pts, color, rows, num):
#     img_size = img.shape
#     rows = rows if rows > 4 else 5
#     graph_h = int(img_size[1] / (rows + 0.1))
#     graph_w = int(graph_h * 1.5)
#     margin_h = int(graph_h / 10)
#     margin_w = int(img_size[0]/40)
#     edge_r = img_size[0] - margin_w
#     edge_b = img_size[1] - margin_h
#
#     for i in range(rows):
#         cv2.rectangle(img, ((edge_r - graph_w), edge_b - int(i) * graph_w),
#                       (edge_r, edge_b - graph_h), (255, 255, 255), 3)
