import cv2
import numpy as np


# Simple COCO-style 17-keypoint skeleton used by many models
COCO_EDGES = [
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
    (5, 6), (11, 12), # shoulders and hips
    (5, 11), (6, 12)  # torso diagonals
]


def draw_keypoints(frame, keypoints, thresh=0.3, color=(0, 255, 0)):
    """keypoints: (N, 3) array [x, y, score] in image coords."""
    img = frame.copy()
    h, w = img.shape[:2]
    pts = []
    for x, y, s in keypoints:
        if s < thresh:
            pts.append(None)
            continue
        cx, cy = int(x), int(y)
        cv2.circle(img, (cx, cy), 3, color, -1)
        pts.append((cx, cy))

    for i, j in COCO_EDGES:
        if i < len(pts) and j < len(pts):
            p1, p2 = pts[i], pts[j]
            if p1 is not None and p2 is not None:
                cv2.line(img, p1, p2, (255, 0, 0), 2)

    return img
