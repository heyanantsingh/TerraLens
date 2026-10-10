"""Object-level analysis on the built-up mask: buildings (blob analysis) and
roads (thin structures + probabilistic Hough transform)."""
import numpy as np
import cv2
from .synthetic import BUILT


def extract_built_features(labels, min_building_area=60, open_ksize=9, min_extent=0.8):
    built = (labels == BUILT).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    buildings_mask = cv2.morphologyEx(built, cv2.MORPH_OPEN, k)      # wide blobs survive
    roads_mask = cv2.morphologyEx(built & (1 - buildings_mask), cv2.MORPH_OPEN,
                                  np.ones((3, 3), np.uint8))          # thin remainder
    n, _, stats, cent = cv2.connectedComponentsWithStats(buildings_mask, connectivity=8)
    buildings = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        extent = a / float(w * h)            # ~1 for rectangles, ~0.65 for a road junction
        if a >= min_building_area and extent >= min_extent:
            buildings.append(dict(bbox=[int(x), int(y), int(w), int(h)], area_px=int(a), extent=round(float(extent), 3),
                                  centroid=[float(cent[i][0]), float(cent[i][1])]))
    lines = cv2.HoughLinesP(roads_mask * 255, 1, np.pi / 180, threshold=60,
                            minLineLength=60, maxLineGap=12)
    segs = [] if lines is None else [tuple(map(int, l[0])) for l in lines]
    horiz = sum(1 for x1, y1, x2, y2 in segs if abs(x2 - x1) >= abs(y2 - y1))
    return dict(buildings=buildings, buildings_mask=buildings_mask, roads_mask=roads_mask,
                road_segments=segs, road_h=horiz, road_v=len(segs) - horiz)
