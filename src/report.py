"""JSON analysis report."""
import json
import numpy as np
from .synthetic import CLASS_NAMES


def build_report(res, f_before, f_after, gsd, technique_log):
    px_ha = gsd * gsd / 1e4
    area = lambda L: {n: round(float((L == c).sum() * px_ha), 2) for c, n in enumerate(CLASS_NAMES)}
    trans = {f"{CLASS_NAMES[i]} -> {CLASS_NAMES[j]}": round(float(res["transitions"][i, j] * px_ha), 2)
             for i in range(4) for j in range(4) if i != j and res["transitions"][i, j] > 0}
    return dict(
        ground_sample_distance_m=gsd,
        estimated_shift_px=[round(res["shift"][0], 2), round(res["shift"][1], 2)],
        area_ha_date1=area(res["labels_before"]), area_ha_date2=area(res["labels_after"]),
        changed_area_ha=round(float(res["change_mask"].sum() * px_ha), 2),
        class_transitions_ha=dict(sorted(trans.items(), key=lambda kv: -kv[1])),
        buildings=dict(date1=len(f_before["buildings"]), date2=len(f_after["buildings"])),
        road_area_ha=dict(date1=round(float(f_before["roads_mask"].sum() * px_ha), 2),
                          date2=round(float(f_after["roads_mask"].sum() * px_ha), 2)),
        hough_road_lines_raw=dict(date1=len(f_before["road_segments"]), date2=len(f_after["road_segments"]),
                                  note="overlapping Hough detections of the same road; not a road count"),
        techniques=technique_log)


def save_json(rep, path):
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
