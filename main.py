"""TerraLens CLI.

  python main.py --demo                       # built-in synthetic scene pair
  python main.py --before a.png --after b.png --gsd 10
"""
import argparse, os, sys
import cv2
from src.synthetic import make_pair
from src.change import detect_change
from src.features import extract_built_features
from src import figures
from src.report import build_report, save_json

TECHNIQUES = ["Bilateral filtering", "ECC image registration (translation)",
              "Radiometric normalisation (robust mean/std)", "K-Means clustering in CIELAB",
              "Colour-rule cluster naming", "Majority (mode) filtering",
              "Morphological opening + connected components", "Probabilistic Hough transform",
              "Lab colour-difference (Delta-E) + Otsu thresholding"]


def load(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"error: cannot read image '{path}'")
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description="TerraLens - classical-CV land-cover & change analyzer")
    ap.add_argument("--demo", action="store_true", help="use built-in synthetic scene pair")
    ap.add_argument("--before"); ap.add_argument("--after")
    ap.add_argument("--gsd", type=float, default=10.0, help="ground sample distance, metres/pixel")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    if a.demo:
        d = make_pair(a.seed); before, after = d["before"], d["after"]
        cv2.imwrite("input/demo_before.png", before); cv2.imwrite("input/demo_after.png", after)
    elif a.before and a.after:
        before, after = load(a.before), load(a.after)
        if after.shape != before.shape:
            after = cv2.resize(after, (before.shape[1], before.shape[0]))
    else:
        ap.error("give --demo or both --before and --after")
    os.makedirs(a.out, exist_ok=True)
    res = detect_change(before, after)
    fb = extract_built_features(res["labels_before"]); fa = extract_built_features(res["labels_after"])
    figures.fig_classification(before, res["after_aligned"], res["labels_before"], res["labels_after"], f"{a.out}/classification.png")
    figures.fig_features(before, fb, f"{a.out}/features_date1.png", "Date 1")
    figures.fig_features(res["after_aligned"], fa, f"{a.out}/features_date2.png", "Date 2")
    figures.fig_change(before, res["after_aligned"], res, f"{a.out}/change.png")
    figures.fig_area(res["labels_before"], res["labels_after"], a.gsd, f"{a.out}/area.png")
    rep = build_report(res, fb, fa, a.gsd, TECHNIQUES)
    save_json(rep, f"{a.out}/analysis_report.json")
    print(f"Done. Changed area: {rep['changed_area_ha']} ha | buildings {rep['buildings']['date1']} -> "
          f"{rep['buildings']['date2']} | estimated shift {rep['estimated_shift_px']} px\nOutputs in '{a.out}/'")
    return rep


if __name__ == "__main__":
    main()
